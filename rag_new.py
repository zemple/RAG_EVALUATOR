import os
import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
from langchain.embeddings.base import Embeddings
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from datasets import load_dataset
from rouge_score import rouge_scorer
from huggingface_hub import login
from langchain.prompts import PromptTemplate
from langchain_huggingface.llms import HuggingFacePipeline
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from typing import Union
import nltk
import faiss

nltk.download('punkt')
nltk.download('punkt_tab')

MODEL_NAME = "google/flan-t5-xl"
USE_GPU = True

@dataclass
class ModelConfig:
    """Configuration for the model and tokenizer"""
    model_name: str = MODEL_NAME
    max_length: int = 256
    device: int = 0 if USE_GPU and torch.cuda.is_available() else -1
    hf_token: str = None

class RAGDataset(Dataset):
    """Dataset for training the retriever"""
    def __init__(self, questions: List[str], contexts: List[str], 
                 positive_indices: List[int], negative_indices: List[List[int]],
                 num_negatives: int = 5):
        # Validate data alignment
        assert len(questions) == len(positive_indices) == len(negative_indices), \
            "Mismatched lengths in dataset components"
        
        self.questions = questions
        self.contexts = contexts
        self.positive_indices = positive_indices
        self.negative_indices = negative_indices
        self.num_negatives = num_negatives
    
    def __len__(self):
        """Return the number of items in the dataset"""
        return len(self.questions)
    
    def __getitem__(self, idx):
        question = self.questions[idx]
        pos_idx = self.positive_indices[idx]
        neg_indices = self.negative_indices[idx]
        
        # Ensure we have exactly num_negatives negative examples
        if len(neg_indices) < self.num_negatives:
            # Sample additional negative examples if needed
            available_indices = list(range(len(self.contexts)))
            available_indices.remove(pos_idx)
            for i in neg_indices:
                if i in available_indices:
                    available_indices.remove(i)
            additional_neg = np.random.choice(
                available_indices,
                size=self.num_negatives - len(neg_indices),
                replace=False
            )
            neg_indices = neg_indices + list(additional_neg)
        elif len(neg_indices) > self.num_negatives:
            # Subsample if we have too many negatives
            neg_indices = list(np.random.choice(neg_indices, size=self.num_negatives, replace=False))
        
        positive_context = self.contexts[pos_idx]
        negative_contexts = [self.contexts[i] for i in neg_indices]
        
        return {
            'question': question,
            'positive_context': positive_context,
            'negative_contexts': negative_contexts
        }

class TrainableEmbeddings(nn.Module, Embeddings):
    """Enhanced embeddings model with fine-tuning capabilities"""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", device: Optional[str] = None):
        # Initialize nn.Module
        nn.Module.__init__(self)
        # Initialize device
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name

        print(f"Initializing embeddings model on {self.device}")
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

        # Projection layer for embedding enhancement
        self.projection = nn.Linear(
            self.model.config.hidden_size,
            self.model.config.hidden_size
        ).to(self.device)

    def tokenize_batch(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize a batch of texts"""
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

    def forward(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """Forward pass with proper input handling"""
        if isinstance(texts, str):
            texts = [texts]

        # Move inputs to device
        inputs = self.tokenize_batch(texts)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Use CLS token embeddings
        embeddings = outputs.last_hidden_state[:, 0, :]

        # Project embeddings
        projected = self.projection(embeddings)

        # Normalize
        return F.normalize(projected, p=2, dim=1)

    def encode(self, texts: List[str], batch_size: int = 32, **kwargs) -> np.ndarray:
        """Encode texts into embeddings with batching"""
        self.eval()  # Set model to evaluation mode
        all_embeddings = []

        # Process batches on GPU
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            with torch.no_grad():
                # This runs on GPU
                embeddings = self.forward(batch_texts)
                # Only move to CPU for numpy conversion at the end
                all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Interface for FAISS document embedding"""
        embeddings = self.encode(texts)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Interface for FAISS query embedding"""
        embedding = self.encode([text])
        return embedding[0].tolist()


class EnhancedDocumentProcessor:
    """Enhanced document processor with trainable components"""
    def __init__(self, chunk_size=384, chunk_overlap=50):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        self.embeddings_model = TrainableEmbeddings()

    def train_embeddings(self, train_dataset: RAGDataset, 
                    val_dataset: Optional[RAGDataset] = None,
                    num_epochs: int = 3,
                    batch_size: int = 16,
                    learning_rate: float = 2e-5):
        """Train the embeddings model"""
        device = self.embeddings_model.device
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            drop_last=True,  # Avoid issues with last incomplete batch
            collate_fn=self._collate_fn  # Add custom collate function
        )
        
        optimizer = torch.optim.AdamW(
            self.embeddings_model.parameters(),
            lr=learning_rate
        )
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=len(train_loader) * num_epochs
        )
        
        for epoch in range(num_epochs):
            self.embeddings_model.train()
            total_loss = 0
            valid_batches = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
                try:
                    optimizer.zero_grad()
                    
                    questions = batch['questions']
                    pos_contexts = batch['positive_contexts']
                    neg_contexts = batch['negative_contexts']
                    
                    # Get embeddings
                    q_embeddings = self.embeddings_model(questions)
                    pos_embeddings = self.embeddings_model(pos_contexts)
                    neg_embeddings = self.embeddings_model(neg_contexts)
                    
                    # Compute similarity scores
                    pos_scores = torch.sum(q_embeddings * pos_embeddings, dim=1)
                    neg_scores = torch.matmul(q_embeddings, neg_embeddings.t())
                    
                    # Compute loss using InfoNCE
                    temperature = 0.1
                    logits = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1) / temperature
                    labels = torch.zeros(len(logits), dtype=torch.long, device=device)
                    
                    loss = F.cross_entropy(logits, labels)
                    loss.backward()
                    
                    optimizer.step()
                    scheduler.step()
                    
                    total_loss += loss.item()
                    valid_batches += 1
                
                except Exception as e:
                    print(f"Error processing batch: {str(e)}")
                    continue
            
            if valid_batches > 0:
                avg_loss = total_loss / valid_batches
                print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")
            else:
                print(f"Epoch {epoch + 1}: No valid batches processed")

    @staticmethod
    def _collate_fn(batch):
        """Custom collate function to handle batches"""
        questions = [item['question'] for item in batch]
        positive_contexts = [item['positive_context'] for item in batch]
        # Ensure all items have the same number of negative contexts
        negative_contexts = [item['negative_contexts'][0] for item in batch]  # Take first negative only
        
        return {
            'questions': questions,
            'positive_contexts': positive_contexts,
            'negative_contexts': negative_contexts
        }
    
    def process_dataset(self, dataset_name="microsoft/ms_marco", version="v2.1", 
                       split="train[:100000]"):
        """Process dataset and create trainable vectorstore"""
        ds = load_dataset(dataset_name, version, split=split)
        
        documents = []
        questions = []
        contexts = []
        positive_indices = []
        negative_indices = []
        
        for idx, item in enumerate(tqdm(ds, desc="Processing dataset")):
            passages = item['passages']['passage_text']
            is_selected = item['passages']['is_selected']
            
            # Find positive passages first
            pos_passages = [(i, p) for i, (p, s) in enumerate(zip(passages, is_selected)) if s == 1]
            
            if pos_passages:  # Only process items with at least one positive passage
                # Add question
                questions.append(item['query'])
                
                # Process positive passage
                pos_idx, pos_passage = pos_passages[0]  # Take first positive passage
                pos_text = self._preprocess_text(pos_passage)
                contexts.append(pos_text)
                current_pos_idx = len(contexts) - 1
                positive_indices.append(current_pos_idx)
                
                # Add document for positive passage
                documents.append(Document(
                    page_content=pos_text,
                    metadata={
                        "id": idx,
                        "passage_idx": pos_idx,
                        "source": "ms_marco",
                        "query": item['query'],
                        "is_positive": True
                    }
                ))
                
                # Process negative passages
                current_neg_indices = []
                for i, (passage, selected) in enumerate(zip(passages, is_selected)):
                    if not selected:  # Only process negative passages
                        neg_text = self._preprocess_text(passage)
                        contexts.append(neg_text)
                        current_neg_idx = len(contexts) - 1
                        current_neg_indices.append(current_neg_idx)
                        
                        # Add document for negative passage
                        documents.append(Document(
                            page_content=neg_text,
                            metadata={
                                "id": idx,
                                "passage_idx": i,
                                "source": "ms_marco",
                                "query": item['query'],
                                "is_positive": False
                            }
                        ))
                
                # Sample negative passages (up to 5)
                sampled_neg = current_neg_indices[:5] if current_neg_indices else []
                negative_indices.append(sampled_neg)
        
        print(f"Processed examples:")
        print(f"Questions: {len(questions)}")
        print(f"Contexts: {len(contexts)}")
        print(f"Positive indices: {len(positive_indices)}")
        print(f"Negative indices: {len(negative_indices)}")
        
        if not questions:
            raise ValueError("No valid examples found in dataset")
            
        # Verify lengths match
        assert len(questions) == len(positive_indices) == len(negative_indices), \
            f"Mismatched lengths: questions={len(questions)}, positive={len(positive_indices)}, negative={len(negative_indices)}"
        
        # Create training dataset
        train_dataset = RAGDataset(
            questions=questions,
            contexts=contexts,
            positive_indices=positive_indices,
            negative_indices=negative_indices
        )
        
        # Train embeddings model
        print("Training embeddings model...")
        self.train_embeddings(train_dataset)
        
        self.embeddings_model.eval()
        print("Creating vectorstore...")
        try:
            vectorstore = FAISS.from_documents(
                documents,
                self.embeddings_model,
                normalize_L2=True
            )
            return vectorstore
        except Exception as e:
            print(f"Error creating vectorstore: {str(e)}")
            raise
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize text"""
        text = text.strip()
        text = " ".join(text.split())
        return text

# Modified setup_qa_chain to work with the enhanced document processor
def setup_qa_chain(vectorstore, model_config: ModelConfig) -> RetrievalQA:
    """Setup QA chain with enhanced retriever"""
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_config.model_name,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    generation_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=model_config.max_length,
        truncation=True,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        top_k=50
    )
    llm = HuggingFacePipeline(pipeline=generation_pipeline)
    
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""Use the following passages to answer the question accurately and concisely.
If you cannot find the answer in the passages, say "I cannot find a specific answer in the provided context."

Relevant passages:
{context}

Question: {question}

Answer: Let me provide a specific answer based on the given context."""
    )
    
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "score_threshold": 0.4,
            "fetch_k": 20,
            "lambda_mult": 0.7
        }
    )
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={
            "prompt": prompt_template,
            "verbose": True
        },
        return_source_documents=True
    )

class RAGEvaluator:
    """Handles the evaluation of RAG system with proper tensor device handling"""
    def __init__(self, qa_chain):
        self.qa_chain = qa_chain
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )
        self.smoothing = SmoothingFunction().method1
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.semantic_model = SentenceTransformer(
            'BAAI/bge-small-en-v1.5',
            device=self.device
        )
        self.metric_types = ['rouge', 'bleu', 'exact_match', 'semantic_similarity', 'answer_relevance']

    def compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts with proper device handling"""
        with torch.no_grad():
            # Encode texts and get embeddings
            embeddings1 = self.semantic_model.encode([text1], convert_to_tensor=True)
            embeddings2 = self.semantic_model.encode([text2], convert_to_tensor=True)
            
            # Move tensors to CPU and convert to numpy
            embeddings1_np = embeddings1.cpu().numpy()
            embeddings2_np = embeddings2.cpu().numpy()
            
            # Compute cosine similarity
            similarity = float(
                cosine_similarity(embeddings1_np.reshape(1, -1), 
                                embeddings2_np.reshape(1, -1))[0][0]
            )
            return similarity

    def evaluate_sample(self, question: str, reference_answer: str) -> Dict:
        """Evaluate a single sample with proper error handling and device management"""
        try:
            result = self.qa_chain({"query": question})
            generated_answer = result["result"]
            source_docs = result.get("source_documents", [])
            
            metrics = {}
                
            if 'rouge' in self.metric_types:
                rouge_scores = self.rouge_scorer.score(reference_answer, generated_answer)
                metrics.update({
                    'rouge1_f1': rouge_scores['rouge1'].fmeasure,
                    'rouge2_f1': rouge_scores['rouge2'].fmeasure,
                    'rougeL_f1': rouge_scores['rougeL'].fmeasure
                })
                
            if 'bleu' in self.metric_types:
                reference_tokens = [word_tokenize(reference_answer.lower())]
                hypothesis_tokens = word_tokenize(generated_answer.lower())
                metrics['bleu_score'] = sentence_bleu(
                    reference_tokens, 
                    hypothesis_tokens, 
                    smoothing_function=self.smoothing
                )
                
            if 'exact_match' in self.metric_types:
                metrics['exact_match'] = int(
                    generated_answer.lower().strip() == reference_answer.lower().strip()
                )
                
            if 'semantic_similarity' in self.metric_types:
                metrics['semantic_similarity'] = self.compute_semantic_similarity(
                    reference_answer, 
                    generated_answer
                )
                
            if 'answer_relevance' in self.metric_types:
                metrics['answer_relevance'] = self.compute_semantic_similarity(
                    question, 
                    generated_answer
                )
            
            return {
                'question': question,
                'reference_answer': reference_answer,
                'generated_answer': generated_answer,
                'source_documents': [doc.page_content for doc in source_docs],
                'metrics': metrics
            }
            
        except Exception as e:
            print(f"Error processing question: {question}")
            print(f"Error details: {str(e)}")
            return None

    def evaluate_dataset(self, eval_dataset: List[Tuple[str, str]]) -> Dict:
        """Evaluate entire dataset with error handling"""
        results = []
        for question, reference_answer in tqdm(eval_dataset):
            result = self.evaluate_sample(question, reference_answer)
            if result is not None:
                results.append(result)
                
        return {
            'individual_results': results,
            'aggregated_metrics': self._aggregate_metrics(results)
        }

    def _aggregate_metrics(self, results: List[Dict]) -> Dict:
        """Aggregate metrics with proper error handling"""
        aggregated = {}
        metric_keys = set()
        
        # Collect all metric keys
        for result in results:
            if result and 'metrics' in result:
                metric_keys.update(result['metrics'].keys())
            
        # Calculate statistics for each metric
        for key in metric_keys:
            values = [r['metrics'].get(key) for r in results 
                     if r and 'metrics' in r and key in r['metrics']]
            if values:
                aggregated[f'mean_{key}'] = np.mean(values)
                aggregated[f'std_{key}'] = np.std(values)
                
        return aggregated

def load_evaluation_dataset(n_samples=100):
    """Load evaluation dataset from MS MARCO"""
    ds = load_dataset("microsoft/ms_marco", "v2.1", split=f"validation[:{n_samples}]")
    eval_pairs = []
    
    for item in ds:
        question = item['query']
        selected_passages = [p for p, is_selected in zip(item['passages']['passage_text'],
                                                       item['passages']['is_selected'])
                           if is_selected == 1]
        if selected_passages:
            eval_pairs.append((question, selected_passages[0]))
    
    return eval_pairs

def main():
    print("\nInitializing Enhanced RAG Evaluation...")
    print(f"Using device: {'cuda' if torch.cuda.is_available() and USE_GPU else 'cpu'}")
    print(f"Model: {MODEL_NAME}")
    
    # Configuration
    model_config = ModelConfig(
        model_name=MODEL_NAME,
        hf_token="your_token_here"
    )
    
    try:
        print("\nProcessing documents and creating vectorstore...")
        doc_processor = EnhancedDocumentProcessor()
        vectorstore = doc_processor.process_dataset()
        
        print("\nSetting up QA chain...")
        qa_chain = setup_qa_chain(vectorstore, model_config)
        
        print("\nInitializing evaluator...")
        evaluator = RAGEvaluator(qa_chain)
        
        print("\nLoading evaluation dataset...")
        eval_dataset = load_evaluation_dataset(n_samples=100)
        
        print("\nRunning evaluation...")
        results = evaluator.evaluate_dataset(eval_dataset)
        
        # Print results
        print("\nDetailed Metrics Summary:")
        print("-" * 50)
        metrics_order = [
            'rouge1_f1', 'rouge2_f1', 'rougeL_f1',
            'bleu_score',
            'exact_match',
            'semantic_similarity',
            'answer_relevance',
        ]
        
        for metric in metrics_order:
            mean_key = f'mean_{metric}'
            std_key = f'std_{metric}'
            if mean_key in results['aggregated_metrics']:
                print(f"{metric:.<30} {results['aggregated_metrics'][mean_key]:.4f} ± {results['aggregated_metrics'][std_key]:.4f}")
    
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()