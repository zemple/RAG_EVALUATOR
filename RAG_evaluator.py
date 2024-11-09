import os
import time
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from tqdm import tqdm
import torch

# Dependencies for model and evaluation
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

import nltk
nltk.download('punkt')

MODEL_NAME = ""
USE_GPU = True

@dataclass
class ModelConfig:
    """Configuration for the model and tokenizer"""
    model_name: str = MODEL_NAME
    max_length: int = 256
    device: int = 0 if USE_GPU and torch.cuda.is_available() else -1
    hf_token: str = None

class DocumentProcessor:
    """Enhanced document processor with better chunking and embedding strategies"""
    def __init__(self, chunk_size=384, chunk_overlap=50):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

    def process_dataset(self, dataset_name="microsoft/ms_marco", version="v2.1", 
                       split="train[:100000]"):
        ds = load_dataset(dataset_name, version, split=split)
        
        documents = []
        for idx, item in enumerate(ds):
            passages = item['passages']['passage_text']
            is_selected = item['passages']['is_selected']
            
            for i, (passage, selected) in enumerate(zip(passages, is_selected)):
                if selected == 1:
                    metadata = {
                        "id": idx,
                        "passage_idx": i,
                        "source": "ms_marco",
                        "query": item['query'],
                    }
                    cleaned_passage = self._preprocess_text(passage)
                    documents.append(Document(page_content=cleaned_passage, metadata=metadata))

        docs = []
        for doc in documents:
            splits = self.text_splitter.split_text(doc.page_content)
            for i, split in enumerate(splits):
                new_metadata = doc.metadata.copy()
                new_metadata.update({"chunk": i, "total_chunks": len(splits)})
                new_doc = Document(page_content=split, metadata=new_metadata)
                docs.append(new_doc)

        vectorstore = FAISS.from_documents(
            docs, 
            self.embeddings,
            distance_strategy="COSINE"
        )
        return vectorstore

    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize text"""
        text = text.strip()
        text = " ".join(text.split())
        return text

def setup_qa_chain(vectorstore, model_config: ModelConfig) -> RetrievalQA:
    """Enhanced RAG chain setup with better prompting and retrieval"""
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_config.model_name,
        device_map="auto",  # This handles device placement automatically
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    # Remove device argument from pipeline creation
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
    
    # Rest of the function remains the same
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
    """Handles the evaluation of RAG system"""
    def __init__(self, qa_chain):
        self.qa_chain = qa_chain
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )
        self.smoothing = SmoothingFunction().method1
        self.semantic_model = SentenceTransformer(
            'BAAI/bge-small-en-v1.5',  # Using the same model as embeddings for consistency
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.metric_types = ['rouge', 'bleu', 'exact_match', 'semantic_similarity', 'answer_relevance']

    def evaluate_sample(self, question: str, reference_answer: str) -> Dict:
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
            metrics['bleu_score'] = sentence_bleu(reference_tokens, hypothesis_tokens, 
                                                smoothing_function=self.smoothing)
            
        if 'exact_match' in self.metric_types:
            metrics['exact_match'] = int(
                generated_answer.lower().strip() == reference_answer.lower().strip()
            )
            
        if 'semantic_similarity' in self.metric_types:
            ref_embedding = self.semantic_model.encode([reference_answer])
            gen_embedding = self.semantic_model.encode([generated_answer])
            metrics['semantic_similarity'] = float(
                cosine_similarity(ref_embedding, gen_embedding)[0][0]
            )
            
        if 'answer_relevance' in self.metric_types:
            question_embedding = self.semantic_model.encode([question])
            answer_embedding = self.semantic_model.encode([generated_answer])
            metrics['answer_relevance'] = float(
                cosine_similarity(question_embedding, answer_embedding)[0][0]
            )
        
        return {
            'question': question,
            'reference_answer': reference_answer,
            'generated_answer': generated_answer,
            'source_documents': [doc.page_content for doc in source_docs],
            'metrics': metrics
        }

    def evaluate_dataset(self, eval_dataset: List[Tuple[str, str]]) -> Dict:
        results = []
        for question, reference_answer in tqdm(eval_dataset):
            try:
                result = self.evaluate_sample(question, reference_answer)
                results.append(result)
            except Exception as e:
                print(f"Error processing question: {question}")
                print(f"Error: {str(e)}")
                
        return {
            'individual_results': results,
            'aggregated_metrics': self._aggregate_metrics(results)
        }

    def _aggregate_metrics(self, results: List[Dict]) -> Dict:
        aggregated = {}
        metric_keys = set()
        for result in results:
            metric_keys.update(result['metrics'].keys())
            
        for key in metric_keys:
            values = [r['metrics'].get(key) for r in results if key in r['metrics']]
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
        hf_token="your_token_here"  # Replace with your token
    )
    
    print("\nProcessing documents and creating vectorstore...")
    doc_processor = DocumentProcessor()
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


if __name__ == "__main__":
    main()