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
from langchain.chains import LLMChain
from langchain_huggingface.llms import HuggingFacePipeline
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

import nltk
nltk.download('punkt')

MODEL_NAME = "google/flan-t5-small"  # Define model name here

USE_GPU = True  # Set this to False if you want to use CPU instead

@dataclass
class ModelConfig:
    """Configuration for the model and tokenizer"""
    model_name: str = MODEL_NAME
    max_length: int = 256
    device: int = 0 if USE_GPU and torch.cuda.is_available() else -1  # 0 for GPU, -1 for CPU
    hf_token: str = None

@dataclass
class EvalConfig:
    """Configuration for evaluation settings"""
    metric_types: List[str] = None
    n_samples: int = 100
    dataset_name: str = "microsoft/ms_marco"
    dataset_version: str = "v2.1"

    def __post_init__(self):
        if self.metric_types is None:
            self.metric_types = ['rouge', 'bleu', 'exact_match', 'semantic_similarity', 'answer_relevance']

class ModelHandler:
    """Handles model initialization and pipeline setup"""
    def __init__(self, config: ModelConfig):
        self.config = config
        if config.hf_token:
            login(token=config.hf_token)

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name)
        self.pipeline = self._create_pipeline()
        self.llm = HuggingFacePipeline(pipeline=self.pipeline)

    def _create_pipeline(self):
        return pipeline(
            "text2text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_length=self.config.max_length,
            device=self.config.device,
            truncation=True
        )

class DatasetHandler:
    """Handles dataset loading and preprocessing"""
    @staticmethod
    def load_evaluation_dataset(config: EvalConfig) -> List[Tuple[str, str]]:
        ds = load_dataset(
            config.dataset_name,
            config.dataset_version,
            split=f"validation[:{config.n_samples}]"
        )

        eval_pairs = []
        for item in ds:
            question = item['query']
            selected_passages = [
                p for p, is_selected in zip(
                    item['passages']['passage_text'],
                    item['passages']['is_selected']
                ) if is_selected == 1
            ]
            if selected_passages:
                eval_pairs.append((question, selected_passages[0]))

        return eval_pairs

class QAEvaluator:
    """Handles the evaluation of QA models"""
    def __init__(self, config: EvalConfig):
        self.config = config
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )
        self.smoothing = SmoothingFunction().method1
        self.semantic_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    def evaluate_sample(self, qa_chain: LLMChain, question: str, reference_answer: str) -> Dict:
        generated_answer = qa_chain.run(question=question)
        metrics = {}

        if 'rouge' in self.config.metric_types:
            rouge_scores = self.rouge_scorer.score(reference_answer, generated_answer)
            metrics.update({
                'rouge1_f1': rouge_scores['rouge1'].fmeasure,
                'rouge2_f1': rouge_scores['rouge2'].fmeasure,
                'rougeL_f1': rouge_scores['rougeL'].fmeasure
            })

        if 'bleu' in self.config.metric_types:
            reference_tokens = [word_tokenize(reference_answer.lower())]
            hypothesis_tokens = word_tokenize(generated_answer.lower())
            metrics['bleu_score'] = sentence_bleu(reference_tokens, hypothesis_tokens,
                                                smoothing_function=self.smoothing)

        if 'exact_match' in self.config.metric_types:
            metrics['exact_match'] = int(
                generated_answer.lower().strip() == reference_answer.lower().strip()
            )

        if 'semantic_similarity' in self.config.metric_types:
            ref_embedding = self.semantic_model.encode([reference_answer])
            gen_embedding = self.semantic_model.encode([generated_answer])
            metrics['semantic_similarity'] = float(
                cosine_similarity(ref_embedding, gen_embedding)[0][0]
            )

        if 'answer_relevance' in self.config.metric_types:
            question_embedding = self.semantic_model.encode([question])
            answer_embedding = self.semantic_model.encode([generated_answer])
            metrics['answer_relevance'] = float(
                cosine_similarity(question_embedding, answer_embedding)[0][0]
            )

        return {
            'question': question,
            'reference_answer': reference_answer,
            'generated_answer': generated_answer,
            'metrics': metrics
        }

    def evaluate_dataset(self, qa_chain: LLMChain, eval_dataset: List[Tuple[str, str]]) -> Dict:
        results = []
        for question, reference_answer in tqdm(eval_dataset):
            try:
                result = self.evaluate_sample(qa_chain, question, reference_answer)
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

def create_qa_chain(llm) -> LLMChain:
    """Creates a QA chain with the specified prompt template"""
    prompt_template = PromptTemplate(
        input_variables=["question"],
        template="""Answer the following question concisely and accurately:

Question: {question}

Answer:"""
    )
    return LLMChain(llm=llm, prompt=prompt_template)

def main():
    # Configuration
    model_config = ModelConfig(
        model_name=MODEL_NAME,
        hf_token="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"
    )

    eval_config = EvalConfig(
        n_samples=100,
        metric_types=['rouge', 'bleu', 'exact_match', 'semantic_similarity', 'answer_relevance']
    )

    # Initialize components
    model_handler = ModelHandler(model_config)
    evaluator = QAEvaluator(eval_config)

    # Load dataset
    eval_dataset = DatasetHandler.load_evaluation_dataset(eval_config)

    # Create QA chain
    qa_chain = create_qa_chain(model_handler.llm)

    # Run evaluation
    results = evaluator.evaluate_dataset(qa_chain, eval_dataset)

    # Print summary with all metrics
    print(MODEL_NAME)
    print("\nAggregated Metrics:")
    metrics_order = [
        'rouge1_f1', 'rouge2_f1', 'rougeL_f1',
        'bleu_score',
        'exact_match',
        'semantic_similarity',
        'answer_relevance'
    ]

    print("\nDetailed Metrics Summary:")
    print("-" * 50)
    for metric in metrics_order:
        mean_key = f'mean_{metric}'
        std_key = f'std_{metric}'
        if mean_key in results['aggregated_metrics']:
            print(f"{metric:.<30} {results['aggregated_metrics'][mean_key]:.4f} ± {results['aggregated_metrics'][std_key]:.4f}")

if __name__ == "__main__":
  main()