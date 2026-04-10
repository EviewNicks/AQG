"""Metrics Calculator untuk evaluation metrics."""

from evaluate import load
from typing import Dict, List, Optional
import numpy as np


class MetricsCalculator:
    """
    Calculate evaluation metrics untuk text generation quality.
    
    Supports:
    - BLEU (1-4)
    - ROUGE (1, 2, L)
    - BERTScore (Precision, Recall, F1)
    - Diversity (Distinct-1, Distinct-2)
    """
    
    def __init__(self, lang: str = "id"):
        """
        Initialize MetricsCalculator.
        
        Args:
            lang: Language code untuk BERTScore (default: "id" for Indonesian)
        """
        self.lang = lang
        self._bleu = None
        self._rouge = None
        self._bertscore = None
    
    @property
    def bleu(self):
        """Lazy load BLEU metric."""
        if self._bleu is None:
            self._bleu = load("bleu")
        return self._bleu
    
    @property
    def rouge(self):
        """Lazy load ROUGE metric."""
        if self._rouge is None:
            self._rouge = load("rouge")
        return self._rouge
    
    @property
    def bertscore(self):
        """Lazy load BERTScore metric."""
        if self._bertscore is None:
            self._bertscore = load("bertscore")
        return self._bertscore
    
    def compute_bleu(
        self, 
        predictions: List[str], 
        references: List[str]
    ) -> Dict[str, float]:
        """
        Compute BLEU scores (BLEU-1 to BLEU-4).
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts
            
        Returns:
            Dict dengan BLEU scores
        """
        if not predictions or not references:
            return {"bleu": 0.0, "precisions": [0.0, 0.0, 0.0, 0.0]}
        
        # Format references untuk BLEU (list of list)
        formatted_refs = [[ref] for ref in references]
        
        try:
            results = self.bleu.compute(
                predictions=predictions,
                references=formatted_refs
            )
            
            return {
                "bleu": results.get("bleu", 0.0),
                "bleu_1": results.get("precisions", [0.0])[0] if results.get("precisions") else 0.0,
                "bleu_2": results.get("precisions", [0.0, 0.0])[1] if len(results.get("precisions", [])) > 1 else 0.0,
                "bleu_3": results.get("precisions", [0.0, 0.0, 0.0])[2] if len(results.get("precisions", [])) > 2 else 0.0,
                "bleu_4": results.get("precisions", [0.0, 0.0, 0.0, 0.0])[3] if len(results.get("precisions", [])) > 3 else 0.0,
                "brevity_penalty": results.get("brevity_penalty", 0.0),
                "length_ratio": results.get("length_ratio", 0.0)
            }
        except Exception as e:
            print(f"Warning: BLEU computation failed: {e}")
            return {"bleu": 0.0, "bleu_1": 0.0, "bleu_2": 0.0, "bleu_3": 0.0, "bleu_4": 0.0}
    
    def compute_rouge(
        self, 
        predictions: List[str], 
        references: List[str]
    ) -> Dict[str, float]:
        """
        Compute ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L).
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts
            
        Returns:
            Dict dengan ROUGE scores
        """
        if not predictions or not references:
            return {"rouge_1": 0.0, "rouge_2": 0.0, "rouge_l": 0.0}
        
        try:
            results = self.rouge.compute(
                predictions=predictions,
                references=references
            )
            
            return {
                "rouge_1": results.get("rouge1", 0.0),
                "rouge_2": results.get("rouge2", 0.0),
                "rouge_l": results.get("rougeL", 0.0),
                "rouge_1_fmeasure": results.get("rouge1", 0.0),
                "rouge_2_fmeasure": results.get("rouge2", 0.0),
                "rouge_l_fmeasure": results.get("rougeL", 0.0)
            }
        except Exception as e:
            print(f"Warning: ROUGE computation failed: {e}")
            return {"rouge_1": 0.0, "rouge_2": 0.0, "rouge_l": 0.0}
    
    def compute_bertscore(
        self, 
        predictions: List[str], 
        references: List[str],
        lang: Optional[str] = None,
        model_type: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Compute BERTScore (Precision, Recall, F1).
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts
            lang: Language code (default: use self.lang)
            model_type: BERT model type untuk BERTScore
            
        Returns:
            Dict dengan BERTScore values
        """
        if not predictions or not references:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        lang = lang or self.lang
        
        try:
            results = self.bertscore.compute(
                predictions=predictions,
                references=references,
                lang=lang,
                model_type=model_type
            )
            
            # Calculate mean values
            precision = np.mean(results.get("precision", [0.0]))
            recall = np.mean(results.get("recall", [0.0]))
            f1 = np.mean(results.get("f1", [0.0]))
            
            return {
                "bertscore_precision": float(precision),
                "bertscore_recall": float(recall),
                "bertscore_f1": float(f1)
            }
        except Exception as e:
            print(f"Warning: BERTScore computation failed: {e}")
            return {"bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0}
    
    def compute_diversity(
        self, 
        predictions: List[str]
    ) -> Dict[str, float]:
        """
        Compute diversity metrics (Distinct-1, Distinct-2).
        
        Distinct-N measures the proportion of unique n-grams in the generated text.
        Higher values indicate more diverse outputs.
        
        Args:
            predictions: List of predicted texts
            
        Returns:
            Dict dengan Distinct-1 dan Distinct-2 values
        """
        if not predictions:
            return {"distinct_1": 0.0, "distinct_2": 0.0}
        
        # Tokenize all predictions
        all_tokens = []
        for pred in predictions:
            tokens = pred.lower().split()
            all_tokens.extend(tokens)
        
        if not all_tokens:
            return {"distinct_1": 0.0, "distinct_2": 0.0}
        
        # Compute Distinct-1 (unique unigrams)
        unique_unigrams = set(all_tokens)
        distinct_1 = len(unique_unigrams) / len(all_tokens) if all_tokens else 0.0
        
        # Compute Distinct-2 (unique bigrams)
        bigrams = []
        for i in range(len(all_tokens) - 1):
            bigrams.append((all_tokens[i], all_tokens[i + 1]))
        
        unique_bigrams = set(bigrams)
        distinct_2 = len(unique_bigrams) / len(bigrams) if bigrams else 0.0
        
        return {
            "distinct_1": float(distinct_1),
            "distinct_2": float(distinct_2)
        }
    
    def compute_all_metrics(
        self,
        predictions: List[str],
        references: List[str],
        include_bertscore: bool = True
    ) -> Dict[str, float]:
        """
        Compute all metrics at once.
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts
            include_bertscore: Whether to compute BERTScore (can be slow)
            
        Returns:
            Dict dengan all metric values
        """
        print(f"Computing metrics for {len(predictions)} samples...")
        
        metrics = {}
        
        # BLEU
        print("  Computing BLEU...")
        bleu_metrics = self.compute_bleu(predictions, references)
        metrics.update(bleu_metrics)
        
        # ROUGE
        print("  Computing ROUGE...")
        rouge_metrics = self.compute_rouge(predictions, references)
        metrics.update(rouge_metrics)
        
        # BERTScore (optional, can be slow)
        if include_bertscore:
            print("  Computing BERTScore...")
            bertscore_metrics = self.compute_bertscore(predictions, references)
            metrics.update(bertscore_metrics)
        
        # Diversity
        print("  Computing Diversity...")
        diversity_metrics = self.compute_diversity(predictions)
        metrics.update(diversity_metrics)
        
        print("✓ All metrics computed")
        
        return metrics
    
    def print_metrics_report(
        self,
        metrics: Dict[str, float],
        title: str = "Evaluation Metrics"
    ) -> None:
        """
        Print formatted metrics report.
        
        Args:
            metrics: Dict dengan metric values
            title: Report title
        """
        print("\n" + "=" * 60)
        print(title)
        print("=" * 60)
        
        # BLEU
        if "bleu" in metrics:
            print("\nBLEU Scores:")
            print(f"  BLEU:     {metrics.get('bleu', 0.0):.4f}")
            print(f"  BLEU-1:   {metrics.get('bleu_1', 0.0):.4f}")
            print(f"  BLEU-2:   {metrics.get('bleu_2', 0.0):.4f}")
            print(f"  BLEU-3:   {metrics.get('bleu_3', 0.0):.4f}")
            print(f"  BLEU-4:   {metrics.get('bleu_4', 0.0):.4f}")
        
        # ROUGE
        if "rouge_1" in metrics:
            print("\nROUGE Scores:")
            print(f"  ROUGE-1:  {metrics.get('rouge_1', 0.0):.4f}")
            print(f"  ROUGE-2:  {metrics.get('rouge_2', 0.0):.4f}")
            print(f"  ROUGE-L:  {metrics.get('rouge_l', 0.0):.4f}")
        
        # BERTScore
        if "bertscore_f1" in metrics:
            print("\nBERTScore:")
            print(f"  Precision: {metrics.get('bertscore_precision', 0.0):.4f}")
            print(f"  Recall:    {metrics.get('bertscore_recall', 0.0):.4f}")
            print(f"  F1:        {metrics.get('bertscore_f1', 0.0):.4f}")
        
        # Diversity
        if "distinct_1" in metrics:
            print("\nDiversity:")
            print(f"  Distinct-1: {metrics.get('distinct_1', 0.0):.4f}")
            print(f"  Distinct-2: {metrics.get('distinct_2', 0.0):.4f}")
        
        print("\n" + "=" * 60)