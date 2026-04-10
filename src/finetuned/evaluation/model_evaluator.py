"""Model Evaluator untuk evaluate trained models."""

import torch
from datasets import Dataset
from typing import Dict, Any, Optional, List
import random
import json
from pathlib import Path
import numpy as np


class ModelEvaluator:
    """
    Evaluate trained model pada test set dan generate sample outputs.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        metrics_calculator,
        max_length: int = 512,
        device: str = None
    ):
        """
        Initialize ModelEvaluator.
        
        Args:
            model: Trained model (PeftModel atau T5ForConditionalGeneration)
            tokenizer: PreTrainedTokenizer
            metrics_calculator: MetricsCalculator instance
            max_length: Maximum sequence length
            device: Device untuk inference (auto-detect if None)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.metrics = metrics_calculator
        self.max_length = max_length
        
        # Auto-detect device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Move model to device
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def generate_prediction(
        self,
        input_text: str,
        num_beams: int = 4,
        max_length: Optional[int] = None,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95
    ) -> str:
        """
        Generate prediction untuk single input.
        
        Args:
            input_text: Input text
            num_beams: Number of beams for beam search
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p sampling
            
        Returns:
            Generated text
        """
        max_length = max_length or self.max_length
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=3,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
        
        # Decode
        prediction = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
        
        return prediction
    
    def evaluate_on_test_set(
        self,
        test_dataset: Dataset,
        batch_size: int = 8,
        num_beams: int = 4,
        include_bertscore: bool = True,
        max_samples: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Evaluate model pada entire test set.
        
        Args:
            test_dataset: Test dataset
            batch_size: Batch size untuk evaluation
            num_beams: Number of beams for generation
            include_bertscore: Whether to compute BERTScore
            max_samples: Maximum samples to evaluate (None = all)
            
        Returns:
            Dict dengan all metrics
        """
        print("\n" + "=" * 60)
        print("EVALUATING ON TEST SET")
        print("=" * 60 + "\n")
        
        # Limit samples if specified
        if max_samples is not None:
            test_dataset = test_dataset.select(range(min(max_samples, len(test_dataset))))
        
        predictions = []
        references = []
        
        print(f"Evaluating {len(test_dataset)} samples...")
        
        for i, sample in enumerate(test_dataset):
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(test_dataset)} samples...")
            
            input_text = sample["input"]
            reference = sample["target"]
            
            # Generate prediction
            prediction = self.generate_prediction(
                input_text,
                num_beams=num_beams
            )
            
            predictions.append(prediction)
            references.append(reference)
        
        print(f"✓ Generated {len(predictions)} predictions")
        
        # Compute metrics
        metrics = self.metrics.compute_all_metrics(
            predictions,
            references,
            include_bertscore=include_bertscore
        )
        
        # Print report
        self.metrics.print_metrics_report(metrics, title="Test Set Evaluation Results")
        
        return metrics
    
    def generate_samples(
        self,
        test_dataset: Dataset,
        num_samples: int = 20,
        num_beams: int = 4,
        random_seed: int = 42,
        save_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate output untuk random samples.
        
        Args:
            test_dataset: Test dataset
            num_samples: Number of samples to generate
            num_beams: Number of beams for generation
            random_seed: Random seed for reproducibility
            save_path: Path to save samples (optional)
            
        Returns:
            List of dicts dengan input, reference, prediction, bleu_score
        """
        print(f"\nGenerating {num_samples} sample outputs...")
        
        # Set random seed
        random.seed(random_seed)
        
        # Select random samples
        indices = random.sample(range(len(test_dataset)), min(num_samples, len(test_dataset)))
        
        samples = []
        
        for i, idx in enumerate(indices):
            sample = test_dataset[idx]
            input_text = sample["input"]
            reference = sample["target"]
            
            # Generate prediction
            prediction = self.generate_prediction(
                input_text,
                num_beams=num_beams
            )
            
            # Compute BLEU for this sample
            try:
                bleu_result = self.metrics.compute_bleu([prediction], [reference])
                bleu_score = bleu_result.get("bleu", 0.0)
            except:
                bleu_score = 0.0
            
            sample_result = {
                "index": idx,
                "input": input_text,
                "reference": reference,
                "prediction": prediction,
                "bleu_score": bleu_score
            }
            
            samples.append(sample_result)
            
            # Print sample
            print(f"\n--- Sample {i + 1} ---")
            print(f"Input: {input_text[:150]}...")
            print(f"Reference: {reference[:150]}...")
            print(f"Prediction: {prediction[:150]}...")
            print(f"BLEU: {bleu_score:.4f}")
        
        # Save to file if specified
        if save_path:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(samples, f, indent=2, ensure_ascii=False)
            print(f"\n✓ Samples saved to {save_path}")
        
        return samples
    
    def compare_with_baseline(
        self,
        finetuned_metrics: Dict[str, float],
        baseline_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate improvement percentage.
        
        Args:
            finetuned_metrics: Metrics from fine-tuned model
            baseline_metrics: Metrics from baseline model
            
        Returns:
            Dict dengan improvement untuk setiap metric
        """
        print("\n" + "=" * 60)
        print("COMPARING WITH BASELINE")
        print("=" * 60 + "\n")
        
        improvements = {}
        
        # Calculate improvement for each metric
        for key in finetuned_metrics:
            if key in baseline_metrics:
                baseline_value = baseline_metrics[key]
                finetuned_value = finetuned_metrics[key]
                
                if baseline_value != 0:
                    improvement = ((finetuned_value - baseline_value) / abs(baseline_value)) * 100
                else:
                    improvement = 0.0 if finetuned_value == 0 else float('inf')
                
                improvements[f"{key}_improvement_pct"] = improvement
                improvements[f"{key}_baseline"] = baseline_value
                improvements[f"{key}_finetuned"] = finetuned_value
        
        # Print comparison
        print(f"{'Metric':<25} {'Baseline':>12} {'Fine-tuned':>12} {'Improvement':>12}")
        print("-" * 65)
        
        for key in finetuned_metrics:
            if key in baseline_metrics:
                baseline = baseline_metrics[key]
                finetuned = finetuned_metrics[key]
                improvement = improvements.get(f"{key}_improvement_pct", 0.0)
                
                print(f"{key:<25} {baseline:>12.4f} {finetuned:>12.4f} {improvement:>11.2f}%")
        
        return improvements
    
    def generate_evaluation_report(
        self,
        test_dataset: Dataset,
        baseline_metrics: Optional[Dict[str, float]] = None,
        num_samples: int = 20,
        output_dir: str = "./evaluation_results"
    ) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report.
        
        Args:
            test_dataset: Test dataset
            baseline_metrics: Baseline metrics for comparison (optional)
            num_samples: Number of sample outputs to generate
            output_dir: Directory to save report
            
        Returns:
            Dict dengan complete evaluation results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "=" * 60)
        print("GENERATING COMPREHENSIVE EVALUATION REPORT")
        print("=" * 60 + "\n")
        
        # Evaluate on test set
        test_metrics = self.evaluate_on_test_set(
            test_dataset,
            include_bertscore=True
        )
        
        # Generate samples
        samples_path = output_dir / "sample_outputs.json"
        samples = self.generate_samples(
            test_dataset,
            num_samples=num_samples,
            save_path=str(samples_path)
        )
        
        # Compare with baseline if provided
        comparison = {}
        if baseline_metrics:
            comparison = self.compare_with_baseline(test_metrics, baseline_metrics)
        
        # Compile report
        report = {
            "test_metrics": test_metrics,
            "baseline_metrics": baseline_metrics,
            "comparison": comparison,
            "num_samples": len(samples),
            "samples": samples[:5]  # Include first 5 samples in report
        }
        
        # Save report
        report_path = output_dir / "evaluation_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n✓ Evaluation report saved to {report_path}")
        
        # Generate summary
        self._print_summary(report)
        
        return report
    
    def _print_summary(self, report: Dict[str, Any]) -> None:
        """Print evaluation summary."""
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        
        test_metrics = report.get("test_metrics", {})
        
        print(f"\nKey Metrics:")
        print(f"  BLEU-4:   {test_metrics.get('bleu_4', 0.0):.4f}")
        print(f"  ROUGE-L:  {test_metrics.get('rouge_l', 0.0):.4f}")
        print(f"  BERTScore F1: {test_metrics.get('bertscore_f1', 0.0):.4f}")
        
        if report.get("comparison"):
            print(f"\nImprovement over Baseline:")
            for key, value in report["comparison"].items():
                if "improvement_pct" in key:
                    metric_name = key.replace("_improvement_pct", "")
                    print(f"  {metric_name}: {value:+.2f}%")
        
        print(f"\nSamples Generated: {report.get('num_samples', 0)}")
        print("=" * 60)