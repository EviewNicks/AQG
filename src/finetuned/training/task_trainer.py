"""Task-Specific Trainer untuk Stage 2 fine-tuning (AQG)."""

import torch
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from datasets import Dataset
from typing import Dict, Any, Optional, List, Callable
import numpy as np
from pathlib import Path
import json


class TaskSpecificTrainer:
    """
    Trainer untuk task-specific AQG stage.
    
    Train model untuk AQG task dengan format output spesifik.
    Uses BLEU, ROUGE, dan BERTScore untuk evaluation.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        output_dir: str = "./checkpoints/aqg",
        max_length: int = 512,
        metrics_calculator=None
    ):
        """
        Initialize TaskSpecificTrainer.
        
        Args:
            model: PeftModel dengan LoRA adapters
            tokenizer: PreTrainedTokenizer
            output_dir: Directory untuk save checkpoints
            max_length: Maximum sequence length
            metrics_calculator: MetricsCalculator instance (optional)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = Path(output_dir)
        self.max_length = max_length
        self.metrics_calculator = metrics_calculator
        self.trainer = None
        self.training_history = []
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def preprocess_dataset(
        self,
        dataset: Dataset,
        batch_size: int = 8
    ) -> Dataset:
        """
        Preprocess dataset untuk training.
        
        Args:
            dataset: Raw dataset dengan 'input' dan 'target' fields
            batch_size: Batch size untuk processing
            
        Returns:
            Preprocessed dataset dengan tokenized fields
        """
        print(f"Preprocessing {len(dataset)} samples...")
        
        def tokenize_function(examples):
            # Tokenize inputs
            model_inputs = self.tokenizer(
                examples["input"],
                max_length=self.max_length,
                truncation=True,
                padding="max_length"
            )
            
            # Tokenize targets (text_target parameter - modern API)
            labels = self.tokenizer(
                text_target=examples["target"],
                max_length=self.max_length,
                truncation=True,
                padding="max_length"
            )
            
            model_inputs["labels"] = labels["input_ids"]
            
            # Replace padding token id dengan -100 untuk loss calculation
            model_inputs["labels"] = [
                [(l if l != self.tokenizer.pad_token_id else -100) for l in label]
                for label in model_inputs["labels"]
            ]
            
            return model_inputs
        
        # Tokenize dataset
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing"
        )
        
        print(f"✓ Preprocessed {len(tokenized_dataset)} samples")
        return tokenized_dataset
    
    def compute_metrics(self, eval_preds) -> Dict[str, float]:
        """
        Compute evaluation metrics (BLEU, ROUGE, BERTScore).
        
        Args:
            eval_preds: Evaluation predictions from trainer
            
        Returns:
            Dict dengan metric values
        """
        predictions, labels = eval_preds
        
        # Decode predictions
        decoded_preds = self.tokenizer.batch_decode(
            predictions, 
            skip_special_tokens=True
        )
        
        # Decode labels (replace -100 dengan pad token)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(
            labels, 
            skip_special_tokens=True
        )
        
        # Clean up
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]
        
        metrics = {}
        
        # Compute BLEU if metrics_calculator available
        if self.metrics_calculator is not None:
            try:
                bleu_results = self.metrics_calculator.compute_bleu(
                    decoded_preds, 
                    decoded_labels
                )
                metrics["bleu_1"] = bleu_results.get("bleu", 0.0)
                metrics["bleu_4"] = bleu_results.get("bleu", 0.0)
                
                rouge_results = self.metrics_calculator.compute_rouge(
                    decoded_preds,
                    decoded_labels
                )
                metrics["rouge_l"] = rouge_results.get("rougeL", 0.0)
                
            except Exception as e:
                print(f"Warning: Could not compute metrics: {e}")
                metrics["bleu_4"] = 0.0
                metrics["rouge_l"] = 0.0
        else:
            # Simple BLEU approximation using NLTK
            try:
                from nltk.translate.bleu_score import corpus_bleu
                references = [[label.split()] for label in decoded_labels]
                hypotheses = [pred.split() for pred in decoded_preds]
                metrics["bleu_4"] = corpus_bleu(references, hypotheses)
            except:
                metrics["bleu_4"] = 0.0
            metrics["rouge_l"] = 0.0
        
        return metrics
    
    def get_training_args(
        self,
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 8,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 1e-4,
        warmup_steps: int = 30,
        logging_steps: int = 50,
        eval_steps: int = None,
        save_steps: int = None,
        fp16: bool = True,
        early_stopping_patience: int = 2
    ) -> Seq2SeqTrainingArguments:
        """
        Get default training arguments untuk task-specific training.
        
        Args:
            num_train_epochs: Number of training epochs
            per_device_train_batch_size: Batch size per device
            gradient_accumulation_steps: Gradient accumulation steps
            learning_rate: Learning rate
            warmup_steps: Warmup steps
            logging_steps: Steps between logging
            eval_steps: Steps between evaluations (None = epoch)
            save_steps: Steps between saves (None = epoch)
            fp16: Use mixed precision training
            early_stopping_patience: Patience for early stopping
            
        Returns:
            Seq2SeqTrainingArguments
        """
        return Seq2SeqTrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            eval_strategy="epoch" if eval_steps is None else "steps",
            eval_steps=eval_steps,
            save_strategy="epoch" if save_steps is None else "steps",
            save_steps=save_steps,
            logging_steps=logging_steps,
            fp16=fp16,
            predict_with_generate=True,
            generation_max_length=self.max_length,
            load_best_model_at_end=True,
            metric_for_best_model="eval_bleu_4",
            greater_is_better=True,
            save_total_limit=2,
            report_to=["none"],
            dataloader_num_workers=2,
            dataloader_pin_memory=True,
        )
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        training_args: Optional[Seq2SeqTrainingArguments] = None,
        early_stopping: bool = True,
        early_stopping_patience: int = 2
    ) -> Dict[str, Any]:
        """
        Train model untuk task-specific AQG.
        
        Args:
            train_dataset: Training dataset (raw, will be preprocessed)
            eval_dataset: Evaluation dataset (optional)
            training_args: Custom training arguments (optional)
            early_stopping: Enable early stopping
            early_stopping_patience: Patience for early stopping
            
        Returns:
            Dict dengan training results
        """
        print("\n" + "=" * 60)
        print("STARTING TASK-SPECIFIC AQG TRAINING")
        print("=" * 60 + "\n")
        
        # Preprocess datasets
        print("Preprocessing datasets...")
        train_dataset = self.preprocess_dataset(train_dataset)
        if eval_dataset is not None:
            eval_dataset = self.preprocess_dataset(eval_dataset)
        
        # Get training args if not provided
        if training_args is None:
            training_args = self.get_training_args(
                early_stopping_patience=early_stopping_patience
            )
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
            max_length=self.max_length
        )
        
        # Callbacks
        callbacks = []
        if early_stopping and eval_dataset is not None:
            callbacks.append(EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience
            ))
        
        # Initialize trainer
        self.trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=callbacks
        )
        
        # Print training info
        print("\n=== Training Configuration ===")
        print(f"Epochs: {training_args.num_train_epochs}")
        print(f"Batch size: {training_args.per_device_train_batch_size}")
        print(f"Gradient accumulation: {training_args.gradient_accumulation_steps}")
        print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
        print(f"Learning rate: {training_args.learning_rate}")
        print(f"Warmup steps: {training_args.warmup_steps}")
        print(f"FP16: {training_args.fp16}")
        print(f"Train samples: {len(train_dataset)}")
        if eval_dataset:
            print(f"Eval samples: {len(eval_dataset)}")
        print(f"Metrics: BLEU-4, ROUGE-L")
        print()
        
        # Train
        print("Starting training...")
        train_result = self.trainer.train()
        
        # Save training history
        self.training_history = self.trainer.state.log_history
        
        # Print results
        print("\n=== Training Complete ===")
        print(f"Final training loss: {train_result.training_loss:.4f}")
        print(f"Training time: {train_result.metrics['train_runtime']:.2f} seconds")
        print(f"Training samples per second: {train_result.metrics['train_samples_per_second']:.2f}")
        
        # Print final metrics
        if eval_dataset:
            final_metrics = self.trainer.evaluate()
            print("\n=== Final Evaluation Metrics ===")
            for key, value in final_metrics.items():
                if key.startswith("eval_"):
                    print(f"{key}: {value:.4f}")
        
        # Save training results
        self._save_training_results(train_result)
        
        return {
            "training_loss": train_result.training_loss,
            "training_time": train_result.metrics['train_runtime'],
            "training_history": self.training_history,
            "metrics": train_result.metrics
        }
    
    def _save_training_results(self, train_result) -> None:
        """Save training results to file."""
        results_path = self.output_dir / "training_results.json"
        
        results = {
            "training_loss": train_result.training_loss,
            "metrics": train_result.metrics,
            "training_history": self.training_history
        }
        
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"✓ Training results saved to {results_path}")
    
    def save_final_model(self, output_name: str = "indot5-python-aqg") -> str:
        """
        Save final trained model.
        
        Args:
            output_name: Name untuk saved model
            
        Returns:
            Path ke saved model
        """
        if self.trainer is None:
            raise ValueError("No trainer found. Run train() first.")
        
        # Save model
        save_path = self.output_dir / output_name
        self.trainer.save_model(str(save_path))
        
        # Save tokenizer
        self.tokenizer.save_pretrained(str(save_path))
        
        print(f"\n✓ Final model saved to: {save_path}")
        return str(save_path)
    
    def get_training_history(self) -> List[Dict]:
        """
        Get training history.
        
        Returns:
            List of training logs
        """
        return self.training_history
    
    def plot_training_curves(self, save_path: Optional[str] = None) -> None:
        """
        Plot training curves.
        
        Args:
            save_path: Path to save plot (optional)
        """
        try:
            import matplotlib.pyplot as plt
            
            if not self.training_history:
                print("No training history available")
                return
            
            # Extract values
            train_loss = []
            eval_loss = []
            eval_bleu = []
            epochs = []
            
            for log in self.training_history:
                if "loss" in log:
                    train_loss.append(log["loss"])
                    epochs.append(log.get("epoch", len(train_loss)))
                if "eval_loss" in log:
                    eval_loss.append(log["eval_loss"])
                if "eval_bleu_4" in log:
                    eval_bleu.append(log["eval_bleu_4"])
            
            # Create subplots
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Plot loss
            if train_loss:
                axes[0].plot(epochs[:len(train_loss)], train_loss, label="Training Loss", marker="o")
            if eval_loss:
                axes[0].plot(range(1, len(eval_loss) + 1), eval_loss, label="Validation Loss", marker="s")
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Loss")
            axes[0].set_title("Training & Validation Loss")
            axes[0].legend()
            axes[0].grid(True)
            
            # Plot BLEU
            if eval_bleu:
                axes[1].plot(range(1, len(eval_bleu) + 1), eval_bleu, label="BLEU-4", marker="s", color="green")
                axes[1].set_xlabel("Epoch")
                axes[1].set_ylabel("BLEU-4")
                axes[1].set_title("BLEU-4 Score")
                axes[1].legend()
                axes[1].grid(True)
            
            plt.suptitle("Task-Specific AQG Training Curves")
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                print(f"✓ Training curves saved to {save_path}")
            else:
                plt.show()
            
            plt.close()
            
        except ImportError:
            print("matplotlib not installed. Skipping plot generation.")