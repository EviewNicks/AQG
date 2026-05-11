"""Task-Specific Trainer untuk Stage 2 fine-tuning (AQG)."""

import torch
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from datasets import Dataset
from typing import Dict, Any, Optional, List, Callable, Union
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
            # Tokenize inputs - NO PADDING (collator will handle it)
            model_inputs = self.tokenizer(
                examples["input"],
                max_length=self.max_length,
                truncation=True
            )
            
            # Support both 'target' (v2) and 'output' (v3) field names
            target_field = "target" if "target" in examples else "output"
            
            # Tokenize targets - NO PADDING (collator will handle it)
            labels = self.tokenizer(
                text_target=examples[target_field],
                max_length=self.max_length,
                truncation=True
            )
            
            model_inputs["labels"] = labels["input_ids"]
            
            # NO manual masking - DataCollatorForSeq2Seq will handle it
            # with label_pad_token_id=-100 parameter
            
            return model_inputs
        
        # Tokenize dataset
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing"
        )
        
        print(f"✓ Preprocessed {len(tokenized_dataset)} samples")
        print(f"  Note: Padding and label masking will be handled by DataCollatorForSeq2Seq")
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
        
        # ✅ FIX: Handle logits from predict_with_generate
        # Predictions might be logits (3D) or token IDs (2D)
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        
        # If predictions are 3D (batch, seq_len, vocab_size), get argmax
        if len(predictions.shape) == 3:
            predictions = np.argmax(predictions, axis=-1)
        
        # Handle predictions yang mungkin mengandung nilai negatif dari padding
        if hasattr(predictions, '__iter__'):
            predictions = np.where(predictions < 0, self.tokenizer.pad_token_id, predictions)
        
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
        
        # ✅ DEBUG: Print sample predictions (first 2 samples)
        if len(decoded_preds) > 0:
            print(f"\n[DEBUG] Sample predictions:")
            for i in range(min(2, len(decoded_preds))):
                print(f"  Pred {i+1}: {decoded_preds[i][:100]}...")
                print(f"  Label {i+1}: {decoded_labels[i][:100]}...")
        
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
                
                print(f"[DEBUG] Computed metrics: BLEU-4={metrics['bleu_4']:.4f}, ROUGE-L={metrics['rouge_l']:.4f}")
                
            except Exception as e:
                print(f"⚠️ Warning: Could not compute metrics: {e}")
                import traceback
                traceback.print_exc()
                metrics["bleu_4"] = 0.0
                metrics["rouge_l"] = 0.0
        else:
            # Simple BLEU approximation using NLTK
            try:
                from nltk.translate.bleu_score import corpus_bleu
                references = [[label.split()] for label in decoded_labels]
                hypotheses = [pred.split() for pred in decoded_preds]
                metrics["bleu_4"] = corpus_bleu(references, hypotheses)
                print(f"[DEBUG] NLTK BLEU-4: {metrics['bleu_4']:.4f}")
            except Exception as e:
                print(f"⚠️ Warning: NLTK BLEU failed: {e}")
                metrics["bleu_4"] = 0.0
            metrics["rouge_l"] = 0.0
        
        return metrics
    
    def get_training_args(
        self,
        num_train_epochs: int = 10,
        per_device_train_batch_size: int = 16,  # ✅ Increased from 8 to 16
        gradient_accumulation_steps: int = 2,   # ✅ Reduced from 4 to 2 (same effective batch size)
        learning_rate: float = 1e-4,
        warmup_steps: int = 50,
        logging_steps: int = 50,
        eval_steps: int = None,
        save_steps: int = None,
        fp16: bool = True,
        weight_decay=0.01,
        optim="adamw_torch_fused",
        early_stopping_patience: int = 5  # ✅ Increased from 2 to 5 (more patience for improvement)
    ) -> Seq2SeqTrainingArguments:
        """
        Get default training arguments untuk task-specific training.
        
        Args:
            num_train_epochs: Number of training epochs
            per_device_train_batch_size: Batch size per device (default: 16, optimized for T4 GPU)
            gradient_accumulation_steps: Gradient accumulation steps (default: 2)
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
            gradient_checkpointing=False,  # ✅ Disabled - not needed with LoRA (only 0.36% trainable params)
            predict_with_generate=True,
            generation_max_length=self.max_length,
            load_best_model_at_end=True,
            metric_for_best_model="eval_bleu_4",
            greater_is_better=True,
            save_total_limit=2,
            report_to=["none"],
            weight_decay=0.01,           # sesuai referensi resmi LazarusNLP
            optim="adamw_torch_fused",   # sesuai referensi resmi LazarusNLP
            dataloader_num_workers=4,    # ✅ Increased from 2 to 4 for faster data loading
            dataloader_pin_memory=True,
            dataloader_prefetch_factor=2,  # ✅ Added prefetching for better GPU utilization
        )
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        training_args: Optional[Seq2SeqTrainingArguments] = None,
        early_stopping: bool = False,  # ✅ DISABLED: Let training run to completion
        early_stopping_patience: int = 5,
        resume_from_checkpoint: Union[bool, str] = False
    ) -> Dict[str, Any]:
        """
        Train model untuk task-specific AQG.
        
        Args:
            train_dataset: Training dataset (raw, will be preprocessed)
            eval_dataset: Evaluation dataset (optional)
            training_args: Custom training arguments (optional)
            early_stopping: Enable early stopping
            early_stopping_patience: Patience for early stopping
            resume_from_checkpoint: Resume training from checkpoint
                - True: Auto-detect last checkpoint in output_dir
                - False: Start fresh training
                - str: Path to specific checkpoint directory
            
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
        
        # Data collator - sesuai dokumentasi HuggingFace
        # https://huggingface.co/docs/transformers/main_classes/data_collator#transformers.DataCollatorForSeq2Seq
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            label_pad_token_id=-100,  # Padding labels akan di-mask dengan -100
            padding=True,  # Dynamic padding
            pad_to_multiple_of=8  # Untuk efisiensi GPU (Tensor Cores)
            # TIDAK menggunakan max_length - biarkan dynamic padding bekerja
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
            processing_class=self.tokenizer,
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
        
        # Handle checkpoint resumption
        checkpoint_path = None
        if resume_from_checkpoint:
            if isinstance(resume_from_checkpoint, str):
                # Specific checkpoint path provided
                checkpoint_path = resume_from_checkpoint
                print(f"🔄 Resuming from checkpoint: {checkpoint_path}")
            elif resume_from_checkpoint is True:
                # Auto-detect last checkpoint
                checkpoint_path = self._get_last_checkpoint()
                if checkpoint_path:
                    print(f"🔄 Auto-detected checkpoint: {checkpoint_path}")
                else:
                    print("⚠️ No checkpoint found - starting fresh training")
        else:
            print("🆕 Starting fresh training (no resume)")
        
        # Train
        print("Starting training...")
        train_result = self.trainer.train(resume_from_checkpoint=checkpoint_path)
        
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
    
    def _get_last_checkpoint(self) -> Optional[str]:
        """
        Get path to last checkpoint in output_dir.
        
        Returns:
            Path to last checkpoint, or None if no checkpoints found
        """
        import os
        import re
        
        if not self.output_dir.exists():
            return None
        
        # List all checkpoint directories
        checkpoints = []
        for item in os.listdir(self.output_dir):
            if item.startswith('checkpoint-'):
                # Extract step number
                match = re.search(r'checkpoint-(\d+)', item)
                if match:
                    step = int(match.group(1))
                    checkpoints.append((step, item))
        
        if not checkpoints:
            return None
        
        # Sort by step number and get the last one
        checkpoints.sort(key=lambda x: x[0])
        last_checkpoint = checkpoints[-1][1]
        
        return str(self.output_dir / last_checkpoint)