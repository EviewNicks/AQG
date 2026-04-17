"""Domain Adaptation Trainer untuk Stage 1 fine-tuning."""

import torch
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from datasets import Dataset
from typing import Dict, Any, Optional, List
import numpy as np
from pathlib import Path
import json


class DomainAdaptationTrainer:
    """
    Trainer untuk domain adaptation stage.
    
    Train model pada domain Python untuk adaptasi terminologi dan konteks.
    Uses HuggingFace Seq2SeqTrainer dengan LoRA adapters.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        output_dir: str = "./checkpoints/domain",
        max_length: int = 512
    ):
        """
        Initialize DomainAdaptationTrainer.
        
        Args:
            model: PeftModel dengan LoRA adapters
            tokenizer: PreTrainedTokenizer
            output_dir: Directory untuk save checkpoints
            max_length: Maximum sequence length
        """
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = Path(output_dir)
        self.max_length = max_length
        self.trainer = None
        self.training_history = []
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def preprocess_dataset(
        self,
        dataset: Dataset,
        batch_size: int = 32
    ) -> Dataset:
        """
        Preprocess dataset untuk training.
        
        Args:
            dataset: Raw dataset dengan 'input' dan 'target' fields
            batch_size: Batch size untuk processing (default: 32)
            
        Returns:
            Preprocessed dataset dengan tokenized fields
            
        Raises:
            ValueError: Jika dataset size mismatch setelah preprocessing
        """
        print(f"Preprocessing {len(dataset)} samples...")
        
        # Debug: cek kolom yang ada
        print(f"  Columns: {dataset.column_names}")
        
        # Verify kolom metadata sudah di-drop
        if 'metadata' in dataset.column_names:
            print("  ⚠ WARNING: Kolom 'metadata' masih ada!")
            print("  Recommendation: Jalankan dataset.remove_columns(['metadata']) sebelum training")
        
        def tokenize_function(examples):
            # Tokenize inputs as-is — tidak menggunakan prefix maupun .lower().
            # Alasan:
            #   1. IndoNanoT5 dilatih pada CulturaX yang case-sensitive; lowercase paksa
            #      merusak token teknis Python seperti True/False/None, nama class (ValueError,
            #      DataFrame), dan proper noun bahasa Indonesia.
            #   2. Domain dataset hanya berisi format qa_generic setelah span_corruption
            #      dihapus — tidak ada kebutuhan task prefix.
            #   3. Konsistensi: inference juga tidak perlu lowercase.
            model_inputs = self.tokenizer(
                examples["input"],
                max_length=self.max_length,
                truncation=True,
                padding=False
            )
            
            # Tokenize targets (transformers 5.x: gunakan text_target)
            labels = self.tokenizer(
                text_target=examples["target"],
                max_length=self.max_length,
                truncation=True,
                padding=False
            )
            
            # CRITICAL: Replace pad_token_id dengan -100 agar padding tidak
            # dihitung dalam cross-entropy loss. Tanpa ini loss bisa 30-40+
            pad_id = self.tokenizer.pad_token_id
            model_inputs["labels"] = [
                [(token if token != pad_id else -100) for token in seq]
                for seq in labels["input_ids"]
            ]
            return model_inputs
        
        # Tokenize dataset
        cols_to_remove = [c for c in dataset.column_names if c not in ["input_ids", "attention_mask", "labels"]]
        
        print(f"  Batch size: {batch_size}")
        print(f"  Removing columns: {cols_to_remove}")
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            batch_size=batch_size,
            remove_columns=cols_to_remove,
            load_from_cache_file=False,
            desc="Tokenizing"
        )
        
        print(f"✓ Preprocessed {len(tokenized_dataset)} samples")
        
        # Debug: verify label masking benar
        sample_labels = tokenized_dataset[0]["labels"]
        n_valid = sum(1 for l in sample_labels if l != -100)
        n_masked = sum(1 for l in sample_labels if l == -100)
        print(f"  Sample label check: {n_valid} valid tokens, {n_masked} masked (-100)")        
        # FAIL FAST: Assertion untuk detect bug preprocessing
        if len(tokenized_dataset) != len(dataset):
            raise ValueError(
                f"❌ CRITICAL: Dataset size mismatch after preprocessing!\n"
                f"   Input:  {len(dataset)} samples\n"
                f"   Output: {len(tokenized_dataset)} samples\n"
                f"   Lost:   {len(dataset) - len(tokenized_dataset)} samples\n\n"
                f"Possible causes:\n"
                f"  1. Kolom 'metadata' dengan nested dict inconsistent schema\n"
                f"  2. HuggingFace datasets 4.0 bug dengan batched=True\n\n"
                f"Solution:\n"
                f"  Jalankan dataset.remove_columns(['metadata']) SEBELUM training.\n"
                f"  Contoh: train_dataset = train_dataset.remove_columns(['metadata'])"
            )
        
        return tokenized_dataset
    
    def get_training_args(
        self,
        num_train_epochs: int = 6,
        per_device_train_batch_size: int = 8,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        warmup_steps: int = 10,
        logging_steps: int = 50,
        eval_steps: int = None,
        save_steps: int = None,
        fp16: bool = True,
        early_stopping_patience: int = 2
    ) -> Seq2SeqTrainingArguments:
        """
        Get default training arguments untuk domain adaptation.
        
        Hyperparameter mengikuti referensi resmi LazarusNLP/IndoT5:
        - learning_rate: 1e-3 (dari run_summarization.py resmi)
        - weight_decay: 0.01 (dari run_summarization.py resmi)
        - optim: adamw_torch_fused (dari run_summarization.py resmi)
        
        Args:
            num_train_epochs: Number of training epochs
            per_device_train_batch_size: Batch size per device
            gradient_accumulation_steps: Gradient accumulation steps
            learning_rate: Learning rate (default 1e-3 sesuai referensi resmi)
            weight_decay: Weight decay untuk regularization (default 0.01)
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
            weight_decay=weight_decay,
            optim="adamw_torch_fused",
            warmup_steps=warmup_steps,
            eval_strategy="epoch" if eval_steps is None else "steps",
            eval_steps=eval_steps,
            save_strategy="epoch" if save_steps is None else "steps",
            save_steps=save_steps,
            logging_steps=logging_steps,
            fp16=fp16,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            predict_with_generate=False,  # False untuk domain adaptation — loss-based training
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=3,
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
        Train model untuk domain adaptation.
        
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
        print("STARTING DOMAIN ADAPTATION TRAINING")
        print("=" * 60 + "\n")
        
        # Pastikan model di GPU sebelum training
        import torch
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            # Wajib untuk gradient_checkpointing + PEFT/LoRA agar gradient flow benar
            self.model.enable_input_require_grads()
            print(f"✓ Model moved to GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠ GPU not available, training on CPU (will be slow)")
        
        # Verifikasi device — fail fast jika model masih di CPU padahal GPU tersedia
        device = next(self.model.parameters()).device
        print(f"  Model device: {device}")
        if torch.cuda.is_available() and str(device) == 'cpu':
            raise RuntimeError("Model masih di CPU padahal GPU tersedia! Cek notebook cell yang memanggil .cpu()")
        
        # Preprocess datasets
        print("Preprocessing datasets...")
        train_dataset = self.preprocess_dataset(train_dataset)
        if eval_dataset is not None:
            eval_dataset = self.preprocess_dataset(eval_dataset)
        
        # Verifikasi ukuran dataset setelah preprocessing
        print(f"\n=== Dataset Size After Preprocessing ===")
        print(f"Train samples (actual): {len(train_dataset)}")
        if eval_dataset is not None:
            print(f"Eval samples (actual):  {len(eval_dataset)}")
        if len(train_dataset) < 100:
            print(f"⚠ WARNING: Train dataset sangat kecil ({len(train_dataset)} sampel)!")
            print(f"  Kemungkinan ada bug di preprocessing. Cek kolom dataset.")
        
        # Get training args if not provided
        if training_args is None:
            training_args = self.get_training_args(
                early_stopping_patience=early_stopping_patience
            )
        
        # Data collator — label_pad_token_id=-100 wajib agar padding
        # tidak dihitung dalam cross-entropy loss
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
            label_pad_token_id=-100,
            pad_to_multiple_of=8
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
    
    def save_best_model(self, output_name: str = "indot5-python-domain") -> str:
        """
        Save best checkpoint sebagai final model.
        
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
        
        print(f"\n✓ Best model saved to: {save_path}")
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
            
            # Extract loss values
            train_loss = []
            eval_loss = []
            epochs = []
            
            for log in self.training_history:
                if "loss" in log:
                    train_loss.append(log["loss"])
                    epochs.append(log.get("epoch", len(train_loss)))
                if "eval_loss" in log:
                    eval_loss.append(log["eval_loss"])
            
            # Plot
            plt.figure(figsize=(10, 6))
            if train_loss:
                plt.plot(epochs[:len(train_loss)], train_loss, label="Training Loss", marker="o")
            if eval_loss:
                plt.plot(range(1, len(eval_loss) + 1), eval_loss, label="Validation Loss", marker="s")
            
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Domain Adaptation Training Curves")
            plt.legend()
            plt.grid(True)
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                print(f"✓ Training curves saved to {save_path}")
            else:
                plt.show()
            
            plt.close()
            
        except ImportError:
            print("matplotlib not installed. Skipping plot generation.")