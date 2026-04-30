"""
Adapter Trainer for Task-Specific Fine-tuning
Handles training configuration and execution for adapter-based fine-tuning.

COMPATIBILITY FIX: This module includes fixes for transformers 4.46+ compatibility
where num_items_in_batch parameter causes issues with adapter models.
"""

import numpy as np
import matplotlib.pyplot as plt
from transformers import (
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from typing import Dict, Any, Optional


class CompatibleSeq2SeqTrainer(Seq2SeqTrainer):
    """
    Custom Seq2SeqTrainer that handles num_items_in_batch parameter compatibility.
    
    Fixes compatibility issue between transformers 4.46+ and adapters library
    where 'num_items_in_batch' parameter causes TypeError.
    
    Solution: Override compute_loss to accept but ignore num_items_in_batch parameter.
    """
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute loss with compatibility for num_items_in_batch parameter.
        
        Args:
            model: The model to compute loss for
            inputs: Input batch
            return_outputs: Whether to return model outputs
            num_items_in_batch: Number of items in batch (ignored for compatibility)
            
        Returns:
            Loss value (and optionally outputs)
        """
        # Call parent's compute_loss WITHOUT num_items_in_batch parameter
        # This fixes the TypeError with adapters library
        return super().compute_loss(model, inputs, return_outputs=return_outputs)


class AdapterTrainer:
    """
    Trainer for adapter-based fine-tuning of seq2seq models.
    
    Handles:
    - Dataset preprocessing
    - Training configuration
    - Metrics computation
    - Training execution
    - Model saving
    - Visualization
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        metrics_calculator,
        output_dir: str,
        max_length: int = 512
    ):
        """
        Initialize adapter trainer.
        
        Args:
            model: Model with adapter layers
            tokenizer: Tokenizer for the model
            metrics_calculator: MetricsCalculator instance
            output_dir: Directory to save checkpoints
            max_length: Maximum sequence length
        """
        self.model = model
        self.tokenizer = tokenizer
        self.metrics_calculator = metrics_calculator
        self.output_dir = output_dir
        self.max_length = max_length
        self.trainer = None
        
    def preprocess_dataset(self, dataset):
        """
        Tokenize dataset with backward compatibility for v2/v3 formats.
        
        Args:
            dataset: HuggingFace dataset
            
        Returns:
            Tokenized dataset
        """
        def preprocess_function(examples):
            # Support both 'target' (v2) and 'output' (v3)
            target_field = "target" if "target" in examples else "output"
            
            # Tokenize inputs
            model_inputs = self.tokenizer(
                examples["input"],
                max_length=self.max_length,
                truncation=True
            )
            
            # Tokenize targets
            labels = self.tokenizer(
                text_target=examples[target_field],
                max_length=self.max_length,
                truncation=True
            )
            
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
        
        return dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing"
        )
    
    def setup_training(
        self,
        num_train_epochs: int = 8,
        per_device_train_batch_size: int = 4,
        per_device_eval_batch_size: int = 8,
        gradient_accumulation_steps: int = 2,
        learning_rate: float = 1e-4,
        warmup_steps: int = 50,
        weight_decay: float = 0.01,
        save_total_limit: int = 2,
        **kwargs
    ) -> Seq2SeqTrainingArguments:
        """
        Setup training arguments for adapter fine-tuning.
        
        Args:
            num_train_epochs: Number of training epochs
            per_device_train_batch_size: Batch size per device
            per_device_eval_batch_size: Eval batch size per device
            gradient_accumulation_steps: Gradient accumulation steps
            learning_rate: Learning rate
            warmup_steps: Warmup steps
            weight_decay: Weight decay for regularization
            save_total_limit: Maximum number of checkpoints to keep
            **kwargs: Additional training arguments
            
        Returns:
            Seq2SeqTrainingArguments instance
        """
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            
            # Training configuration
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            
            # Optimizer
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            optim="adamw_torch_fused",
            
            # Memory optimization
            gradient_checkpointing=True,
            fp16=True,
            
            # Evaluation
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_bleu_4",
            greater_is_better=True,
            save_total_limit=save_total_limit,
            
            # Logging
            logging_steps=10,
            report_to=["none"],
            
            # Generation
            predict_with_generate=True,
            generation_max_length=self.max_length,
            
            # DataLoader
            dataloader_num_workers=2,
            dataloader_pin_memory=True,
            
            **kwargs
        )
        
        print(f"\n{'='*60}")
        print("TRAINING CONFIGURATION")
        print(f"{'='*60}")
        print(f"Epochs: {training_args.num_train_epochs}")
        print(f"Batch size: {training_args.per_device_train_batch_size}")
        print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
        print(f"Learning rate: {training_args.learning_rate}")
        print(f"Warmup steps: {training_args.warmup_steps}")
        print(f"FP16: {training_args.fp16}")
        print(f"Gradient checkpointing: {training_args.gradient_checkpointing}")
        
        return training_args
    
    def compute_metrics(self, eval_preds):
        """
        Compute BLEU and ROUGE metrics during evaluation.
        
        Args:
            eval_preds: Tuple of (predictions, labels)
            
        Returns:
            Dictionary of metrics
        """
        predictions, labels = eval_preds
        
        # Handle negative values
        if hasattr(predictions, '__iter__'):
            predictions = np.where(predictions < 0, 0, predictions)
        
        # Decode predictions
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        # Decode labels (replace -100 with pad token)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Strip whitespace
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]
        
        try:
            bleu_results = self.metrics_calculator.compute_bleu(decoded_preds, decoded_labels)
            rouge_results = self.metrics_calculator.compute_rouge(decoded_preds, decoded_labels)
            
            return {
                "bleu_4": bleu_results.get("bleu", 0.0),
                "rouge_l": rouge_results.get("rougeL", 0.0)
            }
        except Exception as e:
            print(f"Warning: Metrics computation failed: {e}")
            return {"bleu_4": 0.0, "rouge_l": 0.0}
    
    def train(
        self,
        train_dataset,
        eval_dataset,
        training_args: Optional[Seq2SeqTrainingArguments] = None,
        early_stopping_patience: int = 2,
        resume_from_checkpoint: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Train the model with adapter layers.
        
        Args:
            train_dataset: Training dataset (raw, will be preprocessed)
            eval_dataset: Evaluation dataset (raw, will be preprocessed)
            training_args: Training arguments (if None, will use defaults)
            early_stopping_patience: Patience for early stopping
            resume_from_checkpoint: Whether to resume from checkpoint (True/False/path)
                - True: Auto-detect and resume from last checkpoint
                - False/None: Start fresh training
                - str: Resume from specific checkpoint path
            
        Returns:
            Training results dictionary
        """
        import os
        
        print(f"\n{'='*60}")
        print("PREPROCESSING DATASETS")
        print(f"{'='*60}")
        
        # Preprocess datasets
        train_processed = self.preprocess_dataset(train_dataset)
        eval_processed = self.preprocess_dataset(eval_dataset)
        print(f"✓ Datasets tokenized")
        
        # Setup training args if not provided
        if training_args is None:
            training_args = self.setup_training()
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            label_pad_token_id=-100,
            padding=True,
            pad_to_multiple_of=8
        )
        print(f"✓ Data collator configured")
        
        # Initialize trainer with compatibility fix for transformers 4.46+
        self.trainer = CompatibleSeq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_processed,
            eval_dataset=eval_processed,
            processing_class=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]
        )
        print(f"✓ Trainer initialized (with transformers 4.46+ compatibility fix)")
        
        # Handle resume_from_checkpoint logic
        checkpoint_to_resume = None
        if resume_from_checkpoint is True:
            # Auto-detect last checkpoint
            if os.path.exists(self.output_dir):
                checkpoints = [
                    d for d in os.listdir(self.output_dir) 
                    if d.startswith('checkpoint-') and os.path.isdir(os.path.join(self.output_dir, d))
                ]
                if checkpoints:
                    # Sort by checkpoint number
                    checkpoints_sorted = sorted(checkpoints, key=lambda x: int(x.split('-')[1]))
                    checkpoint_to_resume = os.path.join(self.output_dir, checkpoints_sorted[-1])
                    print(f"📂 Found {len(checkpoints)} checkpoint(s): {checkpoints_sorted}")
                    print(f"🔄 Resuming from: {checkpoints_sorted[-1]}")
                else:
                    print("🆕 No checkpoints found - starting fresh training")
        elif isinstance(resume_from_checkpoint, str):
            # Use specific checkpoint path
            checkpoint_to_resume = resume_from_checkpoint
            print(f"🔄 Resuming from specified checkpoint: {checkpoint_to_resume}")
        else:
            print("🆕 Starting fresh training (no resume)")
        
        # Train
        print(f"\n{'='*60}")
        print("STARTING TRAINING")
        print(f"{'='*60}")
        print(f"Training with Adapter Layers (d=64, ~3.6% trainable params)")
        print(f"Expected time: 6-8 hours on T4 GPU")
        print(f"Total epochs: {training_args.num_train_epochs}")
        print(f"{'='*60}\n")
        
        # Pass checkpoint_to_resume to underlying HuggingFace Trainer
        # This ensures training continues from checkpoint to completion
        train_result = self.trainer.train(resume_from_checkpoint=checkpoint_to_resume)
        
        return {
            "training_loss": train_result.training_loss,
            "metrics": train_result.metrics
        }
    
    def save_adapter(self, adapter_name: str = "mcq_generation", save_config: Optional[Dict] = None):
        """
        Save adapter weights and configuration.
        
        Args:
            adapter_name: Name of the adapter to save
            save_config: Optional configuration dictionary to save
            
        Returns:
            Path where adapter was saved
        """
        import json
        from pathlib import Path
        
        adapter_path = f'{self.output_dir}/adapter_{adapter_name}'
        
        print(f"\n{'='*60}")
        print("SAVING ADAPTER WEIGHTS")
        print(f"{'='*60}")
        
        # Save adapter
        self.model.save_adapter(adapter_path, adapter_name)
        print(f'✓ Adapter weights saved to: {adapter_path}')
        
        # Save tokenizer
        self.tokenizer.save_pretrained(adapter_path)
        print(f'✓ Tokenizer saved')
        
        # Save config if provided
        if save_config:
            config_path = f'{adapter_path}/training_config.json'
            with open(config_path, 'w') as f:
                json.dump(save_config, f, indent=2)
            print(f'✓ Config saved')
        
        return adapter_path
    
    def plot_training_curves(self, save_path: Optional[str] = None):
        """
        Plot training loss and evaluation metrics.
        
        Args:
            save_path: Path to save the plot (if None, only displays)
        """
        if self.trainer is None:
            print("⚠ No training history available")
            return
        
        training_history = self.trainer.state.log_history
        
        train_loss = []
        eval_bleu = []
        epochs = []
        
        for log in training_history:
            if "loss" in log:
                train_loss.append(log["loss"])
                epochs.append(log.get("epoch", len(train_loss)))
            if "eval_bleu_4" in log:
                eval_bleu.append(log["eval_bleu_4"])
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss plot
        if train_loss:
            axes[0].plot(epochs[:len(train_loss)], train_loss, marker="o")
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Loss")
            axes[0].set_title("Training Loss")
            axes[0].grid(True)
        
        # BLEU plot
        if eval_bleu:
            axes[1].plot(range(1, len(eval_bleu) + 1), eval_bleu, marker="s", color="green")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("BLEU-4")
            axes[1].set_title("BLEU-4 Score")
            axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f'✓ Plot saved to {save_path}')
        
        plt.show()
