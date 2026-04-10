# IndoNanoT5 Fine-tuning untuk AQG

Pipeline fine-tuning IndoNanoT5 dengan LoRA untuk Automatic Question Generation (AQG) pada domain Python.

## Overview

Sistem ini mengimplementasikan two-stage fine-tuning:

1. **Domain Adaptation (Stage 1)**: Adaptasi model ke terminologi dan konteks Python
2. **Task-Specific AQG (Stage 2)**: Training untuk generasi soal kuis dengan format spesifik

### Model

- **Base Model**: LazarusNLP/IndoNanoT5-base (~248M parameters)
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Trainable Parameters**: ~1.24M (~0.5% of total)
- **Target Platform**: Google Colab T4 GPU (15GB VRAM)

## Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install transformers>=4.35.0 peft>=0.7.0 datasets>=2.15.0 accelerate>=0.25.0
pip install evaluate rouge_score bert_score
```

### 2. Prepare Datasets

Pastikan datasets tersedia di:
- Domain: `dataset_aqg/output_domain/` (340 entries)
- Task-Specific: `dataset_aqg/dataset-task-spesifc/` (1,262 entries)

### 3. Run Validation

```python
from finetuned.data.dataset_loader import DatasetLoader
from finetuned.model.model_setup import ModelSetup

# Load dataset
loader = DatasetLoader()
dataset = loader.load_dataset("dataset_aqg/output_domain/", split="train")

# Validate
validation_results = loader.validate_dataset(dataset)

# Setup model
model_setup = ModelSetup()
peft_model, tokenizer = model_setup.setup_model_for_training()
```

### 4. Training

Gunakan notebook yang tersedia:
- `notebooks/01_setup_and_validation.ipynb`: Setup dan validasi
- `notebooks/02_domain_adaptation.ipynb`: Domain adaptation training
- `notebooks/03_task_specific_training.ipynb`: Task-specific training

## Project Structure

```
src/finetuned/
├── data/
│   ├── dataset_loader.py       # Load dan validate datasets
│   └── tokenizer_tester.py     # Test tokenizer compatibility
├── model/
│   └── model_setup.py          # Setup model dengan LoRA
├── training/
│   ├── domain_trainer.py       # Domain adaptation trainer
│   └── task_trainer.py         # Task-specific trainer
├── evaluation/
│   ├── metrics_calculator.py   # Calculate metrics
│   └── model_evaluator.py      # Evaluate model
├── utils/
│   ├── checkpoint_manager.py   # Manage checkpoints
│   └── colab_helper.py         # Colab utilities
├── config/
│   └── training_config.yaml    # Training configuration
└── notebooks/
    ├── 01_setup_and_validation.ipynb
    ├── 02_domain_adaptation.ipynb
    └── 03_task_specific_training.ipynb
```

## Usage Examples

### Load Dataset

```python
from finetuned.data.dataset_loader import DatasetLoader

loader = DatasetLoader()
dataset = loader.load_dataset("dataset_aqg/output_domain/", split="train")
validation = loader.validate_dataset(dataset)
```

### Test Tokenizer

```python
from transformers import T5Tokenizer
from finetuned.data.tokenizer_tester import TokenizerTester

tokenizer = T5Tokenizer.from_pretrained("LazarusNLP/IndoNanoT5-base")
tester = TokenizerTester(tokenizer)

# Test markdown handling
markdown_results = tester.test_markdown_handling()

# Analyze token distribution
token_stats = loader.analyze_token_distribution(dataset, tokenizer)
```

### Setup Model

```python
from finetuned.model.model_setup import ModelSetup

model_setup = ModelSetup()
peft_model, tokenizer = model_setup.setup_model_for_training(
    model_name="LazarusNLP/IndoNanoT5-base"
)

# Check trainable parameters
stats = model_setup.print_trainable_parameters(peft_model)
# Expected: ~0.5% trainable
```

### Training (Domain Adaptation)

```python
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="./checkpoints/domain",
    num_train_epochs=6,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_steps=50,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    fp16=True,
)

trainer = Seq2SeqTrainer(
    model=peft_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

trainer.train()
```

## Configuration

Edit `config/training_config.yaml` untuk customize training parameters:

```yaml
domain_adaptation:
  model_name: "LazarusNLP/IndoNanoT5-base"
  dataset_dir: "dataset_aqg/output_domain/"
  
  lora:
    r: 8
    lora_alpha: 16
    lora_dropout: 0.1
    target_modules: ["q", "v"]
  
  training:
    num_train_epochs: 6
    per_device_train_batch_size: 8
    learning_rate: 2.0e-4
```

## Expected Results

### Domain Adaptation (Stage 1)
- Training loss: ~3.0 → ~1.5
- Validation loss: ~2.8 → ~1.8
- Training time: ~2-3 hours on T4 GPU

### Task-Specific AQG (Stage 2)
- BLEU-4: ~0.10 → ~0.35-0.45
- ROUGE-L: ~0.40-0.50
- BERTScore F1: ~0.75-0.85
- Training time: ~1-2 hours on T4 GPU

## Troubleshooting

### CUDA Out of Memory

```python
# Reduce batch size
training_args.per_device_train_batch_size = 4
training_args.gradient_accumulation_steps = 8

# Enable gradient checkpointing
peft_model.gradient_checkpointing_enable()
```

### Session Disconnect

Checkpoints disimpan setiap epoch di `./checkpoints/`. Resume training:

```python
# Load checkpoint
checkpoint_path = "./checkpoints/domain/checkpoint-epoch-3"
peft_model = PeftModel.from_pretrained(base_model, checkpoint_path)

# Continue training
trainer.train(resume_from_checkpoint=checkpoint_path)
```

### Dataset Loading Error

```python
# Verify file exists
from pathlib import Path
assert Path("dataset_aqg/output_domain/train.jsonl").exists()

# Check JSONL format
import json
with open("dataset_aqg/output_domain/train.jsonl") as f:
    for line in f:
        json.loads(line)  # Should not raise error
```

## Testing

Run unit tests:

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_dataset_loader.py

# Run with coverage
pytest --cov=src/finetuned tests/
```

## License

MIT License

## Citation

```bibtex
@misc{indonanot5-aqg,
  title={IndoNanoT5 Fine-tuning untuk Automatic Question Generation},
  author={Your Name},
  year={2026}
}
```
