# Design Document: IndoNanoT5 Fine-tuning untuk AQG

## Overview

Sistem ini mengimplementasikan two-stage fine-tuning pipeline untuk IndoNanoT5 model menggunakan LoRA (Low-Rank Adaptation) untuk task Automatic Question Generation (AQG) pada domain Python. Design ini mengoptimalkan untuk Google Colab environment dengan T4 GPU (15GB VRAM) dan menggunakan modular architecture untuk reusability dan maintainability.

**Model Selection Rationale:** IndoNanoT5-base dipilih karena:
- Monolingual Indonesian (pre-trained dari nol pada CulturaX 23M dokumen)
- Parameter lebih efisien (~248M vs ~580M IndoT5-base)
- Performa NLG lebih unggul (BLEU: 4.07 vs 1.89 pada XPersona)
- Lebih cocok untuk task generatif bahasa Indonesia

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     FINE-TUNING PIPELINE                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 0: PREPARATION                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   Dataset    │  │  Tokenizer   │  │   Model      │         │
│  │   Loader     │  │   Tester     │  │   Setup      │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              STAGE 1: DOMAIN ADAPTATION                         │
│  ┌──────────────────────────────────────────────────┐          │
│  │  Domain Dataset (340 entries)                    │          │
│  │  - span_corruption: 170 entries                  │          │
│  │  - qa_generic: 170 entries                       │          │
│  └──────────────────────────────────────────────────┘          │
│                       │                                         │
│                       ▼                                         │
│  ┌──────────────────────────────────────────────────┐          │
│  │  Trainer (6 epochs, lr=2e-4)                     │          │
│  │  - Batch size: 8                                 │          │
│  │  - Gradient accumulation: 4                      │          │
│  │  - Warmup steps: 50                              │          │
│  └──────────────────────────────────────────────────┘          │
│                       │                                         │
│                       ▼                                         │
│  ┌──────────────────────────────────────────────────┐          │
│  │  Output: indot5-python-domain                    │          │
│  └──────────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│            STAGE 2: TASK-SPECIFIC AQG                           │
│  ┌──────────────────────────────────────────────────┐          │
│  │  Task-Specific Dataset (1,262 entries)           │          │
│  │  - MCQ: 631 entries                              │          │
│  │  - Code Completion: 631 entries                  │          │
│  └──────────────────────────────────────────────────┘          │
│                       │                                         │
│                       ▼                                         │
│  ┌──────────────────────────────────────────────────┐          │
│  │  Trainer (3 epochs, lr=1e-4)                     │          │
│  │  - Batch size: 8                                 │          │
│  │  - Gradient accumulation: 4                      │          │
│  │  - Warmup steps: 30                              │          │
│  └──────────────────────────────────────────────────┘          │
│                       │                                         │
│                       ▼                                         │
│  ┌──────────────────────────────────────────────────┐          │
│  │  Output: indot5-python-aqg                       │          │
│  └──────────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EVALUATION & TESTING                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   Metrics    │  │   Sample     │  │   Report     │         │
│  │  Calculator  │  │  Generator   │  │  Generator   │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

## Component Descriptions

### 1. Data Module (`data/`)

#### 1.1 DatasetLoader

**Purpose:** Load dan validate JSONL datasets untuk domain adaptation dan task-specific training.

**Interface:**
```python
class DatasetLoader:
    def load_dataset(
        self, 
        data_dir: str, 
        split: str = "train"
    ) -> Dataset:
        """
        Load dataset dari JSONL files.
        
        Args:
            data_dir: Path ke directory dataset (e.g., "dataset_aqg/output_domain/")
            split: "train", "validation", atau "test"
            
        Returns:
            HuggingFace Dataset object
        """
        pass
    
    def validate_dataset(self, dataset: Dataset) -> Dict[str, Any]:
        """
        Validate dataset structure dan integrity.
        
        Returns:
            Dict dengan validation results:
            - total_entries: int
            - missing_fields: List[str]
            - duplicate_count: int
            - avg_input_length: float
            - avg_target_length: float
        """
        pass
    
    def analyze_token_distribution(
        self, 
        dataset: Dataset, 
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512
    ) -> Dict[str, Any]:
        """
        Analyze distribusi panjang token.
        
        Returns:
            Dict dengan statistics:
            - mean_length: float
            - median_length: float
            - max_length: int
            - pct_exceeding_limit: float
            - histogram: List[int]
        """
        pass
```

**Data Model:**
```python
# Domain Adaptation Entry
{
    "input": str,      # Context atau masked text
    "target": str,     # Expected output
    "metadata": {
        "format": str,           # "span_corruption" atau "qa_generic"
        "source_file": str,      # Original markdown file
        "section_heading": str,  # Section title
        "chunk_id": int         # Chunk identifier
    }
}

# Task-Specific AQG Entry
{
    "input": str,      # Context + instruction prompt
    "target": str,     # Structured output (question + answer + distractors)
    "metadata": {
        "format": str,           # "MCQ" atau "Code Completion"
        "concept": str,          # Main concept
        "difficulty": str,       # "easy", "medium", "hard"
        "source_file": str,
        "section_heading": str
    }
}
```

#### 1.2 TokenizerTester

**Purpose:** Test tokenizer compatibility dengan markdown formatting dan code blocks.

**Interface:**
```python
class TokenizerTester:
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
    
    def test_markdown_handling(self) -> Dict[str, bool]:
        """
        Test tokenization untuk markdown special characters.
        
        Returns:
            Dict dengan test results untuk:
            - headings (#, ##)
            - bold (**text**)
            - code (`code`)
            - newlines (\n)
        """
        pass
    
    def test_code_block_integrity(self, samples: List[str]) -> List[Dict]:
        """
        Test apakah code blocks tetap intact setelah tokenization.
        
        Returns:
            List of dicts dengan:
            - original: str
            - tokenized: List[int]
            - decoded: str
            - integrity_preserved: bool
        """
        pass
    
    def detect_oov_tokens(self, dataset: Dataset) -> Dict[str, int]:
        """
        Detect out-of-vocabulary tokens.
        
        Returns:
            Dict mapping OOV token -> frequency
        """
        pass
```

### 2. Model Module (`model/`)

#### 2.1 ModelSetup

**Purpose:** Setup IndoT5 model dengan LoRA adapters untuk parameter-efficient fine-tuning.

**Interface:**
```python
class ModelSetup:
    def load_base_model(
        self, 
        model_name: str = "LazarusNLP/IndoNanoT5-base"
    ) -> T5ForConditionalGeneration:
        """
        Load pre-trained IndoNanoT5 base model.
        
        Returns:
            T5ForConditionalGeneration model (~248M parameters)
        """
        pass
    
    def apply_lora(
        self, 
        model: T5ForConditionalGeneration,
        lora_config: LoraConfig
    ) -> PeftModel:
        """
        Apply LoRA adapters ke model.
        
        Args:
            lora_config: LoraConfig dengan:
                - r (rank): 8
                - lora_alpha: 16
                - lora_dropout: 0.1
                - target_modules: ["q", "v"]
                
        Returns:
            PeftModel dengan ~1.25M trainable parameters (~0.5%)
        """
        pass
    
    def print_trainable_parameters(self, model: PeftModel) -> None:
        """
        Print summary trainable vs total parameters.
        """
        pass
    
    def check_gpu_memory(self) -> Dict[str, float]:
        """
        Check GPU memory availability.
        
        Returns:
            Dict dengan:
            - total_memory_gb: float
            - allocated_memory_gb: float
            - free_memory_gb: float
        """
        pass
```

**LoRA Configuration:**
```python
lora_config = LoraConfig(
    r=8,                          # Rank untuk low-rank matrices
    lora_alpha=16,                # Scaling factor (alpha/r = 2.0)
    lora_dropout=0.1,             # Dropout untuk regularization
    target_modules=["q", "v"],    # Apply LoRA ke query dan value projections
    bias="none",                  # Tidak train bias parameters
    task_type="SEQ_2_SEQ_LM"      # Task type untuk T5
)
```

### 3. Training Module (`training/`)

#### 3.1 DomainAdaptationTrainer

**Purpose:** Train model pada domain Python untuk adaptasi terminologi dan konteks.

**Interface:**
```python
class DomainAdaptationTrainer:
    def __init__(
        self,
        model: PeftModel,
        tokenizer: PreTrainedTokenizer,
        output_dir: str = "./checkpoints/domain"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = output_dir
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        training_args: TrainingArguments
    ) -> TrainingOutput:
        """
        Train model untuk domain adaptation.
        
        Args:
            training_args: TrainingArguments dengan:
                - num_train_epochs: 6
                - per_device_train_batch_size: 8
                - gradient_accumulation_steps: 4
                - learning_rate: 2e-4
                - warmup_steps: 50
                - evaluation_strategy: "epoch"
                - save_strategy: "epoch"
                - logging_steps: 50
                
        Returns:
            TrainingOutput dengan training history
        """
        pass
    
    def save_best_model(self, output_name: str = "indot5-python-domain") -> None:
        """
        Save best checkpoint sebagai final model.
        """
        pass
```

**Expected Training Metrics:**
- Initial training loss: ~3.0
- Final training loss: ~1.5
- Initial validation loss: ~2.8
- Final validation loss: ~1.8
- Perplexity: Decreases dari ~16.0 ke ~6.0

#### 3.2 TaskSpecificTrainer

**Purpose:** Train model untuk AQG task dengan format output spesifik.

**Interface:**
```python
class TaskSpecificTrainer:
    def __init__(
        self,
        model: PeftModel,
        tokenizer: PreTrainedTokenizer,
        output_dir: str = "./checkpoints/aqg"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = output_dir
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        training_args: TrainingArguments
    ) -> TrainingOutput:
        """
        Train model untuk task-specific AQG.
        
        Args:
            training_args: TrainingArguments dengan:
                - num_train_epochs: 3
                - per_device_train_batch_size: 8
                - gradient_accumulation_steps: 4
                - learning_rate: 1e-4
                - warmup_steps: 30
                - evaluation_strategy: "epoch"
                - save_strategy: "epoch"
                - load_best_model_at_end: True
                - metric_for_best_model: "eval_bleu"
                
        Returns:
            TrainingOutput dengan training history
        """
        pass
    
    def save_final_model(self, output_name: str = "indot5-python-aqg") -> None:
        """
        Save final trained model.
        """
        pass
```

**Expected Training Metrics:**
- Initial BLEU-4: ~0.10
- Final BLEU-4: ~0.35-0.45
- ROUGE-L: ~0.40-0.50
- BERTScore F1: ~0.75-0.85

### 4. Evaluation Module (`evaluation/`)

#### 4.1 MetricsCalculator

**Purpose:** Calculate evaluation metrics untuk text generation quality.

**Interface:**
```python
class MetricsCalculator:
    def __init__(self):
        self.bleu = load_metric("bleu")
        self.rouge = load_metric("rouge")
        self.bertscore = load_metric("bertscore")
    
    def compute_bleu(
        self, 
        predictions: List[str], 
        references: List[str]
    ) -> Dict[str, float]:
        """
        Compute BLEU-1, BLEU-2, BLEU-3, BLEU-4.
        """
        pass
    
    def compute_rouge(
        self, 
        predictions: List[str], 
        references: List[str]
    ) -> Dict[str, float]:
        """
        Compute ROUGE-1, ROUGE-2, ROUGE-L.
        """
        pass
    
    def compute_bertscore(
        self, 
        predictions: List[str], 
        references: List[str],
        lang: str = "id"
    ) -> Dict[str, float]:
        """
        Compute BERTScore (Precision, Recall, F1).
        """
        pass
    
    def compute_diversity(
        self, 
        predictions: List[str]
    ) -> Dict[str, float]:
        """
        Compute Distinct-1 dan Distinct-2 untuk diversity.
        """
        pass
```

#### 4.2 ModelEvaluator

**Purpose:** Evaluate trained model pada test set dan generate sample outputs.

**Interface:**
```python
class ModelEvaluator:
    def __init__(
        self,
        model: PeftModel,
        tokenizer: PreTrainedTokenizer,
        metrics_calculator: MetricsCalculator
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.metrics = metrics_calculator
    
    def evaluate_on_test_set(
        self, 
        test_dataset: Dataset
    ) -> Dict[str, float]:
        """
        Evaluate model pada entire test set.
        
        Returns:
            Dict dengan all metrics:
            - bleu_1, bleu_2, bleu_3, bleu_4
            - rouge_1, rouge_2, rouge_l
            - bertscore_precision, bertscore_recall, bertscore_f1
            - distinct_1, distinct_2
        """
        pass
    
    def generate_samples(
        self, 
        test_dataset: Dataset, 
        num_samples: int = 20
    ) -> List[Dict[str, str]]:
        """
        Generate output untuk random samples.
        
        Returns:
            List of dicts dengan:
            - input: str
            - reference: str
            - prediction: str
            - bleu_score: float
        """
        pass
    
    def compare_with_baseline(
        self,
        finetuned_metrics: Dict[str, float],
        baseline_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate improvement percentage.
        
        Returns:
            Dict dengan improvement untuk setiap metric
        """
        pass
```

### 5. Utilities Module (`utils/`)

#### 5.1 CheckpointManager

**Purpose:** Manage checkpoint saving, loading, dan cleanup.

**Interface:**
```python
class CheckpointManager:
    def __init__(
        self,
        checkpoint_dir: str,
        max_checkpoints: int = 3,
        drive_backup: bool = True
    ):
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
        self.drive_backup = drive_backup
    
    def save_checkpoint(
        self,
        model: PeftModel,
        optimizer: Optimizer,
        epoch: int,
        metrics: Dict[str, float]
    ) -> str:
        """
        Save checkpoint dengan metadata.
        
        Returns:
            Path ke saved checkpoint
        """
        pass
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load checkpoint untuk resume training.
        
        Returns:
            Dict dengan model, optimizer, epoch, metrics
        """
        pass
    
    def cleanup_old_checkpoints(self) -> None:
        """
        Keep only last N checkpoints.
        """
        pass
    
    def backup_to_drive(self, checkpoint_path: str) -> None:
        """
        Copy checkpoint ke Google Drive.
        """
        pass
```

#### 5.2 ColabHelper

**Purpose:** Utility functions untuk Google Colab environment.

**Interface:**
```python
class ColabHelper:
    @staticmethod
    def check_gpu() -> Dict[str, Any]:
        """
        Check GPU availability dan specifications.
        
        Returns:
            Dict dengan:
            - gpu_available: bool
            - gpu_name: str
            - gpu_memory_gb: float
        """
        pass
    
    @staticmethod
    def mount_drive() -> bool:
        """
        Mount Google Drive untuk persistent storage.
        """
        pass
    
    @staticmethod
    def install_dependencies() -> None:
        """
        Install required packages:
        - transformers
        - peft
        - datasets
        - accelerate
        - bitsandbytes
        - evaluate
        - rouge_score
        - bert_score
        """
        pass
    
    @staticmethod
    def setup_wandb(project_name: str = "indot5-aqg") -> None:
        """
        Setup Weights & Biases untuk experiment tracking.
        """
        pass
```

## Error Handling Strategy

### 1. CUDA Out of Memory

**Detection:** `torch.cuda.OutOfMemoryError`

**Recovery Strategy:**
1. Clear CUDA cache: `torch.cuda.empty_cache()`
2. Reduce batch size by 50%
3. Increase gradient accumulation steps proportionally
4. Retry training
5. If still fails, suggest using gradient checkpointing

### 2. Session Disconnect

**Detection:** Colab session timeout atau network interruption

**Recovery Strategy:**
1. Checkpoints saved setiap epoch automatically
2. Resume script loads latest checkpoint
3. Continue training dari last completed epoch
4. Backup checkpoints ke Google Drive setiap epoch

### 3. Dataset Loading Failure

**Detection:** `FileNotFoundError`, `JSONDecodeError`

**Recovery Strategy:**
1. Log error dengan file path yang bermasalah
2. Verify file exists dan readable
3. Check JSONL format validity
4. Provide clear error message dengan suggested fix

### 4. Model Loading Failure

**Detection:** `OSError`, `ConnectionError`

**Recovery Strategy:**
1. Check internet connection
2. Retry download dengan exponential backoff
3. Fallback ke cached model jika available
4. Provide manual download instructions

### 5. Validation Metrics Degradation

**Detection:** Validation loss increases untuk 2 consecutive epochs

**Recovery Strategy:**
1. Trigger early stopping
2. Load best checkpoint berdasarkan validation loss
3. Log warning dengan metrics history
4. Suggest hyperparameter adjustments

## Testing Strategy

### Unit Tests

1. **DatasetLoader Tests:**
   - Test loading valid JSONL files
   - Test handling missing fields
   - Test duplicate detection
   - Test token distribution analysis

2. **TokenizerTester Tests:**
   - Test markdown character handling
   - Test code block preservation
   - Test OOV token detection

3. **ModelSetup Tests:**
   - Test LoRA application
   - Test trainable parameter calculation
   - Test GPU memory check

4. **MetricsCalculator Tests:**
   - Test BLEU calculation dengan known examples
   - Test ROUGE calculation
   - Test diversity metrics

### Integration Tests

1. **End-to-End Domain Adaptation:**
   - Load domain dataset
   - Setup model dengan LoRA
   - Train untuk 1 epoch
   - Verify checkpoint saved
   - Verify metrics logged

2. **End-to-End Task-Specific Training:**
   - Load checkpoint dari Stage 1
   - Load task-specific dataset
   - Train untuk 1 epoch
   - Verify final model saved
   - Verify evaluation metrics computed

### Manual Testing

1. **Qualitative Output Analysis:**
   - Generate 20 sample outputs
   - Manually review untuk:
     - Format correctness
     - Semantic coherence
     - Distractor plausibility
     - Code syntax validity

2. **Colab Environment Testing:**
   - Test pada fresh Colab session
   - Verify GPU detection
   - Verify Drive mounting
   - Verify dependency installation
   - Verify training completes successfully

## Performance Considerations

### Memory Optimization

1. **LoRA Efficiency:**
   - Only ~0.5% parameters trainable
   - Reduces memory footprint significantly
   - Enables training pada T4 GPU (15GB)

2. **Gradient Accumulation:**
   - Effective batch size: 8 × 4 = 32
   - Reduces memory usage per step
   - Maintains training stability

3. **Mixed Precision Training:**
   - Use fp16 untuk faster training
   - Reduces memory usage by ~50%
   - Minimal impact pada model quality

### Training Speed

1. **Expected Training Time:**
   - Domain Adaptation (6 epochs): ~2-3 hours
   - Task-Specific (3 epochs): ~1-2 hours
   - Total: ~3-5 hours pada T4 GPU

2. **Optimization Techniques:**
   - DataLoader num_workers: 2
   - Pin memory: True
   - Gradient checkpointing: Optional (jika OOM)

### Checkpoint Storage

1. **Checkpoint Size:**
   - LoRA adapters only: ~5MB per checkpoint
   - Full model: ~1GB per checkpoint
   - Strategy: Save only LoRA adapters

2. **Storage Management:**
   - Keep last 3 checkpoints untuk domain
   - Keep last 2 checkpoints untuk task-specific
   - Backup best checkpoint ke Drive

## Correctness Properties

### Property 1: Dataset Integrity

**Property:** All loaded dataset entries MUST have required fields (`input`, `target`, `metadata`).

**Verification:** DatasetLoader.validate_dataset() checks for missing fields dan raises ValueError jika found.

### Property 2: Token Length Constraint

**Property:** At least 95% of samples MUST fit within max_length=512 tokens after tokenization.

**Verification:** DatasetLoader.analyze_token_distribution() computes percentage dan warns jika < 95%.

### Property 3: LoRA Parameter Efficiency

**Property:** Trainable parameters MUST be ≤ 1% of total model parameters.

**Verification:** ModelSetup.print_trainable_parameters() verifies trainable percentage ≤ 1%.

### Property 4: Training Loss Convergence

**Property:** Training loss MUST decrease monotonically over epochs (allowing small fluctuations).

**Verification:** Trainer logs loss setiap epoch dan triggers warning jika loss increases untuk 2 consecutive epochs.

### Property 5: Output Format Validity

**Property:** Generated outputs MUST follow expected format structure (question + answer + distractors).

**Verification:** ModelEvaluator parses generated outputs dan checks for required components.

## Dependencies

### Core Libraries

```python
transformers>=4.35.0      # HuggingFace Transformers untuk T5
peft>=0.7.0               # Parameter-Efficient Fine-Tuning
datasets>=2.15.0          # HuggingFace Datasets
accelerate>=0.25.0        # Training acceleration
torch>=2.1.0              # PyTorch
```

### Evaluation Libraries

```python
evaluate>=0.4.0           # HuggingFace Evaluate
rouge_score>=0.1.2        # ROUGE metrics
bert_score>=0.3.13        # BERTScore
nltk>=3.8.1               # BLEU calculation
```

### Utilities

```python
wandb>=0.16.0             # Experiment tracking (optional)
matplotlib>=3.8.0         # Plotting
pandas>=2.1.0             # Data analysis
tqdm>=4.66.0              # Progress bars
```

## Configuration Files

### training_config.yaml

```yaml
# Domain Adaptation Configuration
domain_adaptation:
  model_name: "LazarusNLP/IndoNanoT5-base"
  dataset_dir: "dataset_aqg/output_domain/"
  output_dir: "./checkpoints/domain"
  
  lora:
    r: 8
    lora_alpha: 16
    lora_dropout: 0.1
    target_modules: ["q", "v"]
  
  training:
    num_train_epochs: 6
    per_device_train_batch_size: 8
    gradient_accumulation_steps: 4
    learning_rate: 2.0e-4
    warmup_steps: 50
    max_length: 512
    evaluation_strategy: "epoch"
    save_strategy: "epoch"
    logging_steps: 50
    fp16: true

# Task-Specific AQG Configuration
task_specific:
  checkpoint_path: "./checkpoints/domain/best"
  dataset_dir: "dataset_aqg/dataset-task-spesifc/"
  output_dir: "./checkpoints/aqg"
  
  training:
    num_train_epochs: 3
    per_device_train_batch_size: 8
    gradient_accumulation_steps: 4
    learning_rate: 1.0e-4
    warmup_steps: 30
    max_length: 512
    evaluation_strategy: "epoch"
    save_strategy: "epoch"
    load_best_model_at_end: true
    metric_for_best_model: "eval_bleu"
    logging_steps: 50
    fp16: true

# Evaluation Configuration
evaluation:
  metrics: ["bleu", "rouge", "bertscore", "diversity"]
  num_samples: 20
  generation_config:
    max_length: 512
    num_beams: 4
    early_stopping: true
    no_repeat_ngram_size: 3
```

## Deployment Considerations

### Model Export

1. **Save Format:** PyTorch state dict + LoRA adapters
2. **Model Size:** ~1GB (base model) + ~5MB (LoRA adapters)
3. **Inference Requirements:** transformers, peft, torch

### Inference Pipeline

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration
from peft import PeftModel

# Load model
base_model = T5ForConditionalGeneration.from_pretrained("LazarusNLP/IndoNanoT5-base")
model = PeftModel.from_pretrained(base_model, "path/to/lora/adapters")
tokenizer = T5Tokenizer.from_pretrained("LazarusNLP/IndoNanoT5-base")

# Generate
input_text = "Context: ... Instruction: Generate MCQ question about variables"
inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
outputs = model.generate(**inputs, max_length=512, num_beams=4)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### Performance Benchmarks

- **Inference Speed:** ~2-3 seconds per question pada T4 GPU
- **Batch Inference:** ~10-15 questions per minute
- **Memory Usage:** ~2GB VRAM untuk inference

