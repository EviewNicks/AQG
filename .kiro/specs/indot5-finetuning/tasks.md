# Implementation Tasks: IndoT5 Fine-tuning untuk AQG

## Task Breakdown

### Phase 0: Project Setup (Estimated: 30 minutes)

#### Task 0.1: Create Project Structure
**Status:** Not Started  
**Dependencies:** None  
**Estimated Time:** 10 minutes

**Description:** Create directory structure untuk modular code organization.

**Acceptance Criteria:**
- [ ] Create directories: `data/`, `model/`, `training/`, `evaluation/`, `utils/`
- [ ] Create `__init__.py` files untuk setiap module
- [ ] Create `config/training_config.yaml` untuk configuration
- [ ] Create `notebooks/` directory untuk Colab notebooks

**Files to Create:**
```
src/finetuning/
├── data/
│   ├── __init__.py
│   ├── dataset_loader.py
│   └── tokenizer_tester.py
├── model/
│   ├── __init__.py
│   └── model_setup.py
├── training/
│   ├── __init__.py
│   ├── domain_trainer.py
│   └── task_trainer.py
├── evaluation/
│   ├── __init__.py
│   ├── metrics_calculator.py
│   └── model_evaluator.py
├── utils/
│   ├── __init__.py
│   ├── checkpoint_manager.py
│   └── colab_helper.py
├── config/
│   └── training_config.yaml
└── notebooks/
    ├── 01_setup_and_validation.ipynb
    ├── 02_domain_adaptation.ipynb
    └── 03_task_specific_training.ipynb
```

#### Task 0.2: Setup Colab Environment
**Status:** Not Started  
**Dependencies:** None  
**Estimated Time:** 10 minutes

**Description:** Create Colab setup script untuk install dependencies dan mount Drive.

**Acceptance Criteria:**
- [ ] Implement `ColabHelper.check_gpu()` untuk verify T4 GPU
- [ ] Implement `ColabHelper.install_dependencies()` untuk install packages
- [ ] Implement `ColabHelper.mount_drive()` untuk persistent storage
- [ ] Test pada fresh Colab session

**Implementation Notes:**
```python
# utils/colab_helper.py
class ColabHelper:
    @staticmethod
    def check_gpu():
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("GPU not available!")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        return {
            "gpu_available": True,
            "gpu_name": gpu_name,
            "gpu_memory_gb": gpu_memory
        }
```

#### Task 0.3: Create Configuration File
**Status:** Not Started  
**Dependencies:** None  
**Estimated Time:** 10 minutes

**Description:** Create YAML configuration file dengan all training parameters.

**Acceptance Criteria:**
- [ ] Define domain adaptation config
- [ ] Define task-specific config
- [ ] Define evaluation config
- [ ] Add comments untuk setiap parameter

---

### Phase 1: Data Module Implementation (Estimated: 2 hours)

#### Task 1.1: Implement DatasetLoader
**Status:** Not Started  
**Dependencies:** Task 0.1  
**Estimated Time:** 45 minutes

**Description:** Implement class untuk load dan validate JSONL datasets.

**Acceptance Criteria:**
- [ ] Implement `load_dataset()` method
- [ ] Implement `validate_dataset()` method
- [ ] Implement `analyze_token_distribution()` method
- [ ] Handle missing files gracefully
- [ ] Write unit tests

**Implementation Notes:**
```python
# data/dataset_loader.py
from datasets import load_dataset
import json
from pathlib import Path

class DatasetLoader:
    def load_dataset(self, data_dir: str, split: str = "train"):
        file_path = Path(data_dir) / f"{split}.jsonl"
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        dataset = load_dataset("json", data_files=str(file_path), split="train")
        return dataset
```

**Testing:**
- Test dengan domain dataset: `dataset_aqg/output_domain/`
- Test dengan task-specific dataset: `dataset_aqg/dataset-task-spesifc/`
- Test error handling untuk missing files

#### Task 1.2: Implement TokenizerTester
**Status:** Not Started  
**Dependencies:** Task 0.1  
**Estimated Time:** 45 minutes

**Description:** Implement class untuk test tokenizer compatibility.

**Acceptance Criteria:**
- [ ] Implement `test_markdown_handling()` method
- [ ] Implement `test_code_block_integrity()` method
- [ ] Implement `detect_oov_tokens()` method
- [ ] Generate test report dengan examples
- [ ] Write unit tests

**Implementation Notes:**
```python
# data/tokenizer_tester.py
class TokenizerTester:
    def test_markdown_handling(self):
        test_cases = {
            "headings": "# Heading 1\n## Heading 2",
            "bold": "**bold text**",
            "code": "`inline code`",
            "newlines": "line1\nline2\nline3"
        }
        results = {}
        for name, text in test_cases.items():
            tokens = self.tokenizer(text)
            decoded = self.tokenizer.decode(tokens["input_ids"])
            results[name] = (decoded == text)
        return results
```

#### Task 1.3: Create Data Validation Notebook
**Status:** Not Started  
**Dependencies:** Task 1.1, Task 1.2  
**Estimated Time:** 30 minutes

**Description:** Create Colab notebook untuk validate datasets dan tokenizer.

**Acceptance Criteria:**
- [ ] Load both datasets (domain + task-specific)
- [ ] Run validation checks
- [ ] Display token distribution histograms
- [ ] Test tokenizer dengan sample data
- [ ] Generate validation report

**Notebook Sections:**
1. Setup (install packages, mount drive)
2. Load datasets
3. Validate dataset structure
4. Analyze token distribution
5. Test tokenizer compatibility
6. Summary report

---

### Phase 2: Model Module Implementation (Estimated: 1.5 hours)

#### Task 2.1: Implement ModelSetup
**Status:** Not Started  
**Dependencies:** Task 0.1  
**Estimated Time:** 1 hour

**Description:** Implement class untuk setup IndoT5 dengan LoRA.

**Acceptance Criteria:**
- [ ] Implement `load_base_model()` method
- [ ] Implement `apply_lora()` method
- [ ] Implement `print_trainable_parameters()` method
- [ ] Implement `check_gpu_memory()` method
- [ ] Verify trainable parameters ≤ 1%
- [ ] Write unit tests

**Implementation Notes:**
```python
# model/model_setup.py
from transformers import T5ForConditionalGeneration
from peft import LoraConfig, get_peft_model

class ModelSetup:
    def load_base_model(self, model_name="Wikidepia/IndoT5-base"):
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        return model
    
    def apply_lora(self, model, lora_config):
        peft_model = get_peft_model(model, lora_config)
        return peft_model
    
    def print_trainable_parameters(self, model):
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
        print(f"Total: {total_params:,}")
```

#### Task 2.2: Test Model Setup
**Status:** Not Started  
**Dependencies:** Task 2.1  
**Estimated Time:** 30 minutes

**Description:** Test model loading dan LoRA application pada Colab.

**Acceptance Criteria:**
- [ ] Load IndoT5-base successfully
- [ ] Apply LoRA configuration
- [ ] Verify trainable parameters ~0.5%
- [ ] Check GPU memory usage < 10GB
- [ ] Test forward pass dengan dummy input

**Testing Script:**
```python
# Test model setup
setup = ModelSetup()
model = setup.load_base_model()

lora_config = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.1,
    target_modules=["q", "v"], task_type="SEQ_2_SEQ_LM"
)
peft_model = setup.apply_lora(model, lora_config)
setup.print_trainable_parameters(peft_model)

# Test forward pass
dummy_input = tokenizer("test input", return_tensors="pt")
output = peft_model(**dummy_input)
print("Forward pass successful!")
```

---

### Phase 3: Training Module Implementation (Estimated: 3 hours)

#### Task 3.1: Implement DomainAdaptationTrainer
**Status:** Not Started  
**Dependencies:** Task 2.1, Task 1.1  
**Estimated Time:** 1.5 hours

**Description:** Implement trainer untuk domain adaptation stage.

**Acceptance Criteria:**
- [ ] Implement `train()` method dengan Seq2SeqTrainer
- [ ] Implement `save_best_model()` method
- [ ] Configure training arguments (6 epochs, lr=2e-4)
- [ ] Setup logging untuk loss dan perplexity
- [ ] Implement checkpoint saving setiap epoch
- [ ] Write integration test

**Implementation Notes:**
```python
# training/domain_trainer.py
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

class DomainAdaptationTrainer:
    def train(self, train_dataset, eval_dataset, training_args):
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
        )
        
        train_result = trainer.train()
        return train_result
```

**Training Arguments:**
```python
training_args = Seq2SeqTrainingArguments(
    output_dir="./checkpoints/domain",
    num_train_epochs=6,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_steps=50,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    fp16=True,
    predict_with_generate=True,
)
```

#### Task 3.2: Implement TaskSpecificTrainer
**Status:** Not Started  
**Dependencies:** Task 2.1, Task 1.1  
**Estimated Time:** 1.5 hours

**Description:** Implement trainer untuk task-specific AQG stage.

**Acceptance Criteria:**
- [ ] Implement `train()` method dengan custom metrics
- [ ] Implement `save_final_model()` method
- [ ] Configure training arguments (3 epochs, lr=1e-4)
- [ ] Setup evaluation dengan BLEU, ROUGE, BERTScore
- [ ] Implement best model selection based on BLEU
- [ ] Write integration test

**Implementation Notes:**
```python
# training/task_trainer.py
def compute_metrics(eval_preds):
    predictions, labels = eval_preds
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Compute BLEU
    bleu = metrics_calculator.compute_bleu(decoded_preds, decoded_labels)
    return {"bleu": bleu["bleu_4"]}
```

---

### Phase 4: Evaluation Module Implementation (Estimated: 2 hours)

#### Task 4.1: Implement MetricsCalculator
**Status:** Not Started  
**Dependencies:** Task 0.1  
**Estimated Time:** 1 hour

**Description:** Implement class untuk calculate evaluation metrics.

**Acceptance Criteria:**
- [ ] Implement `compute_bleu()` method
- [ ] Implement `compute_rouge()` method
- [ ] Implement `compute_bertscore()` method
- [ ] Implement `compute_diversity()` method
- [ ] Write unit tests dengan known examples

**Implementation Notes:**
```python
# evaluation/metrics_calculator.py
from evaluate import load

class MetricsCalculator:
    def __init__(self):
        self.bleu = load("bleu")
        self.rouge = load("rouge")
        self.bertscore = load("bertscore")
    
    def compute_bleu(self, predictions, references):
        results = self.bleu.compute(
            predictions=predictions,
            references=[[ref] for ref in references]
        )
        return results
```

#### Task 4.2: Implement ModelEvaluator
**Status:** Not Started  
**Dependencies:** Task 4.1  
**Estimated Time:** 1 hour

**Description:** Implement class untuk evaluate trained model.

**Acceptance Criteria:**
- [ ] Implement `evaluate_on_test_set()` method
- [ ] Implement `generate_samples()` method
- [ ] Implement `compare_with_baseline()` method
- [ ] Generate evaluation report dengan visualizations
- [ ] Write integration test

**Implementation Notes:**
```python
# evaluation/model_evaluator.py
class ModelEvaluator:
    def evaluate_on_test_set(self, test_dataset):
        predictions = []
        references = []
        
        for sample in test_dataset:
            input_text = sample["input"]
            reference = sample["target"]
            
            inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
            outputs = self.model.generate(**inputs, max_length=512, num_beams=4)
            prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            predictions.append(prediction)
            references.append(reference)
        
        metrics = self.metrics.compute_bleu(predictions, references)
        metrics.update(self.metrics.compute_rouge(predictions, references))
        metrics.update(self.metrics.compute_bertscore(predictions, references))
        
        return metrics
```

---

### Phase 5: Utilities Module Implementation (Estimated: 1.5 hours)

#### Task 5.1: Implement CheckpointManager
**Status:** Not Started  
**Dependencies:** Task 0.1  
**Estimated Time:** 1 hour

**Description:** Implement class untuk manage checkpoints.

**Acceptance Criteria:**
- [ ] Implement `save_checkpoint()` method
- [ ] Implement `load_checkpoint()` method
- [ ] Implement `cleanup_old_checkpoints()` method
- [ ] Implement `backup_to_drive()` method
- [ ] Write unit tests

**Implementation Notes:**
```python
# utils/checkpoint_manager.py
import shutil
from pathlib import Path

class CheckpointManager:
    def save_checkpoint(self, model, optimizer, epoch, metrics):
        checkpoint_path = Path(self.checkpoint_dir) / f"checkpoint-epoch-{epoch}"
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model.save_pretrained(checkpoint_path)
        
        # Save metadata
        metadata = {
            "epoch": epoch,
            "metrics": metrics
        }
        with open(checkpoint_path / "metadata.json", "w") as f:
            json.dump(metadata, f)
        
        # Cleanup old checkpoints
        self.cleanup_old_checkpoints()
        
        # Backup to Drive
        if self.drive_backup:
            self.backup_to_drive(str(checkpoint_path))
        
        return str(checkpoint_path)
```

#### Task 5.2: Complete ColabHelper Implementation
**Status:** Not Started  
**Dependencies:** Task 0.2  
**Estimated Time:** 30 minutes

**Description:** Complete remaining ColabHelper methods.

**Acceptance Criteria:**
- [ ] Implement `setup_wandb()` method (optional)
- [ ] Add helper untuk download final model
- [ ] Add helper untuk zip checkpoints
- [ ] Test all methods pada Colab

---

### Phase 6: Notebook Creation (Estimated: 2 hours)

#### Task 6.1: Create Setup and Validation Notebook
**Status:** Not Started  
**Dependencies:** Task 1.3  
**Estimated Time:** 30 minutes

**Description:** Create `01_setup_and_validation.ipynb` untuk initial setup.

**Notebook Sections:**
1. Install dependencies
2. Mount Google Drive
3. Check GPU availability
4. Load datasets
5. Validate data quality
6. Test tokenizer
7. Setup model dengan LoRA
8. Baseline evaluation

**Acceptance Criteria:**
- [ ] All cells run without errors
- [ ] Clear documentation dan comments
- [ ] Validation report generated
- [ ] Baseline metrics computed

#### Task 6.2: Create Domain Adaptation Notebook
**Status:** Not Started  
**Dependencies:** Task 3.1, Task 5.1  
**Estimated Time:** 45 minutes

**Description:** Create `02_domain_adaptation.ipynb` untuk Stage 1 training.

**Notebook Sections:**
1. Load domain dataset
2. Setup model dan tokenizer
3. Configure training arguments
4. Train untuk 6 epochs
5. Monitor training metrics
6. Save best model
7. Evaluate on validation set

**Acceptance Criteria:**
- [ ] Training completes successfully
- [ ] Checkpoints saved setiap epoch
- [ ] Training loss decreases dari ~3.0 ke ~1.5
- [ ] Validation loss decreases dari ~2.8 ke ~1.8
- [ ] Best model saved sebagai `indot5-python-domain`

#### Task 6.3: Create Task-Specific Training Notebook
**Status:** Not Started  
**Dependencies:** Task 3.2, Task 4.2  
**Estimated Time:** 45 minutes

**Description:** Create `03_task_specific_training.ipynb` untuk Stage 2 training.

**Notebook Sections:**
1. Load checkpoint dari Stage 1
2. Load task-specific dataset
3. Configure training arguments
4. Train untuk 3 epochs
5. Monitor BLEU, ROUGE, BERTScore
6. Save final model
7. Comprehensive evaluation
8. Generate sample outputs
9. Compare dengan baseline

**Acceptance Criteria:**
- [ ] Training completes successfully
- [ ] BLEU-4 increases dari ~0.10 ke ~0.35-0.45
- [ ] Final model saved sebagai `indot5-python-aqg`
- [ ] Evaluation report generated
- [ ] 20 sample outputs saved

---

### Phase 7: Testing and Validation (Estimated: 2 hours)

#### Task 7.1: Write Unit Tests
**Status:** Not Started  
**Dependencies:** All implementation tasks  
**Estimated Time:** 1 hour

**Description:** Write comprehensive unit tests untuk all modules.

**Test Files:**
```
tests/
├── test_dataset_loader.py
├── test_tokenizer_tester.py
├── test_model_setup.py
├── test_metrics_calculator.py
├── test_checkpoint_manager.py
└── test_colab_helper.py
```

**Acceptance Criteria:**
- [ ] All modules have unit tests
- [ ] Test coverage > 80%
- [ ] All tests pass
- [ ] Edge cases covered

#### Task 7.2: Run Integration Tests
**Status:** Not Started  
**Dependencies:** Task 7.1  
**Estimated Time:** 1 hour

**Description:** Run end-to-end integration tests pada Colab.

**Test Scenarios:**
1. Complete domain adaptation pipeline (1 epoch)
2. Complete task-specific pipeline (1 epoch)
3. Checkpoint save/load cycle
4. Error recovery scenarios

**Acceptance Criteria:**
- [ ] All integration tests pass
- [ ] No memory leaks detected
- [ ] Checkpoints properly saved/loaded
- [ ] Error handling works correctly

---

### Phase 8: Documentation and Finalization (Estimated: 1 hour)

#### Task 8.1: Write README
**Status:** Not Started  
**Dependencies:** All tasks  
**Estimated Time:** 30 minutes

**Description:** Create comprehensive README dengan usage instructions.

**README Sections:**
1. Project overview
2. Installation instructions
3. Dataset preparation
4. Training instructions
5. Evaluation instructions
6. Troubleshooting guide
7. Expected results

**Acceptance Criteria:**
- [ ] Clear step-by-step instructions
- [ ] Code examples included
- [ ] Expected outputs documented
- [ ] Common issues addressed

#### Task 8.2: Create Quick Start Guide
**Status:** Not Started  
**Dependencies:** Task 8.1  
**Estimated Time:** 30 minutes

**Description:** Create quick start guide untuk immediate execution.

**Quick Start Steps:**
1. Open Colab notebook
2. Run setup cells
3. Start training
4. Download results

**Acceptance Criteria:**
- [ ] Can be completed dalam < 10 minutes
- [ ] No prior knowledge required
- [ ] Links to all resources provided

---

## Task Dependencies Graph

```
Phase 0: Project Setup
├── Task 0.1 ──┬──> Task 1.1 (DatasetLoader)
│              ├──> Task 1.2 (TokenizerTester)
│              ├──> Task 2.1 (ModelSetup)
│              ├──> Task 4.1 (MetricsCalculator)
│              └──> Task 5.1 (CheckpointManager)
├── Task 0.2 ──┴──> Task 5.2 (ColabHelper)
└── Task 0.3

Phase 1: Data Module
├── Task 1.1 ──┬──> Task 1.3 (Validation Notebook)
└── Task 1.2 ──┘

Phase 2: Model Module
├── Task 2.1 ──┬──> Task 2.2 (Model Testing)
               ├──> Task 3.1 (DomainTrainer)
               └──> Task 3.2 (TaskTrainer)

Phase 3: Training Module
├── Task 3.1 ──┬──> Task 6.2 (Domain Notebook)
└── Task 3.2 ──┴──> Task 6.3 (Task Notebook)

Phase 4: Evaluation Module
├── Task 4.1 ──┬──> Task 4.2 (ModelEvaluator)
└── Task 4.2 ──┴──> Task 6.3 (Task Notebook)

Phase 5: Utilities Module
├── Task 5.1 ──┬──> Task 6.2 (Domain Notebook)
└── Task 5.2 ──┘

Phase 6: Notebooks
├── Task 6.1 (Setup Notebook)
├── Task 6.2 (Domain Notebook)
└── Task 6.3 (Task Notebook)

Phase 7: Testing
├── Task 7.1 (Unit Tests)
└── Task 7.2 (Integration Tests)

Phase 8: Documentation
├── Task 8.1 (README)
└── Task 8.2 (Quick Start)
```

## Time Estimates Summary

| Phase | Estimated Time | Tasks |
|-------|---------------|-------|
| Phase 0: Project Setup | 30 min | 3 |
| Phase 1: Data Module | 2 hours | 3 |
| Phase 2: Model Module | 1.5 hours | 2 |
| Phase 3: Training Module | 3 hours | 2 |
| Phase 4: Evaluation Module | 2 hours | 2 |
| Phase 5: Utilities Module | 1.5 hours | 2 |
| Phase 6: Notebooks | 2 hours | 3 |
| Phase 7: Testing | 2 hours | 2 |
| Phase 8: Documentation | 1 hour | 2 |
| **TOTAL** | **15.5 hours** | **21 tasks** |

## Accelerated Timeline (Target: 2-3 days)

### Day 1: Core Implementation (6-8 hours)
- Morning: Phase 0, Phase 1, Phase 2 (4 hours)
- Afternoon: Phase 3, Phase 5 (4 hours)

### Day 2: Training and Evaluation (6-8 hours)
- Morning: Phase 4, Phase 6 (4 hours)
- Afternoon: Run training notebooks, monitor progress (4 hours)

### Day 3: Testing and Documentation (3-4 hours)
- Morning: Phase 7 (2 hours)
- Afternoon: Phase 8 (1 hour)

## Critical Path

The critical path untuk fastest completion:

1. Task 0.1 → Task 1.1 → Task 2.1 → Task 3.1 → Task 6.2 (Domain Training)
2. Task 6.2 → Task 3.2 → Task 4.2 → Task 6.3 (Task-Specific Training)
3. Task 6.3 → Task 7.2 → Task 8.1 (Finalization)

**Minimum viable path:** ~12 hours of focused work

## Risk Mitigation

### High-Risk Tasks

1. **Task 3.1 & 3.2 (Training):** May encounter OOM errors
   - Mitigation: Implement gradient checkpointing, reduce batch size
   
2. **Task 6.2 & 6.3 (Notebooks):** Training may take longer than expected
   - Mitigation: Use smaller subset untuk testing, optimize hyperparameters

3. **Task 7.2 (Integration Tests):** May reveal unexpected issues
   - Mitigation: Allocate buffer time, have fallback strategies

### Checkpoint Tasks

After completing these tasks, validate before proceeding:

- ✓ **After Task 1.3:** Verify datasets load correctly
- ✓ **After Task 2.2:** Verify model fits dalam GPU memory
- ✓ **After Task 6.2:** Verify domain adaptation improves perplexity
- ✓ **After Task 6.3:** Verify task-specific training improves BLEU

## Success Criteria

Project is considered complete when:

1. ✓ All 21 tasks marked as "Completed"
2. ✓ Domain adaptation achieves validation loss < 2.0
3. ✓ Task-specific training achieves BLEU-4 > 0.35
4. ✓ Final model saved dan downloadable
5. ✓ Documentation complete dan tested
6. ✓ All notebooks run end-to-end without errors

