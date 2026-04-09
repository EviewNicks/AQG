# Panduan Fine-tuning IndoT5 dengan Google Colab

## Overview

Dokumen ini menjelaskan cara menggunakan Google Colab untuk fine-tuning model IndoT5 dengan LoRA. Colab menyediakan GPU gratis (T4 15GB) yang cukup untuk training model kita.

## 1. Akses Google Colab di VS Code / Kiro

### Opsi A: Menggunakan Extension (TIDAK TERSEDIA)

Sayangnya, **tidak ada extension resmi Google Colab untuk VS Code**. Extension yang ada di marketplace adalah third-party dan tidak reliable.

### Opsi B: Workflow Hybrid (REKOMENDASI) ✅

Gunakan kombinasi VS Code (development) + Colab (training):

**Workflow:**
```
VS Code/Kiro → Develop code → Export to .py → Upload to Colab → Train → Download results
```

**Keuntungan:**
- Development di VS Code dengan autocomplete, debugging, git
- Training di Colab dengan GPU gratis
- Best of both worlds

## 2. Export/Import Notebook dan Python Files

### 2.1. Dari Jupyter Notebook (.ipynb) ke Python (.py)

**Di VS Code/Kiro:**
```bash
# Install jupyter jika belum ada
pip install jupyter

# Convert notebook to Python script
jupyter nbconvert --to script your_notebook.ipynb

# Output: your_notebook.py
```

**Manual (lebih kontrol):**
- Buka notebook di VS Code
- Copy cell-by-cell ke file .py baru
- Tambahkan comments untuk struktur

### 2.2. Dari Python (.py) ke Colab

**Cara 1: Upload Manual**
1. Buka Google Colab: https://colab.research.google.com/
2. File → Upload notebook (atau drag & drop .py file)
3. Colab akan convert otomatis ke notebook format

**Cara 2: Via Google Drive**
```python
# Di Colab, mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Import Python file
import sys
sys.path.append('/content/drive/MyDrive/your_project/')
from your_module import your_function
```

**Cara 3: Via GitHub (BEST PRACTICE)**
```python
# Di Colab, clone repository
!git clone https://github.com/your-username/your-repo.git
%cd your-repo

# Install dependencies
!pip install -r requirements.txt

# Import modules
from src.training import train_model
```

### 2.3. Dari Colab ke VS Code

**Download hasil training:**
```python
# Di Colab, setelah training selesai
from google.colab import files

# Download model checkpoint
files.download('model_checkpoint.pth')

# Atau save ke Google Drive
!cp -r ./checkpoints /content/drive/MyDrive/aqg_checkpoints/
```

## 3. Struktur Project untuk Colab

### 3.1. Organize Code sebagai Python Modules

**Struktur yang direkomendasikan:**
```
aqg-project/
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # Load dataset dari JSONL
│   ├── model_setup.py      # Setup IndoT5 + LoRA
│   ├── trainer.py          # Training loop
│   └── evaluator.py        # Evaluation metrics
├── notebooks/
│   └── colab_training.ipynb  # Notebook untuk Colab
├── requirements.txt
└── README.md
```

**Keuntungan:**
- Code reusable (bisa dipakai di Colab dan local)
- Mudah di-test dan di-debug
- Version control dengan git

### 3.2. Contoh: data_loader.py

```python
# src/data_loader.py
from datasets import load_dataset
from transformers import AutoTokenizer

def load_aqg_dataset(data_dir, tokenizer_name="Wikidepia/IndoT5-base"):
    """
    Load AQG dataset dari JSONL files
    
    Args:
        data_dir: Path ke folder dataset (train.jsonl, val.jsonl, test.jsonl)
        tokenizer_name: Nama tokenizer HuggingFace
    
    Returns:
        dataset: HuggingFace Dataset object
        tokenizer: Tokenizer object
    """
    # Load dataset
    dataset = load_dataset('json', data_files={
        'train': f'{data_dir}/train.jsonl',
        'validation': f'{data_dir}/validation.jsonl',
        'test': f'{data_dir}/test.jsonl'
    })
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Tokenize function
    def tokenize_function(examples):
        model_inputs = tokenizer(
            examples['input'], 
            max_length=512, 
            truncation=True, 
            padding='max_length'
        )
        
        # Tokenize targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples['target'], 
                max_length=512, 
                truncation=True, 
                padding='max_length'
            )
        
        model_inputs['labels'] = labels['input_ids']
        return model_inputs
    
    # Apply tokenization
    tokenized_dataset = dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=dataset['train'].column_names
    )
    
    return tokenized_dataset, tokenizer

# Usage example
if __name__ == "__main__":
    dataset, tokenizer = load_aqg_dataset('./dataset_aqg/dataset-task-spesifc/')
    print(f"Train size: {len(dataset['train'])}")
    print(f"Validation size: {len(dataset['validation'])}")
```

### 3.3. Contoh: model_setup.py

```python
# src/model_setup.py
from transformers import AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model, TaskType

def setup_model_with_lora(
    model_name="Wikidepia/IndoT5-base",
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.1
):
    """
    Setup IndoT5 model dengan LoRA
    
    Args:
        model_name: Nama model HuggingFace
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: Dropout rate
    
    Returns:
        model: Model dengan LoRA adapter
    """
    # Load base model
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q", "v"],  # Target attention layers
        bias="none"
    )
    
    # Apply LoRA
    model = get_peft_model(base_model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model

# Usage example
if __name__ == "__main__":
    model = setup_model_with_lora()
    print("Model setup complete!")
```

### 3.4. Contoh: Colab Notebook

```python
# notebooks/colab_training.ipynb

# ===== CELL 1: Setup Environment =====
# Check GPU
!nvidia-smi

# Install dependencies
!pip install transformers peft datasets accelerate bitsandbytes evaluate rouge-score

# ===== CELL 2: Mount Google Drive (Optional) =====
from google.colab import drive
drive.mount('/content/drive')

# ===== CELL 3: Clone Repository =====
!git clone https://github.com/your-username/aqg-project.git
%cd aqg-project

# ===== CELL 4: Upload Dataset (if not in repo) =====
# Option A: From Google Drive
!cp -r /content/drive/MyDrive/dataset_aqg ./

# Option B: Upload manually
from google.colab import files
uploaded = files.upload()  # Upload train.jsonl, val.jsonl, test.jsonl

# ===== CELL 5: Load Dataset =====
from src.data_loader import load_aqg_dataset

dataset, tokenizer = load_aqg_dataset('./dataset_aqg/dataset-task-spesifc/')
print(f"Dataset loaded: {len(dataset['train'])} train samples")

# ===== CELL 6: Setup Model =====
from src.model_setup import setup_model_with_lora

model = setup_model_with_lora(
    model_name="Wikidepia/IndoT5-base",
    lora_r=8,
    lora_alpha=16
)

# ===== CELL 7: Training Configuration =====
from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="./checkpoints",
    evaluation_strategy="epoch",
    learning_rate=2e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=3,
    predict_with_generate=True,
    fp16=True,  # Mixed precision training
    push_to_hub=False,
    logging_steps=100,
    save_strategy="epoch"
)

# ===== CELL 8: Train Model =====
from transformers import Seq2SeqTrainer, DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Start training
trainer.train()

# ===== CELL 9: Save Model =====
# Save to local
trainer.save_model("./final_model")

# Save to Google Drive
!cp -r ./final_model /content/drive/MyDrive/aqg_model/

# Download to local machine
from google.colab import files
!zip -r final_model.zip ./final_model
files.download('final_model.zip')

# ===== CELL 10: Evaluation =====
results = trainer.evaluate()
print(results)
```

## 4. Best Practices untuk Colab

### 4.1. Manajemen Session

**Problem:** Colab session timeout setelah 12 jam (free tier)

**Solution:**
```python
# Save checkpoint setiap epoch
training_args = Seq2SeqTrainingArguments(
    ...
    save_strategy="epoch",
    save_total_limit=3  # Keep only last 3 checkpoints
)

# Resume dari checkpoint jika session terputus
trainer.train(resume_from_checkpoint="./checkpoints/checkpoint-1000")
```

### 4.2. Monitor Training

```python
# Install tensorboard
!pip install tensorboard

# Load tensorboard extension
%load_ext tensorboard

# Start tensorboard
%tensorboard --logdir ./checkpoints/runs
```

### 4.3. Memory Management

```python
# Clear GPU memory jika OOM
import torch
torch.cuda.empty_cache()

# Reduce batch size jika masih OOM
training_args = Seq2SeqTrainingArguments(
    per_device_train_batch_size=4,  # Reduce from 8
    gradient_accumulation_steps=2   # Maintain effective batch size
)
```

## 5. Workflow Lengkap: VS Code → Colab → VS Code

### Step 1: Development di VS Code
```bash
# Develop dan test locally
python src/data_loader.py
python src/model_setup.py

# Commit ke git
git add .
git commit -m "Add training modules"
git push origin main
```

### Step 2: Training di Colab
1. Buka Colab: https://colab.research.google.com/
2. Clone repo: `!git clone https://github.com/your-username/aqg-project.git`
3. Run training cells
4. Save model ke Google Drive

### Step 3: Download Results ke VS Code
```bash
# Download dari Google Drive
# Atau via Colab files.download()

# Extract model
unzip final_model.zip

# Test inference locally
python src/inference.py --model ./final_model --input "test input"
```

## 6. Troubleshooting

### Issue 1: "CUDA out of memory"
**Solution:**
- Reduce batch size: `per_device_train_batch_size=4`
- Enable gradient checkpointing: `gradient_checkpointing=True`
- Use 8-bit quantization: `load_in_8bit=True`

### Issue 2: "Session disconnected"
**Solution:**
- Save checkpoints frequently
- Use Google Drive untuk persistent storage
- Resume training: `trainer.train(resume_from_checkpoint=True)`

### Issue 3: "Module not found"
**Solution:**
```python
# Add project root to Python path
import sys
sys.path.append('/content/aqg-project')
```

## 7. Kesimpulan

**Rekomendasi Workflow:**
1. ✅ Develop code sebagai Python modules di VS Code
2. ✅ Version control dengan git
3. ✅ Training di Colab dengan GPU gratis
4. ✅ Save results ke Google Drive
5. ✅ Download dan evaluate di VS Code

**Tidak Direkomendasikan:**
- ❌ Develop langsung di Colab (no autocomplete, no git integration)
- ❌ Menggunakan extension third-party untuk Colab di VS Code

**Next Steps:**
1. Buat struktur project dengan Python modules
2. Test data loading dan model setup locally
3. Push ke GitHub
4. Clone di Colab dan mulai training
