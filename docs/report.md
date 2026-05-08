```

import time
import os
from pathlib import Path

start_time = time.time()

# Ensure checkpoint directory exists
Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)

# Check for existing checkpoints
checkpoints = []
if os.path.exists(CHECKPOINT_DIR):
    checkpoints = [d for d in os.listdir(CHECKPOINT_DIR) if d.startswith('checkpoint-')]

# Decide whether to resume
if checkpoints:
    print(f"📂 Found {len(checkpoints)} checkpoint(s): {sorted(checkpoints)}")
    print(f"🔄 Resuming from last checkpoint")
    resume = True
else:
    print("🆕 No checkpoints found - starting fresh training")
    resume = False

print('Starting task-specific AQG training...')
print('='*60)

# Train (all logic in task_trainer.py)
results = trainer.train(
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    early_stopping=True,
    early_stopping_patience=2,
    resume_from_checkpoint=resume  # ✅ Auto-resume jika ada checkpoint
)

elapsed = (time.time() - start_time) / 3600
print(f'\n✓ Training completed in {elapsed:.2f} hours')
print(f'  Final training loss: {results["training_loss"]:.4f}')

```


=== Final Evaluation Metrics ===
eval_loss: 9.3682
eval_bleu_1: 0.0000
eval_bleu_4: 0.0000
eval_rouge_l: 0.0000
eval_runtime: 340.6204
eval_samples_per_second: 1.0300
eval_steps_per_second: 0.1290
✓ Training results saved to /content/drive/MyDrive/dataset_aqg/checkpoints/05-indonanoot5-report/training_results.json

✓ Training completed in 0.57 hours
  Final training loss: 18.6591

