============================================================
COMPARING WITH BASELINE
============================================================

Metric                        Baseline   Fine-tuned  Improvement
-----------------------------------------------------------------
bleu                            0.0286       0.2907      914.77%
bleu_1                          0.1304       0.6357      387.39%
bleu_2                          0.0338       0.4389     1197.60%
bleu_3                          0.0173       0.3196     1749.91%
bleu_4                          0.0088       0.2632     2880.83%
brevity_penalty                 1.0000       0.7426      -25.74%
length_ratio                    1.0000       0.7706      -22.94%
rouge_1                         0.1827       0.5325      191.47%
rouge_2                         0.0714       0.3472      386.57%
rouge_l                         0.1682       0.4826      186.94%
rouge_1_fmeasure                0.1827       0.5325      191.47%
rouge_2_fmeasure                0.0714       0.3472      386.57%
rouge_l_fmeasure                0.1682       0.4826      186.94%
distinct_1                      0.3979       0.1470      -63.05%
distinct_2                      0.6996       0.4510      -35.53%

============================================================
ADAPTER-BASED AQG TRAINING SUMMARY
============================================================
Method: Adapter Layers (d=64)
Training Time: 0.02 hours
Trainable: 1.88%

Metrics Comparison:
  BLEU-4:  0.0088 → 0.2632
  ROUGE-L: 0.1682 → 0.4826

BLEU-4 Improvement: +2880.8%

✓ SUCCESS: BLEU-4 target achieved (>= 0.20)

✓ Fine-tuning pipeline complete!
  Adapter: /content/drive/MyDrive/dataset_aqg/checkpoints/08-indonanoot5-report/adapter_mcq_generation
  Report: /content/drive/MyDrive/dataset_aqg/evaluation_results/08-indonanoot5-report/evaluation_report.json
  Samples: /content/drive/MyDrive/dataset_aqg/evaluation_results/08-indonanoot5-report/sample_outputs.json

============================================================
HOW TO LOAD TRAINED ADAPTER
============================================================
from adapters import AutoAdapterModel
from transformers import AutoTokenizer

model = AutoAdapterModel.from_pretrained("LazarusNLP/IndoNanoT5-base")
tokenizer = AutoTokenizer.from_pretrained("LazarusNLP/IndoNanoT5-base")
model.load_adapter("/content/drive/MyDrive/dataset_aqg/checkpoints/08-indonanoot5-report/adapter_mcq_generation")
model.set_active_adapters("mcq_generation")

# Generate
inputs = tokenizer("generate_mcq: [CONTEXT]", return_tensors="pt")
outputs = model.generate(**inputs, max_length=512, num_beams=4)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))