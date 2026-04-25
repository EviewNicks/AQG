# 8 final Evaluation 

```
# Re-initialize evaluator with trained model
evaluator_final = ModelEvaluator(
    model=peft_model,
    tokenizer=tokenizer,
    metrics_calculator=metrics_calc
)

print('Running comprehensive evaluation on test set...')
final_metrics = evaluator_final.evaluate_on_test_set(
    test_dataset=test_dataset,
    num_beams=4,
    include_bertscore=True,
    max_samples=None
)

print('\n=== Evaluation Results ===')
for key, value in final_metrics.items():
    print(f'{key}: {value:.4f}')

```

Running comprehensive evaluation on test set...

============================================================
EVALUATING ON TEST SET
============================================================

Evaluating 168 samples...
  Processed 10/168 samples...
  Processed 20/168 samples...
  Processed 30/168 samples...
  Processed 40/168 samples...
  Processed 50/168 samples...
  Processed 60/168 samples...
  Processed 70/168 samples...
  Processed 80/168 samples...
  Processed 90/168 samples...
  Processed 100/168 samples...
  Processed 110/168 samples...
  Processed 120/168 samples...
  Processed 130/168 samples...
  Processed 140/168 samples...
  Processed 150/168 samples...
  Processed 160/168 samples...
✓ Generated 168 predictions
Computing metrics for 168 samples...
  Computing BLEU...
  Computing ROUGE...
  Computing BERTScore...

BertModel LOAD REPORT from: bert-base-multilingual-cased
Key                                        | Status     |  | 
-------------------------------------------+------------+--+-
cls.predictions.transform.dense.bias       | UNEXPECTED |  | 
cls.predictions.transform.dense.weight     | UNEXPECTED |  | 
cls.predictions.bias                       | UNEXPECTED |  | 
cls.seq_relationship.weight                | UNEXPECTED |  | 
cls.seq_relationship.bias                  | UNEXPECTED |  | 
cls.predictions.transform.LayerNorm.bias   | UNEXPECTED |  | 
cls.predictions.transform.LayerNorm.weight | UNEXPECTED |  | 

Notes:
- UNEXPECTED	:can be ignored when loading from different task/architecture; not ok if you expect identical arch.
  Computing Diversity...
✓ All metrics computed

============================================================
Test Set Evaluation Results
============================================================

BLEU Scores:
  BLEU:     0.0336
  BLEU-1:   0.1261
  BLEU-2:   0.0449
  BLEU-3:   0.0217
  BLEU-4:   0.0104

ROUGE Scores:
  ROUGE-1:  0.1800
  ROUGE-2:  0.0699
  ROUGE-L:  0.1469

BERTScore:
  Precision: 0.6349
  Recall:    0.6429
  F1:        0.6386

Diversity:
  Distinct-1: 0.2068
  Distinct-2: 0.5988

============================================================

=== Evaluation Results ===
bleu: 0.0336
bleu_1: 0.1261
bleu_2: 0.0449
bleu_3: 0.0217
bleu_4: 0.0104
brevity_penalty: 1.0000
length_ratio: 1.9803
rouge_1: 0.1800
rouge_2: 0.0699
rouge_l: 0.1469
rouge_1_fmeasure: 0.1800
rouge_2_fmeasure: 0.0699
rouge_l_fmeasure: 0.1469
bertscore_precision: 0.6349
bertscore_recall: 0.6429
bertscore_f1: 0.6386
distinct_1: 0.2068
distinct_2: 0.5988