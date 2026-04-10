IndoNanoT5 Base

https://huggingface.co/LazarusNLP/IndoNanoT5-base


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("LazarusNLP/IndoNanoT5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("LazarusNLP/IndoNanoT5-base")



IndoNanoT5 Base is an Indonesian sequence-to-sequence language model based on the T5 architecture. We conducted pre-training on an open-source Indonesian corpus of uonlp/CulturaX. On a held-out subset of the corpus, our model achieved an evaluation loss of 2.082 or a perplexity of about 8.02.

This model was trained using the nanoT5 PyTorch framework. All training was done on an NVIDIA H100 GPU. LazarusNLP/IndoNanoT5-base is released under Apache 2.0 license.

Model Detail
Developed by: LazarusNLP
Model type: Encoder-decoder T5 transformer language model
Language(s): Indonesian
License: Apache 2.0
Contact: Wilson Wongso
Use in 🤗Transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_checkpoint = "LazarusNLP/IndoNanoT5-base"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
Training Datasets
Around 4B tokens from the following corpora were used during pre-training.

Cleaned, Enormous, and Public: The Multilingual Fuel to Democratize Large Language Models for 167 Languages
Training Hyperparameters
The following hyperparameters were used during training:

total_steps: 65536
input_length: 512
batch_size: 128
grad_acc: 1
base_lr: 5e-3
optimizer: AdamWScaled with betas=(0.9,0.999) and epsilon=1e-08
weight_decay: 0.0
lr_scheduler: cosine
warmup_steps: 10000
final_cosine: 1e-5
grad_clip: 1.0
precision: bf16
Acknowledgements
We would like to acknowledge nanoT5 for inspiring this project.