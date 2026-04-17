Fine-tune T5
NanoT5 supports fine-tuning to a downstream dataset like Super Natural-Instructions (SNI). However, since this requires further customization of fine-tuning code to other downstream datasets, we opted to develop our own fine-tuning script based on Hugging Face's sample fine-tuning code.

In particular, we developed fine-tuning scripts for 3 IndoNLG tasks, namely: summarization, question-answering, and chit-chat (conversational), which you can find in scripts.

Summarization
To fine-tune for summarization, run the following command and modify accordingly:

python scripts/run_summarization.py \
    --model-checkpoint LazarusNLP/IndoNanoT5-base \ # pre-trained model checkpoint
    --dataset-name LazarusNLP/indonlg \ # Hugging Face 🤗 dataset name
    --dataset-config indosum \ # dataset config
    --input-column-name input \ # input column (text passage) name in dataset
    --target-column-name target \ # target column (summary) name in dataset
    --input-max-length 512 \
    --target-max-length 512 \
    --num-beams 5 \ # beam width during beam search
    --output-dir outputs/indo-nanot5-indosum \
    --num-train-epochs 5 \
    --optim adamw_torch_fused \ # any optimizer supported in Hugging Face 🤗 transformers
    --learning-rate 1e-3 \
    --weight-decay 0.01 \
    --per-device-train-batch-size 8 \
    --per-device-eval-batch-size 16 \
    --hub-model-id LazarusNLP/IndoNanoT5-base-IndoSum # Hugging Face 🤗 Hub repo name
IndoNLG summarization recipes are provided here.

Question-Answering
To fine-tune for question-answering, run the following command and modify accordingly:

python scripts/run_qa.py \
    --model-checkpoint LazarusNLP/IndoNanoT5-base \
    --dataset-name LazarusNLP/indonlg \
    --dataset-config question_answering \
    --context-column-name context \ # context/passage column name
    --question-column-name input \ # question column name
    --answer-column-name references \ # answer column name, must be list
    --id-column-name gem_id \ # question-answer pair id
    --input-max-length 512 \
    --target-max-length 512 \
    --num-beams 5 \
    --output-dir outputs/indo-nanot5-tydiqa \
    --num-train-epochs 50 \
    --optim adamw_torch_fused \
    --learning-rate 1e-5 \
    --weight-decay 0.01 \
    --per-device-train-batch-size 8 \
    --per-device-eval-batch-size 16 \
    --hub-model-id LazarusNLP/IndoNanoT5-base-TyDiQA
IndoNLG question-answering recipe is provided here.

Chit-chat
To fine-tune for chit-chat, run the following command and modify accordingly:

python scripts/run_chitchat.py \
    --model-checkpoint LazarusNLP/IndoNanoT5-base \
    --dataset-name LazarusNLP/indonlg \
    --dataset-config xpersona \
    --context-column-name context \ # context/persona column name
    --question-column-name input \ # conversation history/dialogues column name
    --answer-column-name references \ # response column name
    --use-persona \ # whether to use persona or not
    --input-max-length 512 \
    --target-max-length 512 \
    --num-beams 5 \
    --output-dir outputs/indo-nanot5-xpersona \
    --num-train-epochs 50 \
    --optim adamw_torch_fused \
    --learning-rate 1e-5 \
    --weight-decay 0.01 \
    --per-device-train-batch-size 8 \
    --per-device-eval-batch-size 16 \
    --hub-model-id LazarusNLP/IndoNanoT5-base-XPersona