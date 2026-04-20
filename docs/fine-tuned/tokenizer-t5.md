# T5

[T5](https://huggingface.co/papers/1910.10683) is a encoder-decoder transformer available in a range of sizes from 60M to 11B parameters. It is designed to handle a wide range of NLP tasks by treating them all as text-to-text problems. This eliminates the need for task-specific architectures because T5 converts every NLP task into a text generation task.

To formulate every task as text generation, each task is prepended with a task-specific prefix (e.g., translate English to German: ..., summarize: ...). This enables T5 to handle tasks like translation, summarization, question answering, and more.

You can find all official T5 checkpoints under the [T5](https://huggingface.co/collections/google/t5-release-65005e7c520f8d7b4d037918) collection.

> [!TIP]
> Click on the T5 models in the right sidebar for more examples of how to apply T5 to different language tasks.

The example below demonstrates how to generate text with [Pipeline](/docs/transformers/main/en/main_classes/pipelines#transformers.Pipeline), [AutoModel](/docs/transformers/main/en/model_doc/auto#transformers.AutoModel), and how to translate with T5 from the command line.

```py
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "google-t5/t5-base"
    )
model = AutoModelForSeq2SeqLM.from_pretrained(
    "google-t5/t5-base",
    dtype=torch.float16,
    device_map="auto"
    )

input_ids = tokenizer("translate English to French: The weather is nice today.", return_tensors="pt").to(model.device)

output = model.generate(**input_ids, cache_implementation="static")
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [torchao](../quantization/torchao) to only quantize the weights to int4.

```py
# pip install torchao
import torch
from transformers import TorchAoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

quantization_config = TorchAoConfig("int4_weight_only", group_size=128)
model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/t5-v1_1-xl",
    dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=quantization_config
)

tokenizer = AutoTokenizer.from_pretrained("google/t5-v1_1-xl")
input_ids = tokenizer("translate English to French: The weather is nice today.", return_tensors="pt").to(model.device)

output = model.generate(**input_ids, cache_implementation="static")
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Notes

- You can pad the encoder inputs on the left or right because T5 uses relative scalar embeddings.
- T5 models need a slightly higher learning rate than the default used in [Trainer](/docs/transformers/main/en/main_classes/trainer#transformers.Trainer). Typically, values of `1e-4` and `3e-4` work well for most tasks.

## T5Config[[transformers.T5Config]]

#### transformers.T5Config[[transformers.T5Config]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/configuration_t5.py#L24)

This is the configuration class to store the configuration of a T5Model. It is used to instantiate a T5
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [google-t5/t5-small](https://huggingface.co/google-t5/t5-small)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

**Parameters:**

is_encoder_decoder (`bool`, *optional*, defaults to `True`) : Whether the model is used as an encoder/decoder or not.

vocab_size (`int`, *optional*, defaults to `32128`) : Vocabulary size of the model. Defines the number of different tokens that can be represented by the `input_ids`.

d_model (`int`, *optional*, defaults to `512`) : Size of the encoder layers and the pooler layer.

d_kv (`int`, *optional*, defaults to `64`) : Size of the key, query, value projections per attention head. The `inner_dim` of the projection layer will be defined as `num_heads * d_kv`.

d_ff (`int`, *optional*, defaults to `2048`) : Dimension of the MLP representations.

num_layers (`int`, *optional*, defaults to `6`) : Number of hidden layers in the Transformer decoder.

num_decoder_layers (`int`, *optional*) : Number of hidden layers in the Transformer decoder. Will use the same value as `num_layers` if not set.

num_heads (`int`, *optional*, defaults to `8`) : Number of attention heads for each attention layer in the Transformer decoder.

relative_attention_num_buckets (`int`, *optional*, defaults to 32) : The number of buckets to use for each attention layer.

relative_attention_max_distance (`int`, *optional*, defaults to 128) : The maximum distance of the longer sequences for the bucket separation.

dropout_rate (`Union[float, int]`, *optional*, defaults to `0.1`) : The ratio for all dropout layers.

layer_norm_epsilon (`float`, *optional*, defaults to `1e-06`) : The epsilon used by the layer normalization layers.

initializer_factor (`float`, *optional*, defaults to `1.0`) : A factor for initializing all weight matrices (should be kept to 1, used internally for initialization testing).

feed_forward_proj (`string`, *optional*, defaults to `"relu"`) : Type of feed forward layer to be used. Should be one of `"relu"` or `"gated-gelu"`. T5v1.1 uses the `"gated-gelu"` feed forward projection. Original T5 uses `"relu"`.

use_cache (`bool`, *optional*, defaults to `True`) : Whether or not the model should return the last key/values attentions (not used by all models). Only relevant if `config.is_decoder=True` or when the model is a decoder-only generative model.

pad_token_id (`int`, *optional*, defaults to `0`) : Token id used for padding in the vocabulary.

eos_token_id (`Union[int, list[int]]`, *optional*, defaults to `1`) : Token id used for end-of-stream in the vocabulary.

classifier_dropout (`Union[float, int]`, *optional*, defaults to `0.0`) : The dropout ratio for classifier.

is_decoder (`bool`, *optional*, defaults to `False`) : Whether the model is used as a decoder or not. If `False`, the model is used as an encoder.


## T5TokenizerFast[[transformers.T5Tokenizer]]

#### transformers.T5Tokenizer[[transformers.T5Tokenizer]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/tokenization_t5.py#L30)

Construct a T5 tokenizer (backed by HuggingFace's *tokenizers* library). Based on
[Unigram](https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=unigram#models).

This tokenizer inherits from [TokenizersBackend](/docs/transformers/main/en/main_classes/tokenizer#transformers.TokenizersBackend) which contains most of the main methods. Users should
refer to this superclass for more information regarding those methods.

get_sentinel_token_idstransformers.T5Tokenizer.get_sentinel_token_idshttps://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/tokenization_t5.py#L154[]
Get the token IDs for sentinel tokens.

**Parameters:**

vocab_file (`str`, *optional*) : [SentencePiece](https://github.com/google/sentencepiece) file (generally has a *.spm* extension) that contains the vocabulary necessary to instantiate a tokenizer.

eos_token (`str`, *optional*, defaults to `""`) : The end of sequence token.    When building a sequence using special tokens, this is not the token that is used for the end of sequence. The token used is the `sep_token`.   

unk_token (`str`, *optional*, defaults to `""`) : The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this token instead.

pad_token (`str`, *optional*, defaults to `""`) : The token used for padding, for example when batching sequences of different lengths.

extra_ids (`int`, *optional*, defaults to 100) : Add a number of extra ids added to the vocabulary for use as sentinels. These tokens are accessible as "" where "{%d}" is a number between 0 and extra_ids-1. These tokens can be retrieved by calling get_sentinel_tokens method and token ids can be by calling get_sentinel_token_ids method

additional_special_tokens (`list[str]`, *optional*) : Additional special tokens used by the tokenizer.

vocab (`str`, `dict` or `list`, *optional*) : Custom vocabulary dict. If not provided, a minimal vocabulary is created using the special tokens.
#### get_sentinel_tokens[[transformers.T5Tokenizer.get_sentinel_tokens]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/tokenization_t5.py#L148)

Get the list of sentinel tokens (extra_id tokens) from additional_special_tokens.

## T5ForConditionalGeneration[[transformers.T5ForConditionalGeneration]]

#### transformers.T5ForConditionalGeneration[[transformers.T5ForConditionalGeneration]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py#L952)

T5 Model with a `language modeling` head on top.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.T5ForConditionalGeneration.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py#L991[{"name": "input_ids", "val": ": torch.LongTensor | None = None"}, {"name": "attention_mask", "val": ": torch.FloatTensor | None = None"}, {"name": "decoder_input_ids", "val": ": torch.LongTensor | None = None"}, {"name": "decoder_attention_mask", "val": ": torch.BoolTensor | None = None"}, {"name": "encoder_outputs", "val": ": tuple[tuple[torch.Tensor]] | None = None"}, {"name": "past_key_values", "val": ": transformers.cache_utils.Cache | None = None"}, {"name": "inputs_embeds", "val": ": torch.FloatTensor | None = None"}, {"name": "decoder_inputs_embeds", "val": ": torch.FloatTensor | None = None"}, {"name": "labels", "val": ": torch.LongTensor | None = None"}, {"name": "use_cache", "val": ": bool | None = None"}, {"name": "output_attentions", "val": ": bool | None = None"}, {"name": "output_hidden_states", "val": ": bool | None = None"}, {"name": "return_dict", "val": ": bool | None = None"}, {"name": "**kwargs", "val": ""}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) --
  Indices of input sequence tokens in the vocabulary. T5 is a model with relative position embeddings so you
  should be able to pad the inputs on both the right and the left.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for detail.

  [What are input IDs?](../glossary#input-ids)

  To know more on how to prepare `input_ids` for pretraining take a look a [T5 Training](./t5#training).
- **attention_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **decoder_input_ids** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*) --
  Indices of decoder input sequence tokens in the vocabulary.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are decoder input IDs?](../glossary#decoder-input-ids)

  T5 uses the `pad_token_id` as the starting token for `decoder_input_ids` generation. If `past_key_values`
  is used, optionally only the last `decoder_input_ids` have to be input (see `past_key_values`).

  To know more on how to prepare `decoder_input_ids` for pretraining take a look at [T5
  Training](./t5#training).
- **decoder_attention_mask** (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*) --
  Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
  be used by default.
- **encoder_outputs** (`tuple[tuple[torch.Tensor]]`, *optional*) --
  Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
  `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of
  hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
- **past_key_values** (`~cache_utils.Cache`, *optional*) --
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/main/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/main/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
- **inputs_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) --
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model's internal embedding lookup matrix.
- **decoder_inputs_embeds** (`torch.FloatTensor` of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*) --
  Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded
  representation. If `past_key_values` is used, optionally only the last `decoder_inputs_embeds` have to be
  input (see `past_key_values`). This is useful if you want more control over how to convert
  `decoder_input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

  If `decoder_input_ids` and `decoder_inputs_embeds` are both unset, `decoder_inputs_embeds` takes the value
  of `inputs_embeds`.
- **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) --
  Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
  config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
  labels in `[0, ..., config.vocab_size]`
- **use_cache** (`bool`, *optional*) --
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.0[Seq2SeqLMOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput) or `tuple(torch.FloatTensor)`A [Seq2SeqLMOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([T5Config](/docs/transformers/main/en/model_doc/t5#transformers.T5Config)) and inputs.
The [T5ForConditionalGeneration](/docs/transformers/main/en/model_doc/t5#transformers.T5ForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Language modeling loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) -- Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
- **past_key_values** (`EncoderDecoderCache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [EncoderDecoderCache](/docs/transformers/main/en/internal/generation_utils#transformers.EncoderDecoderCache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
- **decoder_hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
- **decoder_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
- **cross_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
  weighted average in the cross-attention heads.
- **encoder_last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) -- Sequence of hidden-states at the output of the last layer of the encoder of the model.
- **encoder_hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
- **encoder_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.

Examples:

```python
>>> from transformers import AutoTokenizer, T5ForConditionalGeneration

>>> tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
>>> model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")

>>> # training
>>> input_ids = tokenizer("The  walks in  park", return_tensors="pt").input_ids
>>> labels = tokenizer(" cute dog  the ", return_tensors="pt").input_ids
>>> outputs = model(input_ids=input_ids, labels=labels)
>>> loss = outputs.loss
>>> logits = outputs.logits

>>> # inference
>>> input_ids = tokenizer(
...     "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
... ).input_ids  # Batch size 1
>>> outputs = model.generate(input_ids)
>>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
>>> # studies have shown that owning a dog is good for you.
```

**Parameters:**

config ([T5Config](/docs/transformers/main/en/model_doc/t5#transformers.T5Config)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

`[Seq2SeqLMOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput) or `tuple(torch.FloatTensor)``

A [Seq2SeqLMOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([T5Config](/docs/transformers/main/en/model_doc/t5#transformers.T5Config)) and inputs.

## T5ForQuestionAnswering[[transformers.T5ForQuestionAnswering]]

#### transformers.T5ForQuestionAnswering[[transformers.T5ForQuestionAnswering]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py#L1433)

The T5 transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute `span start logits` and `span end logits`).

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.T5ForQuestionAnswering.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py#L1470[{"name": "input_ids", "val": ": torch.LongTensor | None = None"}, {"name": "attention_mask", "val": ": torch.FloatTensor | None = None"}, {"name": "decoder_input_ids", "val": ": torch.LongTensor | None = None"}, {"name": "decoder_attention_mask", "val": ": torch.BoolTensor | None = None"}, {"name": "encoder_outputs", "val": ": tuple[tuple[torch.Tensor]] | None = None"}, {"name": "start_positions", "val": ": torch.LongTensor | None = None"}, {"name": "end_positions", "val": ": torch.LongTensor | None = None"}, {"name": "inputs_embeds", "val": ": torch.FloatTensor | None = None"}, {"name": "decoder_inputs_embeds", "val": ": torch.FloatTensor | None = None"}, {"name": "use_cache", "val": ": bool | None = None"}, {"name": "output_attentions", "val": ": bool | None = None"}, {"name": "output_hidden_states", "val": ": bool | None = None"}, {"name": "return_dict", "val": ": bool | None = None"}, {"name": "**kwargs", "val": ""}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) --
  Indices of input sequence tokens in the vocabulary. T5 is a model with relative position embeddings so you
  should be able to pad the inputs on both the right and the left.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for detail.

  [What are input IDs?](../glossary#input-ids)

  To know more on how to prepare `input_ids` for pretraining take a look a [T5 Training](./t5#training).
- **attention_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **decoder_input_ids** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*) --
  Indices of decoder input sequence tokens in the vocabulary.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are decoder input IDs?](../glossary#decoder-input-ids)

  T5 uses the `pad_token_id` as the starting token for `decoder_input_ids` generation. If `past_key_values`
  is used, optionally only the last `decoder_input_ids` have to be input (see `past_key_values`).

  To know more on how to prepare `decoder_input_ids` for pretraining take a look at [T5
  Training](./t5#training).
- **decoder_attention_mask** (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*) --
  Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
  be used by default.
- **encoder_outputs** (`tuple[tuple[torch.Tensor]]`, *optional*) --
  Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
  `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of
  hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
- **start_positions** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) --
  Labels for position (index) of the start of the labelled span for computing the token classification loss.
  Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
  are not taken into account for computing the loss.
- **end_positions** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) --
  Labels for position (index) of the end of the labelled span for computing the token classification loss.
  Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
  are not taken into account for computing the loss.
- **inputs_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) --
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model's internal embedding lookup matrix.
- **decoder_inputs_embeds** (`torch.FloatTensor` of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*) --
  Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded
  representation. If `past_key_values` is used, optionally only the last `decoder_inputs_embeds` have to be
  input (see `past_key_values`). This is useful if you want more control over how to convert
  `decoder_input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

  If `decoder_input_ids` and `decoder_inputs_embeds` are both unset, `decoder_inputs_embeds` takes the value
  of `inputs_embeds`.
- **use_cache** (`bool`, *optional*) --
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.0[Seq2SeqQuestionAnsweringModelOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.Seq2SeqQuestionAnsweringModelOutput) or `tuple(torch.FloatTensor)`A [Seq2SeqQuestionAnsweringModelOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.Seq2SeqQuestionAnsweringModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([T5Config](/docs/transformers/main/en/model_doc/t5#transformers.T5Config)) and inputs.
The [T5ForQuestionAnswering](/docs/transformers/main/en/model_doc/t5#transformers.T5ForQuestionAnswering) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
- **start_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) -- Span-start scores (before SoftMax).
- **end_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) -- Span-end scores (before SoftMax).
- **past_key_values** (`EncoderDecoderCache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [EncoderDecoderCache](/docs/transformers/main/en/internal/generation_utils#transformers.EncoderDecoderCache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
- **decoder_hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
- **decoder_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
- **cross_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
  weighted average in the cross-attention heads.
- **encoder_last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) -- Sequence of hidden-states at the output of the last layer of the encoder of the model.
- **encoder_hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
- **encoder_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.

Example:

```python
>>> from transformers import AutoTokenizer, T5ForQuestionAnswering
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
>>> model = T5ForQuestionAnswering.from_pretrained("google-t5/t5-small")

>>> question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"

>>> inputs = tokenizer(question, text, return_tensors="pt")
>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> answer_start_index = outputs.start_logits.argmax()
>>> answer_end_index = outputs.end_logits.argmax()

>>> predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
>>> tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)
...

>>> # target is "nice puppet"
>>> target_start_index = torch.tensor([14])
>>> target_end_index = torch.tensor([15])

>>> outputs = model(**inputs, start_positions=target_start_index, end_positions=target_end_index)
>>> loss = outputs.loss
>>> round(loss.item(), 2)
...
```

**Parameters:**

config ([T5Config](/docs/transformers/main/en/model_doc/t5#transformers.T5Config)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

`[Seq2SeqQuestionAnsweringModelOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.Seq2SeqQuestionAnsweringModelOutput) or `tuple(torch.FloatTensor)``

A [Seq2SeqQuestionAnsweringModelOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.Seq2SeqQuestionAnsweringModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([T5Config](/docs/transformers/main/en/model_doc/t5#transformers.T5Config)) and inputs.

