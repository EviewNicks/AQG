# Load Model with LoRA 

```
from src.finetuned.utils.model_loader import load_model_with_lora, print_model_info

# Load model with LoRA - UPDATED: Using IndoT5 (580M params) instead of IndoNanoT5 (248M)
# IndoNanoT5 was insufficient for complex AQG task
peft_model, tokenizer = load_model_with_lora(
    model_name='LazarusNLP/IndoNanoT5-base',  
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=['q', 'v']
)

# Print detailed info
print_model_info(peft_model, tokenizer)
```



---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_41690/4014487330.py in <cell line: 0>()
      3 # Load model with LoRA - UPDATED: Using IndoT5 (580M params) instead of IndoNanoT5 (248M)
      4 # IndoNanoT5 was insufficient for complex AQG task
----> 5 peft_model, tokenizer = load_model_with_lora(
      6     model_name='LazarusNLP/IndoNanoT5-base',
      7     lora_r=8,

/content/src/finetuned/utils/model_loader.py in load_model_with_lora(model_name, lora_r, lora_alpha, lora_dropout, target_modules, device)
     48 
     49     # Apply LoRA
---> 50     peft_model = get_peft_model(base_model, lora_config)
     51 
     52     # Print parameter statistics

/usr/local/lib/python3.12/dist-packages/peft/mapping_func.py in get_peft_model(model, peft_config, adapter_name, mixed, autocast_adapter_dtype, revision, low_cpu_mem_usage)
    120         )
    121 
--> 122     return MODEL_TYPE_TO_PEFT_MODEL_MAPPING[peft_config.task_type](
    123         model,
    124         peft_config,

/usr/local/lib/python3.12/dist-packages/peft/peft_model.py in __init__(self, model, peft_config, adapter_name, **kwargs)
   2312         self, model: torch.nn.Module, peft_config: PeftConfig, adapter_name: str = "default", **kwargs
   2313     ) -> None:
-> 2314         super().__init__(model, peft_config, adapter_name, **kwargs)
   2315         self.base_model_prepare_inputs_for_generation = self.base_model.prepare_inputs_for_generation
   2316         self.base_model_prepare_encoder_decoder_kwargs_for_generation = (

/usr/local/lib/python3.12/dist-packages/peft/peft_model.py in __init__(self, model, peft_config, adapter_name, autocast_adapter_dtype, low_cpu_mem_usage)
    127             ctx = init_empty_weights if low_cpu_mem_usage else nullcontext
    128             with ctx():
--> 129                 self.base_model = cls(model, {adapter_name: peft_config}, adapter_name)
    130 
    131         if hasattr(self.base_model, "_cast_adapter_dtype"):

/usr/local/lib/python3.12/dist-packages/peft/tuners/tuners_utils.py in __init__(self, model, peft_config, adapter_name, low_cpu_mem_usage, state_dict)
    313         self._pre_injection_hook(self.model, self.peft_config[adapter_name], adapter_name)
    314         if peft_config != PeftType.XLORA or peft_config[adapter_name] != PeftType.XLORA:
--> 315             self.inject_adapter(self.model, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage, state_dict=state_dict)
    316 
    317         self._post_injection_hook(self.model, self.peft_config[adapter_name], adapter_name)

/usr/local/lib/python3.12/dist-packages/peft/tuners/tuners_utils.py in inject_adapter(self, model, adapter_name, autocast_adapter_dtype, low_cpu_mem_usage, state_dict)
    911                     ctx = init_empty_weights if low_cpu_mem_usage else nullcontext
    912                     with ctx():
--> 913                         self._create_and_replace(
    914                             peft_config, adapter_name, target, target_name, parent, current_key=key
    915                         )

/usr/local/lib/python3.12/dist-packages/peft/tuners/lora/model.py in _create_and_replace(self, lora_config, adapter_name, target, target_name, parent, current_key, parameter_name)
    267                 )
    268             device_map = get_device_map(self.model)
--> 269             new_module = self._create_new_module(lora_config, adapter_name, target, device_map=device_map, **kwargs)
    270             if adapter_name not in self.active_adapters:
    271                 # adding an additional adapter: it is not automatically trainable

/usr/local/lib/python3.12/dist-packages/peft/tuners/lora/model.py in _create_new_module(lora_config, adapter_name, target, **kwargs)
    416         new_module = None
    417         for dispatcher in dispatchers:
--> 418             new_module = dispatcher(target, adapter_name, config=lora_config, **kwargs)
    419             if new_module is not None:  # first match wins
    420                 break

/usr/local/lib/python3.12/dist-packages/peft/tuners/lora/torchao.py in dispatch_torchao(target, adapter_name, config, **kwargs)
    140         return new_module
    141 
--> 142     if not is_torchao_available():
    143         return new_module
    144 

/usr/local/lib/python3.12/dist-packages/peft/import_utils.py in is_torchao_available()
    141 
    142     if torchao_version < TORCHAO_MINIMUM_VERSION:
--> 143         raise ImportError(
    144             f"Found an incompatible version of torchao. Found version {torchao_version}, "
    145             f"but only versions above {TORCHAO_MINIMUM_VERSION} are supported"

ImportError: Found an incompatible version of torchao. Found version 0.10.0, but only versions above 0.16.0 are supported

---------------------------------------------------------------------------
NOTE: If your import is failing due to a missing package, you can
manually install dependencies using either !pip or !apt.

To view examples of installing some common dependencies, click the
"Open Examples" button below.
---------------------------------------------------------------------------