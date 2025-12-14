import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, PeftConfig

class Seq2Seq(nn.Module):
    def __init__(self, args, accelerator=None, **kwargs) -> None:
        super().__init__()
        if accelerator is not None:
            accelerator.wait_for_everyone()

        if args.load_in_8bit:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            quant_kwargs = {"quantization_config": bnb_config}
        else:
            quant_kwargs = {}
        
        if args.load_model_path is not None:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                args.load_model_path, **quant_kwargs
            )
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                args.model, **quant_kwargs
            )
        
        # LoRA config for encoder-decoder (target encoder/decoder layers)
        if args.apply_classhead_lora:
            target_modules = ["q_proj", "v_proj", "lm_head"]
        elif args.apply_qkv_head_lora:
            target_modules = ["q_proj", "v_proj", "k_proj", "lm_head"]
        else:
            target_modules = ["q_proj", "v_proj"]

        peft_config = LoraConfig(
            task_type="SEQ_2_SEQ_LM",
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
        )
        self.model = get_peft_model(model, peft_config)
        self.peft_config = peft_config