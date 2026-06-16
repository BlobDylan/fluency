import torch
from typing import Any
from transformers import AutoModelForCausalLM, AutoTokenizer
import consts

class TargetLLM:
    def __init__(self, weights_dir: str = consts.TARGET_MODEL_PATH):
        """
        Initializes the victim Llama-3.2-3B model completely offline.
        Binds it to the Apple Silicon MPS backend and freezes all layers.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            weights_dir, 
            local_files_only=True, 
            clean_up_tokenization_spaces=False
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model: Any = AutoModelForCausalLM.from_pretrained(
            weights_dir,
            local_files_only=True,
            dtype=torch.float16
        ).to("mps")
        
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def answer_prompt(self, prompt: str, max_new_tokens: int = 64) -> str:
        """
        Executes a direct offline inference generation pass against the target model.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to("mps")
        input_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=False 
            )
        
        generated_tokens = outputs[0][input_length:]
        completion = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return completion.strip()