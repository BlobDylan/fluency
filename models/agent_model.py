import torch
from typing import Any
from transformers import AutoModelForCausalLM, AutoTokenizer
import consts

class RLAgentPolicy:
    def __init__(self, weights_dir: str = consts.AGENT_MODEL_PATH):
        """
        Initializes the base RL Agent Policy model completely offline.
        Binds it to the Apple Silicon MPS backend.
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

    def generate_prefix(self, prompt: str, max_new_tokens: int = 10) -> str:
        """
        A basic inference generation pass. 
        During training, this will be swapped for a batched token sampling step.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to("mps")
        input_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7
            )
            
        generated_tokens = outputs[0][input_length:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()