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
            dtype=consts.DTYPE
        ).to(consts.DEVICE)

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def answer_prompt(self, prompt: str, max_new_tokens: int = 64) -> str:
        """
        Executes a direct offline inference generation pass against the target model.
        Convenience wrapper around the batched path for a single prompt.
        """
        return self.answer_prompts([prompt], max_new_tokens=max_new_tokens)[0]

    def answer_prompts(self, prompts: list[str], max_new_tokens: int = 64) -> list[str]:
        """
        Batched offline inference against the target model.

        Uses left padding (required for correct batched decoder generation) so
        the newly generated tokens line up across the batch, then strips the
        prompt and returns one completion string per input prompt.
        """
        if not prompts:
            return []

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            padding_side="left",
        ).to(consts.DEVICE)
        input_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=False
            )

        generated_tokens = outputs[:, input_length:]
        completions = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        return [c.strip() for c in completions]