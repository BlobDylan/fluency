import torch
from typing import Any
from transformers import AutoModelForCausalLM, AutoTokenizer
import consts

class RLAgentPolicy:
    def __init__(self, weights_dir: str = consts.AGENT_MODEL_PATH):
        """
        Initializes the RL Agent Policy model offline via Apple Silicon MPS.
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

    def generate_prefix_greedy(self, prompt: str, max_new_tokens: int = 10) -> str:
        """
        INFERENCE MODE: Deterministic, greedy generation.
        Used for testing the model's absolute best guess after training.
        Returns only the generated string.
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
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    def generate_training_rollouts(self, prompt: str, group_size: int = 4, max_new_tokens: int = 10) -> list[dict]:
        """
        TRAINING MODE: Stochastic exploration for GRPO.
        Generates a group of G distinct completions for the same prompt.
        Returns the text, the token IDs, and the log probabilities needed for RL math.
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
                temperature=0.7,
                num_return_sequences=group_size, 
                return_dict_in_generate=True,
                output_scores=True
            )

        transition_scores = self.model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )

        rollouts = []
        for i in range(group_size):
            gen_tokens = outputs.sequences[i][input_length:]
            gen_log_probs = transition_scores[i]
            gen_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

            rollouts.append({
                "text": gen_text,
                "prefix_ids": gen_tokens.tolist(),
                "log_probs": gen_log_probs.tolist()
            })

        return rollouts