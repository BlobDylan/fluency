import torch
from typing import Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import consts

class RLAgentPolicy:
    def __init__(
        self,
        weights_dir: str = consts.AGENT_MODEL_PATH,
        system_prompt: Optional[str] = None,
    ):
        """
        Initializes the RL Agent Policy model offline via Apple Silicon MPS.

        If ``system_prompt`` is provided and the tokenizer exposes a chat
        template, prompts are wrapped with the template so the policy receives a
        coherent instruction describing its task.
        """
        self.system_prompt = system_prompt
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

    def build_prompt(self, base_query: str) -> str:
        """
        Renders the text fed to the policy. When a system prompt is configured
        and a chat template is available, the query is wrapped via the chat
        template; otherwise the raw query is returned (continuation behaviour).
        """
        if self.system_prompt is not None and self.tokenizer.chat_template:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": base_query},
            ]
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return base_query

    def _truncate_at_eos(
        self, token_ids: torch.Tensor, log_probs: torch.Tensor
    ) -> tuple[list[int], list[float]]:
        """
        Trims a generated sequence (and its aligned log-probs) at the first EOS
        token, dropping any padding that follows. The EOS token itself is kept
        as a valid stopping action.
        """
        ids = token_ids.tolist()
        lps = log_probs.tolist()

        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None and eos_id in ids:
            cut = ids.index(eos_id) + 1
            ids = ids[:cut]
            lps = lps[:cut]
        return ids, lps

    def generate_greedy(self, base_query: str, max_new_tokens: int = 10) -> str:
        """
        INFERENCE MODE: Deterministic, greedy generation.
        Used for testing the model's absolute best guess after training.
        Returns only the generated string.
        """
        prompt = self.build_prompt(base_query)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(consts.DEVICE)
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

    def generate_training_rollouts(
        self,
        base_query: str,
        group_size: int = 4,
        max_new_tokens: int = 10,
        temperature: float = 0.7,
    ) -> list[dict]:
        """
        TRAINING MODE: Stochastic exploration for GRPO.
        Generates a group of G distinct completions for the same prompt.

        Returns, per rollout, the decoded ``text`` (for reward evaluation), the
        shared ``prompt_ids`` and the rollout's real ``generated_ids`` plus the
        aligned ``log_probs``. The token IDs are the *actual sampled IDs* (never
        decoded then re-encoded) so the learning phase can score exactly the
        tokens that were sampled. Sequences are truncated at EOS to drop padding.
        """
        prompt = self.build_prompt(base_query)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(consts.DEVICE)
        input_length = inputs["input_ids"].shape[1]
        prompt_ids = inputs["input_ids"][0].tolist()

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=temperature,
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

            generated_ids, log_probs = self._truncate_at_eos(gen_tokens, gen_log_probs)
            gen_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            rollouts.append({
                "text": gen_text,
                "prompt_ids": prompt_ids,
                "generated_ids": generated_ids,
                "log_probs": log_probs,
            })

        return rollouts

    def score_rollouts(self, rollouts: list[dict]) -> torch.Tensor:
        """
        Recomputes, differentiably, the summed log-probability of each rollout's
        generated tokens under the current policy.

        All rollouts in a group share the same ``prompt_ids`` but have variable
        ``generated_ids`` lengths, so sequences are right-padded into a single
        batch and a mask restricts the sum to real generated positions. Operates
        on the actual sampled token IDs (never decode/re-encode), so the scored
        tokens are exactly those that were sampled. Returns a tensor of shape
        (num_rollouts,); rollouts with no generated tokens score 0.
        """
        prompt_ids = rollouts[0]["prompt_ids"]
        prompt_len = len(prompt_ids)
        pad_id = self.tokenizer.pad_token_id

        full_seqs = [prompt_ids + r["generated_ids"] for r in rollouts]
        max_len = max(len(seq) for seq in full_seqs)

        input_ids = torch.full(
            (len(full_seqs), max_len), pad_id, dtype=torch.long, device=consts.DEVICE
        )
        attention_mask = torch.zeros_like(input_ids)
        for row, seq in enumerate(full_seqs):
            input_ids[row, : len(seq)] = torch.tensor(seq, device=consts.DEVICE)
            attention_mask[row, : len(seq)] = 1

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (B, max_len, vocab)

        # Logit at position t predicts token t+1, so shift to align.
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]

        log_probs = torch.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

        # Build a mask selecting only generated-token predictions. After the
        # shift, generated token j of a rollout is predicted at index
        # (prompt_len - 1 + j), and we exclude positions that fall in padding.
        positions = torch.arange(max_len - 1, device=consts.DEVICE).unsqueeze(0)
        seq_lens = torch.tensor([len(seq) for seq in full_seqs], device=consts.DEVICE).unsqueeze(1)
        gen_mask = (positions >= (prompt_len - 1)) & (positions < (seq_lens - 1))

        return (token_log_probs * gen_mask).sum(dim=1)
