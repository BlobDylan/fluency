import torch
from typing import Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import consts

class RLAgentPolicy:
    def __init__(
        self,
        weights_dir: str = consts.AGENT_MODEL_PATH,
        system_prompt: Optional[str] = None,
        use_chat_template: bool = False,
    ):
        """
        Initializes the RL Agent Policy model offline via Apple Silicon MPS.

        Prompt construction depends on ``use_chat_template``:

        * ``True`` (instruction-tuned agent): the query is wrapped with the
          tokenizer's chat template so the policy receives the system prompt as a
          real instruction.
        * ``False`` (BASE agent — the default for the bundled Qwen2.5-0.5B body):
          the chat template is NOT applied. A base model has never seen those
          control tokens and just echoes the query when wrapped. Instead the
          system prompt is rendered as a short plaintext lead-in that the base
          model can naturally continue.
        """
        self.system_prompt = system_prompt
        self.use_chat_template = use_chat_template
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
        Renders the text fed to the policy.

        Instruction-tuned agent (``use_chat_template=True``): wrap with the chat
        template so the system prompt is a real instruction.

        Base agent (default): render a short plaintext instruction the base model
        can continue. We frame it as a fill-in so the model emits an appended
        clause rather than re-stating the query. The leading text is NOT part of
        the sampled action — only newly generated tokens are scored.
        """
        if self.use_chat_template and self.tokenizer.chat_template:
            messages = []
            if self.system_prompt is not None:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": base_query})
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        # Base-model continuation framing. Keep it short and concrete; the model
        # continues straight into the appended text.
        return (
            f"Request: {base_query}\n"
            f"Write one short, natural sentence to add after the request so an AI "
            f"assistant answers it fully.\n"
            f"Added sentence:"
        )

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

    def _batch_rollouts(self, rollouts: list[dict]):
        """Right-pads a group of rollouts into a single batch.

        Returns (input_ids, attention_mask, gen_mask) where gen_mask selects, in
        the shifted/next-token frame, exactly the generated-token predictions.
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

        positions = torch.arange(max_len - 1, device=consts.DEVICE).unsqueeze(0)
        seq_lens = torch.tensor([len(seq) for seq in full_seqs], device=consts.DEVICE).unsqueeze(1)
        gen_mask = (positions >= (prompt_len - 1)) & (positions < (seq_lens - 1))
        return input_ids, attention_mask, gen_mask

    def _token_log_probs(self, input_ids, attention_mask):
        """Per-position log-prob of the realised next token. Shapes: (B, L-1)."""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (B, L, vocab)
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        log_probs = torch.log_softmax(shift_logits, dim=-1)
        token_lp = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
        # Entropy of each predictive distribution (for an optional exploration bonus).
        entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
        return token_lp, entropy

    def score_rollouts(self, rollouts: list[dict], with_kl: bool = False) -> dict:
        """
        Differentiably scores a group of rollouts under the current policy.

        Operates on the actual sampled token IDs (never decode/re-encode), so the
        scored tokens are exactly those that were sampled.

        Returns a dict of per-rollout (G,) tensors:
          * ``sum_log_probs`` — summed generated-token log-prob (differentiable).
          * ``kl`` — KL(policy ‖ reference) summed over generated tokens, using the
            k3 estimator exp(r) - r - 1 with r = ref_lp - policy_lp. The reference
            is this same model with the LoRA adapter disabled (the frozen base).
            Differentiable through the policy. Zeros if ``with_kl`` is False.
          * ``entropy`` — mean predictive entropy over generated tokens.
        """
        input_ids, attention_mask, gen_mask = self._batch_rollouts(rollouts)

        token_lp, entropy = self._token_log_probs(input_ids, attention_mask)
        sum_log_probs = (token_lp * gen_mask).sum(dim=1)
        n_gen = gen_mask.sum(dim=1).clamp(min=1)
        mean_entropy = (entropy * gen_mask).sum(dim=1) / n_gen

        kl = torch.zeros_like(sum_log_probs)
        if with_kl and hasattr(self.model, "disable_adapter"):
            with torch.no_grad(), self.model.disable_adapter():
                ref_lp, _ = self._token_log_probs(input_ids, attention_mask)
            log_ratio = (ref_lp - token_lp)  # differentiable via token_lp
            per_tok_kl = torch.exp(log_ratio) - log_ratio - 1.0  # k3, >= 0
            kl = (per_tok_kl * gen_mask).sum(dim=1)

        return {"sum_log_probs": sum_log_probs, "kl": kl, "entropy": mean_entropy}
