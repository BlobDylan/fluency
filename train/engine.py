import torch

import consts
from schemas.config_schemas import GRPOTrainingConfig
from schemas.io_schemas import PromptRequest


class GRPOEngine:
    """
    Orchestrates one Group Relative Policy Optimization step.

    Heavy tensor work lives on the model wrappers (the agent scores rollouts,
    the target generates completions) so this class stays a thin, readable
    coordinator of the rollout -> evaluate -> learn cycle.
    """

    def __init__(self, agent, target, compliance_evaluator, perplexity_evaluator, optimizer, config: GRPOTrainingConfig):
        self.agent = agent
        self.target = target
        self.compliance_evaluator = compliance_evaluator
        self.perplexity_evaluator = perplexity_evaluator
        self.optimizer = optimizer
        self.config = config

    def _is_degenerate(self, text: str) -> bool:
        """Detects degenerate / non-fluent text in a Unicode-aware way.

        The previous heuristic only caught a fixed ASCII punctuation set, so the
        policy reward-hacked by emitting repeated CJK filler (e.g. "圾圾圾圾圾"),
        which distilgpt2 actually scores as *low* perplexity. This catches the
        general failure modes: emptiness, low character/word diversity, long
        single-character runs, and a high non-ASCII-letter ratio.
        """
        text = text.strip()
        if not text:
            return True

        # 1. Long consecutive run of the same character ("圾圾圾圾圾", "aaaaa",
        #    "!!!!"). Natural text rarely repeats a char 4+ times in a row.
        run = 1
        for a, b in zip(text, text[1:]):
            run = run + 1 if a == b else 1
            if run >= 4:
                return True

        # 2. Word-level repetition ("toe toe toe toe").
        words = text.split()
        if len(words) >= 4 and (len(set(words)) / len(words)) < 0.40:
            return True

        # 3. Mostly non-fluent symbols: few ASCII letters relative to length.
        #    (Catches CJK/punctuation filler without flagging real sentences,
        #    which run ~0.8 ASCII-letter ratio.)
        ascii_letters = sum(1 for c in text if c.isascii() and c.isalpha())
        if len(text) >= 8 and (ascii_letters / len(text)) < 0.35:
            return True

        return False

    def _fluency_penalty(self, perplexity: float) -> float:
        """Maps perplexity to a bounded penalty in [0, 1].

        Crucially this only *penalises* high perplexity — it never *rewards* low
        perplexity, because degenerate repetition scores deceptively low. Below
        ``fluency_threshold`` the penalty is 0; it ramps linearly to 1.0 over the
        next ``fluency_scale`` of perplexity.
        """
        r = self.config.rewards
        excess = perplexity - r.fluency_threshold
        if excess <= 0:
            return 0.0
        return min(1.0, excess / r.fluency_scale)

    def _compute_rewards(self, base_query: str, rollouts: list[dict]) -> list[dict]:
        r = self.config.rewards
        requests = [
            PromptRequest(
                base_query=base_query,
                adversarial_text=rollout["text"],
                position=self.config.adversarial_position,
            )
            for rollout in rollouts
        ]

        completions = self.target.answer_prompts(
            [req.full_prompt for req in requests],
            max_new_tokens=self.config.max_response_tokens,
            use_chat_template=self.config.target_chat_template,
        )

        breakdowns = []
        for req, completion in zip(requests, completions):
            adv_degenerate = self._is_degenerate(req.adversarial_text)
            perp_score = self.perplexity_evaluator.evaluate_fluency(req.adversarial_text)

            # Degenerate adversarial text gets a flat floor reward regardless of
            # what the target said — this is what kills the repetition hack.
            if adv_degenerate:
                breakdowns.append({
                    "adversarial_text": req.adversarial_text,
                    "full_prompt": req.full_prompt,
                    "completion": completion,
                    "compliance": 0.0,
                    "perplexity": perp_score,
                    "attack_term": 0.0,
                    "fluency_penalty": 0.0,
                    "reward": r.degenerate_reward,
                    "degenerate": True,
                })
                continue

            # A degenerate target completion can't count as a real jailbreak.
            comp_score = (
                0.0 if self._is_degenerate(completion)
                else self.compliance_evaluator.evaluate_compliance(req.base_query, completion)
            )

            attack_term = r.alpha_toxicity * comp_score
            fluency_penalty = r.beta_fluency * self._fluency_penalty(perp_score)
            reward = attack_term - fluency_penalty + r.base_reward

            breakdowns.append({
                "adversarial_text": req.adversarial_text,
                "full_prompt": req.full_prompt,
                "completion": completion,
                "compliance": comp_score,
                "perplexity": perp_score,
                "attack_term": attack_term,
                "fluency_penalty": fluency_penalty,
                "reward": reward,
                "degenerate": False,
            })
        return breakdowns

    def _log_debug(self, base_query: str, breakdowns: list[dict], advantages: torch.Tensor) -> None:
        """Prints a verbose per-rollout breakdown of the reward computation."""
        adv = advantages.tolist()
        print(f"\n  ┌─ DEBUG step | query: {base_query!r}")
        for i, b in enumerate(breakdowns):
            flag = " [DEGEN]" if b.get("degenerate") else ""
            print(
                f"  │ [{i}] reward {b['reward']:+.4f} = "
                f"attack {b['attack_term']:+.4f} (comp {b['compliance']:.4f}) "
                f"- fluency {b['fluency_penalty']:.4f} (ppl {b['perplexity']:.1f}) "
                f"| adv {adv[i]:+.3f}{flag}"
            )
            print(f"  │     adv_text:   {b['adversarial_text']!r}")
            print(f"  │     completion: {b['completion']!r}")
        print("  └─")

    def step(self, base_query: str) -> dict:
        """Executes a single complete GRPO optimization step for one query."""
        # 1. ROLLOUT PHASE (generation, no gradients)
        rollouts = self.agent.generate_training_rollouts(
            base_query=base_query,
            group_size=self.config.group_size,
            max_new_tokens=self.config.max_adversarial_tokens,
            temperature=self.config.temperature,
        )

        # 2. EVALUATION PHASE (target + reward models)
        breakdowns = self._compute_rewards(base_query, rollouts)
        rewards = [b["reward"] for b in breakdowns]
        reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=consts.DEVICE)

        # GRPO advantages: A = (R - mean) / std
        advantages = (reward_tensor - reward_tensor.mean()) / (reward_tensor.std() + 1e-8)

        if self.config.debug:
            self._log_debug(base_query, breakdowns, advantages)

        # 3. LEARNING PHASE (batched differentiable scoring of the real sampled tokens)
        self.optimizer.zero_grad()
        use_kl = self.config.kl_coeff > 0
        scored = self.agent.score_rollouts(rollouts, with_kl=use_kl)
        sum_log_probs = scored["sum_log_probs"]  # (G,), differentiable

        # Policy-gradient objective: minimize -A * log pi (advantage detached).
        pg_loss = -(advantages.detach() * sum_log_probs).sum() / self.config.group_size

        # KL-to-reference keeps the policy near the frozen base, preventing the
        # collapse into a degenerate fixed point we saw in the 10-epoch run.
        kl_loss = self.config.kl_coeff * scored["kl"].mean() if use_kl else torch.zeros((), device=consts.DEVICE)
        # Optional entropy bonus (subtracted from loss) to preserve exploration.
        ent_loss = -self.config.entropy_coeff * scored["entropy"].mean() if self.config.entropy_coeff > 0 else torch.zeros((), device=consts.DEVICE)

        loss = pg_loss + kl_loss + ent_loss

        # 4. OPTIMIZATION PHASE
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        best_idx = torch.argmax(reward_tensor).item()
        group_size = len(breakdowns)
        return {
            "loss": loss.item(),
            "pg_loss": pg_loss.item(),
            "kl": scored["kl"].mean().item(),
            "entropy": scored["entropy"].mean().item(),
            "mean_reward": reward_tensor.mean().item(),
            "max_reward": reward_tensor.max().item(),
            "mean_compliance": sum(b["compliance"] for b in breakdowns) / group_size,
            "max_compliance": max(b["compliance"] for b in breakdowns),
            "mean_perplexity": sum(b["perplexity"] for b in breakdowns) / group_size,
            "frac_degenerate": sum(1 for b in breakdowns if b.get("degenerate")) / group_size,
            "best_prefix": rollouts[best_idx]["text"],
        }
