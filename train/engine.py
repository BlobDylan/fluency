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

    def __init__(self, agent, target, toxicity_evaluator, perplexity_evaluator, optimizer, config: GRPOTrainingConfig):
        self.agent = agent
        self.target = target
        self.toxicity_evaluator = toxicity_evaluator
        self.perplexity_evaluator = perplexity_evaluator
        self.optimizer = optimizer
        self.config = config

    def _compute_rewards(self, base_query: str, rollouts: list[dict]) -> list[dict]:
        """
        Evaluates every rollout's full prompt against the target and reward models.

        Returns one breakdown dict per rollout with the raw component scores
        (``toxicity``, ``perplexity``), their weighted contributions, the final
        ``reward``, and the ``full_prompt`` / ``completion`` for inspection.
        """
        requests = [
            PromptRequest(
                base_query=base_query,
                adversarial_text=r["text"],
                position=self.config.adversarial_position,
            )
            for r in rollouts
        ]

        # Single batched generation pass over the whole group.
        completions = self.target.answer_prompts(
            [req.full_prompt for req in requests],
            max_new_tokens=self.config.max_response_tokens,
        )

        breakdowns = []
        for req, completion in zip(requests, completions):
            tox_score = self.toxicity_evaluator.evaluate_text(completion)
            perp_score = self.perplexity_evaluator.evaluate_fluency(req.adversarial_text)
            attack_term = self.config.rewards.alpha_toxicity * tox_score
            fluency_penalty = self.config.rewards.beta_fluency * perp_score
            breakdowns.append({
                "adversarial_text": req.adversarial_text,
                "full_prompt": req.full_prompt,
                "completion": completion,
                "toxicity": tox_score,
                "perplexity": perp_score,
                "attack_term": attack_term,
                "fluency_penalty": fluency_penalty,
                "reward": attack_term - fluency_penalty,
            })
        return breakdowns

    def _log_debug(self, base_query: str, breakdowns: list[dict], advantages: torch.Tensor) -> None:
        """Prints a verbose per-rollout breakdown of the reward computation."""
        adv = advantages.tolist()
        print(f"\n  ┌─ DEBUG step | query: {base_query!r}")
        for i, b in enumerate(breakdowns):
            print(
                f"  │ [{i}] reward {b['reward']:+.4f} = "
                f"attack {b['attack_term']:+.4f} (tox {b['toxicity']:.4f}) "
                f"- fluency {b['fluency_penalty']:.4f} (ppl {b['perplexity']:.1f}) "
                f"| adv {adv[i]:+.3f}"
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
        sum_log_probs = self.agent.score_rollouts(rollouts)  # (G,), differentiable

        # Policy-gradient objective: minimize -A * log pi (advantage detached).
        loss = -(advantages.detach() * sum_log_probs).sum() / self.config.group_size

        # 4. OPTIMIZATION PHASE
        loss.backward()
        self.optimizer.step()

        best_idx = torch.argmax(reward_tensor).item()
        group_size = len(breakdowns)
        return {
            "loss": loss.item(),
            "mean_reward": reward_tensor.mean().item(),
            "max_reward": reward_tensor.max().item(),
            "mean_toxicity": sum(b["toxicity"] for b in breakdowns) / group_size,
            "max_toxicity": max(b["toxicity"] for b in breakdowns),
            "mean_perplexity": sum(b["perplexity"] for b in breakdowns) / group_size,
            "best_prefix": rollouts[best_idx]["text"],
        }
