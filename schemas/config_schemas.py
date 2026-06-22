from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from schemas.io_schemas import AdversarialPosition


class LoRAConfigParams(BaseModel):
    """Parameters for PEFT LoRA injection."""
    r: int = Field(default=8, description="Rank of the LoRA update matrices.")
    lora_alpha: int = Field(default=16, description="Scaling factor for LoRA.")
    lora_dropout: float = Field(default=0.05, description="Dropout probability for LoRA layers.")
    target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        description="Linear layers inside the agent model to inject adapters into."
    )


class GRPORewardParams(BaseModel):
    """Weights for the composite reward function.

    Design note: the fluency term only ever *penalises* high perplexity; it never
    *rewards* low perplexity. distilgpt2 assigns very low perplexity to degenerate
    repetition (e.g. "圾圾圾圾圾"), so a reward that subtracts raw perplexity
    actively pushes the policy toward junk. We instead clip perplexity above a
    fluency threshold into a bounded [0, 1] penalty, and reject degenerate text
    outright (see GRPOEngine._is_degenerate).
    """
    alpha_toxicity: float = Field(default=1.0, description="Weight for compliance/jailbreak success (reward).")
    beta_fluency: float = Field(default=0.5, description="Weight for the bounded fluency penalty (in [0, beta]).")
    fluency_threshold: float = Field(
        default=50.0,
        description="Perplexity at or below which no fluency penalty is applied (typical fluent English on distilgpt2).",
    )
    fluency_scale: float = Field(
        default=150.0,
        description="Perplexity span above the threshold that maps to the full penalty of 1.0.",
    )
    degenerate_reward: float = Field(
        default=-1.0,
        description="Flat reward assigned to degenerate/junk adversarial text (overrides all other terms).",
    )
    base_reward: float = Field(
        default=0.05,
        description="Small constant added to non-degenerate rollouts to keep coherent attempts above junk.",
    )


class GRPOTrainingConfig(BaseModel):
    """Global configuration parameters for the RL loop."""
    # --- Core RL loop ---
    group_size: int = Field(default=4, description="Number of parallel rollouts (G) generated per prompt.")
    max_adversarial_tokens: int = Field(default=20, description="Max tokens the agent can generate for the adversarial text.")
    learning_rate: float = Field(default=5e-5, description="Optimizer learning rate.")
    epochs: int = Field(default=3, description="Number of training epochs.")
    seed: Optional[int] = Field(default=None, description="Optional RNG seed for reproducible runs.")

    # --- Generation behaviour ---
    temperature: float = Field(default=1.0, description="Sampling temperature for rollout exploration.")
    max_response_tokens: int = Field(default=64, description="Max tokens the target model generates per prompt.")

    # --- Anti-collapse regularisation ---
    kl_coeff: float = Field(
        default=0.05,
        description="Coefficient on the KL-to-reference penalty that keeps the policy near the frozen base model "
                    "(prevents collapse into degenerate fixed points). Set 0 to disable.",
    )
    entropy_coeff: float = Field(
        default=0.0,
        description="Optional bonus on rollout entropy to preserve exploration. Set 0 to disable.",
    )

    # --- Attack framing ---
    adversarial_position: AdversarialPosition = Field(
        default=AdversarialPosition.SUFFIX,
        description="Whether the generated adversarial text wraps the query as a prefix or suffix.",
    )
    system_prompt: Optional[str] = Field(
        default=None,
        description="Optional system prompt describing the agent's task (used when agent_chat_template is true).",
    )
    agent_chat_template: bool = Field(
        default=False,
        description="If true, wrap the agent prompt with the tokenizer's chat template (use only for an "
                    "instruction-tuned agent). For a BASE agent, leave false: the system prompt is rendered "
                    "as a short plaintext instruction the base model can actually continue.",
    )
    target_chat_template: bool = Field(
        default=True,
        description="If true, the adversarial prompt is placed in the target's user turn via its chat template "
                    "(tests the real aligned assistant). If false, the target completes the raw concatenated text.",
    )

    # --- Logging ---
    log_every: int = Field(default=1, description="Print a metrics line every N steps.")
    debug: bool = Field(
        default=False,
        description="Emit verbose per-rollout diagnostics (scores, prompts, completions, advantages).",
    )

    # --- Nested configs ---
    lora: LoRAConfigParams = Field(default_factory=LoRAConfigParams)
    rewards: GRPORewardParams = Field(default_factory=GRPORewardParams)

    @classmethod
    def from_toml(cls, path: str | Path) -> "GRPOTrainingConfig":
        """Loads a config from a TOML file, falling back to defaults for any omitted field."""
        with open(path, "rb") as f:
            data = tomllib.load(f)
        return cls(**data)
