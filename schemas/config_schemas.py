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
    """Weights for the composite reward function."""
    alpha_toxicity: float = Field(default=1.0, description="Weight multiplier for successful jailbreaks.")
    beta_fluency: float = Field(default=0.01, description="Penalty multiplier for high perplexity (gibberish).")


class GRPOTrainingConfig(BaseModel):
    """Global configuration parameters for the RL loop."""
    # --- Core RL loop ---
    group_size: int = Field(default=4, description="Number of parallel rollouts (G) generated per prompt.")
    max_adversarial_tokens: int = Field(default=20, description="Max tokens the agent can generate for the adversarial text.")
    learning_rate: float = Field(default=5e-5, description="Optimizer learning rate.")
    epochs: int = Field(default=3, description="Number of training epochs.")
    seed: Optional[int] = Field(default=None, description="Optional RNG seed for reproducible runs.")

    # --- Generation behaviour ---
    temperature: float = Field(default=0.7, description="Sampling temperature for rollout exploration.")
    max_response_tokens: int = Field(default=64, description="Max tokens the target model generates per prompt.")

    # --- Attack framing ---
    adversarial_position: AdversarialPosition = Field(
        default=AdversarialPosition.SUFFIX,
        description="Whether the generated adversarial text wraps the query as a prefix or suffix.",
    )
    system_prompt: Optional[str] = Field(
        default=None,
        description="Optional system prompt describing the agent's task (used when the tokenizer has a chat template).",
    )

    # --- Logging ---
    log_every: int = Field(default=1, description="Print a metrics line every N steps.")

    # --- Nested configs ---
    lora: LoRAConfigParams = Field(default_factory=LoRAConfigParams)
    rewards: GRPORewardParams = Field(default_factory=GRPORewardParams)

    @classmethod
    def from_toml(cls, path: str | Path) -> "GRPOTrainingConfig":
        """Loads a config from a TOML file, falling back to defaults for any omitted field."""
        with open(path, "rb") as f:
            data = tomllib.load(f)
        return cls(**data)
