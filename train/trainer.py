"""Configurable GRPO training entrypoint.

Usage:
    python -m train.trainer                         # run with built-in defaults
    python -m train.trainer --config run.toml       # load a config from TOML
    python -m train.trainer --epochs 10 --group-size 8   # override individual fields
    python -m train.trainer --system-prompt-file configs/system_prompt.txt
"""
import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model

import consts
from models.agent_model import RLAgentPolicy
from models.perplexity_evaluator import PerplexityEvaluator
from models.target_model import TargetLLM
from models.compliance_evaluator import ComplianceEvaluator
from schemas.config_schemas import GRPOTrainingConfig
from train.engine import GRPOEngine


@dataclass
class TrainingStack:
    """Everything needed to run the loop, assembled once."""
    agent: RLAgentPolicy
    engine: GRPOEngine
    config: GRPOTrainingConfig


def build_training_stack(config: GRPOTrainingConfig) -> TrainingStack:
    """Loads the model stack, injects LoRA into the agent, and wires up the engine."""
    if config.seed is not None:
        torch.manual_seed(config.seed)

    agent = RLAgentPolicy(
        system_prompt=config.system_prompt,
        use_chat_template=config.agent_chat_template,
    )
    target = TargetLLM()
    compliance = ComplianceEvaluator()
    perplexity = PerplexityEvaluator()

    lora_config = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        target_modules=config.lora.target_modules,
        lora_dropout=config.lora.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    agent.model = get_peft_model(agent.model, lora_config)
    optimizer = torch.optim.AdamW(agent.model.parameters(), lr=config.learning_rate)

    engine = GRPOEngine(agent, target, compliance, perplexity, optimizer, config)
    return TrainingStack(agent=agent, engine=engine, config=config)


def save_lora_checkpoint(agent: RLAgentPolicy) -> None:
    """Saves the trained LoRA adapter weights to the configured directory."""
    consts.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    agent.model.save_pretrained(str(consts.CHECKPOINT_DIR))
    print(f"💾 Checkpoint saved to {consts.CHECKPOINT_DIR}")


def train(config: GRPOTrainingConfig, dataset: list[str]) -> list[dict]:
    """Runs the full GRPO loop over the dataset and returns per-step metric logs."""
    print(f"Device: {consts.DEVICE} | dtype: {consts.DTYPE}")
    print(f"Dataset: {len(dataset)} prompts | Total steps: {len(dataset) * config.epochs}")
    print("Loading models and injecting LoRA...")

    stack = build_training_stack(config)

    logs: list[dict] = []
    start = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        print(f"\n=== Epoch {epoch}/{config.epochs} ===")
        for step, query in enumerate(dataset, 1):
            step_start = time.perf_counter()
            metrics = stack.engine.step(query)
            step_time = time.perf_counter() - step_start

            if step % config.log_every == 0:
                print(
                    f"E{epoch} S{step:02d}/{len(dataset)} | "
                    f"loss {metrics['loss']:>8.4f} | "
                    f"mean_rwd {metrics['mean_reward']:>7.4f} | "
                    f"max_rwd {metrics['max_reward']:>7.4f} | "
                    f"comp(mean/max) {metrics['mean_compliance']:.3f}/{metrics['max_compliance']:.3f} | "
                    f"ppl {metrics['mean_perplexity']:>6.1f} | "
                    f"kl {metrics['kl']:.3f} | degen {metrics['frac_degenerate']:.2f} | "
                    f"{step_time:.1f}s | best: {metrics['best_prefix']!r}"
                )

            logs.append({"epoch": epoch, "step": step, "query": query, **metrics})

        if consts.DEVICE == "mps":
            torch.mps.empty_cache()

    total = time.perf_counter() - start
    print(f"\n🏁 Training complete in {total / 60:.1f} min")

    consts.LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(consts.METRICS_FILE, "w") as f:
        json.dump(logs, f, indent=2)
    print(f"📈 Metrics saved to {consts.METRICS_FILE}")

    save_lora_checkpoint(stack.agent)
    return logs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO adversarial-prompt training.")
    parser.add_argument("--config", type=str, default=None, help="Path to a TOML config file.")
    parser.add_argument(
        "--system-prompt-file",
        type=str,
        default=None,
        help="Path to a text file whose contents become the agent's system prompt.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs.")
    parser.add_argument("--group-size", type=int, default=None, help="Override GRPO group size.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--seed", type=int, default=None, help="Override RNG seed.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Verbose per-rollout diagnostics (raw scores, prompts, completions).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = GRPOTrainingConfig.from_toml(args.config) if args.config else GRPOTrainingConfig()

    # CLI overrides take precedence over the file / defaults.
    system_prompt = None
    if args.system_prompt_file:
        system_prompt = Path(args.system_prompt_file).read_text(encoding="utf-8").strip()

    overrides = {
        "epochs": args.epochs,
        "group_size": args.group_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "system_prompt": system_prompt,
        # store_true defaults to False; only override when the flag is passed so
        # it never silently disables debug=true set in a TOML config.
        "debug": True if args.debug else None,
    }
    config = config.model_copy(update={k: v for k, v in overrides.items() if v is not None})

    train(config, consts.RESTRICTED_QUERIES)


if __name__ == "__main__":
    main()
