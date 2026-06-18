# Fluency-Constrained Adversarial Prompt Optimization

An RL framework that trains a small LoRA policy to generate **fluent, human-readable**
adversarial text that jailbreaks a frozen target LLM — evading the perplexity filters
that catch gibberish-token attacks (GCG, RLPrompt).

## Method

A GRPO loop optimizes a dual objective:

```
R = alpha * S_attack  -  beta * P_fluency
```

- `S_attack` — toxicity probability of the target's response (`toxic-bert`).
- `P_fluency` — perplexity of the agent's adversarial text (`distilgpt2`); lower is more fluent.

### Model stack (all local, offline)

| Role            | Model                          | State            |
| --------------- | ------------------------------ | ---------------- |
| Target          | `Llama-3.2-3B-Instruct`        | Frozen           |
| RL agent        | `Qwen2.5-0.5B`                 | LoRA (trainable) |
| Reward: attack  | `toxic-bert`                   | Frozen           |
| Reward: fluency | `distilgpt2`                   | Frozen           |

## Layout

```
consts.py              # paths, device/dtype, dataset
schemas/
  io_schemas.py        # PromptRequest (prefix/suffix), RewardSignal, RLExperience
  config_schemas.py    # GRPOTrainingConfig (+ TOML loading)
models/                # one wrapper per model; agent owns batched rollout + scoring
train/
  engine.py            # one GRPO step (rollout -> evaluate -> learn), batched
  trainer.py           # configurable entrypoint (build_training_stack, train, CLI)
configs/example.toml   # sample run config
tests/                 # schema + scoring unit tests, model integration tests
```

## Running

```bash
python -m train.trainer                          # built-in defaults
python -m train.trainer --config configs/example.toml
python -m train.trainer --epochs 10 --group-size 8 --seed 42
```

Config precedence: CLI flags > TOML file > defaults in `config_schemas.py`.
Metrics are written to `logs/`, LoRA adapter to `checkpoints/`.

## Tests

```bash
python -m pytest tests/test_schemas.py tests/test_scoring.py   # fast, no weights
python -m pytest tests                                         # full, loads local models
```
