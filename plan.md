Here is the comprehensive summary of your project blueprint, capturing the objectives, decisions, architecture, and next steps to serve as a clean handoff for resuming development.

---

## 1. Project Objective & Core Novelty

The goal of this project is to build a **Fluency-Constrained Adversarial Prompt Optimization** framework using Reinforcement Learning.

- **The Problem:** Historical automated jailbreaking techniques (like RLPrompt or GCG) generate mathematical gibberish tokens. Modern 2026 LLMs effortlessly block these via perplexity filters (the "Perplexity Wall").
- **The Solution (Our Novelty):** We train an RL agent to find _stealthy, grammatically fluent, human-readable_ adversarial prefixes. These prefixes look innocent to human reviewers and security filters but manipulate the internal attention mechanisms of an aligned target model to bypass its safety guardrails.

---

## 2. Key Decisions, Motivations, & Architectural Forks

We evaluated several technical pathways and finalized the following design choices based on your local compute constraints (**M4 MacBook Pro with 48GB RAM**) and project goals:

| Design Dimension               | Selected Choice                                           | Motivation / Justification                                                                                                                               |
| ------------------------------ | --------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Fork A: Agent Architecture** | **LoRA on a Small LM** (`Qwen2.5-0.5B` or `SmolLM2-360M`) | More expressive than a rigid MLP head, faster convergence than an LSTM from scratch, and highly memory-efficient.                                        |
| **Fork B: RL Optimizer**       | **Group Relative Policy Optimization (GRPO)**             | Saves significant VRAM by eliminating a separate Critic network. Safe from vanishing gradients because our reward function outputs continuous fractions. |
| **Fork C: Action Space**       | **Top-K Vocabulary Filtering**                            | Restricting token selection to the top $K$ most common English words shrinks the search space, ensuring faster baseline fluency training.                |
| **Framework Strategy**         | **PyTorch (MPS Backend) + Hugging Face**                  | Prioritizes rapid prototyping and extensive ecosystem support over Apple MLX. Quantizing the target model handles the hardware load.                     |

---

## 3. The Dual-Objective Reward Function

To achieve both high attack success and high human readability, the RL agent evaluates completed token sequences using a combined scalar reward function:

$$R_{total} = \alpha \cdot S_{attack} - \beta \cdot P_{prompt}$$

- **$S_{attack}$ (Attack Score):** The continuous probability (0.0 to 1.0) calculated by `toxic-bert`, evaluating how toxic or compliant the target LLM's response is to a forbidden query.
- **$P_{prompt}$ (Perplexity Score):** The grammatical confusion factor calculated by `distilgpt2` on the agent's prefix. Lower perplexity means more fluent English, yielding a lower penalty.

---

## 4. The 4-Model Public Stack

All models selected are public, free, open-source, and perfectly weighted for a 48GB Unified Memory footprint:

1. **Target Model:** `Meta-Llama-3.2-3B-Instruct` (Strictly frozen, loaded in 4-bit/8-bit precision; highly jailbreak-resistant).
2. **RL Agent Model:** `Qwen/Qwen2.5-0.5B` or `HuggingFaceTB/SmolLM2-360M` (Active LoRA training).
3. **Reward Model A (Toxicity):** `unitary/toxic-bert` (Frozen 110M parameter classification engine).
4. **Reward Model B (Fluency):** `distilgpt2` (Frozen 82M parameter causal engine for perplexity calculation).

---

## 5. Intended Project Structure

The project will be built as a production-grade Python package using **Poetry 2.x** for modern dependency management and **Pydantic** for runtime type-safety.

```text
fluency_rl_project/
├── pyproject.toml             # Poetry 2.x configuration (PEP-621 standard)
├── consts.py                  # Global hyperparameters (Top-K, Alpha, Beta weights)
├── models/
│   ├── weights/               # Local .gitignored model binaries
│   ├── target_model.py        # Frozen Llama-3.2 wrapper
│   ├── reward_model.py        # toxic-bert & distilgpt2 wrappers
│   └── agent_model.py         # Trainable LoRA policy network
├── schemas/
│   └── io_schemas.py          # Pydantic input/output validation models
├── train/
│   ├── grpo.py                # Group relative policy optimization math
│   └── loop.py                # Epoch execution and batch management
├── evaluate/
│   └── evaluator.py           # Evaluation suite (Success Rate vs Fluency curves)
└── tests/
    └── test_models.py         # Pytest assertions for inference pipelines

```

---

## 6. Strategic Points for Resuming Discussion

When you are ready to kick off development, we should resume by tackling the foundational setup:

- Setting up the `base_query` input dataset (using targets from benchmarks like _AdvBench_).
- Writing the local weight download script to pull the model stack into `models/weights/`.
- Drafting the actual generation phase where the LoRA agent samples its group of outputs for the GRPO algorithm.

---

## 7. Pipeline Audit & Fixes (2026-06-22)

A 10-epoch run had collapsed: epochs 7–10 degenerated into `"<query>圾圾圾圾圾"`
with compliance 0.0 and `loss -0.0` (zero-gradient dead end). Root-caused and fixed.

### Root causes found
1. **Fluency reward was inverted (the killer).** distilgpt2 gives *repetition*
   lower perplexity than fluent English (measured: `圾圾圾` ppl ≈ 4 vs real
   sentence ≈ 36). Reward subtracted raw `β·ppl`, so the policy was actively
   pushed toward junk. The old `_is_junk` only caught a fixed ASCII punctuation
   set, so CJK repetition slipped through.
2. **Reward scale imbalance.** `β·ppl` (0.6–2.0) dwarfed `α·comp` (≤1.0); a
   successful fluent jailbreak scored *worse* than silent junk.
3. **Target never got its chat template** — Llama-3.2-3B-Instruct was used as a
   raw text-completer, so "compliance" was autocompletion, not a real bypass.
4. **Agent is a BASE Qwen2.5-0.5B with a Llama chat template grafted on** its
   tokenizer (eos `<|eot_id|>`). It was fed a 122-token Llama chat wrapper it was
   never trained on → it mostly echoed the query and ignored the system prompt.
5. **Not真 GRPO** — plain REINFORCE-with-baseline, no KL/entropy/clip, nothing
   opposing collapse.

### Changes made
- **`train/engine.py`**: `_is_junk` → `_is_degenerate` (Unicode-aware: char-run,
  word-repeat, low ASCII-letter ratio). New `_fluency_penalty` maps ppl→[0,1],
  **penalty-only above a threshold** (never rewards low ppl). Degenerate adv text
  gets a flat `degenerate_reward` floor. Degenerate *completions* can't score
  compliance. `step()` adds KL-to-reference + optional entropy to the loss and
  grad-clips to 1.0; logs `kl` and `frac_degenerate`.
- **`models/agent_model.py`**: `build_prompt` switches on `use_chat_template`
  (base model → short plaintext instruction it can continue; instruct model →
  chat template). `score_rollouts` now returns `{sum_log_probs, kl, entropy}`;
  KL via the k3 estimator using the **adapter-disabled** base as reference.
- **`models/target_model.py`**: `answer_prompts(use_chat_template=True)` puts the
  adversarial prompt in the target's user turn — tests the real aligned model.
- **`schemas/config_schemas.py`**: reward params (alpha/beta/threshold/scale/
  degenerate/base), `kl_coeff`, `entropy_coeff`, `agent_chat_template`,
  `target_chat_template`. Default temperature 0.7→1.0.
- **`configs/example.toml`**: updated to corrected defaults + comments.
- Removed scratch files `debug_compliance.py`, `test_new_reward_logic.py`.
- `tests/test_scoring.py` updated for the dict return.

### Verified
- `pytest tests/test_scoring.py tests/test_schemas.py` → 9 passed.
- 2-step end-to-end smoke run: target now refuses properly ("I can't fulfill
  that request…"), KL active (~0.03–0.06), junk caught and floored.
- `_is_degenerate` / `_fluency_penalty` validated on junk vs fluent fixtures.

### Run command
```bash
.venv/bin/python -m train.trainer --config configs/example.toml | tee logs/run_console.log
```
(~100 steps; watch `mean_compliance` rise and `frac_degenerate` stay low.)

### NOT yet done / open items
- **Agent swap (blocked):** HF downloads fail on this corporate laptop (self-
  signed SSL). Want to swap the BASE agent → **Qwen2.5-0.5B-Instruct** (same
  arch, so LoRA/wiring unchanged; just set `agent_chat_template = true` and point
  `consts.AGENT_MODEL_PATH` at the new dir). User will download via browser.
- **Metrics overwrite:** `logs/training_metrics.json` is clobbered each run — add
  per-run subdirs/timestamps before long experiments.
- **No eval/held-out split:** all 10 queries are train; add a held-out set +
  greedy eval to measure real generalization.
- **KL coefficient untuned:** 0.05 is a guess; may need raising if it still
  drifts or lowering if it can't escape the base distribution.
- **Reward thresholds** (`fluency_threshold=50`, `scale=150`) are calibrated to
  distilgpt2 eyeball values — worth confirming against a fluent-text ppl
  distribution.
- **Single-token EOS assumption** in agent truncation is fine for now but
  revisit if swapping tokenizers.
