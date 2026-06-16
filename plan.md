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

Would you like to start by generating the dataset loading script for the `base_queries`, or should we write the code to download and verify the model binaries on your Mac?
