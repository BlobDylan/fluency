"""Unit tests for the batched, differentiable rollout scoring math.

These use a tiny deterministic stub model (no weights loaded) so they validate
the alignment/masking logic in RLAgentPolicy.score_rollouts directly.
"""
import torch

import consts
from models.agent_model import RLAgentPolicy


class _StubTokenizer:
    pad_token_id = 0


class _StubOutput:
    def __init__(self, logits):
        self.logits = logits


class _StubModel:
    """Maps each token id to a fixed logit vector via an embedding, so the
    log-probs are deterministic and a reference computation is easy."""

    VOCAB = 12

    def __init__(self):
        torch.manual_seed(0)
        self.table = torch.randn(self.VOCAB, self.VOCAB, device=consts.DEVICE)

    def __call__(self, input_ids, attention_mask=None):
        # logits[b, t] = table[input_ids[b, t]]
        return _StubOutput(self.table[input_ids])


def _make_agent():
    agent = RLAgentPolicy.__new__(RLAgentPolicy)  # bypass heavy __init__
    agent.tokenizer = _StubTokenizer()
    agent.model = _StubModel()
    return agent


def _reference_log_prob(model, prompt_ids, generated_ids):
    """Independent single-sequence implementation of the same quantity."""
    full = torch.tensor([prompt_ids + generated_ids], device=consts.DEVICE)
    logits = model(full).logits[0]
    shift_logits = logits[:-1]
    shift_labels = full[0, 1:]
    lp = torch.log_softmax(shift_logits, dim=-1)
    token_lp = lp.gather(1, shift_labels.unsqueeze(1)).squeeze(1)
    return token_lp[-len(generated_ids):].sum()


def test_score_rollouts_matches_reference_with_ragged_lengths():
    agent = _make_agent()
    prompt_ids = [1, 2, 3]
    rollouts = [
        {"prompt_ids": prompt_ids, "generated_ids": [4, 5]},
        {"prompt_ids": prompt_ids, "generated_ids": [6, 7, 8, 9]},  # longer -> forces padding
        {"prompt_ids": prompt_ids, "generated_ids": [10]},
    ]

    batched = agent.score_rollouts(rollouts)

    assert batched.shape == (3,)
    for i, r in enumerate(rollouts):
        expected = _reference_log_prob(agent.model, prompt_ids, r["generated_ids"])
        assert torch.allclose(batched[i], expected, atol=1e-5), (
            f"rollout {i}: {batched[i].item()} != {expected.item()}"
        )


def test_score_rollouts_is_differentiable():
    agent = _make_agent()
    agent.model.table.requires_grad_(True)
    prompt_ids = [1, 2]
    rollouts = [{"prompt_ids": prompt_ids, "generated_ids": [3, 4]}]

    out = agent.score_rollouts(rollouts).sum()
    out.backward()

    assert agent.model.table.grad is not None
    assert agent.model.table.grad.abs().sum() > 0
