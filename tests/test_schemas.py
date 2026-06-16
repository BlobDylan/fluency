# tests/test_schemas.py
import pytest
from pydantic import ValidationError
from schemas.io_schemas import PromptRequest, TargetResponse, RewardSignal, RLExperience

# ==========================================
# 1. PromptRequest Tests
# ==========================================
def test_prompt_request_property():
    """Verifies that the full_prompt property joins strings correctly."""
    req = PromptRequest(base_query="Do X.", adversarial_prefix="Prefix text")
    assert req.full_prompt == "Prefix text Do X."

# ==========================================
# 2. RewardSignal Tests
# ==========================================
def test_reward_signal_valid_bounds():
    """Verifies RewardSignal accepts accurate mathematical values."""
    signal = RewardSignal(toxicity_score=0.85, perplexity_score=12.4, total_reward=1.2)
    assert signal.toxicity_score == 0.85
    assert signal.perplexity_score == 12.4

def test_reward_signal_invalid_toxicity():
    """Verifies Pydantic catches errors on out-of-bounds toxicity."""
    with pytest.raises(ValidationError):
        RewardSignal(toxicity_score=1.5, perplexity_score=10.0, total_reward=0.0)

    with pytest.raises(ValidationError):
        RewardSignal(toxicity_score=-0.1, perplexity_score=10.0, total_reward=0.0)

def test_reward_signal_invalid_perplexity():
    """Verifies Pydantic catches errors on negative perplexity."""
    with pytest.raises(ValidationError):
        RewardSignal(toxicity_score=0.5, perplexity_score=-5.0, total_reward=0.0)

# ==========================================
# 3. TargetResponse Tests
# ==========================================
def test_target_response_valid():
    """Verifies TargetResponse correctly nests the PromptRequest."""
    req = PromptRequest(base_query="query", adversarial_prefix="prefix")
    resp = TargetResponse(request=req, completion="completion text")
    
    assert resp.completion == "completion text"
    assert resp.request.full_prompt == "prefix query"

# ==========================================
# 4. RLExperience Tests
# ==========================================
def test_rl_experience_defaults():
    """Verifies RLExperience safely defaults missing lists to empty lists."""
    req = PromptRequest(base_query="Q", adversarial_prefix="P")
    resp = TargetResponse(request=req, completion="C")
    rew = RewardSignal(toxicity_score=0.5, perplexity_score=10.0, total_reward=0.5)

    exp = RLExperience(request=req, response=resp, reward=rew)
    
    # Asserting they are empty lists instead of None
    assert exp.prefix_ids == []
    assert exp.log_probs == []

def test_rl_experience_with_tensors():
    """Verifies RLExperience correctly handles populated lists."""
    req = PromptRequest(base_query="Q", adversarial_prefix="P")
    resp = TargetResponse(request=req, completion="C")
    rew = RewardSignal(toxicity_score=0.5, perplexity_score=10.0, total_reward=0.5)

    exp = RLExperience(
        request=req,
        response=resp,
        reward=rew,
        prefix_ids=[1243, 532, 901],
        log_probs=[-0.12, -1.05, -0.01]
    )
    
    assert len(exp.prefix_ids) == 3
    assert exp.log_probs[0] == -0.12