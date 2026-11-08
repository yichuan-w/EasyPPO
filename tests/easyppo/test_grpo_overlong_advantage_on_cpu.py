"""Filtered responses contribute zero reward to GRPO statistics without shrinking groups."""

import numpy as np
import pytest
import torch

from verl import DataProto
from verl.trainer.ppo.core_algos import AdvantageEstimator
from verl.trainer.ppo.ray_trainer import filter_overlong_responses
from verl.trainer.ppo.v1.utils import compute_advantage_for_multi_trajectories


@pytest.mark.parametrize("normalize", [True, False], ids=["grpo", "dr_grpo"])
@pytest.mark.parametrize("enabled", [True, False], ids=["filter", "no_filter"])
@pytest.mark.parametrize("all_first_group_overlong", [False, True])
def test_group_of_32_includes_zero_reward_overlong(normalize, enabled, all_first_group_overlong):
    # Distinct group scales detect accidental mixing; overlong judge scores are
    # deliberately nonzero to distinguish zeroing from simply retaining them.
    rewards = torch.tensor([1.0] * 31 + [4.0] + [2.0] * 31 + [8.0])
    masks = torch.zeros(64, 4, dtype=torch.long)
    masks[:, :2] = 1
    masks[[31, 63]] = 1
    if all_first_group_overlong:
        masks[:32] = 1
    overlong = masks.sum(-1) >= 4
    token_rewards = torch.zeros(64, 4)
    token_rewards[torch.arange(64), masks.sum(-1) - 1] = rewards
    original_rewards = token_rewards.clone()
    data = DataProto.from_dict(
        tensors={"token_level_rewards": token_rewards, "response_mask": masks},
        non_tensors={"uid": np.array(["a"] * 32 + ["b"] * 32, dtype=object)},
    )
    data, _ = filter_overlong_responses(data, enabled, max_response_length=4, mask_dropped_rows=False)
    # Mimic DP reordering; each UID still has 32 sessions.
    permutation = torch.randperm(64, generator=torch.Generator().manual_seed(42))
    data = data.select_idxs(permutation)
    keys = [f"{'a' if i < 32 else 'b'}_{i % 32}_0" for i in permutation.tolist()]
    result = compute_advantage_for_multi_trajectories(
        data,
        batch_keys=keys,
        adv_estimator=AdvantageEstimator.GRPO,
        num_repeat=32,
        norm_adv_by_std_in_grpo=normalize,
    )
    expected_rewards = rewards.clone()
    if enabled:
        expected_rewards[overlong] = 0
    groups = expected_rewards.reshape(2, 32)
    # All 32 samples, including zeros, enter both mean and sample std.
    expected = groups - groups.mean(dim=1, keepdim=True)
    if normalize:
        expected /= groups.std(dim=1, keepdim=True) + 1e-6
    expected = expected.flatten()[permutation, None] * masks[permutation]
    torch.testing.assert_close(result.batch["advantages"], expected)
    torch.testing.assert_close(result.batch["returns"], expected)
    torch.testing.assert_close(result.batch["token_level_rewards"], original_rewards[permutation])
    torch.testing.assert_close(result.batch["response_mask"], masks[permutation])
    assert len(result) == 64


@pytest.mark.parametrize("normalize", [True, False], ids=["grpo", "dr_grpo"])
@pytest.mark.parametrize("enabled", [True, False], ids=["filter", "no_filter"])
def test_v1_trainer_preserves_overlong_keys_after_metadata_loss(monkeypatch, normalize, enabled):
    from types import SimpleNamespace

    from omegaconf import OmegaConf
    from tensordict import TensorDict
    from transfer_queue import KVBatchMeta

    from verl.trainer.ppo.v1 import trainer_base

    # The overlong response moved from the last position to the first; the tags
    # deliberately contain no overlong annotation, as after TransferQueue writes.
    keys = ["prompt_31_0"] + [f"prompt_{i}_0" for i in range(31)]
    masks = [torch.ones(4, dtype=torch.long)] + [torch.ones(2, dtype=torch.long) for _ in range(31)]
    rewards = [torch.tensor([0.0, 0.0, 0.0, 4.0])] + [torch.tensor([0.0, 1.0]) for _ in range(31)]
    td = TensorDict(
        {
            "uid": torch.ones(32, dtype=torch.long),
            "response_mask": torch.nested.as_nested_tensor(masks, layout=torch.jagged),
            "rm_scores": torch.nested.as_nested_tensor(rewards, layout=torch.jagged),
        },
        batch_size=32,
    )
    batch = KVBatchMeta(keys=keys, tags=[{} for _ in keys], partition_id="train")
    written = {}
    monkeypatch.setattr(trainer_base.tq, "kv_batch_get", lambda **kwargs: td.clone())

    def put(**kwargs):
        written.update(kwargs)
        return batch

    monkeypatch.setattr(trainer_base.tq, "kv_batch_put", put)
    trainer = SimpleNamespace(
        config=OmegaConf.create(
            {
                "algorithm": {
                    "adv_estimator": "grpo",
                    "gamma": 1.0,
                    "lam": 1.0,
                    "use_kl_in_reward": False,
                    "norm_adv_by_std_in_grpo": normalize,
                },
                "critic": {"variance_weighted_loss": False},
                "actor_rollout_ref": {"rollout": {"n": 32}},
            }
        )
    )
    trainer_base.PPOTrainer._compute_advantage(
        trainer, batch, metrics={}, overlong_keys=["prompt_31_0"] if enabled else []
    )
    scores = torch.tensor([0.0 if enabled else 4.0] + [1.0] * 31)
    expected = scores - scores.mean()
    if normalize:
        expected /= scores.std() + 1e-6
    actual = torch.stack([row[0] for row in written["fields"]["advantages"].unbind()])
    torch.testing.assert_close(actual, expected)
    assert written["keys"] == keys
