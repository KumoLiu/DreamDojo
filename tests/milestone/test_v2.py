"""Small CPU safety checks for the retained classifier; no old runs required."""

import sys

import numpy as np
import pytest
import torch

from scripts.milestone import dataset, train
from scripts.milestone.labels import Drop, Episode, cumulative_targets, loss_weights
from scripts.milestone.model import MilestoneNet
from scripts.milestone.reward import CONFIRM_COUNT, CONFIRM_WINDOW, THRESHOLDS


def test_drop_targets_and_uncertain_masks():
    milestones = (10, 80, 80)
    targets = cumulative_targets(milestones, 80, drop=Drop(40))
    np.testing.assert_array_equal(targets[10:40, 0], 1)
    np.testing.assert_array_equal(targets[40:], 0)
    weights = loss_weights(milestones, 80, drop=Drop(40, certain=False))
    np.testing.assert_array_equal(weights[35:, 0], 0)
    np.testing.assert_array_equal(weights[:, 1:], 1)


def test_causal_frame_stack_and_colour_order(monkeypatch):
    frames = []

    def read(_dataset, _episode, frame):
        frames.append(frame)
        image = np.zeros((270, 360, 3), dtype=np.uint8)
        image[..., 2] = 255  # BGR red -> first RGB channel.
        return image

    monkeypatch.setattr(dataset, "_read", read)
    samples = dataset.MilestoneFrames(
        [Episode("synthetic", 0, 20, (2, 10, 18), True)], train=False
    )
    stack, target, weights = samples[10]
    assert frames == [10, 6, 2, 0]
    assert stack.shape == (12, 240, 320)
    assert float(stack[0, 0, 0]) == pytest.approx((1 - 0.485) / 0.229)
    assert float(stack[1, 0, 0]) == pytest.approx(-0.456 / 0.224)
    assert target.tolist() == [1, 1, 0]
    assert weights[1] == 0  # Ambiguous transition is masked.


def test_v2_checkpoint_format_and_three_heads(tmp_path):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        model = MilestoneNet(pretrained=False).eval()
        image = torch.zeros(1, 12, 64, 64)
        with torch.no_grad():
            expected = model(image)
        path = tmp_path / "best.pt"
        torch.save({"model": model.state_dict()}, path)
        restored = MilestoneNet(pretrained=False).eval()
        restored.load_state_dict(torch.load(path, weights_only=True)["model"])
        with torch.no_grad():
            torch.testing.assert_close(restored(image), expected, rtol=0, atol=0)
        assert expected.shape == (1, 3)
    finally:
        torch.set_num_threads(old_threads)


def test_final_training_and_reward_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train", "--name", "new_v2_run"])
    args = train.parse_args()
    assert (args.epochs, args.batch_size, args.lr, args.seed) == (8, 128, 3e-4, 0)
    assert args.extra_labels.name == "reviewed_labels_v3.json"
    assert not args.include_rollout_val
    assert THRESHOLDS == (0.8, 0.8, 0.8)
    assert CONFIRM_WINDOW == (15, 15, 2) and CONFIRM_COUNT == (13, 13, 2)


def test_training_refuses_existing_run_before_reading_data(tmp_path, monkeypatch):
    final = tmp_path / "v2"
    final.mkdir()
    checkpoint = final / "best.pt"
    checkpoint.write_bytes(b"preserve-final-checkpoint")
    monkeypatch.setattr(train, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["train", "--name", "v2"])
    with pytest.raises(FileExistsError):
        train.main()
    assert checkpoint.read_bytes() == b"preserve-final-checkpoint"
