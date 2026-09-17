#!/usr/bin/env python3
"""Metrics for world-model rollouts that plain PSNR cannot separate.

PSNR over a whole frame is dominated by whatever fills most of the pixels, and
in these episodes that is a static table and background. A rollout can keep the
scene photometrically almost perfect while the trocar changes shape, and a
rollout can lose several dB purely because the arm moved slightly early. Both
show up as "lower PSNR", which makes the metric close to useless for ranking
runs on the failure mode we care about.

Three additions here, each aimed at one thing PSNR hides:

motion weighting
    Weight the error by how much the ground truth actually moves at that pixel,
    so the score reflects the arm and the object rather than the tablecloth.
    `moving_region_*` is the same idea with a hard top-decile mask, which is
    easier to reason about: "error inside the pixels that move".

per-step error
    Closed-loop error compounds. A single number averaged over the rollout
    cannot distinguish a model that is slightly wrong everywhere from one that
    is perfect for 30 frames and then diverges, yet those are very different
    models. The per-frame curve and its slope separate them.

structural / perceptual similarity
    SSIM and LPIPS respond to an object changing shape far more than mean
    squared error does, which is exactly the trocar-morphing case.

All functions take uint8 arrays shaped (T, H, W, 3) and drop frame 0, matching
image_metrics() in validate_pick_trocar_world_model.py: frame 0 is the
conditioning frame and is identical by construction.
"""

from __future__ import annotations

import numpy as np

MAX_VALUE = 255.0
# Fraction of pixels, ranked by ground-truth motion, that count as "moving".
MOVING_QUANTILE = 0.9


def psnr_from_mse(mse: float) -> float:
    return float(20 * np.log10(MAX_VALUE / np.sqrt(max(float(mse), 1e-12))))


def _pixel_squared_error(reference: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Squared error per pixel, averaged over channels, shaped (T-1, H, W)."""
    ref = reference[1:].astype(np.float32)
    pred = prediction[1:].astype(np.float32)
    return np.square(ref - pred).mean(axis=3)


def motion_magnitude(reference: np.ndarray) -> np.ndarray:
    """Per-pixel |frame_t - frame_{t-1}| of the ground truth, shaped (T-1, H, W).

    Aligned with _pixel_squared_error: index i of both refers to frame i+1.
    """
    ref = reference.astype(np.float32)
    return np.abs(ref[1:] - ref[:-1]).mean(axis=3)


def per_frame_mse(reference: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    return _pixel_squared_error(reference, prediction).mean(axis=(1, 2))


def motion_metrics(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """Error restricted to, or weighted by, where the ground truth moves."""
    error = _pixel_squared_error(reference, prediction)
    motion = motion_magnitude(reference)

    total_motion = float(motion.sum())
    if total_motion <= 0:
        # A completely static episode has no moving pixels to score.
        return {
            "motion_weighted_mse": float("nan"),
            "motion_weighted_psnr": float("nan"),
            "moving_region_mse": float("nan"),
            "moving_region_psnr": float("nan"),
            "static_region_mse": float("nan"),
            "moving_pixel_fraction": 0.0,
            "mean_motion": 0.0,
        }

    weighted_mse = float((motion * error).sum() / total_motion)

    # A hard mask over the whole episode rather than per frame: a per-frame
    # threshold would move with the arm and make the score non-comparable
    # between a still moment and a fast one.
    threshold = float(np.quantile(motion, MOVING_QUANTILE))
    mask = motion > threshold
    if mask.any():
        moving_mse = float(error[mask].mean())
        static_mse = float(error[~mask].mean()) if (~mask).any() else float("nan")
    else:
        moving_mse = static_mse = float("nan")

    return {
        "motion_weighted_mse": weighted_mse,
        "motion_weighted_psnr": psnr_from_mse(weighted_mse),
        "moving_region_mse": moving_mse,
        "moving_region_psnr": psnr_from_mse(moving_mse),
        "static_region_mse": static_mse,
        "moving_pixel_fraction": float(mask.mean()),
        "mean_motion": float(motion.mean()),
    }


def drift_metrics(frame_mse: np.ndarray) -> dict[str, float]:
    """How fast the error grows along the rollout.

    Two different questions, and conflating them is easy:

    mse_slope_per_frame is absolute. A model with uniformly lower error has a
    smaller slope even when its error grows at exactly the same rate, so this
    ranks overall accuracy as much as it ranks drift.

    mse_late_over_early and psnr_drop_db are relative, and are the ones that
    isolate compounding: they ask how much worse the end of the rollout is than
    the beginning, independently of where it started. A change that fixes
    exposure bias should move these; a change that just makes the model better
    everywhere will move only the slope.
    """
    steps = np.arange(1, len(frame_mse) + 1, dtype=np.float64)
    if len(frame_mse) < 4:
        return {
            "mse_slope_per_frame": float("nan"),
            "mse_early": float(frame_mse.mean()),
            "mse_late": float(frame_mse.mean()),
            "psnr_drop_db": 0.0,
            "frames_scored": int(len(frame_mse)),
        }

    slope = float(np.polyfit(steps, frame_mse, 1)[0])
    quarter = max(1, len(frame_mse) // 4)
    early = float(frame_mse[:quarter].mean())
    late = float(frame_mse[-quarter:].mean())
    return {
        "mse_slope_per_frame": slope,
        "mse_early": early,
        "mse_late": late,
        "mse_late_over_early": float(late / early) if early > 0 else float("nan"),
        # Positive means the rollout got worse from start to end.
        "psnr_drop_db": psnr_from_mse(early) - psnr_from_mse(late),
        "frames_scored": int(len(frame_mse)),
    }


def _torch_pairs(reference: np.ndarray, prediction: np.ndarray):
    import torch

    ref = torch.from_numpy(reference[1:].astype(np.float32) / MAX_VALUE)
    pred = torch.from_numpy(prediction[1:].astype(np.float32) / MAX_VALUE)
    return ref.permute(0, 3, 1, 2), pred.permute(0, 3, 1, 2)


def ssim_per_frame(
    reference: np.ndarray, prediction: np.ndarray, *, device: str = "cpu", batch: int = 8
) -> np.ndarray | None:
    """Per-frame SSIM, or None if torchmetrics is unavailable."""
    try:
        import torch
        from torchmetrics.functional import structural_similarity_index_measure
    except Exception:
        return None

    ref, pred = _torch_pairs(reference, prediction)
    values = []
    with torch.no_grad():
        for start in range(0, len(ref), batch):
            a = ref[start : start + batch].to(device)
            b = pred[start : start + batch].to(device)
            score = structural_similarity_index_measure(
                b, a, data_range=1.0, reduction="none"
            )
            values.append(np.atleast_1d(score.cpu().numpy().squeeze()))
    return np.concatenate(values)


def lpips_per_frame(
    reference: np.ndarray,
    prediction: np.ndarray,
    *,
    device: str = "cpu",
    batch: int = 8,
    net_type: str = "squeeze",
) -> np.ndarray | None:
    """Per-frame LPIPS, or None if the metric or its weights are unavailable.

    Returns None rather than raising: the weights come from a download, and a
    missing perceptual metric must not take down an eval that is otherwise fine.
    """
    try:
        import torch
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        metric = LearnedPerceptualImagePatchSimilarity(
            net_type=net_type, normalize=True
        ).to(device)
    except Exception:
        return None

    ref, pred = _torch_pairs(reference, prediction)
    values = []
    with torch.no_grad():
        for start in range(0, len(ref), batch):
            a = ref[start : start + batch].to(device)
            b = pred[start : start + batch].to(device)
            for i in range(len(a)):
                values.append(float(metric(b[i : i + 1], a[i : i + 1])))
    return np.asarray(values)


def evaluate(
    reference: np.ndarray,
    prediction: np.ndarray,
    *,
    device: str = "cpu",
    with_perceptual: bool = True,
) -> dict:
    """Full metric bundle for one rollout against its ground truth."""
    frame_mse = per_frame_mse(reference, prediction)
    mse = float(frame_mse.mean())
    result = {
        "mse": mse,
        "psnr": psnr_from_mse(mse),
        "mae": float(
            np.abs(
                reference[1:].astype(np.float32) - prediction[1:].astype(np.float32)
            ).mean()
        ),
        "per_frame_mse": [round(v, 3) for v in frame_mse.tolist()],
        "per_frame_psnr": [round(psnr_from_mse(v), 3) for v in frame_mse.tolist()],
    }
    result.update(motion_metrics(reference, prediction))
    result.update(drift_metrics(frame_mse))

    if with_perceptual:
        ssim = ssim_per_frame(reference, prediction, device=device)
        if ssim is not None:
            result["ssim"] = float(np.mean(ssim))
            result["per_frame_ssim"] = [round(v, 4) for v in ssim.tolist()]
        lpips = lpips_per_frame(reference, prediction, device=device)
        if lpips is not None:
            result["lpips"] = float(np.mean(lpips))
            result["per_frame_lpips"] = [round(v, 4) for v in lpips.tolist()]
    return result
