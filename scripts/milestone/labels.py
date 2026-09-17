"""Canonical milestone labels for the pick_trocar task.

The task has three milestones, hand annotated on the teleop demonstrations and
stored in `meta/episodes.jsonl` under `source_annotation.cleaned_phase_frames`:
left hand grasps the trocar, the handover to the right hand succeeds, and the
right hand places it on the plate. They partition an episode into four ordered
phases.

Milestones are turned into *cumulative* binary targets ("has this happened
yet") rather than one-hot phase or spike targets. That is what a reinforcement
learning reward needs: at frame 300 a policy must know that the handover is
behind it, which a detector for the instant of handover cannot express. The
instants are still recoverable as the frames where a target flips.

The exact frame of a transition is ambiguous even to the annotator, so frames
within `TRANSITION_MARGIN` of a flip are excluded from the loss rather than
forced to one side of it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATASET_ROOT = Path("/localhome/local-yunl/DreamDojo/datasets")

# Order matters: these are cumulative, so milestone k implies milestone k-1.
ANNOTATION_KEYS = (
    "left_hand_pickup",
    "handover_to_right_hand",
    "placed_on_plate",
)
HEAD_NAMES = ("picked", "handed", "placed")
NUM_HEADS = len(HEAD_NAMES)

TRANSITION_MARGIN = 5

TELEOP_DATASETS = (
    "g1_hf_pick_trocar_teleop_success_train",
    "g1_hf_pick_trocar_teleop_success_val",
)
ROLLOUT_DATASETS = (
    "g1_hf_pick_trocar_rollouts_30k_train",
    "g1_hf_pick_trocar_rollouts_30k_val",
    "g1_hf_pick_trocar_rollouts_10k_train",
    "g1_hf_pick_trocar_rollouts_10k_val",
)

# An annotator pointing at the frame one past the end is a boundary slip and is
# clamped. A milestone far beyond the end means the event was cut out of the
# episode when it was trimmed, and the episode cannot be labelled at all.
CLAMP_TOLERANCE = 1


@dataclass(frozen=True)
class Episode:
    dataset: str
    episode_index: int
    length: int
    milestones: tuple[int, ...] | None
    success: bool | None
    note: str = ""

    @property
    def labelled(self) -> bool:
        return self.milestones is not None

    @property
    def video_path(self) -> Path:
        return (
            DATASET_ROOT
            / self.dataset
            / "videos/chunk-000/observation.images.cam_head"
            / f"episode_{self.episode_index:06d}.mp4"
        )


def _clean(raw: list[int], length: int) -> tuple[tuple[int, ...] | None, str]:
    frames = list(raw)
    notes = []
    for i, f in enumerate(frames):
        if f > length - 1:
            if f - (length - 1) <= CLAMP_TOLERANCE:
                frames[i] = length - 1
                notes.append(f"clamped {ANNOTATION_KEYS[i]} {f}->{length - 1}")
            else:
                return None, f"{ANNOTATION_KEYS[i]}={f} beyond episode end {length - 1}"
        if f < 0:
            return None, f"{ANNOTATION_KEYS[i]}={f} negative"
    if not all(a < b for a, b in zip(frames, frames[1:])):
        return None, f"milestones out of order: {frames}"
    return tuple(frames), "; ".join(notes)


def read_episodes(dataset: str) -> list[Episode]:
    """Read every episode of a dataset with its milestones, if annotated."""
    path = DATASET_ROOT / dataset / "meta/episodes.jsonl"
    episodes = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        length = record["length"]
        phases = (record.get("source_annotation") or {}).get("cleaned_phase_frames")
        if phases is None:
            milestones, note = None, "no manual annotation"
        else:
            milestones, note = _clean([phases[k] for k in ANNOTATION_KEYS], length)
            if milestones is None:
                note = f"rejected: {note}"
        episodes.append(
            Episode(
                dataset=dataset,
                episode_index=record["episode_index"],
                length=length,
                milestones=milestones,
                success=record.get("success"),
                note=note,
            )
        )
    return episodes


def read_labelled(datasets: tuple[str, ...]) -> list[Episode]:
    return [ep for d in datasets for ep in read_episodes(d) if ep.labelled]


@dataclass(frozen=True)
class Drop:
    """The frame at which the trocar stopped being held.

    A purely cumulative target cannot express a drop, and on this corpus that
    is not a corner case: 103 of the 266 rollout training episodes lift the
    trocar and never complete the handover. Labelling the frames after the
    trocar falls `picked` teaches the model that a trocar lying on the table
    with both hands empty still counts, and that is measurably what happened --
    `v1` reports p(picked) = 1.000 over the last 20 frames of the episodes
    where the trocar was dropped, exactly what it reports for the ones where
    the task succeeded. It is the most likely reason a hand brushing the trocar
    latches the head on.

    So `picked` means lifted and not since dropped: monotone except that a drop
    resets it. Successful episodes are unaffected, and the ratchet that an RL
    reward needs lives in `reward.py` rather than here, so nothing downstream
    depends on the target itself being monotone.

    `certain` distinguishes a drop we can point at from one we only know
    happened somewhere later. Where the frame is a guess the following frames
    are masked instead of being called negative.
    """

    frame: int
    certain: bool = True


def cumulative_targets(
    milestones: tuple[int, ...],
    length: int,
    drop: Drop | None = None,
) -> np.ndarray:
    """(length, NUM_HEADS) float32, 1 once the milestone frame is reached.

    A milestone at or past `length` never fires, which is how an episode that
    failed part way through is expressed: the milestones it did reach carry
    their frame, and the rest are all-zero targets. Those zeros are the only
    supervision anywhere in the corpus for what a failed handover looks like,
    since every teleop demonstration succeeds.

    A drop returns every head to zero from that frame on: the trocar is back on
    the table however far the episode had got.
    """
    t = np.arange(length)[:, None]
    targets = (t >= np.asarray(milestones)[None, :]).astype(np.float32)
    if drop is not None and drop.certain:
        targets[drop.frame :] = 0.0
    return targets


def loss_weights(
    milestones: tuple[int, ...],
    length: int,
    margin: int = TRANSITION_MARGIN,
    drop: Drop | None = None,
) -> np.ndarray:
    """(length, NUM_HEADS) float32, 0 near a transition and 1 elsewhere.

    Only the head that is flipping is masked; the other heads are unambiguous
    there and stay supervised. Heads that never fire have no transition to be
    uncertain about and stay supervised throughout.

    A drop is a transition like any other and gets the same margin, and an
    uncertain drop masks the rest of the episode instead. Either way this
    touches only the heads the drop actually moves, the ones standing at 1 when
    it happens. A head still at 0 is unaffected: whether or not the trocar was
    dropped, an episode that never handed it over never handed it over, and
    that zero is supervision worth keeping to the last frame. Masking it too
    would throw away the negatives these failures exist to provide.
    """
    weights = np.ones((length, NUM_HEADS), dtype=np.float32)
    if margin > 0:
        for head, frame in enumerate(milestones):
            if frame >= length:
                continue
            lo = max(0, frame - margin)
            hi = min(length, frame + margin + 1)
            weights[lo:hi, head] = 0.0
    if drop is not None:
        moved = np.asarray(milestones) < drop.frame
        lo = max(0, drop.frame - margin)
        hi = min(length, drop.frame + margin + 1) if drop.certain else length
        weights[lo:hi, moved] = 0.0
    return weights


def pad_milestones(reached: tuple[int, ...], length: int) -> tuple[int, ...]:
    """Extend a partial chain to NUM_HEADS entries with never-reached sentinels."""
    return tuple(reached) + (length,) * (NUM_HEADS - len(reached))


# The sentinel is convenient in array code and misleading in a file someone
# reads: `[69, 131, 192]` on a 192-frame episode looks like three milestones
# and means two. Files use null.
def to_json(milestones: tuple[int, ...], length: int) -> list[int | None]:
    return [None if f >= length else int(f) for f in milestones]


def from_json(values: list[int | None], length: int) -> tuple[int, ...]:
    return tuple(length if v is None else int(v) for v in values)


@dataclass(frozen=True)
class ExternalLabel:
    """A rollout label, possibly supervising only some of the heads.

    Review showed the automatic labels are not uniformly trustworthy: an
    episode's `placed` target can be certain while its `picked` target is a
    coin flip. Dropping the whole episode would throw away the certain part,
    and keeping it whole would train on the coin flip, so each head carries its
    own switch.
    """

    milestones: tuple[int, ...]
    supervise: tuple[bool, ...]
    length: int
    provenance: str = ""
    drop: Drop | None = None


def load_external_labels(path: Path) -> dict[tuple[str, int], ExternalLabel]:
    """Read milestones produced by auto-labelling and human review."""
    labels = {}
    for r in json.loads(Path(path).read_text()):
        length = r["length"]
        drop = r.get("drop")
        labels[(r["dataset"], r["episode_index"])] = ExternalLabel(
            milestones=from_json(r["milestones"], length),
            supervise=tuple(r.get("supervise", [True] * NUM_HEADS)),
            length=length,
            provenance=r.get("provenance", ""),
            drop=Drop(int(drop["frame"]), bool(drop.get("certain", True)))
            if drop else None,
        )
    return labels


def summarise(datasets: tuple[str, ...]) -> None:
    for dataset in datasets:
        episodes = read_episodes(dataset)
        good = [e for e in episodes if e.labelled]
        print(f"\n{dataset}: {len(episodes)} episodes, {len(good)} labelled")
        unannotated = [e for e in episodes if e.note == "no manual annotation"]
        if unannotated:
            print(f"  {len(unannotated)} carry no manual annotation")
        for e in episodes:
            if e.note and e.note != "no manual annotation":
                print(f"  ep{e.episode_index:4d} len={e.length:4d}  {e.note}")
        if not good:
            continue
        frames = sum(e.length for e in good)
        counts = np.zeros(NUM_HEADS)
        for e in good:
            counts += cumulative_targets(e.milestones, e.length).sum(0)
        masked = sum(
            (loss_weights(e.milestones, e.length) == 0).sum() for e in good
        )
        print(f"  {frames} frames, positives: " + ", ".join(
            f"{n} {100 * c / frames:.1f}%" for n, c in zip(HEAD_NAMES, counts)
        ))
        print(f"  {masked} of {frames * NUM_HEADS} targets masked near transitions "
              f"({100 * masked / (frames * NUM_HEADS):.1f}%)")


if __name__ == "__main__":
    summarise(TELEOP_DATASETS + ROLLOUT_DATASETS)
