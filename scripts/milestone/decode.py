"""Turn per-frame milestone probabilities into one milestone frame each.

Thresholding each frame independently produces labels that flip back and forth
and can claim the handover happened before the pickup. The task is a
left-to-right chain: an episode passes through the phases in order and never
goes back, so the whole episode is decoded at once under that constraint. The
result is at most three transition frames, guaranteed ordered.

The rollout datasets record whether the episode succeeded, which pins the state
the chain has to end in: a success reached the plate, a failure did not. That
label is free supervision and it resolves many episodes the probabilities alone
leave ambiguous. Where the unconstrained decode contradicts it, the episode is
worth a human look, so the disagreement is reported rather than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from scripts.milestone.labels import NUM_HEADS, pad_milestones

NUM_STATES = NUM_HEADS + 1
EPS = 1e-6
AMBIGUOUS = (0.1, 0.9)


@dataclass
class Decoded:
    milestones: tuple[int, ...]
    states: np.ndarray
    mean_loglik: float
    reached: int
    ambiguous_frames: tuple[int, ...] = field(default=())
    constraint_conflict: bool = False

    @property
    def complete(self) -> bool:
        return self.reached == NUM_HEADS

    def reached_frames(self) -> tuple[int, ...]:
        """Only the milestones that actually fired, without the sentinels."""
        return self.milestones[: self.reached]


def _emissions(probs: np.ndarray) -> np.ndarray:
    """(T, NUM_STATES) log-likelihood of each frame under each phase."""
    p = np.clip(probs, EPS, 1 - EPS)
    pos, neg = np.log(p), np.log1p(-p)
    emission = np.zeros((len(p), NUM_STATES), dtype=np.float64)
    for state in range(NUM_STATES):
        # In state s the first s milestones have happened and the rest have not.
        emission[:, state] = pos[:, :state].sum(1) + neg[:, state:].sum(1)
    return emission


def _viterbi(
    emission: np.ndarray,
    allowed_final: set[int],
    advance_penalty: float,
) -> tuple[np.ndarray, float]:
    T = len(emission)
    score = np.full((T, NUM_STATES), -np.inf)
    back = np.zeros((T, NUM_STATES), dtype=np.int8)
    score[0, 0] = emission[0, 0]
    for t in range(1, T):
        for s in range(NUM_STATES):
            stay = score[t - 1, s]
            advance = score[t - 1, s - 1] - advance_penalty if s else -np.inf
            if advance > stay:
                score[t, s], back[t, s] = advance + emission[t, s], 1
            else:
                score[t, s], back[t, s] = stay + emission[t, s], 0

    final = [s for s in allowed_final if np.isfinite(score[-1, s])]
    if not final:
        final = [int(np.argmax(score[-1]))]
    end = max(final, key=lambda s: score[-1, s])

    states = np.zeros(T, dtype=np.int8)
    s = end
    for t in range(T - 1, 0, -1):
        states[t] = s
        s -= back[t, s]
    states[0] = s
    return states, float(score[-1, end])


def decode(
    probs: np.ndarray,
    *,
    success: bool | None = None,
    advance_penalty: float = 0.0,
) -> Decoded:
    """Decode (T, NUM_HEADS) probabilities into ordered milestone frames."""
    emission = _emissions(probs)
    unconstrained, _ = _viterbi(emission, set(range(NUM_STATES)), advance_penalty)

    if success is None:
        allowed = set(range(NUM_STATES))
    elif success:
        allowed = {NUM_HEADS}
    else:
        allowed = set(range(NUM_HEADS))
    states, total = _viterbi(emission, allowed, advance_penalty)

    reached = int(states[-1])
    transitions = [int(np.argmax(states >= k + 1)) for k in range(reached)]
    milestones = pad_milestones(tuple(transitions), len(probs))

    ambiguous = tuple(
        int(((probs[:, k] > AMBIGUOUS[0]) & (probs[:, k] < AMBIGUOUS[1])).sum())
        for k in range(NUM_HEADS)
    )
    return Decoded(
        milestones=milestones,
        states=states,
        mean_loglik=total / len(probs),
        reached=reached,
        ambiguous_frames=ambiguous,
        constraint_conflict=bool(int(unconstrained[-1]) != reached),
    )


def review_priority(d: Decoded) -> float:
    """Higher means the automatic label is less trustworthy.

    A conflict with the recorded outcome dominates everything else: the model
    and the ground truth disagree about what happened, so one of them is wrong.
    Below that, episodes are ranked by how poorly the decoded path explains the
    probabilities and by how long the model dithered around each transition.
    """
    dither = sum(d.ambiguous_frames) / max(1, len(d.states))
    return (100.0 if d.constraint_conflict else 0.0) - d.mean_loglik + 3.0 * dither
