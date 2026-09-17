# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small inference-only cache for immutable, whole-batch text embeddings."""

from collections import OrderedDict
from collections.abc import Callable, Hashable

import torch


class TextEmbeddingCache:
    """Cache CPU copies without changing batch composition or tensor values.

    Owners must clear the cache after changing encoder weights in place. Keys
    must include the full ordered prompt batch and encoder/configuration identity.
    Callers receive independent tensors so conditioning cannot corrupt the cache.
    """

    def __init__(self, max_entries: int = 4) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self.max_entries = max_entries
        self.entries: OrderedDict = OrderedDict()
        self.hits = 0
        self.misses = 0

    def clear(self) -> None:
        """Invalidate entries after an encoder/checkpoint change."""
        self.entries.clear()
        self.hits = self.misses = 0

    def get_or_compute(self, key: Hashable, compute: Callable[[], torch.Tensor]) -> torch.Tensor:
        """Return exact embeddings on their original device, or compute a miss."""
        if key in self.entries:
            self.hits += 1
            value, device = self.entries.pop(key)
            self.entries[key] = value, device
            return value.to(device=device, copy=True)
        self.misses += 1
        value = compute()
        self.entries[key] = value.detach().to(device="cpu", copy=True), value.device
        while len(self.entries) > self.max_entries:
            self.entries.popitem(last=False)
        return value
