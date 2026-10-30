from __future__ import annotations

import gzip
import json
from functools import cached_property
from importlib import resources as ir
from pathlib import Path

import numpy as np

DATA = ir.files("hardread") / "data"


class BigramSet:
    """Sorted 32-bit hashes of lowercased word pairs seen in the reference, stored delta-encoded."""

    def __init__(self, hashes: np.ndarray):
        self.hashes = np.unique(hashes.astype(np.uint32))

    def __contains__(self, h: int) -> bool:
        h = np.uint32(h)
        i = self.hashes.searchsorted(h)
        return bool(i < len(self.hashes) and self.hashes[i] == h)

    def __len__(self) -> int:
        return len(self.hashes)

    @classmethod
    def load(cls, path) -> BigramSet:
        with gzip.open(path, "rb") as f:
            deltas = np.frombuffer(f.read(), dtype="<u4")
        return cls(np.cumsum(deltas, dtype=np.uint32))

    def save(self, path) -> None:
        with gzip.open(path, "wb", compresslevel=9) as f:
            f.write(np.diff(self.hashes, prepend=np.uint32(0)).astype("<u4").tobytes())


def _words(path) -> frozenset[str]:
    lines = path.read_text().splitlines()
    return frozenset(l.strip() for l in lines if l.strip() and not l.startswith("#"))


class Resources:
    def __init__(self, data_dir: str | Path | None = None):
        self.dir = Path(data_dir) if data_dir else DATA

    def _path(self, name: str):
        p = self.dir / name
        return p if p.is_file() else DATA / name

    @cached_property
    def animate_nouns(self) -> frozenset[str]:
        return _words(self._path("animate_nouns.txt"))

    @cached_property
    def animate_verbs(self) -> frozenset[str]:
        return _words(self._path("animate_verbs.txt"))

    @cached_property
    def denominal(self) -> frozenset[str]:
        p = self._path("denominal_verbs.txt")
        return _words(p) if p.is_file() else frozenset()

    @cached_property
    def bigrams(self) -> BigramSet:
        p = self._path("bigrams.u32.gz")
        return BigramSet.load(p) if p.is_file() else BigramSet(np.zeros(0, dtype=np.uint32))

    @cached_property
    def baselines(self) -> dict:
        p = self._path("baselines.json")
        return json.loads(p.read_text()) if p.is_file() else {"dep_weights": [0.2] * 5, "corpora": {}}
