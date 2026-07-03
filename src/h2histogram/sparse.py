"""Sparse, columnar representation of an h2 histogram.

Only non-zero buckets are stored, as two parallel arrays ``index`` and
``count`` in ascending index order. This is the form Rezolus uses for its
``:bucket_indices`` / ``:bucket_counts`` parquet columns.
"""

from __future__ import annotations

from typing import Iterator, List, Optional, Sequence, Tuple

from .bucket import Bucket
from .config import Config

__all__ = ["SparseHistogram"]


class SparseHistogram:
    """A histogram stored as ``(index, count)`` pairs for non-zero buckets."""

    __slots__ = ("_config", "_index", "_count")

    def __init__(
        self,
        config: Config,
        index: Optional[Sequence[int]] = None,
        count: Optional[Sequence[int]] = None,
    ) -> None:
        self._config = config
        self._index: List[int] = list(index) if index is not None else []
        self._count: List[int] = list(count) if count is not None else []

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_histogram(cls, histogram) -> "SparseHistogram":
        """Build a sparse histogram from a dense :class:`Histogram`."""
        index: List[int] = []
        count: List[int] = []
        for i, c in enumerate(histogram.buckets):
            if c:
                index.append(i)
                count.append(c)
        return cls(histogram.config, index, count)

    @classmethod
    def from_parts(
        cls,
        config: Config,
        index: Sequence[int],
        count: Sequence[int],
    ) -> "SparseHistogram":
        """Create a sparse histogram from raw parts, validating invariants.

        Raises :class:`ValueError` if the lengths differ, an index is out of
        range, or the indices are not strictly ascending.
        """
        if len(index) != len(count):
            raise ValueError("index and count must have the same length")
        total = config.total_buckets
        prev = -1
        for i in index:
            if i < 0 or i >= total:
                raise ValueError(f"index {i} out of range for config")
            if i <= prev:
                raise ValueError("indices must be strictly ascending")
            prev = i
        return cls(config, index, count)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    @property
    def config(self) -> Config:
        return self._config

    @property
    def index(self) -> List[int]:
        """Non-zero bucket indices, ascending."""
        return self._index

    @property
    def count(self) -> List[int]:
        """Counts corresponding to :attr:`index`."""
        return self._count

    def __len__(self) -> int:
        return len(self._index)

    def is_empty(self) -> bool:
        return not self._index

    def total_count(self) -> int:
        return sum(self._count)

    # ------------------------------------------------------------------
    # Iteration / conversion
    # ------------------------------------------------------------------
    def __iter__(self) -> Iterator[Bucket]:
        cfg = self._config
        for i, c in zip(self._index, self._count):
            start, end = cfg.index_to_range(i)
            yield Bucket(count=c, start=start, end=end)

    def to_dense(self):
        """Convert to a dense :class:`Histogram`."""
        from .histogram import Histogram

        h = Histogram(config=self._config)
        for i, c in zip(self._index, self._count):
            h.buckets[i] = c
        return h

    def to_cumulative(self):
        """Convert to a read-only :class:`CumulativeHistogram`."""
        from .cumulative import CumulativeHistogram

        return CumulativeHistogram.from_sparse(self)

    # ------------------------------------------------------------------
    # Percentiles
    # ------------------------------------------------------------------
    def percentile(self, percentile: float) -> Optional[Bucket]:
        """Compute a percentile via the dense representation."""
        return self.to_dense().percentile(percentile)

    def percentiles(
        self, percentiles: Sequence[float]
    ) -> Optional[List[Tuple[float, Bucket]]]:
        return self.to_dense().percentiles(percentiles)

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SparseHistogram):
            return NotImplemented
        return (
            self._config == other._config
            and self._index == other._index
            and self._count == other._count
        )

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"SparseHistogram(grouping_power={self._config.grouping_power}, "
            f"max_value_power={self._config.max_value_power}, "
            f"nonzero_buckets={len(self._index)})"
        )
