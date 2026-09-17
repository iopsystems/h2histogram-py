"""Sparse, columnar representation of an h2 histogram.

Only non-zero buckets are stored, as two parallel arrays ``index`` and
``count`` in ascending index order. This is the form Rezolus uses for its
``:bucket_indices`` / ``:bucket_counts`` parquet columns.
"""

from __future__ import annotations

from typing import Iterator, List, Optional, Sequence, Tuple

from .bucket import Bucket
from .config import Config
from . import _analytics as analytics

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
        self._index: List[int] = [analytics.integer(i, "indices") for i in index] if index is not None else []
        self._count: List[int] = [analytics.integer(n, "counts") for n in count] if count is not None else []
        analytics.validate_parts(config, self._index, self._count)
        occupied = [(i, n) for i, n in zip(self._index, self._count) if n]
        self._index = [i for i, _ in occupied]
        self._count = [n for _, n in occupied]

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_histogram(cls, histogram) -> "SparseHistogram":
        """Build a sparse histogram from a dense :class:`Histogram`."""
        histogram._validate_storage()
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
        return cls(config, index, count)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    @property
    def config(self) -> Config:
        return self._config

    @property
    def index(self) -> List[int]:
        """A copy of the stored bucket indices, ascending."""
        return self._index.copy()

    @property
    def count(self) -> List[int]:
        """A copy of counts corresponding to :attr:`index`."""
        return self._count.copy()

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
        """Scan stored counts directly, without reconstructing a dense histogram."""
        return analytics.scalar(self._config, zip(self._index, self._count),
                                self.total_count(), percentile)

    def percentiles_into(self, percentiles, output) -> bool:
        """Fill caller Bucket/None slots in request order; return nonempty.

        The output list is reused; Bucket objects and request-order scratch may
        allocate. Invalid requests or short output leave the list unchanged.
        """
        return analytics.batch_into(self._config, zip(self._index, self._count),
                                    self.total_count(), percentiles, output)

    def percentiles(self, percentiles):
        output = [None] * len(percentiles)
        if not self.percentiles_into(percentiles, output):
            return None
        return list(zip(percentiles, output))

    @classmethod
    def _from_pairs(cls, config, pairs):
        indices, counts = [], []
        for index, count in pairs:
            if count:
                indices.append(index)
                counts.append(count)
        return cls(config, indices, counts)

    def merge(self, other: "SparseHistogram") -> "SparseHistogram":
        """Merge sorted counts into independent sparse storage in O(n + m)."""
        if self._config != other._config:
            raise ValueError("histograms have incompatible configurations")
        return self._from_pairs(self._config, analytics.merge_pairs(
            zip(self._index, self._count), zip(other._index, other._count)))

    def downsample(self, grouping_power: int) -> "SparseHistogram":
        """Coalesce stored buckets on a coarser grid without dense storage."""
        config, pairs = analytics.downsample_pairs(
            self._config, zip(self._index, self._count), grouping_power)
        return self._from_pairs(config, pairs)

    def shrink_to_fit(self) -> None:
        """Rebuild owned lists to release optional interpreter spare capacity.

        This is an explicit retention operation, not an RSS or allocator-release
        guarantee. Old and new list storage coexist during each copy.
        """
        self._index = self._index.copy()
        self._count = self._count.copy()

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
