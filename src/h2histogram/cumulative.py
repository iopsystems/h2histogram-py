"""Read-only cumulative histogram for fast quantile queries.

This is the Python analogue of the Rust crate's ``CumulativeROHistogram``. It is
a variant of :class:`~h2histogram.sparse.SparseHistogram` that stores only
non-zero buckets in columnar form, but with **cumulative** counts: ``count[i]``
is the running prefix sum of individual bucket counts, so the last element
equals the total observation count.

Because the counts are cumulative, percentile queries are answered with a binary
search (``O(log n)`` in the number of non-zero buckets) rather than a linear
scan. The histogram is read-only -- it does not accept new observations, since
updating cumulative counts would be expensive. A midpoint-estimated ``mean`` is
computed once at construction.
"""

from __future__ import annotations

import bisect
from typing import Iterator, List, Optional, Sequence, Tuple

from .bucket import Bucket
from .config import Config
from . import _analytics as analytics

__all__ = ["CumulativeHistogram"]


class CumulativeHistogram:
    """A read-only histogram with cumulative counts for fast quantile queries.

    Corresponds to ``CumulativeROHistogram`` in the Rust ``histogram`` crate.
    Build one with :meth:`from_histogram`, :meth:`from_sparse`, or
    :meth:`from_parts`.
    """

    __slots__ = ("_config", "_index", "_count", "_mean")

    def __init__(
        self,
        config: Config,
        index: Sequence[int],
        count: Sequence[int],
        *,
        _validate: bool = True,
    ) -> None:
        self._config = config
        self._index: List[int] = [analytics.integer(i, "indices") for i in index]
        self._count: List[int] = [analytics.integer(n, "counts") for n in count]
        if _validate:
            self._validate()
        self._mean = self._compute_mean()

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_parts(
        cls,
        config: Config,
        index: Sequence[int],
        count: Sequence[int],
    ) -> "CumulativeHistogram":
        """Create from raw parts. ``count`` must be **cumulative** (prefix sums).

        Raises :class:`ValueError` if the lengths differ, an index is out of
        range, the indices are not strictly ascending, the counts are not
        non-decreasing, or any count is zero.
        """
        return cls(config, index, count)

    @classmethod
    def from_histogram(cls, histogram) -> "CumulativeHistogram":
        """Build from a dense :class:`~h2histogram.histogram.Histogram`."""
        histogram._validate_storage()
        index: List[int] = []
        count: List[int] = []
        running = 0
        for i, n in enumerate(histogram.buckets):
            if n:
                running += int(n)
                index.append(i)
                count.append(running)
        return cls(histogram.config, index, count)

    @classmethod
    def from_sparse(cls, sparse) -> "CumulativeHistogram":
        """Build from a :class:`~h2histogram.sparse.SparseHistogram`."""
        index = list(sparse.index)
        cumulative: List[int] = []
        running = 0
        for n in sparse.count:
            running += int(n)
            cumulative.append(running)
        return cls(sparse.config, index, cumulative)

    # ------------------------------------------------------------------
    # Validation / mean
    # ------------------------------------------------------------------
    def _validate(self) -> None:
        analytics.validate_parts(self._config, self._index, self._count, cumulative=True)

    def _individual_count(self, position: int) -> int:
        if position == 0:
            return self._count[0]
        return self._count[position] - self._count[position - 1]

    def _compute_mean(self) -> Optional[float]:
        if not self._count:
            return None
        total = self._count[-1]
        if total == 0:
            return None
        weighted = 0.0
        for i in range(len(self._index)):
            individual = self._individual_count(i)
            start, end = self._config.index_to_range(self._index[i])
            midpoint = (start + end) / 2.0
            weighted += midpoint * (individual / total)
        return weighted

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
        """A copy of cumulative counts; modifying it cannot stale the cached mean."""
        return self._count.copy()

    def __len__(self) -> int:
        return len(self._index)

    def is_empty(self) -> bool:
        return not self._index

    def total_count(self) -> int:
        return self._count[-1] if self._count else 0

    def mean(self) -> Optional[float]:
        """Midpoint-estimated mean of all observations, or ``None`` if empty.

        Computed once at construction; this is a cheap attribute read.
        """
        return self._mean

    # ------------------------------------------------------------------
    # Quantile queries (binary search)
    # ------------------------------------------------------------------
    def _find_quantile_position(self, target: int) -> int:
        # First position where cumulative count >= target.
        pos = bisect.bisect_left(self._count, target)
        return min(pos, len(self._count) - 1)

    def percentile(self, percentile: float) -> Optional[Bucket]:
        """Return the :class:`Bucket` at ``percentile`` in ``[0.0, 1.0]``.

        The returned bucket carries the **individual** (non-cumulative) count.
        Returns ``None`` if the histogram is empty.
        """
        analytics.validate_percentile(percentile)
        if not self._count:
            return None
        pos = self._find_quantile_position(analytics.rank(percentile, self._count[-1]))
        return analytics.bucket(self._config, self._index[pos], self._individual_count(pos))

    def percentiles_into(self, percentiles, output) -> bool:
        """Binary-search each request into caller Bucket/None slots.

        No sorting or dense reconstruction. Selected Bucket objects still allocate.
        Invalid requests or short output leave the list unchanged.
        """
        if percentiles is output:
            raise ValueError("requests and output cannot alias")
        for p in percentiles:
            analytics.validate_percentile(p)
        if len(output) < len(percentiles):
            raise ValueError("output must have at least one slot per percentile")
        for i, p in enumerate(percentiles):
            output[i] = self.percentile(p)
        return bool(self._count)

    def percentiles(self, percentiles):
        """Return (percentile, Bucket) pairs in request order, or None if empty."""
        output = [None] * len(percentiles)
        if not self.percentiles_into(percentiles, output):
            return None
        return list(zip(percentiles, output))

    def quantile(self, quantile: float) -> Optional[Bucket]:
        """Alias for :meth:`percentile`."""
        return self.percentile(quantile)

    def bucket_quantile_range(self, bucket_idx: int) -> Optional[Tuple[float, float]]:
        """Return ``(lower, upper)`` quantile fractions for the ``bucket_idx``-th
        stored bucket.

        ``lower`` is the fraction of observations strictly before this bucket and
        ``upper`` the fraction at or before it, both in ``[0.0, 1.0]``. Returns
        ``None`` if empty or out of range.
        """
        if bucket_idx < 0 or bucket_idx >= len(self._count):
            return None
        total = self._count[-1]
        if total == 0:
            return None
        lower = 0.0 if bucket_idx == 0 else self._count[bucket_idx - 1] / total
        upper = self._count[bucket_idx] / total
        return (lower, upper)

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------
    def __iter__(self) -> Iterator[Bucket]:
        """Iterate non-zero buckets with their **individual** counts."""
        cfg = self._config
        for i in range(len(self._index)):
            start, end = cfg.index_to_range(self._index[i])
            yield Bucket(count=self._individual_count(i), start=start, end=end)

    def iter_with_quantiles(self) -> Iterator[Tuple[Bucket, float, float]]:
        """Iterate ``(Bucket, lower_quantile, upper_quantile)`` per non-zero bucket."""
        cfg = self._config
        total = self._count[-1] if self._count else 0
        for i in range(len(self._index)):
            lower = 0.0 if i == 0 else self._count[i - 1] / total
            upper = self._count[i] / total
            start, end = cfg.index_to_range(self._index[i])
            yield (
                Bucket(count=self._individual_count(i), start=start, end=end),
                lower,
                upper,
            )

    # ------------------------------------------------------------------
    # Conversions
    # ------------------------------------------------------------------
    def to_dense(self):
        """Reconstruct a dense :class:`~h2histogram.histogram.Histogram`."""
        from .histogram import Histogram

        h = Histogram(config=self._config)
        for i in range(len(self._index)):
            h.buckets[self._index[i]] = self._individual_count(i)
        return h

    def _pairs(self):
        previous = 0
        for index, count in zip(self._index, self._count):
            yield index, count - previous
            previous = count

    @classmethod
    def _from_pairs(cls, config, pairs):
        indices, cumulative = [], []
        running = 0
        for index, count in pairs:
            if count:
                running += count
                indices.append(index)
                cumulative.append(running)
        return cls(config, indices, cumulative)

    def to_sparse(self):
        """Copy individual counts to sparse storage without dense reconstruction."""
        from .sparse import SparseHistogram
        return SparseHistogram._from_pairs(self._config, self._pairs())

    def merge(self, other: "CumulativeHistogram") -> "CumulativeHistogram":
        """Merge sorted individual counts, then compute new prefixes and mean."""
        if self._config != other._config:
            raise ValueError("histograms have incompatible configurations")
        return self._from_pairs(self._config,
                                analytics.merge_pairs(self._pairs(), other._pairs()))

    def downsample(self, grouping_power: int) -> "CumulativeHistogram":
        """Coalesce onto a coarser grid, recomputing the midpoint-estimated mean."""
        config, pairs = analytics.downsample_pairs(self._config, self._pairs(), grouping_power)
        return self._from_pairs(config, pairs)

    def shrink_to_fit(self) -> None:
        """Rebuild snapshot lists to release optional spare interpreter capacity.

        Contents and mean stay unchanged. This does not guarantee RSS reduction.
        """
        self._index = self._index.copy()
        self._count = self._count.copy()

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CumulativeHistogram):
            return NotImplemented
        return (
            self._config == other._config
            and self._index == other._index
            and self._count == other._count
        )

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"CumulativeHistogram(grouping_power={self._config.grouping_power}, "
            f"max_value_power={self._config.max_value_power}, "
            f"nonzero_buckets={len(self._index)}, total_count={self.total_count()})"
        )
