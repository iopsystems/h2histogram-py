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
import math
from typing import Iterator, List, Optional, Sequence, Tuple

from .bucket import Bucket
from .config import Config

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
        self._index: List[int] = list(index)
        self._count: List[int] = list(count)
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
        index: List[int] = []
        count: List[int] = []
        running = 0
        for i, n in enumerate(histogram.buckets):
            if n:
                running += n
                index.append(i)
                count.append(running)
        return cls(histogram.config, index, count, _validate=False)

    @classmethod
    def from_sparse(cls, sparse) -> "CumulativeHistogram":
        """Build from a :class:`~h2histogram.sparse.SparseHistogram`."""
        index = list(sparse.index)
        cumulative: List[int] = []
        running = 0
        for n in sparse.count:
            running += n
            cumulative.append(running)
        return cls(sparse.config, index, cumulative, _validate=False)

    # ------------------------------------------------------------------
    # Validation / mean
    # ------------------------------------------------------------------
    def _validate(self) -> None:
        if len(self._index) != len(self._count):
            raise ValueError("index and count must have the same length")
        total_buckets = self._config.total_buckets
        prev = -1
        for i in self._index:
            if i < 0 or i >= total_buckets:
                raise ValueError(f"index {i} out of range for config")
            if i <= prev:
                raise ValueError("indices must be strictly ascending")
            prev = i
        prev_c: Optional[int] = None
        for c in self._count:
            if c == 0:
                raise ValueError("cumulative counts must be non-zero")
            if prev_c is not None and c < prev_c:
                raise ValueError("cumulative counts must be non-decreasing")
            prev_c = c

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
            weighted += midpoint * individual
        return weighted / total

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
        """Cumulative (prefix-sum) counts aligned with :attr:`index`."""
        return self._count

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
        result = self.percentiles([percentile])
        if result is None:
            return None
        return result[0][1]

    def percentiles(
        self, percentiles: Sequence[float]
    ) -> Optional[List[Tuple[float, Bucket]]]:
        """Return ``(percentile, Bucket)`` pairs, one per requested percentile.

        Each percentile must be in ``[0.0, 1.0]``. Returns ``None`` if empty.
        """
        for p in percentiles:
            if not 0.0 <= p <= 1.0:
                raise ValueError("percentiles must be in the range [0.0, 1.0]")
        if not self._count:
            return None
        total = self._count[-1]
        if total == 0:
            return None

        out: List[Tuple[float, Bucket]] = []
        for p in percentiles:
            target = max(1, int(math.ceil(p * total)))
            pos = self._find_quantile_position(target)
            start, end = self._config.index_to_range(self._index[pos])
            out.append(
                (p, Bucket(count=self._individual_count(pos), start=start, end=end))
            )
        return out

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
