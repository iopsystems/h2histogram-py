"""Dense h2 histogram with per-bucket counters."""

from __future__ import annotations

from typing import Iterable, Iterator, List, Optional, Sequence, Tuple, Union

from .bucket import Bucket
from .config import Config

__all__ = ["Histogram"]


class Histogram:
    """A histogram that stores a counter for every bucket.

    Values are quantized into buckets according to a :class:`Config` determined
    by ``grouping_power`` and ``max_value_power``. This is the Python analogue of
    the Rust ``Histogram`` type and produces byte-for-byte identical bucketing,
    so histograms recorded here can be consumed by Rezolus (and vice versa).

    Example::

        h = Histogram(grouping_power=7, max_value_power=64)
        h.increment(42)
        h.record(1000, count=3)
        print(h.percentile(0.5))
    """

    __slots__ = ("_config", "_buckets")

    def __init__(
        self,
        grouping_power: int = 7,
        max_value_power: int = 64,
        *,
        config: Optional[Config] = None,
    ) -> None:
        if config is None:
            config = Config.new(grouping_power, max_value_power)
        self._config = config
        self._buckets: List[int] = [0] * config.total_buckets

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def with_config(cls, config: Config) -> "Histogram":
        """Create an empty histogram from an existing :class:`Config`."""
        return cls(config=config)

    @classmethod
    def from_buckets(
        cls,
        grouping_power: int,
        max_value_power: int,
        buckets: Sequence[int],
    ) -> "Histogram":
        """Create a histogram from a full, dense list of bucket counts.

        The length of ``buckets`` must equal the config's ``total_buckets``.
        """
        config = Config.new(grouping_power, max_value_power)
        if len(buckets) != config.total_buckets:
            raise ValueError(
                f"expected {config.total_buckets} buckets, got {len(buckets)}"
            )
        h = cls(config=config)
        h._buckets = [int(b) for b in buckets]
        return h

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    @property
    def config(self) -> Config:
        """The bucketing configuration."""
        return self._config

    @property
    def buckets(self) -> List[int]:
        """The raw, dense list of bucket counts (one entry per bucket)."""
        return self._buckets

    def __len__(self) -> int:
        return len(self._buckets)

    def total_count(self) -> int:
        """The total number of observations recorded."""
        return sum(self._buckets)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    def increment(self, value: int) -> None:
        """Add one observation of ``value``."""
        self.record(value, 1)

    def record(self, value: int, count: int = 1) -> None:
        """Add ``count`` observations of ``value``."""
        index = self._config.value_to_index(value)
        self._buckets[index] += count

    def record_many(self, values: Iterable[int], counts: Optional[Iterable[int]] = None) -> None:
        """Record many values at once.

        If ``counts`` is provided it must be the same length as ``values`` and
        supplies a weight for each value; otherwise each value counts once.

        Uses NumPy for a vectorized fast path when it is installed and ``counts``
        is omitted; otherwise falls back to a simple loop.
        """
        if counts is not None:
            for value, count in zip(values, counts):
                self.record(value, count)
            return

        try:
            import numpy as np  # noqa: WPS433 (optional acceleration)
        except ImportError:
            for value in values:
                self.increment(value)
            return

        self._record_many_numpy(np, values)

    def _record_many_numpy(self, np, values: Iterable[int]) -> None:
        arr = np.asarray(list(values), dtype=np.uint64)
        if arr.size == 0:
            return
        cfg = self._config
        if int(arr.max()) > cfg.max:
            raise ValueError("a value is out of range for this histogram")

        indices = np.empty(arr.shape, dtype=np.int64)
        linear = arr < cfg.cutoff_value
        indices[linear] = arr[linear].astype(np.int64)

        log = ~linear
        if np.any(log):
            log_vals = arr[log]
            # power = floor(log2(value)) for 64-bit values. float64's log2 is
            # only good to ~1 ULP near large powers of two, so we clamp to the
            # valid u64 exponent range [0, 63] and correct by at most one step
            # in each direction. Clamping first also avoids computing `1 << 64`
            # (which wraps to 0 in uint64 arithmetic).
            power = np.floor(np.log2(log_vals.astype(np.float64))).astype(np.int64)
            power = np.clip(power, 0, 63)

            # Down-correct where 2**power overshot the value.
            too_high = (np.uint64(1) << power.astype(np.uint64)) > log_vals
            power = np.clip(power - too_high.astype(np.int64), 0, 63)

            # Up-correct where value reaches into the next power of two. Guard
            # the shift so we never form `1 << 64`.
            can_up = power < 63
            p_u = power.astype(np.uint64)
            next_pow = np.where(
                can_up, np.uint64(1) << (p_u + np.uint64(1)), np.uint64(0)
            )
            too_low = can_up & (log_vals >= next_pow)
            power = power + too_low.astype(np.int64)

            p_u = power.astype(np.uint64)
            log_bin = power - cfg.cutoff_power
            offset = (log_vals - (np.uint64(1) << p_u)) >> (
                power - cfg.grouping_power
            ).astype(np.uint64)
            indices[log] = (
                cfg.lower_bin_count
                + log_bin * cfg.upper_bin_divisions
                + offset.astype(np.int64)
            )

        counts = np.bincount(indices, minlength=cfg.total_buckets)
        existing = np.asarray(self._buckets, dtype=np.int64)
        self._buckets = (existing + counts).tolist()

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------
    def __iter__(self) -> Iterator[Bucket]:
        cfg = self._config
        for index, count in enumerate(self._buckets):
            start, end = cfg.index_to_range(index)
            yield Bucket(count=count, start=start, end=end)

    def nonzero_buckets(self) -> Iterator[Bucket]:
        """Iterate only over buckets with a non-zero count."""
        cfg = self._config
        for index, count in enumerate(self._buckets):
            if count:
                start, end = cfg.index_to_range(index)
                yield Bucket(count=count, start=start, end=end)

    # ------------------------------------------------------------------
    # Combination
    # ------------------------------------------------------------------
    def _check_compatible(self, other: "Histogram") -> None:
        if self._config != other._config:
            raise ValueError("histograms have incompatible configurations")

    def merge(self, other: "Histogram") -> "Histogram":
        """Return a new histogram that is the element-wise sum of both.

        Both histograms must share the same configuration.
        """
        self._check_compatible(other)
        result = Histogram(config=self._config)
        result._buckets = [a + b for a, b in zip(self._buckets, other._buckets)]
        return result

    def __add__(self, other: "Histogram") -> "Histogram":
        return self.merge(other)

    def subtract(self, other: "Histogram") -> "Histogram":
        """Return a new histogram that is the element-wise difference.

        Raises :class:`ValueError` if any bucket would go negative.
        """
        self._check_compatible(other)
        buckets = []
        for a, b in zip(self._buckets, other._buckets):
            diff = a - b
            if diff < 0:
                raise ValueError("subtraction would produce a negative bucket count")
            buckets.append(diff)
        result = Histogram(config=self._config)
        result._buckets = buckets
        return result

    def __sub__(self, other: "Histogram") -> "Histogram":
        return self.subtract(other)

    def downsample(self, grouping_power: int) -> "Histogram":
        """Return a coarser histogram with a smaller ``grouping_power``.

        Every step down approximately halves the number of buckets while
        doubling the relative error. The new grouping power must be strictly
        less than the current one.
        """
        if grouping_power >= self._config.grouping_power:
            raise ValueError(
                "target grouping_power must be less than the current grouping_power"
            )
        result = Histogram(grouping_power, self._config.max_value_power)
        for index, count in enumerate(self._buckets):
            if count:
                value = self._config.index_to_lower_bound(index)
                result.record(value, count)
        return result

    # ------------------------------------------------------------------
    # Quantiles / percentiles
    # ------------------------------------------------------------------
    def percentile(self, percentile: float) -> Optional[Bucket]:
        """Return the bucket at a single ``percentile`` in ``[0.0, 1.0]``.

        Returns ``None`` if the histogram is empty. ``percentile`` uses the same
        fractional convention as the Rust crate: ``0.5`` is the median.
        """
        result = self.percentiles([percentile])
        if result is None:
            return None
        return result[0][1]

    def percentiles(
        self, percentiles: Sequence[float]
    ) -> Optional[List[Tuple[float, Bucket]]]:
        """Return ``(percentile, Bucket)`` pairs for each requested percentile.

        Each percentile must be in ``[0.0, 1.0]``. Returns ``None`` if the
        histogram is empty. This mirrors the algorithm used by the Rust crate.
        """
        for p in percentiles:
            if not 0.0 <= p <= 1.0:
                raise ValueError("percentiles must be in the range [0.0, 1.0]")

        total = self.total_count()
        if total == 0:
            return None

        # Deduplicate and sort while remembering the original order for output.
        sorted_unique = sorted(set(percentiles))

        cfg = self._config
        results: dict = {}
        bucket_idx = 0
        partial_sum = self._buckets[0]

        for p in sorted_unique:
            target = max(1, _ceil(p * total))
            while True:
                if partial_sum >= target:
                    start, end = cfg.index_to_range(bucket_idx)
                    results[p] = Bucket(
                        count=self._buckets[bucket_idx], start=start, end=end
                    )
                    break
                if bucket_idx == len(self._buckets) - 1:
                    start, end = cfg.index_to_range(bucket_idx)
                    results[p] = Bucket(
                        count=self._buckets[bucket_idx], start=start, end=end
                    )
                    break
                bucket_idx += 1
                partial_sum += self._buckets[bucket_idx]

        return [(p, results[p]) for p in percentiles]

    def quantile(self, quantile: float) -> Optional[Bucket]:
        """Alias for :meth:`percentile` (the crate uses ``quantile``)."""
        return self.percentile(quantile)

    # ------------------------------------------------------------------
    # Conversions
    # ------------------------------------------------------------------
    def to_sparse(self) -> "SparseHistogram":
        """Convert to the sparse (columnar) representation."""
        from .sparse import SparseHistogram

        return SparseHistogram.from_histogram(self)

    def to_cumulative(self) -> "CumulativeHistogram":
        """Convert to a read-only cumulative histogram for fast quantiles."""
        from .cumulative import CumulativeHistogram

        return CumulativeHistogram.from_histogram(self)

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Histogram):
            return NotImplemented
        return self._config == other._config and self._buckets == other._buckets

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"Histogram(grouping_power={self._config.grouping_power}, "
            f"max_value_power={self._config.max_value_power}, "
            f"total_count={self.total_count()})"
        )


def _ceil(x: float) -> int:
    """Ceil that matches Rust's ``f64::ceil() as u128`` for our inputs."""
    import math

    return int(math.ceil(x))
