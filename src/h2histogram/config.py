"""Bucketing configuration for h2 histograms.

This is a faithful port of the ``Config`` type from the Rust
`histogram <https://github.com/iopsystems/histogram>`_ crate. The bucketing
strategy is fully determined by two parameters:

* ``grouping_power`` -- the number of buckets used to span consecutive powers
  of two. It controls the relative error: ``2**-grouping_power``. For example,
  ``grouping_power=7`` gives a relative error of ~0.78%.
* ``max_value_power`` -- the largest representable value is
  ``2**max_value_power - 1``.

The layout has two regions:

* A *linear* region covering ``0 .. 2**(grouping_power + 1)`` where every bucket
  has width 1 (exact representation, no error).
* A *logarithmic* region above the cutoff, subdivided into
  ``2**grouping_power`` buckets per power of two.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

__all__ = ["Config"]

# Maximum inclusive value representable by a u64, i.e. 2**64 - 1.
_U64_MAX = (1 << 64) - 1


@dataclass(frozen=True)
class Config:
    """Immutable bucketing configuration.

    Construct with :meth:`Config.new` (which validates the parameters and mirrors
    the Rust constructor) rather than instantiating the dataclass directly.
    """

    grouping_power: int
    max_value_power: int

    # Derived quantities (computed in ``__post_init__``).
    max: int
    cutoff_power: int
    cutoff_value: int
    lower_bin_count: int
    upper_bin_divisions: int
    upper_bin_count: int

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def new(cls, grouping_power: int, max_value_power: int) -> "Config":
        """Create and validate a :class:`Config`.

        Raises :class:`ValueError` if the parameters are out of range, matching
        the constraints enforced by the Rust crate:

        * ``0 <= max_value_power <= 64``
        * ``grouping_power < max_value_power``
        """
        if not isinstance(grouping_power, int) or not isinstance(max_value_power, int):
            raise TypeError("grouping_power and max_value_power must be integers")
        if max_value_power > 64:
            raise ValueError("max_value_power must be <= 64")
        if max_value_power < 0 or grouping_power < 0:
            raise ValueError("grouping_power and max_value_power must be non-negative")
        if grouping_power >= max_value_power:
            raise ValueError("grouping_power must be less than max_value_power")

        # The cutoff is where the linear divisions and the logarithmic
        # subdivisions have the same width: cutoff_power = grouping_power + 1.
        cutoff_power = grouping_power + 1
        cutoff_value = 1 << cutoff_power
        upper_bin_divisions = 1 << grouping_power

        max_value = _U64_MAX if max_value_power == 64 else (1 << max_value_power) - 1

        lower_bin_count = cutoff_value
        upper_bin_count = (max_value_power - cutoff_power) * upper_bin_divisions

        return cls(
            grouping_power=grouping_power,
            max_value_power=max_value_power,
            max=max_value,
            cutoff_power=cutoff_power,
            cutoff_value=cutoff_value,
            lower_bin_count=lower_bin_count,
            upper_bin_divisions=upper_bin_divisions,
            upper_bin_count=upper_bin_count,
        )

    @classmethod
    def from_total_buckets(cls, total_buckets: int, max_value_power: int = 64) -> "Config":
        """Infer a config from a known bucket count and ``max_value_power``.

        Rezolus/metriken parquet files store dense histograms as a bare list of
        bucket counts without recording ``grouping_power``. Given the number of
        buckets and the (conventionally fixed) ``max_value_power`` the grouping
        power can be recovered uniquely.

        Raises :class:`ValueError` if no grouping power produces ``total_buckets``.
        """
        for grouping_power in range(0, max_value_power):
            candidate = cls.new(grouping_power, max_value_power)
            if candidate.total_buckets == total_buckets:
                return candidate
        raise ValueError(
            f"no grouping_power with max_value_power={max_value_power} yields "
            f"{total_buckets} buckets"
        )

    # ------------------------------------------------------------------
    # Sizing / error
    # ------------------------------------------------------------------
    @property
    def total_buckets(self) -> int:
        """Total number of buckets for this configuration."""
        return self.lower_bin_count + self.upper_bin_count

    @property
    def error(self) -> float:
        """Relative error (as a percentage) of the logarithmic buckets.

        Linear buckets have width 1 and no error. If the config has no
        logarithmic buckets the error is zero.
        """
        if self.grouping_power == self.max_value_power - 1:
            return 0.0
        return 100.0 / (1 << self.grouping_power)

    # ------------------------------------------------------------------
    # Value <-> index
    # ------------------------------------------------------------------
    def value_to_index(self, value: int) -> int:
        """Return the bucket index that ``value`` falls into.

        Raises :class:`ValueError` if the value is negative or greater than the
        configured maximum.
        """
        if value < 0:
            raise ValueError("value must be non-negative")

        if value < self.cutoff_value:
            return value

        if value > self.max:
            raise ValueError(
                f"value {value} is out of range for max {self.max}"
            )

        # power = floor(log2(value)); equivalent to 63 - leading_zeros for u64.
        power = value.bit_length() - 1
        log_bin = power - self.cutoff_power
        offset = (value - (1 << power)) >> (power - self.grouping_power)

        return self.lower_bin_count + log_bin * self.upper_bin_divisions + offset

    def index_to_lower_bound(self, index: int) -> int:
        """Return the inclusive lower bound of the bucket at ``index``."""
        g = index >> self.grouping_power
        h = index - g * (1 << self.grouping_power)
        if g < 1:
            return h
        return (1 << (self.grouping_power + g - 1)) + (1 << (g - 1)) * h

    def index_to_upper_bound(self, index: int) -> int:
        """Return the inclusive upper bound of the bucket at ``index``."""
        if index == self.lower_bin_count + self.upper_bin_count - 1:
            return self.max
        g = index >> self.grouping_power
        h = index - g * (1 << self.grouping_power) + 1
        if g < 1:
            return h - 1
        return (1 << (self.grouping_power + g - 1)) + (1 << (g - 1)) * h - 1

    def index_to_range(self, index: int) -> Tuple[int, int]:
        """Return ``(lower, upper)`` inclusive bounds for the bucket at ``index``."""
        return (self.index_to_lower_bound(index), self.index_to_upper_bound(index))

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"Config(grouping_power={self.grouping_power}, "
            f"max_value_power={self.max_value_power}, "
            f"total_buckets={self.total_buckets})"
        )
