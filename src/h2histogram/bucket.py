"""A single histogram bucket: a count and an inclusive value range."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Bucket"]


@dataclass(frozen=True)
class Bucket:
    """A histogram bucket with its count and inclusive ``[start, end]`` range."""

    count: int
    start: int
    end: int

    @property
    def range(self) -> "tuple[int, int]":
        """The inclusive ``(start, end)`` range of the bucket."""
        return (self.start, self.end)

    @property
    def midpoint(self) -> float:
        """The arithmetic midpoint of the bucket range.

        A reasonable point estimate for values that fell into this bucket.
        """
        return (self.start + self.end) / 2.0

    @property
    def width(self) -> int:
        """The number of distinct integer values the bucket covers."""
        return self.end - self.start + 1

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"Bucket(count={self.count}, range=[{self.start}, {self.end}])"
