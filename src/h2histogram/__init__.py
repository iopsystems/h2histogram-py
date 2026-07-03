"""h2histogram -- a Python implementation of the iopsystems h2 histogram.

This library produces histograms with byte-for-byte identical bucketing to the
Rust `histogram <https://github.com/iopsystems/histogram>`_ crate, so they can
be consumed by (and interoperate with) Rezolus.

Basic use::

    from h2histogram import Histogram

    h = Histogram(grouping_power=7, max_value_power=64)
    h.increment(42)
    h.record(1000, count=5)
    bucket = h.percentile(0.99)

Reading histograms from a Rezolus parquet file (requires the ``parquet`` extra)::

    from h2histogram.arrow import read_histograms
    series = read_histograms("rezolus.parquet", "cpu/usage")
"""

from .bucket import Bucket
from .config import Config
from .cumulative import CumulativeHistogram
from .histogram import Histogram
from .sparse import SparseHistogram

__version__ = "0.1.0"

__all__ = [
    "Bucket",
    "Config",
    "CumulativeHistogram",
    "Histogram",
    "SparseHistogram",
    "__version__",
]
