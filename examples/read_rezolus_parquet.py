"""Open a parquet/arrow column of h2histogram values, as produced by Rezolus.

Rezolus records metric snapshots to a parquet file (``rezolus record ...``)
with one row per sample interval. Histogram metrics are stored as either a
dense ``"{metric}:buckets"`` column or a sparse
``"{metric}:bucket_indices"`` / ``"{metric}:bucket_counts"`` pair, all of type
``List<UInt64>``. Rezolus uses ``grouping_power=3`` and ``max_value_power=64``.

Usage::

    # Read a real recording:
    python examples/read_rezolus_parquet.py path/to/rezolus.parquet

    # Or, with no argument, synthesize a Rezolus-style file and read it back
    # (handy for trying the library without a running Rezolus instance):
    python examples/read_rezolus_parquet.py

Requires the parquet extra: ``pip install h2histogram[parquet]``.
"""

import sys

from h2histogram import Histogram
from h2histogram.arrow import (
    histogram_columns,
    read_histograms,
    write_histograms,
)


def synthesize(path: str) -> None:
    """Write a small Rezolus-shaped parquet file to `path`.

    This mimics what `rezolus record` produces: a dense `:buckets` column with
    grouping_power=3, one histogram per row (sample interval).
    """
    import random

    random.seed(7)
    rows = []
    for _ in range(10):  # 10 sample intervals
        h = Histogram(grouping_power=3, max_value_power=64)
        for _ in range(5_000):
            h.increment(int(random.lognormvariate(10, 1.2)))  # ns latencies
        rows.append(h)

    # timestamps in nanoseconds (as Rezolus records them)
    timestamps = [i * 1_000_000_000 for i in range(len(rows))]
    write_histograms(
        path,
        {"syscall/read/latency": rows},
        timestamps=timestamps,
        histogram_type="standard",
    )
    print(f"wrote synthetic Rezolus-style recording to {path}\n")


def main() -> None:
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = "rezolus-example.parquet"
        synthesize(path)

    # 1. Discover which columns are histograms.
    columns = histogram_columns(path)
    print(f"histogram metrics in {path}:")
    for col in columns:
        print(f"  - {col.name}  ({col.kind})")
    print()

    if not columns:
        print("no histogram columns found")
        return

    # 2. Read one metric's time series. Each row is one sample interval.
    #    Rezolus uses grouping_power=3, max_value_power=64; those are the
    #    defaults, and are also inferable from a dense column's bucket count.
    metric = columns[0].name
    series = read_histograms(path, metric, grouping_power=3, max_value_power=64)

    print(f"metric {metric!r}: {len(series)} sample intervals\n")
    print(f"{'interval':>8}  {'count':>8}  {'p50':>12}  {'p99':>12}  {'p99.9':>12}")
    for i, hist in enumerate(series):
        if hist is None:
            print(f"{i:>8}  {'(missing)':>8}")
            continue
        p50 = hist.percentile(0.5).midpoint
        p99 = hist.percentile(0.99).midpoint
        p999 = hist.percentile(0.999).midpoint
        print(
            f"{i:>8}  {hist.total_count():>8}  "
            f"{p50:>12.0f}  {p99:>12.0f}  {p999:>12.0f}"
        )

    # 3. Merge the whole time series into one aggregate histogram.
    merged = series[0]
    for hist in series[1:]:
        if hist is not None:
            merged = merged.merge(hist)
    agg99 = merged.percentile(0.99)
    print(
        f"\naggregate over all intervals: {merged.total_count()} observations, "
        f"p99 ~ {agg99.midpoint:.0f} ns (bucket {agg99.range})"
    )


if __name__ == "__main__":
    main()
