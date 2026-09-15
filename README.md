# h2histogram-py

[![PyPI version](https://img.shields.io/pypi/v/h2histogram.svg)](https://pypi.org/project/h2histogram/)
[![Python versions](https://img.shields.io/pypi/pyversions/h2histogram.svg)](https://pypi.org/project/h2histogram/)
[![CI](https://github.com/iopsystems/h2histogram-py/actions/workflows/ci.yml/badge.svg)](https://github.com/iopsystems/h2histogram-py/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A pure-Python implementation of the [iopsystems h2 histogram](https://github.com/iopsystems/histogram).

`h2histogram` produces histograms with **byte-for-byte identical bucketing** to the
Rust `histogram` crate, so histograms recorded here can be consumed by
[Rezolus](https://github.com/iopsystems/rezolus) — and, conversely, you can open a
Parquet/Arrow column of h2histogram values produced by Rezolus and analyze it in
Python.

## What is an h2 histogram?

An h2 histogram quantizes values into buckets using two parameters:

- **`grouping_power`** — the number of buckets spanning each power of two. It sets
  the relative error to `2^-grouping_power` (e.g. `grouping_power=7` → ~0.78% error).
- **`max_value_power`** — the largest representable value is `2^max_value_power - 1`.

Values below `2^(grouping_power+1)` are stored **exactly** (linear buckets of width 1);
larger values fall into logarithmic buckets. This gives HDR-histogram-like guarantees
with a simpler, faster bucket index computation. Rezolus records histograms with
`grouping_power=3` and `max_value_power=64`.

## Install

```bash
pip install h2histogram            # core library (no dependencies)
pip install h2histogram[parquet]   # + pyarrow, for the Arrow/Parquet interop
pip install h2histogram[numpy]     # + numpy, for a vectorized bulk-record fast path
```

For local development from a checkout:

```bash
pip install -e ".[dev]"
pytest
```

## Quick start

```python
from h2histogram import Histogram

h = Histogram(grouping_power=7, max_value_power=64)
h.increment(42)
h.record(1000, count=5)
h.record_many([12, 15, 900, 1_000_000])   # bulk (uses numpy if available)

print(h.total_count())          # 8
p99 = h.percentile(0.99)        # a Bucket
print(p99.range, p99.midpoint)  # ((..lo.., ..hi..), midpoint estimate)

# Combine / reduce
merged = h.merge(other_h)       # element-wise sum (also: h + other_h)
coarse = h.downsample(4)        # fewer buckets, higher error, same total count
sparse = h.to_sparse()          # columnar (index, count) form for storage
```

### Fast repeated quantile queries

For a snapshot you'll query many times, convert to a `CumulativeHistogram`
(the crate's `CumulativeROHistogram`). It stores non-zero buckets with
**cumulative** counts, so percentiles are answered with a binary search, and it
precomputes a midpoint-estimated `mean`:

```python
c = h.to_cumulative()           # read-only; also SparseHistogram.to_cumulative()
c.percentile(0.99)              # O(log n) binary search -> Bucket (individual count)
c.mean()                        # midpoint-estimated mean, computed once
c.bucket_quantile_range(0)      # (lower, upper) quantile fraction of a stored bucket
for bucket, lo, hi in c.iter_with_quantiles():
    ...                         # each non-zero bucket with its quantile span
```

## Choose by phase

| Phase | Priorities | Starting representation |
|---|---|---|
| Update | Cheap recording; bounded bucket geometry | Dense `Histogram` |
| Report in place | Reuse storage; query a few percentiles per interval | Dense, with `percentiles_into` and `snapshot_into` / `drain_into` |
| Report offline | Repeated quantiles and cached midpoint mean | `CumulativeHistogram`; amortize conversion across reads |
| Analytics | Merge/downsample occupied buckets; retain compact snapshots | Sparse or cumulative when occupancy is low; dense when most buckets are occupied |

Recording still only changes bucket counts: no cached total, minimum, maximum, or
mean is maintained on the write path. These classes do not synchronize concurrent
writers or readers. Coordinate access externally, or give each writer its own
histogram and merge snapshots at the interval boundary. The GIL does not make a
multi-step snapshot-and-reset atomic.

### Interval reporting and owned merges

```python
active = Histogram(7, 30)
snapshot = Histogram(7, 30)
active.record(42)
active.snapshot_into(snapshot)  # overwrite reusable destination; keep active counts
active.drain_into(snapshot)     # overwrite destination, then reset active counts
active.reset()                 # clear counts, retaining the same bucket list

combined = Histogram.checked_sum([snapshot, snapshot])  # independent owned result
snapshot.checked_add_assign(combined)                   # reuse destination storage
queries = [0.99, 0.5, 1.0, 0.5]                          # order/duplicates preserved
output = [None] * len(queries)
if snapshot.percentiles_into(queries, output):
    print(output[0])  # Bucket; each slot corresponds to the same query position
```

`checked_sum` requires at least one input and matching configurations. It copies
the first input and adds the others to private output; it does not repeatedly use
the transactional in-place API. `checked_add_assign` validates all counts before
changing its destination. Configuration or count validation failures preserve
inputs. A self-snapshot is allowed; a self-drain is rejected. These operations
require exclusive access to the histograms involved.

All three representations provide a direct `percentile` query and
`percentiles_into`. The latter requires at least one output slot per request,
validates requests/capacity before writing, leaves extra slots untouched, and
returns `False` for an empty histogram after filling requested slots with `None`.
Requests and output must be separate lists.
It reuses the output list but still allocates `Bucket` objects; dense/sparse batch
queries also allocate sorting scratch and scan occupied counts in query order.
Cumulative queries use binary search. Invalid percentiles, including NaN, fail
even for an empty histogram.

### Native snapshot analytics

Sparse and cumulative snapshots support `merge` and `downsample` without dense
reconstruction. Merge requires matching configurations; downsampling requires a
strictly lower grouping power. Both return independent snapshots, preserve count
mass, and recompute cumulative means for the resulting geometry.
`CumulativeHistogram.to_sparse()` decumulates directly. `shrink_to_fit()` copies
snapshot columns to discard spare list capacity where the Python implementation
supports it; it allocates temporarily and does not guarantee lower process RSS.

Snapshot `index` and `count` properties return copies. Mutating those returned
lists no longer edits a snapshot or invalidates its cached mean. Dense `buckets`
remains a mutable escape hatch: preserve its configured length and use
non-negative Python integer counts. Raw mutation and recording are not validated
on every write; imported storage and the new checked lifecycle APIs are validated.

Python counts remain arbitrary-precision integers. Integer-like imported counts
(such as NumPy `uint64`) are normalized to Python integers; zero sparse entries are omitted after validation, and fractional or negative
counts are rejected instead of silently coerced. Totals beyond `2**53` use integer
rank arithmetic for the supplied binary floating-point percentile; ordinary-size
ranks retain the existing floating-point rounding behavior. Percentiles zero and
one select the first and last populated buckets exactly, even with huge totals.
Arrow/Rezolus export still uses `UInt64`, so Python counts outside that range
cannot be represented in that interchange format.

### Benchmark the phases separately

```bash
PYTHONPATH=src python benchmarks/reporting.py --grouping-power 7 --occupancy few
PYTHONPATH=src python benchmarks/reporting.py --grouping-power 10 --occupancy full
```

The dependency-free harness prepares inputs before timing and emits CSV with
repeat medians/ranges. It separates scalar/batch reporting, reusable lifecycle
operations, owned merges, snapshot conversion, and native analytics. Allocating
cases include result destruction; reset-plus-add is labeled as a combined
operation. It does not measure recording throughput or process memory, and its
Python timings should not be compared directly with another language's harness.

## Reading histograms from a Rezolus Parquet file

Rezolus writes one row per sample interval. Histogram metrics are stored as a dense
`"{metric}:buckets"` column, or a sparse `"{metric}:bucket_indices"` /
`"{metric}:bucket_counts"` pair — all `List<UInt64>`.

```python
from h2histogram.arrow import histogram_columns, read_histograms

# Discover histogram metrics in the file
for col in histogram_columns("rezolus.parquet"):
    print(col.name, col.kind)   # e.g. "syscall/read/latency standard"

# Read a metric's time series: one Histogram per row (None for missing rows)
series = read_histograms("rezolus.parquet", "syscall/read/latency")

for i, hist in enumerate(series):
    if hist is not None:
        print(i, hist.percentile(0.99).midpoint)

# Aggregate the whole recording
total = series[0]
for hist in series[1:]:
    if hist is not None:
        total = total.merge(hist)
print("overall p99:", total.percentile(0.99).midpoint)
```

The bucketing config is resolved from (in order): an explicit `config=`/`grouping_power=`
argument, `grouping_power`/`max_value_power` recorded in the field metadata, inference
from a dense column's bucket count, and finally the Rezolus defaults
(`grouping_power=3`, `max_value_power=64`).

### Writing a Rezolus-compatible file

```python
from h2histogram.arrow import write_histograms

write_histograms(
    "out.parquet",
    {"syscall/read/latency": series},   # {metric_name: [Histogram, ...]}
    timestamps=timestamps_ns,           # optional; one per row
    histogram_type="standard",          # or "sparse"
)
```

Files written this way match the metriken/Rezolus column layout and additionally
record `grouping_power`/`max_value_power` in the field metadata so they are fully
self-describing on read.

See the runnable examples in [`examples/`](examples):

- [`basic_usage.py`](examples/basic_usage.py) — record and query percentiles
- [`read_rezolus_parquet.py`](examples/read_rezolus_parquet.py) — open a Parquet column
  of h2histogram values (synthesizes a sample file if you don't pass one)

## API overview

| Type | Purpose |
|------|---------|
| `Config` | Bucketing parameters; `value_to_index`, `index_to_range`, `total_buckets`, `error` |
| `Histogram` | Dense histogram; `increment`, `record`, `record_many`, `percentile(s)`, `merge`, `subtract`, `downsample`, `to_sparse`, `to_cumulative`, `from_buckets` |
| `SparseHistogram` | Columnar `(index, count)` form; `from_histogram`, `from_parts`, `to_dense`, `to_cumulative` |
| `CumulativeHistogram` | Read-only cumulative form (crate's `CumulativeROHistogram`); binary-search `percentile(s)`, `mean`, `bucket_quantile_range`, `iter_with_quantiles` |
| `Bucket` | A bucket's `count` and inclusive `[start, end]` range, plus `midpoint`/`width` |
| `h2histogram.arrow` | Read/write the Rezolus Arrow/Parquet layout |

## Correctness

The bucketing math is verified against the exact assertions from the Rust crate's
own unit tests (`src/config.rs`), and the NumPy bulk-record fast path is checked
against the scalar path across the full `u64` range. Run `pytest` to see for yourself.

## Related implementations

The h2 histogram bucketing is implemented in several languages, all producing
byte-for-byte identical buckets so histograms interoperate across them:

- [**Rust**](https://github.com/iopsystems/histogram) — the canonical
  implementation (`histogram` crate)
- [**Python**](https://github.com/iopsystems/h2histogram-py) — this repository
- [**Go**](https://github.com/iopsystems/h2histogram-go)
- [**Java**](https://github.com/iopsystems/h2histogram-java)
- [**JavaScript**](https://github.com/iopsystems/h2histogram-js) (values up to
  `2^53 - 1`)

## Releasing

Releases are published to PyPI automatically via GitHub Actions trusted
publishing. See [RELEASING.md](RELEASING.md) for the steps.

## License

MIT — see [LICENSE](LICENSE).
