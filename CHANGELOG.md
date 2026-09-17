# Changelog

## Unreleased

### Reporting and analytics APIs

Adds reusable reset/snapshot/drain, checked in-place and owned aggregation,
caller-output queries, native sparse/cumulative merge and downsample,
cumulative-to-sparse conversion and opt-in snapshot list compaction.

The unweighted NumPy bulk path now adds batch counts to Python integers rather
than narrowing existing counters to int64; this fixes signed overflow and permits
existing counts above the u64 range. It retains the dense list.

Weighted `record_many` rejects mismatched value/count lengths, including lazy
iterators, rather than silently truncating. Complete pairs before an error remain
recorded; the operation does not materialize inputs or roll back the batch.

### Compatibility changes

- Imported counts/indices must be integer objects; boolean and floating-point
  values are rejected, including integral floats such as `5.0`. Integer-like
  NumPy values normalize to Python integers. Weighted `record_many` applies the
  same count rules before each addition (not transactionally across the batch).
- Sparse imports validate all entries, then omit zero counts. Stored length,
  arrays and equality can differ from an input with explicit zeros. This accepts
  and normalizes zeros rather than following Rust's rejecting import contract.
- Snapshot `index`/`count` accessors return copies; editing them no longer edits
  the snapshot or its cached mean. Dense raw storage remains caller-managed.
- Invalid configurations/storage and NaN percentiles fail validation, including
  on empty snapshots.
- Above `2**53` total count, percentile ranks use exact integer arithmetic for
  the supplied binary float. This intentionally can differ at bucket boundaries
  from Rust's floating-point product, even for totals within the u64 range.
  Counts themselves remain unbounded; p0/p100 select exact endpoints.

The scalar recording path remains unchecked and requires non-negative Python int
weights and Python int bucket storage. Convert external scalar types first, or
use weighted `record_many` for validated normalization. Report-time validation
cannot recover counts already wrapped by unsupported scalar arithmetic.
