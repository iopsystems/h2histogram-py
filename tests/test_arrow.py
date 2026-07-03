import pytest

pa = pytest.importorskip("pyarrow")
import pyarrow.parquet as pq  # noqa: E402

from h2histogram import Config, Histogram  # noqa: E402
from h2histogram.arrow import (  # noqa: E402
    histogram_columns,
    read_all_histograms,
    read_histograms,
    write_histograms,
)


def _make_series(n, grouping_power=3, max_value_power=64):
    series = []
    for i in range(n):
        h = Histogram(grouping_power, max_value_power)
        h.record(i + 1, count=i + 1)
        h.record(1000 * (i + 1), count=2)
        series.append(h)
    return series


def test_write_read_standard_roundtrip(tmp_path):
    path = tmp_path / "std.parquet"
    series = _make_series(4)
    write_histograms(path, {"latency": series}, histogram_type="standard")

    cols = histogram_columns(str(path))
    assert len(cols) == 1
    assert cols[0].name == "latency"
    assert cols[0].kind == "standard"
    assert cols[0].config == Config.new(3, 64)

    read = read_histograms(str(path), "latency")
    assert len(read) == 4
    assert read == series


def test_write_read_sparse_roundtrip(tmp_path):
    path = tmp_path / "sparse.parquet"
    series = _make_series(3)
    write_histograms(path, {"latency": series}, histogram_type="sparse")

    cols = histogram_columns(str(path))
    assert cols[0].kind == "sparse"

    read = read_histograms(str(path), "latency")
    assert read == series


def test_schema_matches_rezolus_standard(tmp_path):
    """Column names and metadata must match the metriken/Rezolus layout."""
    path = tmp_path / "std.parquet"
    series = _make_series(2)
    write_histograms(path, {"cpu/usage": series}, histogram_type="standard")

    schema = pq.read_schema(path)
    assert "timestamp" in schema.names
    assert "duration" in schema.names
    assert "cpu/usage:buckets" in schema.names
    field = schema.field("cpu/usage:buckets")
    assert field.metadata[b"metric_type"] == b"histogram"
    # The dense column is a list of u64.
    assert pa.types.is_list(field.type)
    assert pa.types.is_unsigned_integer(field.type.value_type)


def test_schema_matches_rezolus_sparse(tmp_path):
    path = tmp_path / "sparse.parquet"
    series = _make_series(2)
    write_histograms(path, {"cpu/usage": series}, histogram_type="sparse")

    schema = pq.read_schema(path)
    assert "cpu/usage:bucket_indices" in schema.names
    assert "cpu/usage:bucket_counts" in schema.names
    field = schema.field("cpu/usage:bucket_indices")
    assert field.metadata[b"metric_type"] == b"sparse_histogram"


def test_read_standard_without_metadata_infers_config(tmp_path):
    """A Rezolus-style file with no grouping_power metadata should still read.

    We simulate that by hand-building a table with only a :buckets column.
    """
    config = Config.new(3, 64)
    h = Histogram(config=config)
    h.record(42, 3)

    list_type = pa.list_(pa.uint64())
    table = pa.table(
        {
            "timestamp": pa.array([0], type=pa.uint64()),
            "latency:buckets": pa.array([list(h.buckets)], type=list_type),
        }
    )
    path = tmp_path / "nometa.parquet"
    pq.write_table(table, path)

    # No metadata present; grouping_power inferred from bucket count.
    read = read_histograms(str(path), "latency")
    assert read[0] == h


def test_null_rows(tmp_path):
    path = tmp_path / "nulls.parquet"
    series = [_make_series(1)[0], None, _make_series(1)[0]]
    write_histograms(path, {"latency": series}, histogram_type="standard")
    read = read_histograms(str(path), "latency")
    assert read[1] is None
    assert read[0] is not None and read[2] is not None


def test_read_all_histograms(tmp_path):
    path = tmp_path / "multi.parquet"
    write_histograms(
        path,
        {"a": _make_series(3), "b": _make_series(3)},
        histogram_type="standard",
    )
    result = read_all_histograms(str(path))
    assert set(result.keys()) == {"a", "b"}
    assert len(result["a"]) == 3


def test_read_missing_metric(tmp_path):
    path = tmp_path / "std.parquet"
    write_histograms(path, {"latency": _make_series(1)}, histogram_type="standard")
    with pytest.raises(KeyError):
        read_histograms(str(path), "nope")


def test_timestamps_written(tmp_path):
    path = tmp_path / "ts.parquet"
    ts = [100, 200, 300]
    write_histograms(path, {"latency": _make_series(3)}, timestamps=ts)
    table = pq.read_table(path)
    assert table.column("timestamp").to_pylist() == ts
