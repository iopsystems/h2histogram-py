import pytest

from h2histogram import Bucket, Config, Histogram, SparseHistogram


def test_increment_and_total():
    h = Histogram(7, 64)
    for i in range(101):
        h.increment(i)
    assert h.total_count() == 101


def test_record_with_count():
    h = Histogram(7, 64)
    h.record(100, count=5)
    assert h.total_count() == 5
    idx = h.config.value_to_index(100)
    assert h.buckets[idx] == 5


def test_percentile_exact_low_range():
    # In the linear region (values < cutoff) buckets have width 1, so
    # percentiles are exact.
    h = Histogram(7, 64)
    for i in range(1, 101):
        h.increment(i)
    # Rust semantics: count = max(1, ceil(q*total)); return that bucket.
    p50 = h.percentile(0.5)
    assert p50.start == 50 and p50.end == 50
    p100 = h.percentile(1.0)
    assert p100.start == 100 and p100.end == 100
    p0 = h.percentile(0.0)
    assert p0.start == 1 and p0.end == 1


def test_percentile_empty():
    h = Histogram(7, 64)
    assert h.percentile(0.5) is None
    assert h.percentiles([0.5, 0.9]) is None


def test_percentiles_order_preserved():
    h = Histogram(7, 64)
    for i in range(1000):
        h.increment(i)
    result = h.percentiles([0.9, 0.5, 0.99])
    assert [p for p, _ in result] == [0.9, 0.5, 0.99]


def test_percentile_invalid():
    h = Histogram(7, 64)
    h.increment(1)
    with pytest.raises(ValueError):
        h.percentile(1.5)


def test_merge():
    a = Histogram(7, 64)
    b = Histogram(7, 64)
    a.record(10, 3)
    b.record(10, 4)
    b.record(2000, 1)
    merged = a + b
    assert merged.total_count() == 8
    assert merged.buckets[merged.config.value_to_index(10)] == 7


def test_merge_incompatible():
    a = Histogram(7, 64)
    b = Histogram(6, 64)
    with pytest.raises(ValueError):
        a.merge(b)


def test_subtract():
    a = Histogram(7, 64)
    b = Histogram(7, 64)
    a.record(10, 5)
    b.record(10, 2)
    diff = a - b
    assert diff.total_count() == 3
    with pytest.raises(ValueError):
        b - a  # would go negative


def test_from_buckets_roundtrip():
    h = Histogram(3, 64)
    h.record(5, 2)
    h.record(1000, 7)
    h2 = Histogram.from_buckets(3, 64, h.buckets)
    assert h == h2


def test_from_buckets_wrong_length():
    with pytest.raises(ValueError):
        Histogram.from_buckets(7, 64, [0, 0, 0])


def test_downsample():
    h = Histogram(7, 64)
    for i in range(10000):
        h.increment(i)
    coarse = h.downsample(3)
    assert coarse.config.grouping_power == 3
    assert coarse.total_count() == h.total_count()
    with pytest.raises(ValueError):
        h.downsample(7)


def test_sparse_roundtrip():
    h = Histogram(7, 64)
    h.record(1, 1)
    h.record(500, 3)
    h.record(999999, 2)
    sparse = h.to_sparse()
    assert isinstance(sparse, SparseHistogram)
    assert sparse.total_count() == h.total_count()
    assert len(sparse) == 3  # three non-zero buckets
    # indices strictly ascending
    assert sparse.index == sorted(sparse.index)
    dense = sparse.to_dense()
    assert dense == h


def test_sparse_from_parts_validation():
    c = Config.new(7, 64)
    with pytest.raises(ValueError):
        SparseHistogram.from_parts(c, [1, 2], [1])  # length mismatch
    with pytest.raises(ValueError):
        SparseHistogram.from_parts(c, [2, 1], [1, 1])  # not ascending
    with pytest.raises(ValueError):
        SparseHistogram.from_parts(c, [999999999], [1])  # out of range


def test_record_many_matches_loop():
    values = [0, 1, 2, 300, 255, 256, 1024, 1_000_000, (1 << 50) + 3] * 111
    a = Histogram(7, 64)
    for v in values:
        a.increment(v)
    b = Histogram(7, 64)
    b.record_many(values)
    assert a == b


def test_record_many_with_counts():
    a = Histogram(7, 64)
    a.record_many([10, 20, 10], counts=[2, 3, 5])
    assert a.total_count() == 10
    assert a.buckets[a.config.value_to_index(10)] == 7


def test_iter_buckets():
    h = Histogram(3, 6)
    h.increment(0)
    buckets = list(h)
    assert len(buckets) == h.config.total_buckets
    assert all(isinstance(b, Bucket) for b in buckets)
    nonzero = list(h.nonzero_buckets())
    assert len(nonzero) == 1
    assert nonzero[0].count == 1
