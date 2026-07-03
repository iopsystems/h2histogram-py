import pytest

from h2histogram import Config, CumulativeHistogram, Histogram, SparseHistogram


def test_from_histogram_matches_rust():
    # Mirrors the `from_histogram` unit test in the Rust crate's cumulative.rs.
    h = Histogram(7, 64)
    h.increment(1)
    h.increment(1)
    h.increment(5)
    h.increment(100)

    croh = CumulativeHistogram.from_histogram(h)
    assert croh.config == h.config
    assert len(croh) == 3
    assert croh.count == [2, 3, 4]  # cumulative
    assert croh.total_count() == 4


def test_from_sparse_matches_rust():
    # Mirrors the `from_sparse` unit test in the Rust crate.
    config = Config.new(7, 32)
    sparse = SparseHistogram.from_parts(config, [1, 3, 5], [6, 12, 7])
    croh = CumulativeHistogram.from_sparse(sparse)
    assert croh.config == config
    assert croh.index == [1, 3, 5]
    assert croh.count == [6, 18, 25]  # cumulative
    assert croh.total_count() == 25


def test_quantiles_match_dense_histogram():
    # The Rust crate asserts cumulative quantiles equal the dense histogram's.
    h = Histogram(4, 10)
    for v in range(1, 1024):
        h.increment(v)

    croh = h.to_cumulative()
    quantiles = [0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999, 1.0]

    dense = h.percentiles(quantiles)
    cumulative = croh.percentiles(quantiles)
    assert dense is not None and cumulative is not None
    for (pd, bd), (pc, bc) in zip(dense, cumulative):
        assert pd == pc
        assert bd.range == bc.range


def test_percentile_individual_count():
    h = Histogram(7, 64)
    h.record(10, 5)
    h.record(10, 3)  # same bucket -> 8 total in that bucket
    h.record(1_000_000, 2)
    croh = h.to_cumulative()

    b = croh.percentile(0.5)
    # median falls in the value-10 bucket; the bucket carries its individual count
    assert b.count == 8
    lo, hi = croh.config.index_to_range(croh.config.value_to_index(10))
    assert b.range == (lo, hi)


def test_empty():
    croh = Histogram(7, 64).to_cumulative()
    assert croh.is_empty()
    assert croh.total_count() == 0
    assert croh.mean() is None
    assert croh.percentile(0.5) is None
    assert croh.bucket_quantile_range(0) is None


def test_mean_midpoint_estimate():
    # Two exact (linear-region) buckets: values 2 and 4, equal weight.
    h = Histogram(7, 64)
    h.record(2, 1)
    h.record(4, 1)
    croh = h.to_cumulative()
    # linear buckets have width 1, midpoints == the values -> mean 3.0
    assert croh.mean() == pytest.approx(3.0)


def test_bucket_quantile_range():
    config = Config.new(7, 64)
    croh = CumulativeHistogram.from_parts(config, [1, 3, 5], [2, 6, 8])
    assert croh.bucket_quantile_range(0) == pytest.approx((0.0, 0.25))
    assert croh.bucket_quantile_range(1) == pytest.approx((0.25, 0.75))
    assert croh.bucket_quantile_range(2) == pytest.approx((0.75, 1.0))
    assert croh.bucket_quantile_range(3) is None


def test_iter_individual_counts():
    croh = CumulativeHistogram.from_parts(Config.new(7, 64), [1, 3, 5], [2, 6, 8])
    counts = [b.count for b in croh]
    assert counts == [2, 4, 2]  # de-cumulated


def test_iter_with_quantiles():
    croh = CumulativeHistogram.from_parts(Config.new(7, 64), [1, 3, 5], [2, 6, 8])
    rows = list(croh.iter_with_quantiles())
    assert [b.count for b, _, _ in rows] == [2, 4, 2]
    assert rows[0][1] == pytest.approx(0.0)
    assert rows[-1][2] == pytest.approx(1.0)


def test_from_parts_validation():
    config = Config.new(7, 64)
    with pytest.raises(ValueError):
        CumulativeHistogram.from_parts(config, [1, 2], [1])  # length mismatch
    with pytest.raises(ValueError):
        CumulativeHistogram.from_parts(config, [2, 1], [1, 2])  # not ascending
    with pytest.raises(ValueError):
        CumulativeHistogram.from_parts(config, [1, 2], [5, 3])  # counts decrease
    with pytest.raises(ValueError):
        CumulativeHistogram.from_parts(config, [1], [0])  # zero count
    with pytest.raises(ValueError):
        CumulativeHistogram.from_parts(config, [10 ** 9], [1])  # index out of range


def test_to_dense_roundtrip():
    h = Histogram(7, 64)
    h.record(3, 4)
    h.record(9999, 7)
    croh = h.to_cumulative()
    assert croh.to_dense() == h


def test_binary_search_large():
    # Exercise the binary-search path with many non-zero buckets.
    h = Histogram(7, 64)
    for v in range(1, 5000):
        h.increment(v)
    croh = h.to_cumulative()
    # Cumulative and dense percentiles must agree exactly.
    qs = [i / 100 for i in range(101)]
    for (_, bd), (_, bc) in zip(h.percentiles(qs), croh.percentiles(qs)):
        assert bd.range == bc.range
