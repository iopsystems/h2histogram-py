"""Reporting/analytics contracts independent of representation and ownership."""
import dataclasses
import random

import pytest

from h2histogram import Config, CumulativeHistogram, Histogram, SparseHistogram


def populated(seed=1, gp=3, max_power=10):
    rng = random.Random(seed)
    h = Histogram(gp, max_power)
    for _ in range(80):
        h.record(rng.randrange(1 << max_power), rng.randrange(1, 7))
    return h


def test_dense_reset_snapshot_and_drain_reuse_storage():
    h, dst = populated(), populated(2)
    before = h.buckets.copy()
    own, target = h.buckets, dst.buckets
    h.snapshot_into(dst)
    assert dst.buckets is target and dst.buckets == before
    h.drain_into(dst)
    assert dst.buckets is target and dst.buckets == before
    assert h.buckets is own and h.total_count() == 0
    h.record(42, 2)
    h.reset()
    assert h.buckets is own and h.total_count() == 0
    dst.snapshot_into(dst)  # self-snapshot is a no-op
    with pytest.raises(ValueError):
        dst.drain_into(dst)
    assert dst.buckets == before
    with pytest.raises(ValueError):
        dst.drain_into(Histogram(2, 10))
    assert dst.buckets == before


def test_owned_batch_and_in_place_merge_contract():
    a, b = populated(), populated(2)
    before = a.buckets.copy()
    expected = [x * 2 + y for x, y in zip(a.buckets, b.buckets)]
    merged = Histogram.checked_sum([a, b, a])
    assert merged.buckets == expected and a.buckets == before
    singleton = Histogram.checked_sum([a])
    singleton.reset()
    assert a.buckets == before
    with pytest.raises(ValueError):
        Histogram.checked_sum([])
    with pytest.raises(ValueError):
        Histogram.checked_sum([a, b, Histogram(2, 10)])
    assert a.buckets == before
    storage = a.buckets
    a.checked_add_assign(a)
    assert a.buckets is storage and a.buckets == [2 * x for x in before]
    before = a.buckets.copy()
    bad = populated(); bad.buckets[-1] = -1
    with pytest.raises(ValueError):
        a.checked_add_assign(bad)
    assert a.buckets == before


@pytest.mark.parametrize('form', [lambda h: h, lambda h: h.to_sparse(), lambda h: h.to_cumulative()])
def test_direct_queries_order_duplicates_buffers_and_empty(form, monkeypatch):
    h = populated(); qs = [1.0, 0.0, 0.5, 0.5, 0.99]
    expected = h.percentiles(qs)
    obj = form(h)
    if isinstance(obj, SparseHistogram):
        monkeypatch.setattr(SparseHistogram, 'to_dense', lambda *_: pytest.fail('dense reconstruction'))
    # Scalar must not call batch query or allocate its result collection.
    monkeypatch.setattr(type(obj), 'percentiles', lambda *_: pytest.fail('scalar used batch'))
    assert obj.percentile(.5) == expected[2][1]
    out = [None] * 5 + ['sentinel']
    assert obj.percentiles_into(qs, out) is True
    assert out[:5] == [b for _, b in expected] and out[5] == 'sentinel'
    before = out.copy()
    for invalid in ([.5, float('nan')], [.5, 2], [.5, float('inf')]):
        with pytest.raises(ValueError): obj.percentiles_into(invalid, out)
        assert out == before
    with pytest.raises(ValueError): obj.percentiles_into(qs, [None])
    empty = form(Histogram(3, 10))
    assert empty.percentiles_into([.5], out) is False and out[0] is None
    with pytest.raises(ValueError): empty.percentile(float('nan'))


@pytest.mark.parametrize('form', [lambda h: h.to_sparse(), lambda h: h.to_cumulative()])
def test_native_transforms_conserve_buckets_and_mean(form, monkeypatch):
    for seed in range(5):
        a, b = populated(seed), populated(seed + 10)
        sa, sb = form(a), form(b)
        before = (sa.index, sa.count, sb.index, sb.count)
        expected = a.merge(b)
        with monkeypatch.context() as patch:
            patch.setattr(type(sa), 'to_dense', lambda *_: pytest.fail('dense reconstruction'))
            merged = sa.merge(sb)
            coarse = merged.downsample(1)
            sparse = merged.to_sparse() if isinstance(merged, CumulativeHistogram) else merged
        assert merged.to_dense() == expected
        assert sparse.to_dense() == expected
        assert coarse.to_dense() == expected.downsample(1)
        assert before == (sa.index, sa.count, sb.index, sb.count)
        if isinstance(coarse, CumulativeHistogram):
            assert coarse.mean() == pytest.approx(expected.downsample(1).to_cumulative().mean())
        with pytest.raises(ValueError): sa.merge(form(Histogram(2, 10)))
        with pytest.raises(ValueError): sa.downsample(4)
        assert sa.merge(form(Histogram(3, 10))) == sa


def test_snapshots_are_independent_and_accessors_cannot_stale_mean():
    h = populated(); sparse = h.to_sparse(); cumulative = h.to_cumulative()
    expected, mean = h.buckets.copy(), cumulative.mean()
    for snapshot in (sparse, cumulative):
        snapshot.index.clear(); snapshot.count.clear()
        assert snapshot.to_dense().buckets == expected
    h.buckets[:] = [0] * len(h)
    assert cumulative.mean() == mean and cumulative.to_dense().buckets == expected


@pytest.mark.parametrize('bad', [-1, 1.5, float('nan'), '2', None])
def test_imported_counts_cannot_be_silently_coerced(bad):
    with pytest.raises((ValueError, TypeError)):
        Histogram.from_buckets(0, 1, [1, bad])
    with pytest.raises((ValueError, TypeError)):
        SparseHistogram.from_parts(Config.new(0, 1), [0, 1], [1, bad])
    with pytest.raises((ValueError, TypeError)):
        CumulativeHistogram.from_parts(Config.new(0, 1), [0, 1], [1, bad])


def test_invalid_shapes_and_derived_configuration_rejected():
    cfg = Config.new(1, 4)
    with pytest.raises(ValueError): SparseHistogram(cfg, [1], [1, 2])
    with pytest.raises((ValueError, TypeError)): SparseHistogram(cfg, [1.5], [2])
    corrupt = dataclasses.replace(cfg, lower_bin_count=1)
    with pytest.raises(ValueError): Histogram(config=corrupt)
    with pytest.raises(ValueError): CumulativeHistogram.from_parts(corrupt, [0], [1])


def test_python_counts_remain_arbitrary_precision_and_rank_endpoints_are_exact():
    huge = 10 ** 400
    h = Histogram.from_buckets(0, 1, [huge, 1])
    for obj in (h, h.to_sparse(), h.to_cumulative()):
        assert obj.percentile(1).start == 1
        assert obj.percentile(0).start == 0
        assert obj.percentile(.5).start == 0
        assert obj.total_count() == huge + 1
    assert h.to_cumulative().mean() == pytest.approx(0.0, abs=1e-300)
    merged = Histogram.checked_sum([h, h])
    assert merged.buckets == [2 * huge, 2]
    for obj in (h.to_sparse(), h.to_cumulative()):
        assert obj.merge(obj).to_dense() == merged


def test_equal_prefix_and_zero_sparse_entries_do_not_change_mass():
    cfg = Config.new(1, 4)
    sparse = SparseHistogram.from_parts(cfg, [0, 1, 2], [1, 0, 3])
    cumulative = CumulativeHistogram.from_parts(cfg, [0, 1, 2], [1, 1, 4])
    assert sparse.to_dense() == cumulative.to_dense()
    assert cumulative.to_sparse().count == [1, 3]
    assert sparse.percentile(.5).start == 2


@pytest.mark.parametrize('form', [lambda h: h.to_sparse(), lambda h: h.to_cumulative()])
def test_compaction_preserves_snapshot_and_queries(form):
    h = populated(); snapshot = form(h)
    expected = snapshot.percentiles([0, .5, 1])
    original = snapshot.to_dense()
    snapshot.shrink_to_fit()
    assert snapshot.to_dense() == original
    assert snapshot.percentiles([0, .5, 1]) == expected


def test_buffer_cannot_alias_dense_counts():
    h = populated(); before = h.buckets.copy()
    with pytest.raises(ValueError): h.percentiles_into([.5], h.buckets)
    assert h.buckets == before


def test_numpy_import_counts_normalized_to_python_integers():
    np = pytest.importorskip('numpy')
    n = np.uint64(2 ** 64 - 1)
    dense = Histogram.from_buckets(0, 1, [n, n])
    sparse = SparseHistogram.from_parts(dense.config, [0, 1], [n, n])
    assert sparse.merge(sparse).to_dense().buckets == [2 * int(n), 2 * int(n)]
    assert Histogram.checked_sum([dense, dense]).buckets == [2 * int(n), 2 * int(n)]


def test_arrow_rejects_fractional_imported_counts():
    pa = pytest.importorskip('pyarrow')
    from h2histogram.arrow import read_histograms
    malformed = pa.table({'latency:buckets': [[1.0, 1.5]]})
    with pytest.raises((ValueError, TypeError)):
        read_histograms(malformed, 'latency', grouping_power=0, max_value_power=1)


@pytest.mark.parametrize('conversion', ['to_sparse', 'to_cumulative'])
def test_snapshot_rejects_malformed_dense_storage(conversion):
    h = Histogram(1, 4)
    h.buckets.append(0)
    with pytest.raises(ValueError):
        getattr(h, conversion)()


@pytest.mark.parametrize('counts', [[0, 1], [0, 0]])
def test_sparse_import_normalizes_leading_and_all_zero_counts(counts):
    sparse = SparseHistogram.from_parts(Config.new(0, 1), [0, 1], counts)
    assert sparse.count == [count for count in counts if count]
    cumulative = sparse.to_cumulative()
    assert cumulative.to_dense() == sparse.to_dense()
    assert sparse.is_empty() == (sum(counts) == 0)
    assert cumulative.is_empty() == sparse.is_empty()


@pytest.mark.parametrize('form', [lambda h: h, lambda h: h.to_sparse(), lambda h: h.to_cumulative()])
def test_request_buffer_cannot_alias_output(form):
    requests = [1., .5, 0.]
    with pytest.raises(ValueError):
        form(populated()).percentiles_into(requests, requests)
    assert requests == [1., .5, 0.]


def test_weighted_numpy_recording_and_snapshot_counts_stay_unbounded():
    np = pytest.importorskip('numpy')
    a, b = Histogram(0, 1), Histogram(0, 1)
    for h in (a, b):
        h.record_many([0], np.array([2**63], dtype=np.uint64))
    assert a.merge(b).buckets == [2**64, 0]
    a.record_many([0], np.array([2**63], dtype=np.uint64))
    assert a.buckets == [2**64, 0]
    h = Histogram(0, 1)
    h.record_many([0, 0], np.array([2**64 - 1, 1], dtype=np.uint64))
    assert h.buckets == [2**64, 0]
    # Integer-like raw storage is normalized when copied, without editing source.
    h.buckets[0] = np.uint64(2**63)
    dst = Histogram(0, 1)
    h.snapshot_into(dst)
    assert type(dst.buckets[0]) is int
    assert h.merge(h).buckets == [2**64, 0]


@pytest.mark.parametrize('bad', [True, False, 5.0])
def test_import_counts_require_non_boolean_integer_objects(bad):
    with pytest.raises(TypeError): Histogram.from_buckets(0, 1, [bad, 0])
    with pytest.raises(TypeError): SparseHistogram.from_parts(Config.new(0, 1), [0], [bad])
    with pytest.raises(TypeError): CumulativeHistogram.from_parts(Config.new(0, 1), [0], [bad])


def test_arrow_boolean_counts_rejected():
    pa = pytest.importorskip('pyarrow')
    from h2histogram.arrow import read_histograms
    with pytest.raises(TypeError):
        read_histograms(pa.table({'latency:buckets': [[True, False]]}),
                        'latency', grouping_power=0, max_value_power=1)


def test_large_rank_uses_supplied_binary_float_without_product_rounding():
    # ceil(float(.7) * total) rounds to first_count, but the exact rank is next.
    total = 2**53 + 1
    p = .7
    numerator, denominator = p.as_integer_ratio()
    target = (numerator * total + denominator - 1) // denominator
    h = Histogram.from_buckets(0, 1, [target - 1, total - target + 1])
    for obj in (h, h.to_sparse(), h.to_cumulative()):
        assert obj.percentile(p).start == 1


def test_unweighted_numpy_bulk_recording_never_narrows_existing_counts():
    pytest.importorskip('numpy')
    for count in (2**63 - 1, 2**63, 10**400):
        h = Histogram.from_buckets(0, 1, [count, 0])
        storage = h.buckets
        h.record_many([0, 1, 0])
        assert h.buckets == [count + 2, 1]
        assert h.buckets is storage


@pytest.mark.parametrize('values,counts', [([0, 1], [2]), ([0], [2, 3]), ([], [2]), ([0], [])])
@pytest.mark.parametrize('iterators', [False, True])
def test_weighted_recording_rejects_length_mismatch(values, counts, iterators):
    h = Histogram(0, 1)
    expected = [2, 0] if values and counts else [0, 0]
    if iterators:
        values, counts = iter(values), iter(counts)
    with pytest.raises(ValueError, match='same length'):
        h.record_many(values, counts)
    # Streaming recording retains any complete pair before the mismatch.
    assert h.buckets == expected


def test_weighted_recording_equal_empty_and_generator_inputs():
    h = Histogram(0, 1)
    h.record_many(iter([]), iter([]))
    h.record_many((v for v in [0, 1]), (n for n in [2, 3]))
    assert h.buckets == [2, 3]
