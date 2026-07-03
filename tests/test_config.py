"""Config tests cross-checked against the Rust `histogram` crate's unit tests.

The expected values here are copied verbatim from ``src/config.rs`` in
https://github.com/iopsystems/histogram so we are guaranteed bit-identical
bucketing.
"""

import pytest

from h2histogram import Config


def test_total_buckets():
    assert Config.new(2, 64).total_buckets == 252
    assert Config.new(7, 64).total_buckets == 7424
    assert Config.new(14, 64).total_buckets == 835_584
    assert Config.new(2, 4).total_buckets == 12


def test_value_to_index():
    c = Config.new(7, 64)
    assert c.value_to_index(0) == 0
    assert c.value_to_index(1) == 1
    assert c.value_to_index(256) == 256
    assert c.value_to_index(257) == 256
    assert c.value_to_index(258) == 257
    assert c.value_to_index(512) == 384
    assert c.value_to_index(515) == 384
    assert c.value_to_index(516) == 385
    assert c.value_to_index(1024) == 512
    assert c.value_to_index(1031) == 512
    assert c.value_to_index(1032) == 513
    assert c.value_to_index((1 << 64) - 2) == 7423
    assert c.value_to_index((1 << 64) - 1) == 7423


def test_index_to_lower_bound():
    c = Config.new(7, 64)
    assert c.index_to_lower_bound(0) == 0
    assert c.index_to_lower_bound(1) == 1
    assert c.index_to_lower_bound(256) == 256
    assert c.index_to_lower_bound(384) == 512
    assert c.index_to_lower_bound(512) == 1024
    assert c.index_to_lower_bound(7423) == 18_374_686_479_671_623_680


def test_index_to_upper_bound():
    c = Config.new(7, 64)
    assert c.index_to_upper_bound(0) == 0
    assert c.index_to_upper_bound(1) == 1
    assert c.index_to_upper_bound(256) == 257
    assert c.index_to_upper_bound(384) == 515
    assert c.index_to_upper_bound(512) == 1031
    assert c.index_to_upper_bound(7423) == (1 << 64) - 1


def test_index_to_range():
    c = Config.new(7, 64)
    assert c.index_to_range(0) == (0, 0)
    assert c.index_to_range(256) == (256, 257)
    assert c.index_to_range(384) == (512, 515)
    assert c.index_to_range(512) == (1024, 1031)
    assert c.index_to_range(7423) == (18_374_686_479_671_623_680, (1 << 64) - 1)


def test_roundtrip_value_index_range():
    # Every value must land in a bucket whose range contains it.
    c = Config.new(7, 64)
    for value in [0, 1, 5, 127, 128, 255, 256, 257, 999, 1_000_000, (1 << 40) + 7]:
        idx = c.value_to_index(value)
        lo, hi = c.index_to_range(idx)
        assert lo <= value <= hi


def test_error():
    assert Config.new(7, 64).error == pytest.approx(100.0 / 128)
    # No logarithmic buckets -> zero error.
    assert Config.new(3, 4).error == 0.0


def test_invalid_params():
    with pytest.raises(ValueError):
        Config.new(7, 65)
    with pytest.raises(ValueError):
        Config.new(64, 64)
    with pytest.raises(ValueError):
        Config.new(10, 5)


def test_from_total_buckets():
    c = Config.from_total_buckets(7424, 64)
    assert c.grouping_power == 7
    assert c.max_value_power == 64
    # Rezolus default.
    assert Config.from_total_buckets(496, 64).grouping_power == 3
    with pytest.raises(ValueError):
        Config.from_total_buckets(7425, 64)


def test_out_of_range():
    c = Config.new(2, 4)  # max = 15
    assert c.value_to_index(15) == c.total_buckets - 1
    with pytest.raises(ValueError):
        c.value_to_index(16)
    with pytest.raises(ValueError):
        c.value_to_index(-1)
