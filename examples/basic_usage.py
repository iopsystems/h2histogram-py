"""Basic usage of h2histogram: record values and query percentiles.

Run with::

    python examples/basic_usage.py
"""

from h2histogram import Histogram


def main() -> None:
    # grouping_power=7 -> ~0.78% relative error; max_value_power=64 -> any u64.
    h = Histogram(grouping_power=7, max_value_power=64)

    print(f"config: {h.config}")
    print(f"relative error of log buckets: {h.config.error:.3f}%")
    print(f"total buckets: {h.config.total_buckets}")

    # Record a simulated latency distribution (in nanoseconds).
    import random

    random.seed(42)
    for _ in range(100_000):
        # log-normal-ish latencies
        h.increment(int(random.lognormvariate(7, 1)))

    print(f"\nrecorded {h.total_count()} observations")

    for p in (0.5, 0.9, 0.99, 0.999):
        bucket = h.percentile(p)
        print(f"  p{p * 100:<6g}: ~{bucket.midpoint:>12.0f}  (bucket {bucket.range})")

    # Sparse form is much smaller when most buckets are empty.
    sparse = h.to_sparse()
    print(
        f"\nsparse representation uses {len(sparse)} of "
        f"{h.config.total_buckets} buckets"
    )

    # Downsampling trades error for size.
    coarse = h.downsample(4)
    print(
        f"downsampled to grouping_power=4: "
        f"{coarse.config.total_buckets} buckets, "
        f"{coarse.config.error:.2f}% error, "
        f"count preserved = {coarse.total_count() == h.total_count()}"
    )


if __name__ == "__main__":
    main()
