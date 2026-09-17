"""Prepared-input phase smoke/benchmark; no third-party dependencies."""
import argparse
import csv
import platform
import statistics
import sys
import time

from h2histogram import Histogram


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grouping-power', type=int, default=7)
    parser.add_argument('--occupancy', choices=['few', 'full'], default='few')
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    if args.iterations < 1 or args.repeats < 1:
        parser.error('iterations and repeats must be positive')
    h = Histogram(args.grouping_power, 30)
    stride = max(1, len(h) // 32) if args.occupancy == 'few' else 1
    for i in range(0, len(h), stride):
        h.buckets[i] = i % 17 + 1
    inputs = [Histogram.from_buckets(args.grouping_power, 30, h.buckets) for _ in range(8)]
    sparse, cumulative = h.to_sparse(), h.to_cumulative()
    destination = Histogram(args.grouping_power, 30)
    active = Histogram.from_buckets(args.grouping_power, 30, h.buckets)
    qs = [0, .5, .9, .99, 1]
    output = [None] * len(qs)

    def drain_swap():
        nonlocal active, destination
        active.drain_into(destination)
        active, destination = destination, active

    def reset_add():
        destination.reset()
        destination.checked_add_assign(h)

    cases = {
        'report/dense_scalar': lambda: h.percentile(.99),
        'report/dense_batch': lambda: h.percentiles(qs),
        'report/dense_batch_into': lambda: h.percentiles_into(qs, output),
        'report/sparse_scalar': lambda: sparse.percentile(.99),
        'report/sparse_batch_into': lambda: sparse.percentiles_into(qs, output),
        'report/cumulative_scalar': lambda: cumulative.percentile(.99),
        'report/cumulative_batch_into': lambda: cumulative.percentiles_into(qs, output),
        'report/snapshot_into': lambda: h.snapshot_into(destination),
        'report/drain_into_swap': drain_swap,
        'report/reset_plus_checked_add': reset_add,
        'report/convert_sparse': h.to_sparse,
        'report/convert_cumulative': h.to_cumulative,
        'analytics/dense_owned_sum_8': lambda: Histogram.checked_sum(inputs),
        'analytics/sparse_merge_2': lambda: sparse.merge(sparse),
        'analytics/cumulative_merge_2': lambda: cumulative.merge(cumulative),
        'analytics/cumulative_to_sparse': cumulative.to_sparse,
    }
    if args.grouping_power > 0:
        cases.update({
            'analytics/sparse_downsample': lambda: sparse.downsample(args.grouping_power - 1),
            'analytics/cumulative_downsample': lambda: cumulative.downsample(args.grouping_power - 1),
        })
    print('# Python ' + platform.python_version() + ' / ' + platform.platform())
    writer = csv.writer(sys.stdout)
    writer.writerow(['case', 'grouping_power', 'occupancy', 'iterations', 'repeats',
                     'median_us', 'min_us', 'max_us'])
    for name, operation in cases.items():
        operation()  # warm-up, excluded
        measurements = []
        for _ in range(args.repeats):
            start = time.perf_counter_ns()
            for _ in range(args.iterations):
                operation()  # discard each allocating result within timing
            measurements.append((time.perf_counter_ns() - start) / args.iterations / 1000)
        writer.writerow([name, args.grouping_power, args.occupancy, args.iterations,
                         args.repeats, statistics.median(measurements), min(measurements),
                         max(measurements)])


if __name__ == '__main__':
    main()
