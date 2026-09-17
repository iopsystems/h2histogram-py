"""Shared report-time operations; recording does not call these helpers."""
import math
import operator

from .bucket import Bucket
from .config import Config


def validate_config(config):
    if not isinstance(config, Config):
        raise TypeError('config must be a Config')
    if config != Config.new(config.grouping_power, config.max_value_power):
        raise ValueError('configuration contains inconsistent derived fields')


def integer(value, label):
    if isinstance(value, bool):
        raise TypeError(label + " must contain non-boolean integers")
    try:
        return operator.index(value)
    except TypeError:
        raise TypeError(label + ' must contain integers') from None


def validate_counts(counts):
    for count in counts:
        if integer(count, 'counts') < 0:
            raise ValueError('counts must be non-negative')


def validate_parts(config, indices, counts, cumulative=False):
    validate_config(config)
    if len(indices) != len(counts):
        raise ValueError('index and count must have the same length')
    previous = -1
    for index in indices:
        index = integer(index, 'indices')
        if index <= previous or index >= config.total_buckets:
            raise ValueError('indices must be ascending and within the configuration')
        previous = index
    validate_counts(counts)
    if cumulative:
        previous = 0
        for count in counts:
            if count == 0 or count < previous:
                raise ValueError('cumulative counts must be positive and non-decreasing')
            previous = count


def validate_percentile(p):
    if not 0.0 <= p <= 1.0:
        raise ValueError('percentiles must be in the range [0.0, 1.0]')


def rank(p, total):
    if p == 0:
        return 1
    if p == 1:
        return total
    if total <= 1 << 53:
        return max(1, math.ceil(p * total))
    # Python counts may exceed both u64 and the floating-point exponent range.
    # Keep integer rank arithmetic exact for the supplied binary float there.
    numerator, denominator = p.as_integer_ratio()
    return max(1, (numerator * total + denominator - 1) // denominator)


def bucket(config, index, count):
    start, end = config.index_to_range(index)
    return Bucket(count=count, start=start, end=end)


def scalar(config, pairs, total, p):
    validate_percentile(p)
    if not total:
        return None
    target = rank(p, total)
    partial = 0
    for index, count in pairs:
        partial += count
        if partial >= target:
            return bucket(config, index, count)
    raise ValueError('counts do not match the histogram total')


def batch_into(config, pairs, total, percentiles, output):
    # Validate all caller input before changing any output slot.
    if percentiles is output:
        raise ValueError("requests and output cannot alias")
    for p in percentiles:
        validate_percentile(p)
    if len(output) < len(percentiles):
        raise ValueError('output must have at least one slot per percentile')
    if not total:
        for i in range(len(percentiles)):
            output[i] = None
        return False
    order = sorted(range(len(percentiles)), key=percentiles.__getitem__)
    iterator = iter(pairs)
    partial = 0
    current = None
    for slot in order:
        target = rank(percentiles[slot], total)
        while partial < target:
            index, count = next(iterator)
            partial += count
            current = (index, count)
        output[slot] = bucket(config, *current)
    return True


def merge_pairs(left, right):
    """Linear merge of individual counts on matching sorted bucket grids."""
    a, b = iter(left), iter(right)
    x, y = next(a, None), next(b, None)
    while x is not None or y is not None:
        if y is None or (x is not None and x[0] < y[0]):
            index, count = x
            x = next(a, None)
        elif x is None or y[0] < x[0]:
            index, count = y
            y = next(b, None)
        else:
            index, count = x[0], x[1] + y[1]
            x, y = next(a, None), next(b, None)
        if count:
            yield index, count


def downsample_pairs(config, pairs, grouping_power):
    if grouping_power >= config.grouping_power:
        raise ValueError('target grouping_power must be less than the current grouping_power')
    target = Config.new(grouping_power, config.max_value_power)
    def reduced():
        previous, running = None, 0
        for index, count in pairs:
            if not count:
                continue
            new_index = target.value_to_index(config.index_to_lower_bound(index))
            if previous is not None and new_index != previous:
                yield previous, running
                running = 0
            previous = new_index
            running += count
        if previous is not None:
            yield previous, running
    return target, reduced()
