"""Read and write h2 histograms in the Arrow / Parquet layout used by Rezolus.

Rezolus (via ``metriken-exposition``) records a time series of metric snapshots
to a Parquet file with one row per snapshot. Histograms are stored in one of two
layouts:

* **Standard (dense):** a single column named ``"{metric}:buckets"`` of type
  ``List<UInt64>`` holding every bucket's count.
* **Sparse:** two columns ``"{metric}:bucket_indices"`` and
  ``"{metric}:bucket_counts"``, both ``List<UInt64>``, holding only the
  non-zero buckets.

Each histogram field carries a ``metric_type`` metadata entry
(``"histogram"`` or ``"sparse_histogram"``). The bucketing parameters
(``grouping_power`` / ``max_value_power``) are *not* recorded by Rezolus, which
uses fixed values (``grouping_power=3``, ``max_value_power=64``). This module
defaults to those values, lets you override them, records them in field metadata
when it writes (so round-trips are exact), and can infer ``grouping_power`` for
dense columns from the bucket count.

``pyarrow`` is required for this module; install it with
``pip install h2histogram[parquet]``.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .config import Config
from .histogram import Histogram
from .sparse import SparseHistogram

__all__ = [
    "DEFAULT_GROUPING_POWER",
    "DEFAULT_MAX_VALUE_POWER",
    "HistogramColumn",
    "histogram_columns",
    "read_histograms",
    "read_all_histograms",
    "write_histograms",
]

# Rezolus records histograms with these fixed parameters.
DEFAULT_GROUPING_POWER = 3
DEFAULT_MAX_VALUE_POWER = 64

_BUCKETS_SUFFIX = ":buckets"
_INDICES_SUFFIX = ":bucket_indices"
_COUNTS_SUFFIX = ":bucket_counts"

_META_TYPE = b"metric_type"
_META_GROUPING = b"grouping_power"
_META_MAX_POWER = b"max_value_power"


def _require_pyarrow():
    try:
        import pyarrow as pa  # noqa: WPS433
    except ImportError as exc:  # pragma: no cover - exercised only without pyarrow
        raise ImportError(
            "pyarrow is required for h2histogram.arrow; "
            "install it with `pip install h2histogram[parquet]`"
        ) from exc
    return pa


def _read_table(source):
    """Accept a path, a pyarrow Table, or a pyarrow RecordBatch and return a Table."""
    pa = _require_pyarrow()
    if isinstance(source, pa.Table):
        return source
    if isinstance(source, pa.RecordBatch):
        return pa.Table.from_batches([source])
    # Otherwise treat it as a file path / file-like object.
    import pyarrow.parquet as pq  # noqa: WPS433

    return pq.read_table(source)


class HistogramColumn:
    """Describes a histogram metric discovered in an Arrow schema."""

    __slots__ = ("name", "kind", "config")

    def __init__(self, name: str, kind: str, config: Optional[Config]) -> None:
        #: The metric's base name (without the ``:buckets`` etc. suffix).
        self.name = name
        #: Either ``"standard"`` or ``"sparse"``.
        self.kind = kind
        #: The :class:`Config` recorded in field metadata, or ``None`` if absent.
        self.config = config

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"HistogramColumn(name={self.name!r}, kind={self.kind!r})"


def _config_from_field_metadata(field) -> Optional[Config]:
    meta = field.metadata or {}
    if _META_GROUPING in meta and _META_MAX_POWER in meta:
        return Config.new(
            int(meta[_META_GROUPING]),
            int(meta[_META_MAX_POWER]),
        )
    return None


def histogram_columns(source) -> List[HistogramColumn]:
    """Return the histogram metrics present in ``source``.

    ``source`` may be a file path, a pyarrow ``Table``, ``RecordBatch``, or
    ``Schema``.
    """
    pa = _require_pyarrow()
    if isinstance(source, pa.Schema):
        schema = source
    else:
        schema = _read_table(source).schema

    field_names = set(schema.names)
    columns: List[HistogramColumn] = []
    for field in schema:
        name = field.name
        if name.endswith(_BUCKETS_SUFFIX):
            base = name[: -len(_BUCKETS_SUFFIX)]
            columns.append(
                HistogramColumn(base, "standard", _config_from_field_metadata(field))
            )
        elif name.endswith(_INDICES_SUFFIX):
            base = name[: -len(_INDICES_SUFFIX)]
            if base + _COUNTS_SUFFIX in field_names:
                columns.append(
                    HistogramColumn(base, "sparse", _config_from_field_metadata(field))
                )
    return columns


def _resolve_config(
    explicit: Optional[Config],
    field_config: Optional[Config],
    grouping_power: Optional[int],
    max_value_power: int,
    inferred_total_buckets: Optional[int],
) -> Config:
    if explicit is not None:
        return explicit
    if field_config is not None:
        return field_config
    if grouping_power is not None:
        return Config.new(grouping_power, max_value_power)
    if inferred_total_buckets is not None:
        # Dense columns let us recover grouping_power from the bucket count.
        return Config.from_total_buckets(inferred_total_buckets, max_value_power)
    return Config.new(DEFAULT_GROUPING_POWER, max_value_power)


def read_histograms(
    source,
    name: str,
    *,
    config: Optional[Config] = None,
    grouping_power: Optional[int] = None,
    max_value_power: int = DEFAULT_MAX_VALUE_POWER,
) -> List[Optional[Histogram]]:
    """Read the histogram time series for metric ``name``.

    Returns a list with one entry per row: a :class:`Histogram` for rows that
    recorded the metric, or ``None`` for rows where it was absent (a null cell).

    The bucketing config is resolved in this order: the explicit ``config``
    argument, then ``grouping_power``/``max_value_power`` metadata recorded in
    the field, then an explicit ``grouping_power`` argument, then (for dense
    columns only) inference from the bucket count, and finally the Rezolus
    defaults (``grouping_power=3``, ``max_value_power=64``).
    """
    table = _read_table(source)
    field_names = set(table.schema.names)

    if name + _BUCKETS_SUFFIX in field_names:
        return _read_standard(
            table, name, config, grouping_power, max_value_power
        )
    if name + _INDICES_SUFFIX in field_names and name + _COUNTS_SUFFIX in field_names:
        return _read_sparse(
            table, name, config, grouping_power, max_value_power
        )
    raise KeyError(f"no histogram metric named {name!r} found in source")


def _read_standard(
    table,
    name: str,
    config: Optional[Config],
    grouping_power: Optional[int],
    max_value_power: int,
) -> List[Optional[Histogram]]:
    column = table.column(name + _BUCKETS_SUFFIX)
    field = table.schema.field(name + _BUCKETS_SUFFIX)
    field_config = _config_from_field_metadata(field)

    rows = column.to_pylist()
    out: List[Optional[Histogram]] = []
    resolved: Optional[Config] = None
    for buckets in rows:
        if buckets is None:
            out.append(None)
            continue
        if resolved is None:
            resolved = _resolve_config(
                config, field_config, grouping_power, max_value_power, len(buckets)
            )
        out.append(Histogram.from_buckets(
            resolved.grouping_power, resolved.max_value_power, buckets
        ))
    return out


def _read_sparse(
    table,
    name: str,
    config: Optional[Config],
    grouping_power: Optional[int],
    max_value_power: int,
) -> List[Optional[Histogram]]:
    index_col = table.column(name + _INDICES_SUFFIX).to_pylist()
    count_col = table.column(name + _COUNTS_SUFFIX).to_pylist()
    field = table.schema.field(name + _INDICES_SUFFIX)
    field_config = _config_from_field_metadata(field)

    resolved = _resolve_config(
        config, field_config, grouping_power, max_value_power, None
    )

    out: List[Optional[Histogram]] = []
    for index, count in zip(index_col, count_col):
        if index is None or count is None:
            out.append(None)
            continue
        sparse = SparseHistogram.from_parts(resolved, index, count)
        out.append(sparse.to_dense())
    return out


def read_all_histograms(
    source,
    *,
    grouping_power: Optional[int] = None,
    max_value_power: int = DEFAULT_MAX_VALUE_POWER,
) -> Dict[str, List[Optional[Histogram]]]:
    """Read every histogram metric in ``source`` keyed by metric name."""
    table = _read_table(source)
    result: Dict[str, List[Optional[Histogram]]] = {}
    for column in histogram_columns(table):
        result[column.name] = read_histograms(
            table,
            column.name,
            grouping_power=grouping_power,
            max_value_power=max_value_power,
        )
    return result


# ----------------------------------------------------------------------
# Writing
# ----------------------------------------------------------------------
def _list_u64_type(pa):
    return pa.list_(pa.field("item", pa.uint64(), nullable=True))


def _field_metadata(kind_value: bytes, config: Config) -> Dict[bytes, bytes]:
    return {
        _META_TYPE: kind_value,
        _META_GROUPING: str(config.grouping_power).encode(),
        _META_MAX_POWER: str(config.max_value_power).encode(),
    }


def write_histograms(
    path,
    histograms: Mapping[str, Sequence[Optional[Histogram]]],
    *,
    timestamps: Optional[Sequence[int]] = None,
    durations: Optional[Sequence[int]] = None,
    histogram_type: str = "standard",
    metadata: Optional[Mapping[str, str]] = None,
) -> None:
    """Write histogram time series to a Parquet file in the Rezolus layout.

    ``histograms`` maps a metric name to a per-row sequence of
    :class:`Histogram` (or ``None`` for a missing row). Every sequence must have
    the same length -- the number of rows. ``histogram_type`` is ``"standard"``
    (dense ``:buckets`` column) or ``"sparse"`` (``:bucket_indices`` /
    ``:bucket_counts`` columns).

    The bucketing parameters are recorded in each field's metadata so the file
    is self-describing; readers that ignore that metadata (such as Rezolus) are
    unaffected.
    """
    if histogram_type not in ("standard", "sparse"):
        raise ValueError("histogram_type must be 'standard' or 'sparse'")

    pa = _require_pyarrow()
    import pyarrow.parquet as pq  # noqa: WPS433

    n_rows = _validate_row_counts(histograms, timestamps, durations)

    list_type = _list_u64_type(pa)

    fields = [
        pa.field("timestamp", pa.uint64(), nullable=False),
        pa.field("duration", pa.uint64(), nullable=True),
    ]
    arrays = [
        pa.array(
            timestamps if timestamps is not None else list(range(n_rows)),
            type=pa.uint64(),
        ),
        pa.array(
            durations if durations is not None else [0] * n_rows,
            type=pa.uint64(),
        ),
    ]

    for name, series in histograms.items():
        config = _series_config(series)
        if histogram_type == "standard":
            meta = _field_metadata(b"histogram", config)
            fields.append(
                pa.field(name + _BUCKETS_SUFFIX, list_type, nullable=True, metadata=meta)
            )
            arrays.append(
                pa.array(
                    [None if h is None else list(h.buckets) for h in series],
                    type=list_type,
                )
            )
        else:
            meta = _field_metadata(b"sparse_histogram", config)
            indices = []
            counts = []
            for h in series:
                if h is None:
                    indices.append(None)
                    counts.append(None)
                else:
                    sparse = h.to_sparse()
                    indices.append(list(sparse.index))
                    counts.append(list(sparse.count))
            fields.append(
                pa.field(name + _INDICES_SUFFIX, list_type, nullable=True, metadata=meta)
            )
            arrays.append(pa.array(indices, type=list_type))
            fields.append(
                pa.field(name + _COUNTS_SUFFIX, list_type, nullable=True, metadata=meta)
            )
            arrays.append(pa.array(counts, type=list_type))

    schema_metadata = None
    if metadata:
        schema_metadata = {str(k): str(v) for k, v in metadata.items()}
    schema = pa.schema(fields, metadata=schema_metadata)
    table = pa.Table.from_arrays(arrays, schema=schema)
    pq.write_table(table, path)


def _validate_row_counts(histograms, timestamps, durations) -> int:
    lengths = {len(series) for series in histograms.values()}
    if timestamps is not None:
        lengths.add(len(timestamps))
    if durations is not None:
        lengths.add(len(durations))
    if not lengths:
        raise ValueError("no histograms to write")
    if len(lengths) != 1:
        raise ValueError("all series (and timestamps/durations) must have equal length")
    return lengths.pop()


def _series_config(series: Sequence[Optional[Histogram]]) -> Config:
    config: Optional[Config] = None
    for h in series:
        if h is None:
            continue
        if config is None:
            config = h.config
        elif h.config != config:
            raise ValueError("all histograms for a metric must share a config")
    if config is None:
        # Entirely-null series: fall back to Rezolus defaults so the file is
        # still well-formed.
        return Config.new(DEFAULT_GROUPING_POWER, DEFAULT_MAX_VALUE_POWER)
    return config
