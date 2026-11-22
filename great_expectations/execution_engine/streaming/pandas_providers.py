"""Streaming adapters for the built-in Pandas metrics used by the supported rules.

Each adapter implements the three-method streaming protocol.  Critically, the adapters do
**not** re-implement the rule predicates or the numeric kernels: they look up the already
registered Pandas providers (``ColumnValuesBetween._pandas``, ``ColumnValuesNonNull._pandas``,
``ColumnMean._pandas``, ``ColumnStandardDeviation._pandas``) and call the unwrapped kernels.
This keeps a single source of truth for boundary/null semantics.

The adapters are registered by exact metric name in the
:class:`~great_expectations.execution_engine.streaming.protocol.StreamingAdapterRegistry`,
so the Validator never branches on expectation type -- it asks the metric registry which
metrics are streamable.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Dict, List, MutableMapping, Optional

import numpy as np
import pandas as pd

from great_expectations.core.metric_function_types import (
    SummarizationMetricNameSuffixes,
)
from great_expectations.exceptions import StreamingMetricDataError
from great_expectations.execution_engine.streaming.protocol import (
    register_streaming_adapter,
)

if TYPE_CHECKING:
    from great_expectations.execution_engine.execution_engine import ExecutionEngine
    from great_expectations.validator.metric_configuration import MetricConfiguration


_CONDITION_VALUE_KEYS = ("min_value", "max_value", "strict_min", "strict_max")
# Sample standard deviation needs at least two observations.
_MIN_STD_SAMPLE = 2


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _column_name(metric: MetricConfiguration) -> str:
    return metric.metric_domain_kwargs.get("column")


def _get_column(dataframe: pd.DataFrame, metric: MetricConfiguration) -> pd.Series:
    name = _column_name(metric)
    if name is None:
        raise StreamingMetricDataError(  # noqa: TRY003
            "Streaming column metric requires a 'column' in its domain kwargs.",
            batch_id=metric.metric_domain_kwargs.get("batch_id"),
        )
    if name not in dataframe.columns:
        raise StreamingMetricDataError(  # noqa: TRY003
            f"Column {name!r} is not present in the chunk.",
            batch_id=metric.metric_domain_kwargs.get("batch_id"),
            column=name,
        )
    return dataframe[name]


def _kernel(metric_name: str, execution_engine: ExecutionEngine):
    from great_expectations.expectations.registry import get_metric_provider

    provider_class, provider_fn = get_metric_provider(metric_name, execution_engine)
    return provider_class, inspect.unwrap(provider_fn)


def _sample_limit(metric: MetricConfiguration, default: int = 20) -> int:
    result_format = metric.metric_value_kwargs.get("result_format") or {}
    try:
        return max(int(result_format.get("partial_unexpected_count", default)), 0)
    except (TypeError, ValueError):
        return default


class UnexpectedIndexSample(list):
    """A list of original index labels that also carries 0-based full-stream positions.

    ``_format_map_output`` slices the value down to ``partial_unexpected_count``; slicing
    preserves the parallel positions so the streaming result builder can report them
    without re-scanning.
    """

    positions: List[int]

    def __init__(self, labels: List[Any], positions: List[int]) -> None:
        super().__init__(labels)
        self.positions = list(positions)

    def __getitem__(self, item):  # type: ignore[override] # FIXME
        sliced = super().__getitem__(item)
        if isinstance(item, slice):
            return UnexpectedIndexSample(list(sliced), self.positions[item])
        return sliced


class UnexpectedValuesSample(list):
    """Unexpected values carrying their parallel 0-based full-stream positions.

    Used for BASIC, where the index-list metric is not requested but positions are still
    reported alongside ``partial_unexpected_list``.
    """

    positions: List[int]

    def __init__(self, values: List[Any], positions: List[int]) -> None:
        super().__init__(values)
        self.positions = list(positions)

    def __getitem__(self, item):  # type: ignore[override] # FIXME
        sliced = super().__getitem__(item)
        if isinstance(item, slice):
            return UnexpectedValuesSample(list(sliced), self.positions[item])
        return sliced


# --------------------------------------------------------------------------------------
# table.row_count
# --------------------------------------------------------------------------------------


class RowCountStreamingAdapter:
    metric_name = "table.row_count"
    streaming_supported = True

    @classmethod
    def create_streaming_state(cls, *, metric, execution_engine) -> Dict[str, int]:
        return {"count": 0}

    @classmethod
    def accumulate_streaming_state(  # noqa: PLR0913
        cls, *, state, metric, dataframe, execution_engine, chunk_index, row_offset, scratch=None
    ) -> None:
        state["count"] += int(dataframe.shape[0])

    @classmethod
    def finalize_streaming_state(cls, *, state, metric, execution_engine) -> int:
        return state["count"]


# --------------------------------------------------------------------------------------
# table schema metadata (declared by the first chunk)
# --------------------------------------------------------------------------------------


class _SchemaState:
    __slots__ = ("columns", "dtypes")

    def __init__(self) -> None:
        self.columns: Optional[List[Any]] = None
        self.dtypes: Optional[List[Any]] = None


class _SchemaStreamingAdapter:
    streaming_supported = True

    @classmethod
    def create_streaming_state(cls, *, metric, execution_engine) -> _SchemaState:
        return _SchemaState()

    @classmethod
    def accumulate_streaming_state(  # noqa: PLR0913
        cls, *, state, metric, dataframe, execution_engine, chunk_index, row_offset, scratch=None
    ) -> None:
        if chunk_index == 0:
            state.columns = list(dataframe.columns)
            state.dtypes = list(dataframe.dtypes)

    @classmethod
    def finalize_streaming_state(cls, *, state, metric, execution_engine):
        raise NotImplementedError


class TableColumnsStreamingAdapter(_SchemaStreamingAdapter):
    @classmethod
    def finalize_streaming_state(cls, *, state, metric, execution_engine):
        return list(state.columns or [])


class TableColumnTypesStreamingAdapter(_SchemaStreamingAdapter):
    @classmethod
    def finalize_streaming_state(cls, *, state, metric, execution_engine):
        return [
            {"name": name, "type": dtype}
            for name, dtype in zip(state.columns or [], state.dtypes or [], strict=False)
        ]


# --------------------------------------------------------------------------------------
# numeric column aggregates: column.mean / column.standard_deviation
# --------------------------------------------------------------------------------------


def _numeric_non_missing(
    column: pd.Series,
    *,
    metric: MetricConfiguration,
    chunk_index: int,
) -> pd.Series:
    """Return the non-missing values of a real-numeric column as float64.

    Aggregatable numeric columns are limited to finite real numbers and missing values.
    Booleans, complex, datetimes, objects holding non-numbers and non-finite floats all
    raise a locatable :class:`StreamingMetricDataError`.
    """
    batch_id = metric.metric_domain_kwargs.get("batch_id")
    name = _column_name(metric)

    dtype = column.dtype
    if pd.api.types.is_bool_dtype(dtype) or not pd.api.types.is_numeric_dtype(dtype):
        raise StreamingMetricDataError(  # noqa: TRY003
            f"Streaming numeric aggregate on column {name!r} requires a numeric dtype, "
            f"got {dtype!r}.",
            batch_id=batch_id,
            column=name,
            chunk_index=chunk_index,
        )

    non_missing = column[~column.isna()]
    if len(non_missing) == 0:
        return pd.Series([], dtype="float64", name=column.name)
    try:
        float_values = non_missing.to_numpy(dtype="float64", copy=True)
    except (TypeError, ValueError) as exc:
        raise StreamingMetricDataError(  # noqa: TRY003
            f"Column {name!r} cannot be interpreted as real numbers (dtype {dtype!r}).",
            batch_id=batch_id,
            column=name,
            chunk_index=chunk_index,
        ) from exc

    finite = np.isfinite(float_values)
    if not bool(finite.all()):
        first_bad = float_values[~finite][0]
        raise StreamingMetricDataError(  # noqa: TRY003
            f"Column {name!r} contains a non-finite value ({first_bad!r}); streaming numeric "
            "aggregates require finite real numbers or missing values.",
            batch_id=batch_id,
            column=name,
            chunk_index=chunk_index,
        )
    return pd.Series(float_values, name=column.name, copy=False)


class _MeanState:
    __slots__ = ("count", "mean")

    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0


class MeanStreamingAdapter:
    metric_name = "column.mean"
    streaming_supported = True

    @classmethod
    def create_streaming_state(cls, *, metric, execution_engine):
        return _MeanState()

    @classmethod
    def accumulate_streaming_state(  # noqa: PLR0913
        cls, *, state, metric, dataframe, execution_engine, chunk_index, row_offset, scratch=None
    ) -> None:
        series = _numeric_non_missing(
            _get_column(dataframe, metric), metric=metric, chunk_index=chunk_index
        )
        n = len(series)
        if n == 0:
            return
        provider_class, kernel = _kernel("column.mean", execution_engine)
        chunk_mean = float(kernel(provider_class, column=series))
        total = state.count + n
        state.mean = (state.mean * state.count + chunk_mean * n) / total
        state.count = total

    @classmethod
    def finalize_streaming_state(cls, *, state, metric, execution_engine):
        if state.count == 0:
            return float("nan")
        return state.mean


class _StdState:
    __slots__ = ("count", "m2", "mean")

    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0


class StdevStreamingAdapter:
    """Whole-batch sample standard deviation (ddof=1), matching ``Series.std()``."""

    metric_name = "column.standard_deviation"
    streaming_supported = True

    @classmethod
    def create_streaming_state(cls, *, metric, execution_engine):
        return _StdState()

    @classmethod
    def accumulate_streaming_state(  # noqa: PLR0913
        cls, *, state, metric, dataframe, execution_engine, chunk_index, row_offset, scratch=None
    ) -> None:
        series = _numeric_non_missing(
            _get_column(dataframe, metric), metric=metric, chunk_index=chunk_index
        )
        n_b = len(series)
        if n_b == 0:
            return
        provider_class, kernel = _kernel("column.standard_deviation", execution_engine)
        chunk_std = float(kernel(provider_class, column=series))
        chunk_mean = float(series.mean())
        chunk_m2 = 0.0 if n_b < _MIN_STD_SAMPLE else chunk_std * chunk_std * (n_b - 1)
        n_a = state.count
        if n_a == 0:
            state.count, state.mean, state.m2 = n_b, chunk_mean, chunk_m2
            return
        n = n_a + n_b
        delta = chunk_mean - state.mean
        state.m2 = state.m2 + chunk_m2 + delta * delta * n_a * n_b / n
        state.mean = state.mean + delta * n_b / n
        state.count = n

    @classmethod
    def finalize_streaming_state(cls, *, state, metric, execution_engine):
        if state.count < _MIN_STD_SAMPLE:
            return float("nan")
        return float(np.sqrt(state.m2 / (state.count - 1)))


# --------------------------------------------------------------------------------------
# map conditions: column_values.<condition>
# --------------------------------------------------------------------------------------


class _MapConditionState:
    __slots__ = (
        "condition_key",
        "element_count",
        "filter_column_isnull",
        "missing_count",
        "sample_limit",
        "unexpected_count",
        "unexpected_labels",
        "unexpected_positions",
        "unexpected_values",
    )

    def __init__(self, sample_limit: int, condition_key: str, filter_column_isnull: bool):
        self.element_count = 0
        self.missing_count = 0
        self.unexpected_count = 0
        self.sample_limit = sample_limit
        self.unexpected_values: List[Any] = []
        self.unexpected_labels: List[Any] = []
        self.unexpected_positions: List[int] = []
        self.condition_key = condition_key
        self.filter_column_isnull = filter_column_isnull


class MapConditionStreamingAdapter:
    """Generic adapter for ``column_values.<condition>.condition`` and derived metrics.

    ``base_metric_name`` is set on the concrete adapter classes registered below.  The
    predicate is fetched from the registered Pandas provider for
    ``<base_metric_name>.condition`` and reused verbatim (including its dtype branches).
    """

    streaming_supported = True
    base_metric_name: str = ""

    @classmethod
    def _condition_metric_name(cls) -> str:
        return f"{cls.base_metric_name}.condition"

    @classmethod
    def _condition_key(cls, metric: MetricConfiguration) -> str:
        """Cache key for the per-chunk condition evaluation.

        Must include the column: two metrics for the same predicate on different columns
        evaluate different booleans within one chunk and must not share a cache entry.
        """
        column = metric.metric_domain_kwargs.get("column")
        value_kwargs = sorted(
            (k, metric.metric_value_kwargs.get(k))
            for k in _CONDITION_VALUE_KEYS
            if k in metric.metric_value_kwargs
        )
        return f"{cls.base_metric_name}|{column!r}|{value_kwargs!r}"

    @classmethod
    def create_streaming_state(cls, *, metric, execution_engine):
        condition_provider, _ = _kernel(cls._condition_metric_name(), execution_engine)
        filter_column_isnull = bool(
            getattr(condition_provider, "filter_column_isnull", False)
        )
        return _MapConditionState(
            sample_limit=_sample_limit(metric),
            condition_key=cls._condition_key(metric),
            filter_column_isnull=filter_column_isnull,
        )

    @classmethod
    def _meets_condition(  # noqa: PLR0913
        cls,
        *,
        state: _MapConditionState,
        metric: MetricConfiguration,
        dataframe: pd.DataFrame,
        execution_engine: ExecutionEngine,
        chunk_index: int,
        scratch: MutableMapping[str, Any],
    ) -> np.ndarray:
        """Boolean ndarray aligned to ``dataframe`` rows: True == meets the condition.

        The result is cached per chunk under the condition key so count/values/index_list
        for the same condition share one evaluation.
        """
        if state.condition_key in scratch:
            return scratch[state.condition_key]

        provider_class, kernel = _kernel(cls._condition_metric_name(), execution_engine)
        column = _get_column(dataframe, metric)
        value_kwargs = {
            k: metric.metric_value_kwargs.get(k)
            for k in _CONDITION_VALUE_KEYS
            if k in metric.metric_value_kwargs
        }

        try:
            if state.filter_column_isnull:
                # The provider's decorator normally drops nulls before calling the kernel;
                # replicate that here (object-column predicates raise on None otherwise).
                non_null = column.notnull().to_numpy()
                evaluated = kernel(
                    provider_class, column[non_null], **value_kwargs, _metrics={}
                )
                evaluated_arr = np.asarray(evaluated, dtype=bool)
                if len(evaluated_arr) != int(non_null.sum()):
                    raise ValueError(  # noqa: TRY003, TRY301
                        "condition kernel returned a result of unexpected length"
                    )
                meets = np.zeros(len(dataframe), dtype=bool)
                meets[non_null] = evaluated_arr
            else:
                evaluated = kernel(provider_class, column, **value_kwargs, _metrics={})
                meets = np.asarray(evaluated, dtype=bool)
                if len(meets) != len(dataframe):
                    raise ValueError(  # noqa: TRY003, TRY301
                        "condition kernel returned a result of unexpected length"
                    )
        except StreamingMetricDataError:
            raise
        except Exception as exc:
            raise StreamingMetricDataError(  # noqa: TRY003
                f"Failed to evaluate condition {cls._condition_metric_name()!r}: {exc}",
                batch_id=metric.metric_domain_kwargs.get("batch_id"),
                column=_column_name(metric),
                chunk_index=chunk_index,
            ) from exc

        scratch[state.condition_key] = meets
        return meets

    @classmethod
    def accumulate_streaming_state(  # noqa: PLR0913
        cls, *, state, metric, dataframe, execution_engine, chunk_index, row_offset, scratch=None
    ) -> None:
        if scratch is None:
            scratch = {}
        column = _get_column(dataframe, metric)
        column_missing = column.isna().to_numpy()
        meets = cls._meets_condition(
            state=state,
            metric=metric,
            dataframe=dataframe,
            execution_engine=execution_engine,
            chunk_index=chunk_index,
            scratch=scratch,
        )

        state.element_count += len(dataframe)
        if state.filter_column_isnull:
            considered = ~column_missing
            state.missing_count += int(column_missing.sum())
        else:
            considered = np.ones(len(dataframe), dtype=bool)

        unexpected_mask = considered & ~meets
        chunk_unexpected = int(unexpected_mask.sum())
        state.unexpected_count += chunk_unexpected

        if chunk_unexpected == 0 or len(state.unexpected_values) >= state.sample_limit:
            return
        positions_in_chunk = np.flatnonzero(unexpected_mask)
        remaining = state.sample_limit - len(state.unexpected_values)
        selected = positions_in_chunk[:remaining]
        state.unexpected_values.extend(column.to_numpy()[selected].tolist())
        state.unexpected_labels.extend(dataframe.index.to_numpy()[selected].tolist())
        state.unexpected_positions.extend((row_offset + selected).tolist())


class _BetweenConditionAdapter(MapConditionStreamingAdapter):
    base_metric_name = "column_values.between"

    @classmethod
    def finalize_streaming_state(cls, *, state, metric, execution_engine):
        # Internal condition metric; its raw value is never surfaced to an expectation.
        return None


class _NonNullConditionAdapter(MapConditionStreamingAdapter):
    base_metric_name = "column_values.nonnull"

    @classmethod
    def finalize_streaming_state(cls, *, state, metric, execution_engine):
        return None


class _UnexpectedCountAdapter:
    @classmethod
    def finalize_streaming_state(cls, *, state, metric, execution_engine):
        return state.unexpected_count


class _UnexpectedValuesAdapter:
    @classmethod
    def finalize_streaming_state(cls, *, state, metric, execution_engine):
        return UnexpectedValuesSample(
            list(state.unexpected_values), list(state.unexpected_positions)
        )


class _UnexpectedIndexListAdapter:
    @classmethod
    def finalize_streaming_state(cls, *, state, metric, execution_engine):
        return UnexpectedIndexSample(
            list(state.unexpected_labels), list(state.unexpected_positions)
        )


class _ConditionValueAdapter:
    @classmethod
    def finalize_streaming_state(cls, *, state, metric, execution_engine):
        # Internal condition metric; its raw value is never surfaced to an expectation.
        return None


# --------------------------------------------------------------------------------------
# registration
# --------------------------------------------------------------------------------------

_MAP_ADAPTERS = {
    "column_values.between": _BetweenConditionAdapter,
    "column_values.nonnull": _NonNullConditionAdapter,
}


def _derived(base_condition_adapter, finalize_adapter):
    """Build a derived adapter that inherits accumulation from the condition adapter but
    takes ``finalize_streaming_state`` from ``finalize_adapter`` explicitly (so the order
    is independent of MRO quirks)."""
    return type(
        f"{base_condition_adapter.__name__}__{finalize_adapter.__name__}",
        (finalize_adapter, base_condition_adapter),
        {
            "streaming_supported": True,
            "base_metric_name": base_condition_adapter.base_metric_name,
        },
    )


def register_builtin_streaming_adapters() -> None:
    register_streaming_adapter("table.row_count", RowCountStreamingAdapter)
    register_streaming_adapter("table.columns", TableColumnsStreamingAdapter)
    register_streaming_adapter("table.column_types", TableColumnTypesStreamingAdapter)
    register_streaming_adapter("column.mean", MeanStreamingAdapter)
    register_streaming_adapter("column.standard_deviation", StdevStreamingAdapter)

    for base, condition_adapter in _MAP_ADAPTERS.items():
        register_streaming_adapter(f"{base}.condition", condition_adapter)
        register_streaming_adapter(
            f"{base}.{SummarizationMetricNameSuffixes.UNEXPECTED_COUNT.value}",
            _derived(condition_adapter, _UnexpectedCountAdapter),
        )
        register_streaming_adapter(
            f"{base}.{SummarizationMetricNameSuffixes.UNEXPECTED_VALUES.value}",
            _derived(condition_adapter, _UnexpectedValuesAdapter),
        )
        register_streaming_adapter(
            f"{base}.{SummarizationMetricNameSuffixes.UNEXPECTED_INDEX_LIST.value}",
            _derived(condition_adapter, _UnexpectedIndexListAdapter),
        )
        register_streaming_adapter(
            f"{base}.{SummarizationMetricNameSuffixes.UNEXPECTED_INDEX_QUERY.value}",
            _derived(condition_adapter, _ConditionValueAdapter),
        )


register_builtin_streaming_adapters()


__all__ = [
    "MapConditionStreamingAdapter",
    "MeanStreamingAdapter",
    "RowCountStreamingAdapter",
    "StdevStreamingAdapter",
    "UnexpectedIndexSample",
    "register_builtin_streaming_adapters",
]
