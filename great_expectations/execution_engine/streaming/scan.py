from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from great_expectations.exceptions import (
    StreamBatchStructureError,
    StreamPreSourceValidationError,
)
from great_expectations.execution_engine.streaming.protocol import (
    StreamingAdapter,
    StreamingAdapterRegistry,
    provider_declares_streaming,
)

if TYPE_CHECKING:
    from great_expectations.execution_engine.execution_engine import ExecutionEngine
    from great_expectations.validator.metric_configuration import MetricConfiguration

logger = logging.getLogger(__name__)


class _MetricStreamState:
    """Per-metric-identity accumulation bookkeeping for one scan."""

    __slots__ = ("adapter", "configuration", "state")

    def __init__(
        self,
        adapter: StreamingAdapter,
        configuration: MetricConfiguration,
        state: Any,
    ) -> None:
        self.adapter = adapter
        self.configuration = configuration
        self.state = state


class StreamScanResult:
    """Outcome of one pass over the chunk source.

    Attributes:
        resolved_metrics: metric id -> value for metrics that finished successfully.
        failures: metric id -> (configuration, exception) for metric-level failures.
        row_count: rows accumulated (only meaningful on a fully consumed stream).
        chunk_count: number of chunks the source yielded.
        columns: column names declared by the first chunk (or ``None`` if there was none).
        dtypes: dtypes declared by the first chunk keyed by name (or ``None``).
    """

    def __init__(self) -> None:
        self.resolved_metrics: Dict[Any, Any] = {}
        self.failures: Dict[Any, Tuple[MetricConfiguration, BaseException]] = {}
        self.row_count = 0
        self.chunk_count = 0
        self.columns: Optional[Tuple[Any, ...]] = None
        self.dtypes: Optional[Tuple[Any, ...]] = None


def resolve_streaming_adapter(
    metric_configuration: MetricConfiguration,
    execution_engine: ExecutionEngine,
) -> Optional[StreamingAdapter]:
    """Return a streaming adapter for ``metric_configuration`` or ``None``.

    Resolution order:

    1. An exact-name adapter registered via
       :func:`great_expectations.execution_engine.streaming.protocol.register_streaming_adapter`.
    2. The metric's own provider class when it declares ``streaming_supported`` (a
       :class:`StreamingMetricProvider` subclass, typically custom).

    A registered adapter takes precedence so that built-in metrics can be adapted without
    editing their provider classes.
    """
    from great_expectations.expectations.registry import get_metric_provider

    metric_name = metric_configuration.metric_name
    adapter = StreamingAdapterRegistry.get(metric_name)
    if adapter is None:
        # Ensure the built-in adapters are registered even if the submodule was imported
        # directly (before the streaming package ``__init__`` ran).
        from great_expectations.execution_engine.streaming.pandas_providers import (
            register_builtin_streaming_adapters,
        )

        register_builtin_streaming_adapters()
        adapter = StreamingAdapterRegistry.get(metric_name)
    if adapter is not None:
        return adapter

    # Derived summarisation metrics (e.g. ``column_values.nonnull.unexpected_count``) are
    # not registered themselves on some engines; the built-in adapters pre-register them.
    try:
        provider_class, _provider_fn = get_metric_provider(metric_name, execution_engine)
    except Exception:
        return None

    if provider_declares_streaming(provider_class):
        return provider_class
    return None


def stream_capable_metric_names(execution_engine: ExecutionEngine) -> set[str]:
    """All metric names that can be streamed (registered adapters + custom providers)."""
    from great_expectations.expectations.registry import _registered_metrics

    names = set(StreamingAdapterRegistry._registry)
    for metric_name, definition in _registered_metrics.items():
        providers = definition.get("providers", {})
        provider_tuple = providers.get(type(execution_engine).__name__)
        if provider_tuple is None:
            continue
        provider_class = provider_tuple[0]
        if provider_declares_streaming(provider_class):
            names.add(metric_name)
    return names


def required_streaming_metrics(
    metric_configurations: Sequence[MetricConfiguration],
    execution_engine: ExecutionEngine,
) -> Tuple[Dict[Any, _MetricStreamState], List[str]]:
    """Classify the graph's metric configurations into streamable and unsupported.

    Returns:
        (streamable, unsupported_names): ``streamable`` maps metric id -> a freshly created
        :class:`_MetricStreamState`; ``unsupported_names`` lists metric names that cannot be
        computed on the streaming path.  The caller rejects *before* touching the source if
        ``unsupported_names`` is non-empty.
    """
    from great_expectations.validator.metric_configuration import (
        MetricConfiguration,
    )

    streamable: Dict[Any, _MetricStreamState] = {}
    unsupported: List[str] = []
    for metric_configuration in metric_configurations:
        if not isinstance(metric_configuration, MetricConfiguration):
            continue
        if metric_configuration.id in streamable:
            continue
        adapter = resolve_streaming_adapter(metric_configuration, execution_engine)
        if adapter is None:
            unsupported.append(metric_configuration.metric_name)
            continue
        state = adapter.create_streaming_state(
            metric=metric_configuration,
            execution_engine=execution_engine,
        )
        streamable[metric_configuration.id] = _MetricStreamState(
            adapter=adapter,
            configuration=metric_configuration,
            state=state,
        )
    return streamable, sorted(set(unsupported))


def _check_chunk(
    chunk: Any,
    chunk_index: int,
    expected_columns: Optional[Tuple[Any, ...]],
    expected_dtypes: Optional[Tuple[Any, ...]],
) -> Tuple[Tuple[Any, ...], Tuple[Any, ...]]:
    """Validate one chunk against the batch structure contract.

    Returns the (columns, dtypes) signature when this is the first chunk.

    Raises:
        StreamBatchStructureError: if the chunk is not a DataFrame, changes columns or
            column order, or changes a dtype.
    """
    if not isinstance(chunk, pd.DataFrame):
        raise StreamBatchStructureError(  # noqa: TRY003
            f"expected a pandas.DataFrame chunk, got {type(chunk).__name__}",
            chunk_index=chunk_index,
        )

    columns = tuple(chunk.columns)
    dtypes = tuple(str(dtype) for dtype in chunk.dtypes)

    if chunk_index == 0:
        # The first chunk declares the schema; zero rows are allowed.
        return columns, dtypes

    assert expected_columns is not None and expected_dtypes is not None
    if columns != expected_columns:
        raise StreamBatchStructureError(  # noqa: TRY003
            f"column order/names changed from {list(expected_columns)!r} to "
            f"{list(columns)!r}",
            chunk_index=chunk_index,
        )
    if dtypes != expected_dtypes:
        changed = [
            f"{name}: {old} -> {new}"
            for name, old, new in zip(expected_columns, expected_dtypes, dtypes, strict=True)
            if old != new
        ]
        raise StreamBatchStructureError(  # noqa: TRY003
            f"dtype changed ({'; '.join(changed)})",
            chunk_index=chunk_index,
        )
    return columns, dtypes


def _close_iterator(iterator: Any) -> None:
    close = getattr(iterator, "close", None)
    if callable(close):
        close()


def _safe_close_iterator(iterator: Any) -> None:
    """Close ``iterator`` if supported, never masking an in-flight exception."""
    import sys

    try:
        _close_iterator(iterator)
    except Exception:
        # If we are already unwinding because of an error, the original error must win.
        if sys.exc_info()[0] is None:
            raise
        logger.debug("Iterator close() raised while handling another exception", exc_info=True)


def scan_dataframe_chunks(  # noqa: C901, PLR0912
    *,
    source: Any,
    streamable: Dict[Any, _MetricStreamState],
    execution_engine: ExecutionEngine,
) -> StreamScanResult:
    """Consume ``source`` exactly once, folding every chunk into every metric state.

    Source errors and structure errors propagate unchanged; they are batch/source errors,
    never routed through ``catch_exceptions``.  Metric-level exceptions raised by an
    adapter are recorded in the result and do not stop the scan -- their state is simply
    finalized as failed.

    The iterator is always closed if the source's iterator supports ``close``, on both
    normal completion and failure; a close error never masks the original error.
    """
    result = StreamScanResult()
    iterator = source.acquire_iterator()
    failed_ids: set[Any] = set()
    row_offset = 0
    chunk_index = 0
    expected_columns: Optional[Tuple[Any, ...]] = None
    expected_dtypes: Optional[Tuple[Any, ...]] = None

    try:
        for chunk in iterator:
            expected_columns, expected_dtypes = _check_chunk(
                chunk, chunk_index, expected_columns, expected_dtypes
            )
            if chunk_index == 0:
                result.columns = expected_columns
                result.dtypes = expected_dtypes
            elif len(chunk) == 0:
                # A later empty chunk is structurally fine (same schema) and contributes
                # nothing, including no rows.
                pass

            # Per-chunk cache so derived metrics (e.g. unexpected_count built on a base
            # condition) can reuse work another adapter already did for this chunk.
            scratch: Dict[str, Any] = {}
            for metric_id, metric_state in streamable.items():
                if metric_id in failed_ids:
                    continue
                try:
                    metric_state.adapter.accumulate_streaming_state(
                        state=metric_state.state,
                        metric=metric_state.configuration,
                        dataframe=chunk,
                        execution_engine=execution_engine,
                        chunk_index=chunk_index,
                        row_offset=row_offset,
                        scratch=scratch,
                    )
                except StreamBatchStructureError:
                    raise
                except StreamPreSourceValidationError:
                    raise
                except Exception as exc:
                    result.failures[metric_id] = (metric_state.configuration, exc)
                    failed_ids.add(metric_id)

            row_offset += len(chunk)
            chunk_index += 1

        result.row_count = row_offset
        result.chunk_count = chunk_index
    finally:
        _safe_close_iterator(iterator)

    if result.chunk_count == 0:
        raise StreamBatchStructureError(  # noqa: TRY003
            "the chunk source yielded no chunks; a streaming batch requires at least one "
            "chunk (a legitimate empty batch is a single zero-row chunk, not an empty stream)"
        )

    # Finalize the metrics that never failed mid-stream.
    for metric_id, metric_state in streamable.items():
        if metric_id in result.failures:
            continue
        try:
            result.resolved_metrics[metric_id] = (
                metric_state.adapter.finalize_streaming_state(
                    state=metric_state.state,
                    metric=metric_state.configuration,
                    execution_engine=execution_engine,
                )
            )
        except Exception as exc:
            result.failures[metric_id] = (metric_state.configuration, exc)

    return result


__all__ = [
    "StreamScanResult",
    "required_streaming_metrics",
    "resolve_streaming_adapter",
    "scan_dataframe_chunks",
    "stream_capable_metric_names",
]
