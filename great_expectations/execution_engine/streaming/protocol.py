"""Streaming provider protocol for dataframe-chunk validation.

A dataframe-chunk *stream* is a synchronous iterable that yields :class:`pandas.DataFrame`
objects.  Each chunk is a fragment of one logical batch: row counts, missing rates and
numeric statistics must be interpreted over the concatenation of all chunks, never per
chunk.  The source is one-shot -- it is only guaranteed to be consumable once -- and is
never concatenated back into a single ``DataFrame`` by this code.

Capability declaration
======================

A metric provider opts in to the streaming path by declaring
``streaming_supported = True`` and implementing three classmethods (see
:class:`StreamingMetricProvider`):

``create_streaming_state``
    Build the (bounded) accumulator for one ``MetricConfiguration``.  This must not read
    any data.
``accumulate_streaming_state``
    Fold one chunk into the accumulator.  Called once per chunk in source order.
``finalize_streaming_state``
    Turn the accumulator into the metric value reported for the whole logical batch.

The same three methods are consulted for built-in Pandas providers through the
:mod:`great_expectations.execution_engine.streaming.pandas_providers` adapters, which
reuse the *existing* Pandas aggregate and map condition kernels rather than duplicating
the predicates.

State protocol guarantees
=========================

* Accumulator state is created at most once per metric identity (``MetricConfiguration.id``)
  per validation run and lives only for that run -- it is never persisted on the
  ``BatchDefinition``, the suite, or a datasource.
* ``accumulate_streaming_state`` sees chunks in source order.  It must only retain
  quantities whose size is bounded by the metric's definition (counts, sums, and a
  bounded diagnostic sample), never the whole stream.
* A provider may raise an ordinary (metric) exception from ``accumulate`` or
  ``finalize``; those are recorded per metric identity and surfaced through the normal
  ``catch_exceptions`` machinery.  Errors raised while *reading* the source or while
  checking the chunk structure are never routed through this protocol.

Sharing a scan
==============

All metrics required by a validation run are accumulated during a single pass over the
source.  A provider that implements this protocol participates automatically: it is
discovered from the metric registry (``streaming_supported``) and does not require any
Validator dispatch keyed on expectation type.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Dict, MutableMapping, Optional, Type

from great_expectations.expectations.metrics.metric_provider import MetricProvider

if TYPE_CHECKING:
    import pandas as pd

    from great_expectations.execution_engine.execution_engine import ExecutionEngine
    from great_expectations.validator.metric_configuration import MetricConfiguration


class StreamingMetricProvider(MetricProvider):
    """Optional mixin for providers that can compute a metric incrementally over chunks.

    Subclasses keep the normal ``MetricProvider`` registration (``metric_name``,
    ``domain_keys`` ...), so the metric remains usable on ordinary whole-DataFrame
    batches.  The three streaming classmethods below are only invoked by the chunk scan.
    """

    streaming_supported: ClassVar[bool] = True

    @classmethod
    def create_streaming_state(
        cls,
        *,
        metric: MetricConfiguration,
        execution_engine: ExecutionEngine,
    ) -> Any:
        """Create an accumulator for ``metric``; must not read data."""
        raise NotImplementedError

    @classmethod
    def accumulate_streaming_state(  # noqa: PLR0913
        cls,
        *,
        state: Any,
        metric: MetricConfiguration,
        dataframe: pd.DataFrame,
        execution_engine: ExecutionEngine,
        chunk_index: int,
        row_offset: int,
        scratch: MutableMapping[str, Any] | None = None,
    ) -> None:
        """Fold ``dataframe`` into ``state``.

        ``row_offset`` is the number of rows in earlier chunks, so a provider can record
        a stable 0-based position for a row without materialising the prefix.

        ``scratch`` is a per-chunk, per-scan cache shared by adapters; it is cleared for
        every chunk and released when the scan ends.  Derived metrics use it to reuse a
        base condition that another adapter already evaluated for the same chunk.
        """
        raise NotImplementedError

    @classmethod
    def finalize_streaming_state(
        cls,
        *,
        state: Any,
        metric: MetricConfiguration,
        execution_engine: ExecutionEngine,
    ) -> Any:
        """Return the whole-batch metric value for ``metric``."""
        raise NotImplementedError


def provider_declares_streaming(provider_class: Type[Any]) -> bool:
    """Whether ``provider_class`` opted in to the streaming protocol."""
    return bool(getattr(provider_class, "streaming_supported", False)) and all(
        hasattr(provider_class, name)
        for name in (
            "create_streaming_state",
            "accumulate_streaming_state",
            "finalize_streaming_state",
        )
    )


# Type alias used by the scan coordinator.  The adapter object exposes the same three
# classmethods as ``StreamingMetricProvider``; ``metric_names`` lets an adapter stand in
# for one or more derived metric names (for example the summarisation metrics derived
# from a map condition).
StreamingAdapter = Type[Any]


class StreamingAdapterRegistry:
    """Exact-name registry for built-in streaming adapters.

    Custom providers normally do not need to register here: setting
    ``streaming_supported = True`` on the provider class (e.g. by inheriting
    :class:`StreamingMetricProvider`) is enough.  The registry exists so that built-in
    providers can be adapted without editing every metric class.
    """

    _registry: ClassVar[Dict[str, StreamingAdapter]] = {}

    @classmethod
    def register(cls, metric_name: str, adapter: StreamingAdapter) -> None:
        cls._registry[metric_name] = adapter

    @classmethod
    def get(cls, metric_name: str) -> Optional[StreamingAdapter]:
        return cls._registry.get(metric_name)

    @classmethod
    def clear(cls) -> None:  # pragma: no cover - test helper
        cls._registry.clear()


def register_streaming_adapter(metric_name: str, adapter: StreamingAdapter) -> None:
    """Register a streaming adapter for an exact metric name.

    This is a public extension point: local adapters registered here participate in the
    shared single-pass scan without any Validator-level dispatch.
    """
    StreamingAdapterRegistry.register(metric_name, adapter)


__all__ = [
    "StreamingAdapter",
    "StreamingAdapterRegistry",
    "StreamingMetricProvider",
    "provider_declares_streaming",
    "register_streaming_adapter",
]
