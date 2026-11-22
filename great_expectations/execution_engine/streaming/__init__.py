"""Dataframe-chunk streaming validation support.

See :mod:`great_expectations.execution_engine.streaming.protocol` for the provider
capability declaration and state protocol documentation.
"""

from great_expectations.execution_engine.streaming.batch_data import StreamingBatchData
from great_expectations.execution_engine.streaming.pandas_providers import (
    register_builtin_streaming_adapters,
)
from great_expectations.execution_engine.streaming.protocol import (
    StreamingMetricProvider,
    register_streaming_adapter,
)

# Populate the adapter registry for the built-in rules on import.
register_builtin_streaming_adapters()

__all__ = [
    "StreamingBatchData",
    "StreamingMetricProvider",
    "register_builtin_streaming_adapters",
    "register_streaming_adapter",
]
