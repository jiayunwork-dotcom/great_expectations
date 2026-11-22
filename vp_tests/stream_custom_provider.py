"""A custom streaming aggregate used to prove shared-scan registration.

``ExpectVpStreamSum`` and its provider demonstrate that a locally registered streaming
adapter participates in the same single scan as the built-in rules, with no Validator-level
dispatch.  The provider also records which chunks it consumed so tests can verify that all
metrics share one pass.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, Tuple

from great_expectations.core import ExpectationValidationResult
from great_expectations.execution_engine import PandasExecutionEngine
from great_expectations.execution_engine.streaming.protocol import StreamingMetricProvider
from great_expectations.expectations.expectation import BatchExpectation
from great_expectations.expectations.metrics.metric_provider import (
    metric_value,
)

# chunk_index -> set of metric names observed on that chunk
CHUNK_CONSUMERS: Dict[int, set] = {}


def reset_consumers() -> None:
    CHUNK_CONSUMERS.clear()


class ColumnSumStreaming(StreamingMetricProvider):
    """Sum of a numeric column over the whole logical batch."""

    metric_name = "vp.column_sum"
    domain_keys = ("batch_id", "column")
    value_keys = ()

    @metric_value(engine=PandasExecutionEngine)
    def _pandas(
        cls,
        execution_engine,
        metric_domain_kwargs,
        metric_value_kwargs,
        metrics,
        runtime_configuration,
    ):
        # Non-streaming fallback (whole DataFrame path) - not exercised here.
        raise NotImplementedError("vp.column_sum is only implemented for streaming")

    @classmethod
    def create_streaming_state(cls, *, metric, execution_engine):
        return {"sum": 0.0, "count": 0}

    @classmethod
    def accumulate_streaming_state(
        cls,
        *,
        state,
        metric,
        dataframe,
        execution_engine,
        chunk_index,
        row_offset,
        scratch=None,
    ):
        CHUNK_CONSUMERS.setdefault(chunk_index, set()).add(cls.metric_name)
        column = metric.metric_domain_kwargs["column"]
        series = dataframe[column].dropna()
        state["sum"] += float(series.sum())
        state["count"] += len(series)

    @classmethod
    def finalize_streaming_state(cls, *, state, metric, execution_engine):
        return state["sum"]


class ExpectVpStreamSum(BatchExpectation):
    """Whole-batch sum of a column, computed via the streaming protocol."""

    column: str
    tag: str = ""
    domain_keys: ClassVar[Tuple[str, ...]] = ("batch_id", "column", "tag")
    metric_dependencies = ("vp.column_sum",)
    success_keys: ClassVar[Tuple[str, ...]] = ("column",)
    args_keys: ClassVar[Tuple[str, ...]] = ("column",)
    library_metadata: ClassVar[Dict[str, Any]] = {
        "maturity": "experimental",
        "tags": [],
        "contributors": [],
        "requirements": [],
        "has_full_test_suite": False,
        "manually_reviewed_code": True,
    }

    class Config:
        arbitrary_types_allowed = True

    def _validate(self, metrics, runtime_configuration=None, execution_engine=None):
        observed = metrics.get("vp.column_sum")
        return ExpectationValidationResult(
            success=observed is not None,
            result={"observed_value": observed},
        )


def expected_sum(frame) -> float:
    return float(frame["a"].dropna().sum())
