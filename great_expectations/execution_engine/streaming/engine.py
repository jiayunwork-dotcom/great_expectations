"""Single-pass validation engine for dataframe-chunk streams.

This is the bridge between the fluent Batch/Validator world and the streaming metric
adapters.  It deliberately reuses the *existing* dependency graph and expectation result
construction:

1. Build each expectation's metric dependency sub-graph with the same code path ordinary
   validation uses, and merge them into a suite-level ``ValidationGraph``.  The set of
   metrics to compute therefore comes from the existing dependency graph -- not a parallel
   list of rules keyed on expectation name.
2. Verify -- before touching the source -- that the engine is Pandas, there is no
   non-empty ``validation_plan`` and no ``row_condition``, the result format is supported,
   and every metric in the graph has a streaming adapter.  Any violation raises a
   ``StreamPreSourceValidationError``; ``source.__iter__`` is never called.
3. Claim the one-shot source and perform exactly one scan.
4. Feed the resolved metrics through the ordinary ``Expectation.metrics_validate`` path,
   so thresholds, ``strict_min``/``strict_max``, ``mostly``, the missing-value denominator
   and empty-data semantics are exactly the ones each expectation defines -- then graft
   the streaming-only diagnostics (0-based full-stream positions, SUMMARY
   ``partial_unexpected_counts`` suppression, metadata) onto each result.

Metric-level failures are handed to the existing ``catch_exceptions`` machinery; source and
structure errors propagate unchanged.
"""

from __future__ import annotations

import logging
import traceback
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

from great_expectations.exceptions import (
    StreamEngineError,
    StreamResultFormatError,
    StreamRowConditionError,
    StreamUnsupportedMetricError,
    StreamValidationPlanError,
)
from great_expectations.execution_engine.streaming.pandas_providers import (
    UnexpectedIndexSample,
)
from great_expectations.execution_engine.streaming.scan import (
    required_streaming_metrics,
    scan_dataframe_chunks,
)
from great_expectations.validator.exception_info import ExceptionInfo

if TYPE_CHECKING:
    from great_expectations.core.expectation_validation_result import (
        ExpectationValidationResult,
    )
    from great_expectations.execution_engine.execution_engine import ExecutionEngine
    from great_expectations.expectations.expectation_configuration import (
        ExpectationConfiguration,
    )
    from great_expectations.validator.metric_configuration import MetricConfiguration
    from great_expectations.validator.validation_graph import ExpectationValidationGraph

logger = logging.getLogger(__name__)

_COMPLETE = "COMPLETE"
_SUMMARY = "SUMMARY"
_BOOLEAN_ONLY = "BOOLEAN_ONLY"

POSITIONS_KEY = "unexpected_index_positions"
STREAMING_META_KEY = "streaming"


class StreamingNotApplicable(Exception):
    """Internal signal: the batch is not a streaming batch."""


def _expectation_result_format(configuration: ExpectationConfiguration) -> str:
    from great_expectations.expectations.expectation_configuration import (
        parse_result_format,
    )

    raw = configuration.kwargs.get("result_format")
    if raw is None:
        return _SUMMARY
    parsed = parse_result_format(raw)
    return str(parsed.get("result_format"))


def precheck_configurations(  # noqa: C901
    *,
    execution_engine: ExecutionEngine,
    expectation_configurations: Sequence[ExpectationConfiguration],
    validation_plan: Any,
    runtime_configuration: Optional[dict] = None,
) -> None:
    """Reject unsupported streaming configurations before the source is touched."""
    from great_expectations.execution_engine.pandas_execution_engine import (
        PandasExecutionEngine,
    )
    from great_expectations.expectations.expectation_configuration import (
        parse_result_format,
    )

    if not isinstance(execution_engine, PandasExecutionEngine):
        raise StreamEngineError(type(execution_engine).__name__)

    rules = getattr(validation_plan, "rules", validation_plan)
    if rules:
        raise StreamValidationPlanError()

    def _reject_unsupported_format(parsed: dict) -> None:
        if parsed.get("include_unexpected_rows"):
            raise StreamResultFormatError(  # noqa: TRY003
                "include_unexpected_rows requires the full unexpected row set, which is not "
                "retained by a bounded single pass"
            )
        if parsed.get("return_unexpected_index_query"):
            raise StreamResultFormatError(  # noqa: TRY003
                "return_unexpected_index_query requires full-batch index labels, which a "
                "bounded single pass over a one-shot source cannot reconstruct"
            )
        if str(parsed.get("result_format")) == _COMPLETE:
            raise StreamResultFormatError(  # noqa: TRY003
                "COMPLETE requires the full unexpected detail, which is not retained by a "
                "bounded single pass"
            )

    # The run-wide result format (e.g. ValidationDefinition.run(result_format=...)) applies
    # to expectations that do not override it themselves.
    run_result_format = (runtime_configuration or {}).get("result_format")
    if run_result_format is not None:
        _reject_unsupported_format(parse_result_format(run_result_format))

    for configuration in expectation_configurations:
        if configuration.kwargs.get("row_condition") is not None:
            raise StreamRowConditionError(expectation_type=str(configuration.type))
        raw_format = configuration.kwargs.get("result_format")
        if raw_format is None:
            continue
        _reject_unsupported_format(parse_result_format(raw_format))


def _build_subgraph(
    validator: Any,
    configuration: ExpectationConfiguration,
    runtime_configuration: dict,
) -> Tuple[ExpectationValidationGraph, ExpectationConfiguration]:
    """Build one expectation's dependency sub-graph and evaluated configuration.

    No metric is computed here.  The returned configuration has the same ``batch_id``
    injected into its kwargs as ordinary graph validation does, so the metric identities
    the expectation rebuilds in ``metrics_validate`` match the graph vertices.
    """
    produced = validator._generate_metric_dependency_subgraphs_for_each_expectation_configuration(
        expectation_configurations=[configuration],
        processed_configurations=[],
        catch_exceptions=False,
        runtime_configuration=runtime_configuration,
    )
    subgraphs, evrs, processed = produced
    if evrs or not subgraphs or not processed:
        raise _SubgraphBuildError(evrs)
    return subgraphs[0], processed[0]


class _SubgraphBuildError(Exception):
    def __init__(self, evrs: List[ExpectationValidationResult]) -> None:
        super().__init__("Failed to build metric dependency sub-graph")
        self.evrs = evrs


def _all_metric_configurations(
    subgraphs: Sequence[ExpectationValidationGraph],
) -> List[MetricConfiguration]:
    seen: Dict[Any, MetricConfiguration] = {}
    for subgraph in subgraphs:
        for edge in subgraph.graph.edges:
            for vertex in (edge.left, edge.right):
                if vertex is not None and vertex.id not in seen:
                    seen[vertex.id] = vertex
    return list(seen.values())


def run_streaming_validation(  # noqa: C901, PLR0913
    *,
    validator: Any,
    streaming_batch_data: Any,
    expectation_configurations: Sequence[ExpectationConfiguration],
    runtime_configuration: dict,
    catch_exceptions: bool,
    validation_plan: Any = None,
) -> Tuple[List[ExpectationValidationResult], Dict[str, Any]]:
    """Validate ``expectation_configurations`` against the streaming batch in one pass.

    Returns:
        (results, streaming_meta): one result per configuration in input order, plus
        metadata describing the scan.

    Raises:
        StreamPreSourceValidationError: before the source is iterated.
        StreamBatchStructureError / source exceptions: during iteration, unchanged.
    """
    execution_engine = validator.execution_engine

    # ---- Phase 1: build the dependency graph (no data access) -------------------------
    runtime_for_deps = dict(runtime_configuration)
    subgraphs: List[Optional[ExpectationValidationGraph]] = []
    processed_configs: List[Optional[ExpectationConfiguration]] = []
    subgraph_build_failures: Dict[int, List[ExpectationValidationResult]] = {}
    for index, configuration in enumerate(expectation_configurations):
        try:
            subgraph, processed = _build_subgraph(validator, configuration, runtime_for_deps)
            subgraphs.append(subgraph)
            processed_configs.append(processed)
        except _SubgraphBuildError as err:
            subgraph_build_failures[index] = err.evrs
            subgraphs.append(None)
            processed_configs.append(None)

    built_subgraphs = [sg for sg in subgraphs if sg is not None]
    metric_configurations = _all_metric_configurations(built_subgraphs)

    # ---- Phase 2: pre-source capability checks ---------------------------------------
    checkable_configurations = [
        cfg
        for i, cfg in enumerate(expectation_configurations)
        if i not in subgraph_build_failures
    ]
    precheck_configurations(
        execution_engine=execution_engine,
        expectation_configurations=checkable_configurations,
        validation_plan=validation_plan,
        runtime_configuration=runtime_configuration,
    )

    streamable, unsupported = required_streaming_metrics(metric_configurations, execution_engine)
    if unsupported:
        raise StreamUnsupportedMetricError(", ".join(unsupported))

    # ---- Phase 3: one pass over the source -------------------------------------------
    scan = scan_dataframe_chunks(
        source=streaming_batch_data,
        streamable=streamable,
        execution_engine=execution_engine,
    )

    resolved: Dict[Any, Any] = dict(scan.resolved_metrics)
    failed_metric_ids = set(scan.failures)

    # ---- Phase 4: ordinary per-expectation validation over resolved metrics ----------
    results: List[Optional[ExpectationValidationResult]] = [None] * len(expectation_configurations)

    for index, configuration in enumerate(expectation_configurations):
        if index in subgraph_build_failures:
            results[index] = _failure_result(
                configuration,
                exception_info_list=_evrs_to_exception_info(subgraph_build_failures[index]),
            )
            continue
        subgraph = subgraphs[index]
        processed_config = processed_configs[index]
        assert subgraph is not None and processed_config is not None

        # Attribute metric failures that touch this expectation's sub-graph.
        failing_ids = _subgraph_failed_ids(subgraph, failed_metric_ids)
        if failing_ids:
            exception_info = {
                str(metric_id): _exception_info_for(scan.failures[metric_id][1])
                for metric_id in failing_ids
                if metric_id in scan.failures
            }
            if exception_info and not catch_exceptions:
                _raise_first_failure(scan.failures, failing_ids)
            results[index] = _failure_result(
                processed_config, exception_info=exception_info
            )
            continue

        try:
            expectation = processed_config.to_domain_obj()
            result = expectation.metrics_validate(
                metrics=resolved,
                execution_engine=execution_engine,
                runtime_configuration=deepcopy(runtime_configuration),
            )
        except Exception as err:
            if not catch_exceptions:
                raise
            results[index] = _failure_result(
                processed_config, exception_info=_exception_info_for(err)
            )
            continue

        _decorate_streaming_result(result, subgraph, resolved)
        results[index] = result

    meta = {
        "chunk_count": scan.chunk_count,
        "row_count": scan.row_count,
        "columns": list(scan.columns) if scan.columns is not None else None,
        "complete_frequencies_computed": False,
    }
    return [r for r in results if r is not None], meta


# --------------------------------------------------------------------------------------
# result helpers
# --------------------------------------------------------------------------------------


def _subgraph_failed_ids(
    subgraph: ExpectationValidationGraph, failed_ids: set[Any]
) -> List[Any]:
    ids: List[Any] = []
    for edge in subgraph.graph.edges:
        for vertex in (edge.left, edge.right):
            if vertex is not None and vertex.id in failed_ids and vertex.id not in ids:
                ids.append(vertex.id)
    return ids


def _raise_first_failure(failures: Dict[Any, Any], failing_ids: Sequence[Any]) -> None:
    for metric_id in failing_ids:
        if metric_id in failures:
            _configuration, exc = failures[metric_id]
            raise exc


def _exception_info_for(exc: BaseException) -> ExceptionInfo:
    return ExceptionInfo(
        exception_traceback="".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        ),
        exception_message=str(exc),
    )


def _evrs_to_exception_info(evrs: Sequence[ExpectationValidationResult]) -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    for i, evr in enumerate(evrs):
        info[str(i)] = evr.exception_info
    return info


def _failure_result(
    configuration: ExpectationConfiguration,
    *,
    exception_info: Optional[Dict[str, Any]] = None,
    exception_info_list: Optional[Dict[str, Any]] = None,
) -> ExpectationValidationResult:
    from great_expectations.core.expectation_validation_result import (
        ExpectationValidationResult,
    )

    return ExpectationValidationResult(
        success=False,
        exception_info=exception_info or exception_info_list or {},
        expectation_config=configuration,
    )


def _decorate_streaming_result(
    result: ExpectationValidationResult,
    subgraph: ExpectationValidationGraph,
    resolved: Dict[Any, Any],
) -> None:
    """Add streaming diagnostics that the ordinary path cannot express.

    * ``partial_unexpected_counts`` is removed for SUMMARY (the bounded sample cannot
      stand in for whole-batch frequencies); the result meta records that full frequencies
      were not computed.
    * For map expectations, the 0-based full-stream positions of the sampled unexpected rows
      are added alongside the preserved original index labels (``partial_unexpected_list``
      for BASIC, ``partial_unexpected_index_list`` for SUMMARY).
    """
    from great_expectations.execution_engine.streaming.pandas_providers import (
        UnexpectedValuesSample,
    )

    result_meta = dict(result.meta or {})
    result_dict = result.result or {}

    result_dict.pop("partial_unexpected_counts", None)

    positions: Optional[List[int]] = None
    values = result_dict.get("partial_unexpected_list")
    labels = result_dict.get("partial_unexpected_index_list")

    # Prefer the index-list sample when present (SUMMARY), matching the labels length;
    # otherwise fall back to the unexpected-values sample (BASIC).
    if isinstance(labels, UnexpectedIndexSample):
        positions = list(labels.positions)
    elif isinstance(values, UnexpectedValuesSample):
        positions = list(values.positions)

    if positions is not None:
        # Keep positions aligned with whatever sample length survived formatting.
        sample_len = len(values) if values is not None else len(labels or [])
        if sample_len:
            positions = positions[:sample_len]
        result_dict[POSITIONS_KEY] = positions

    if "element_count" in result_dict:
        result_meta[STREAMING_META_KEY] = {
            "complete_frequencies_computed": False,
            "unexpected_index_positions": POSITIONS_KEY if POSITIONS_KEY in result_dict else None,
        }
    result.meta = result_meta


__all__ = [
    "POSITIONS_KEY",
    "STREAMING_META_KEY",
    "StreamingNotApplicable",
    "precheck_configurations",
    "run_streaming_validation",
]
