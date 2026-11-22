"""Custom streaming aggregate shares the single scan with the built-in rules."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import stream_custom_provider as custom
import stream_harness as sh

import great_expectations as gx
from great_expectations.core.expectation_suite import ExpectationSuite
from great_expectations.expectations.core import (
    ExpectColumnMeanToBeBetween,
    ExpectColumnValuesToBeBetween,
)

pytestmark = pytest.mark.unit

FRAME = pd.DataFrame(
    {"a": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0], "b": ["x"] * 8},
    index=[0, 0, 1, 2, 3, 3, 4, 5],
)


@pytest.fixture(autouse=True)
def _reset():
    custom.reset_consumers()
    yield
    custom.reset_consumers()


def test_custom_provider_coexists_with_builtin_metrics_in_one_scan():
    context = gx.get_context(mode="ephemeral")
    _, _, bd = sh.make_asset_batch_definition(context, name="custom_ds")

    suite = ExpectationSuite(name="custom_suite")
    suite.add_expectation(ExpectColumnMeanToBeBetween(column="a", min_value=0, max_value=10))
    suite.add_expectation(
        ExpectColumnValuesToBeBetween(column="a", min_value=0, max_value=5)
    )
    suite.add_expectation(custom.ExpectVpStreamSum(column="a", tag="sum"))

    source = sh.CountingSource([FRAME.iloc[:3], FRAME.iloc[3:5], FRAME.iloc[5:]])
    batch = bd.get_batch(batch_parameters={"dataframe_chunks": source})
    result = batch.validate(suite, result_format="BASIC")

    assert source.iter_count == 1
    by_type = {r.expectation_config.type: r for r in result.results}
    assert by_type["expect_vp_stream_sum"].result["observed_value"] == pytest.approx(
        custom.expected_sum(FRAME)
    )
    assert by_type["expect_column_mean_to_be_between"].result["observed_value"] == pytest.approx(
        FRAME["a"].mean()
    )
    # The custom provider observed every chunk (0, 1, 2) in the same pass.
    assert set(custom.CHUNK_CONSUMERS) == {0, 1, 2}


def test_custom_provider_already_registered_participates_without_validator_dispatch():
    # The provider declares streaming_supported via its base class; the engine discovers it
    # from the metric registry alone.
    from great_expectations.execution_engine.streaming.scan import (
        required_streaming_metrics,
        resolve_streaming_adapter,
    )
    from great_expectations.validator.metric_configuration import MetricConfiguration

    engine = (
        gx.get_context(mode="ephemeral")
        .data_sources.add_pandas(name="probe_ds")
        .get_execution_engine()
    )
    metric = MetricConfiguration("vp.column_sum", {"batch_id": "b", "column": "a"})
    adapter = resolve_streaming_adapter(metric, engine)
    assert adapter is custom.ColumnSumStreaming
    streamable, unsupported = required_streaming_metrics([metric], engine)
    assert metric.id in streamable and not unsupported


def test_identical_metrics_are_deduplicated_across_expectations():
    from great_expectations.execution_engine.streaming.engine import (
        _all_metric_configurations,
        _build_subgraph,
    )
    from great_expectations.execution_engine.streaming.scan import required_streaming_metrics
    from great_expectations.validator.v1_validator import Validator

    context = gx.get_context(mode="ephemeral")
    _, _, bd = sh.make_asset_batch_definition(context, name="dedup_ds")
    suite = ExpectationSuite(name="dedup_suite")
    suite.add_expectation(
        ExpectColumnValuesToBeBetween(column="a", min_value=0, max_value=5)
    )
    suite.add_expectation(
        ExpectColumnValuesToBeBetween(column="a", min_value=0, max_value=5)
    )
    validator = Validator(
        batch_definition=bd,
        batch_parameters={"dataframe_chunks": [FRAME]},
        result_format="BASIC",
    )
    wrapped = validator._wrapped_validator
    engine = wrapped.execution_engine
    processed = wrapped.process_expectations_for_validation(
        suite.expectation_configurations, None
    )
    subgraphs = [
        _build_subgraph(wrapped, cfg, {"result_format": "BASIC"})[0] for cfg in processed
    ]
    streamable, _ = required_streaming_metrics(
        _all_metric_configurations(subgraphs), engine
    )
    # A single identical expectation yields the same identity count as two.
    single = ExpectationSuite(name="single_suite")
    single.add_expectation(
        ExpectColumnValuesToBeBetween(column="a", min_value=0, max_value=5)
    )
    single_processed = wrapped.process_expectations_for_validation(
        single.expectation_configurations, None
    )
    single_subgraphs = [
        _build_subgraph(wrapped, cfg, {"result_format": "BASIC"})[0]
        for cfg in single_processed
    ]
    single_streamable, _ = required_streaming_metrics(
        _all_metric_configurations(single_subgraphs), engine
    )
    assert len(streamable) == len(single_streamable)
