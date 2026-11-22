"""Fluent asset + ValidationDefinition.run + Validation Results Store round-trip."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import stream_harness as sh

import great_expectations as gx
from great_expectations.core.expectation_suite import ExpectationSuite
from great_expectations.core.validation_definition import ValidationDefinition
from great_expectations.expectations.core import (
    ExpectColumnMeanToBeBetween,
    ExpectColumnValuesToBeBetween,
    ExpectTableRowCountToBeBetween,
)

pytestmark = pytest.mark.unit

DF = pd.DataFrame({"a": [1.0, 7.0, np.nan, 2.0, 9.0, 3.0], "b": ["x", "y", "z", "w", "v", "u"]})


def _suite() -> ExpectationSuite:
    suite = ExpectationSuite(name="store_suite")
    suite.add_expectation(ExpectTableRowCountToBeBetween(min_value=1, max_value=100))
    suite.add_expectation(ExpectColumnMeanToBeBetween(column="a", min_value=0, max_value=10))
    suite.add_expectation(
        ExpectColumnValuesToBeBetween(
            column="a", min_value=0, max_value=5, result_format={"result_format": "SUMMARY"}
        )
    )
    return suite


def _chunks(frame=DF):
    for i in range(0, len(frame), 2):
        yield frame.iloc[i : i + 2]


def test_asset_and_batch_definition_are_persisted_without_source(context=None):
    context = gx.get_context(mode="ephemeral")
    datasource = context.data_sources.add_pandas(name="persist_ds")
    asset = datasource.add_dataframe_stream_asset(name="persist_asset")
    assert asset.type == "dataframe_stream"
    batch_definition = asset.add_batch_definition_whole_dataframe_stream(name="persist_bd")
    assert batch_definition.name == "persist_bd"

    # Re-fetch through the context: the stream asset survives a config round trip.
    reloaded_ds = context.data_sources.get("persist_ds")
    reloaded_asset = reloaded_ds.get_asset("persist_asset")
    assert type(reloaded_asset).__name__ == "DataFrameStreamAsset"
    reloaded_bd = reloaded_asset.get_batch_definition("persist_bd")
    assert reloaded_bd.name == "persist_bd"

    # The datasource config never embeds a source object.
    config_text = json.dumps(
        context.data_sources.get("persist_ds").dict(exclude_unset=True), default=str
    )
    assert "dataframe_chunks" not in config_text


def test_validation_definition_run_and_store_round_trip():
    context = gx.get_context(mode="ephemeral")
    vd = sh.build_stream_validation_definition(context, _suite(), name="rt_vd")

    result = vd.run(batch_parameters={"dataframe_chunks": _chunks()})
    assert result.meta["batch_parameters"] == {"dataframe_chunks": "<DATAFRAME_CHUNKS>"}
    assert result.meta["streaming"]["chunk_count"] == 3
    assert result.meta["streaming"]["row_count"] == len(DF)

    # Read back from the Validation Results Store.
    keys = context.validation_results_store.list_keys()
    assert keys
    stored = context.validation_results_store.get(keys[-1])
    assert len(stored.results) == 3
    assert stored.meta["streaming"]["row_count"] == len(DF)
    # JSON round-trip keeps the streaming meta and result extras.
    serialized = stored.to_json_dict()
    assert serialized["meta"]["streaming"]["chunk_count"] == 3


def test_reloaded_definition_runs_with_new_source():
    context = gx.get_context(mode="ephemeral")
    vd = sh.build_stream_validation_definition(context, _suite(), name="reload_vd")
    vd.run(batch_parameters={"dataframe_chunks": _chunks()})

    reloaded = context.validation_definitions.get("reload_vd")
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": ["a", "b", "c", "d"]})
    result = reloaded.run(
        batch_parameters={"dataframe_chunks": [frame.iloc[:2], frame.iloc[2:]]}
    )
    assert result.meta["streaming"]["row_count"] == 4


def test_source_reuse_after_failure_then_new_batch_succeeds():
    context = gx.get_context(mode="ephemeral")
    vd = sh.build_stream_validation_definition(context, _suite(), name="fail_vd")

    def bad_source():
        yield DF.iloc[:2]
        raise RuntimeError("stream broke")

    source = bad_source()
    with pytest.raises(RuntimeError, match="stream broke"):
        vd.run(batch_parameters={"dataframe_chunks": source})

    # Re-using the same (now consumed) generator is rejected before iteration.
    with pytest.raises(gx.exceptions.StreamSourceReuseError):
        vd.run(batch_parameters={"dataframe_chunks": source})

    # A new source runs normally.
    ok = vd.run(batch_parameters={"dataframe_chunks": _chunks()})
    assert ok.meta["streaming"]["row_count"] == len(DF)


def test_batch_spec_placeholder_never_contains_source():
    context = gx.get_context(mode="ephemeral")
    _, _, bd = sh.make_asset_batch_definition(context, name="spec_ds")
    source = sh.GeneratorSource([DF])
    batch = bd.get_batch(batch_parameters={"dataframe_chunks": source})
    spec_json = batch.batch_spec.to_json_dict()
    assert spec_json == {"batch_data": "<DATAFRAME_CHUNKS>"}
    # Preparing did not touch the source.
    assert source.iter_count == 0


def test_file_context_persistence_and_cross_context_reload(tmp_path):
    context = gx.get_context(mode="file", project_root_dir=str(tmp_path))
    datasource = context.data_sources.add_pandas(name="file_ds")
    asset = datasource.add_dataframe_stream_asset(name="file_asset")
    bd = asset.add_batch_definition_whole_dataframe_stream(name="file_bd")
    suite = ExpectationSuite(name="file_suite")
    suite.add_expectation(ExpectTableRowCountToBeBetween(min_value=0, max_value=100))
    suite.add_expectation(ExpectColumnMeanToBeBetween(column="a", min_value=0, max_value=100))
    vd = context.validation_definitions.add(
        ValidationDefinition(name="file_vd", data=bd, suite=context.suites.add(suite))
    )

    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    first = vd.run(batch_parameters={"dataframe_chunks": [frame.iloc[:2], frame.iloc[2:]]})
    assert first.success is True

    # A brand new context on the same project root reloads the asset/definition.
    reloaded_context = gx.get_context(mode="file", project_root_dir=str(tmp_path))
    reloaded_asset = reloaded_context.data_sources.get("file_ds").get_asset("file_asset")
    assert type(reloaded_asset).__name__ == "DataFrameStreamAsset"
    reloaded_vd = reloaded_context.validation_definitions.get("file_vd")
    second = reloaded_vd.run(batch_parameters={"dataframe_chunks": [frame]})
    assert second.meta["streaming"]["row_count"] == 4

    stored = reloaded_context.validation_results_store.get(
        reloaded_context.validation_results_store.list_keys()[-1]
    )
    assert stored.meta["streaming"]["row_count"] == 4
