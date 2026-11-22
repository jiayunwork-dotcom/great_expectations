"""Result-format contracts, empty/all-null semantics, and JSON round trips."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import stream_harness as sh

import great_expectations as gx
from great_expectations.core.expectation_suite import ExpectationSuite
from great_expectations.expectations.core import (
    ExpectColumnMeanToBeBetween,
    ExpectColumnValuesToBeBetween,
    ExpectColumnValuesToNotBeNull,
    ExpectTableRowCountToBeBetween,
)

pytestmark = pytest.mark.unit

RTOL = 1e-10
ATOL = 1e-10


@pytest.fixture()
def bd():
    context = gx.get_context(mode="ephemeral")
    _, _, batch_definition = sh.make_asset_batch_definition(context)
    return batch_definition


def _suite(*expectations) -> ExpectationSuite:
    suite = ExpectationSuite(name="format_suite")
    for e in expectations:
        suite.add_expectation(e)
    return suite


def test_zero_row_first_chunk_declares_schema_then_data_follows(bd):
    empty = pd.DataFrame({"a": pd.Series([], dtype="float64"), "b": pd.Series([], dtype="str")})
    chunks = [empty, pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]})]
    suite = _suite(
        ExpectTableRowCountToBeBetween(min_value=1, max_value=10),
        ExpectColumnValuesToNotBeNull(column="b"),
    )
    batch = bd.get_batch(batch_parameters={"dataframe_chunks": chunks})
    result = batch.validate(suite, result_format="BASIC")
    by_type = {r.expectation_config.type: r for r in result.results}
    assert by_type["expect_table_row_count_to_be_between"].result["observed_value"] == 2
    assert result.success is True


def test_zero_row_single_chunk_is_legitimate_empty_table(bd):
    empty = pd.DataFrame({"a": pd.Series([], dtype="float64")})
    suite = _suite(ExpectTableRowCountToBeBetween(min_value=0, max_value=10))
    batch = bd.get_batch(batch_parameters={"dataframe_chunks": [empty]})
    result = batch.validate(suite, result_format="BASIC")
    assert result.success is True
    assert result.results[0].result["observed_value"] == 0


def test_all_null_mean_matches_pandas_nan(bd):
    frame = pd.DataFrame({"a": [np.nan, np.nan]})
    suite = _suite(ExpectColumnMeanToBeBetween(column="a", min_value=0, max_value=10))
    batch = bd.get_batch(batch_parameters={"dataframe_chunks": [frame]})
    result = batch.validate(suite, result_format="BASIC")
    # pandas mean of all-missing is NaN; the expectation reports failure with a null
    # observed value, exactly as the whole-DataFrame path does.
    assert result.success is False
    assert result.results[0].result["observed_value"] is None


def test_all_null_nonnull_counts_all_rows(bd):
    frame = pd.DataFrame({"a": [np.nan, None, pd.NA]}, dtype="object")
    suite = _suite(ExpectColumnValuesToNotBeNull(column="a"))
    batch = bd.get_batch(batch_parameters={"dataframe_chunks": [frame]})
    result = batch.validate(suite, result_format="BASIC")
    item = result.results[0]
    assert item.result["element_count"] == 3
    assert item.result["unexpected_count"] == 3
    assert item.success is False


def test_boolean_only_strips_result_but_keeps_success(bd):
    chunks = [pd.DataFrame({"a": [1.0, 9.0, 2.0]})]
    suite = _suite(
        ExpectColumnValuesToBeBetween(column="a", min_value=0, max_value=5),
        ExpectTableRowCountToBeBetween(min_value=0),
    )
    batch = bd.get_batch(batch_parameters={"dataframe_chunks": chunks})
    result = batch.validate(suite, result_format="BOOLEAN_ONLY")
    by_type = {r.expectation_config.type: r for r in result.results}
    assert by_type["expect_column_values_to_be_between"].success is False
    assert by_type["expect_column_values_to_be_between"].result == {}
    assert by_type["expect_table_row_count_to_be_between"].result == {}


def test_expectation_level_partial_unexpected_count_is_respected(bd):
    frame = pd.DataFrame({"a": [1.0, 9.0, 2.0, 8.0, 3.0, 7.0]})
    suite = _suite(
        ExpectColumnValuesToBeBetween(column="a", min_value=0, max_value=5)
    )
    batch = bd.get_batch(
        batch_parameters={"dataframe_chunks": [frame.iloc[:2], frame.iloc[2:4], frame.iloc[4:]]}
    )
    # A dict result format carries partial_unexpected_count through to the map metrics.
    item = batch.validate(
        suite, result_format={"result_format": "SUMMARY", "partial_unexpected_count": 2}
    ).results[0]
    assert item.result["partial_unexpected_list"] == [9.0, 8.0]
    assert item.result["partial_unexpected_index_list"] == [1, 3]
    assert item.result["unexpected_index_positions"] == [1, 3]


def test_aggregate_numeric_dtype_error_is_locatable(bd):
    chunks = [pd.DataFrame({"a": ["x", "y"]})]
    suite = _suite(ExpectColumnMeanToBeBetween(column="a", min_value=0, max_value=10))
    batch = bd.get_batch(batch_parameters={"dataframe_chunks": chunks})
    item = batch.validate(suite, result_format="BASIC").results[0]
    assert item.success is False
    message = next(iter(item.exception_info.values()))["exception_message"]
    assert "column='a'" in message
    assert "chunk_index=0" in message


def test_non_finite_value_is_rejected_for_numeric_aggregate(bd):
    chunks = [pd.DataFrame({"a": [1.0, np.inf]})]
    suite = _suite(ExpectColumnMeanToBeBetween(column="a", min_value=0, max_value=10))
    batch = bd.get_batch(batch_parameters={"dataframe_chunks": chunks})
    item = batch.validate(suite, result_format="BASIC").results[0]
    assert item.success is False
    message = next(iter(item.exception_info.values()))["exception_message"]
    assert "non-finite" in message


def test_statistics_and_json_round_trip(bd):
    frame = pd.DataFrame({"a": [1.0, 9.0, np.nan, 2.0], "b": ["x", "y", "z", None]})
    suite = _suite(
        ExpectTableRowCountToBeBetween(min_value=1, max_value=100),
        ExpectColumnValuesToNotBeNull(column="b"),
        ExpectColumnValuesToBeBetween(
            column="a", min_value=0, max_value=5, result_format={"result_format": "SUMMARY"}
        ),
    )
    batch = bd.get_batch(batch_parameters={"dataframe_chunks": [frame.iloc[:2], frame.iloc[2:]]})
    result = batch.validate(suite, result_format="SUMMARY")
    stats = result.statistics
    assert stats["evaluated_expectations"] == 3
    assert stats["declared_expectations"] == 3
    assert stats["unsuccessful_expectations"] == 2

    serialized = result.to_json_dict()
    # Streaming meta survives; no partial_unexpected_counts fabricated.
    assert serialized["meta"]["streaming"]["complete_frequencies_computed"] is False
    for item in serialized["results"]:
        assert "partial_unexpected_counts" not in item["result"]
