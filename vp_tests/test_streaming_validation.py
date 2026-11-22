"""Dataframe-chunk streaming validation: semantics, invariance, and result shape."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import stream_harness as sh

import great_expectations as gx
from great_expectations.core.expectation_suite import ExpectationSuite
from great_expectations.expectations.core import (
    ExpectColumnMeanToBeBetween,
    ExpectColumnStdevToBeBetween,
    ExpectColumnValuesToBeBetween,
    ExpectColumnValuesToNotBeNull,
    ExpectTableRowCountToBeBetween,
)

pytestmark = pytest.mark.unit

BASIC = {"result_format": "BASIC", "partial_unexpected_count": 2}
SUMMARY = {"result_format": "SUMMARY", "partial_unexpected_count": 2}
RTOL = 1e-10
ATOL = 1e-10


@pytest.fixture()
def context():
    return gx.get_context(mode="ephemeral")


@pytest.fixture()
def stream_batch_definition(context):
    _, _, bd = sh.make_asset_batch_definition(context)
    return bd


SMALL = pd.DataFrame(
    {
        "a": [1.0, 7.0, np.nan, 2.0, 9.0, 3.0, 2.0, np.nan, 4.0, 5.0],
        "b": ["x", "y", None, "w", "x", "z", "y", None, "w", "x"],
    },
    index=[0, 0, 1, 2, 3, 3, 4, 5, 6, 7],  # repeated index labels
)


def _suite(*expectations) -> ExpectationSuite:
    suite = ExpectationSuite(name="stream_suite")
    for expectation in expectations:
        suite.add_expectation(expectation)
    return suite


def _run(bd, chunks, suite, result_format=BASIC):
    source = sh.CountingSource(chunks)
    batch = bd.get_batch(batch_parameters={"dataframe_chunks": source})
    result = batch.validate(suite, result_format=result_format)
    return result, source


def _result_by_type(result):
    return {item.expectation_config.type: item for item in result.results}


def test_table_row_count_and_aggregates_match_pandas(stream_batch_definition):
    suite = _suite(
        ExpectTableRowCountToBeBetween(min_value=1, max_value=100),
        ExpectColumnMeanToBeBetween(column="a", min_value=0, max_value=100),
        ExpectColumnStdevToBeBetween(column="a", min_value=0, max_value=100),
    )
    result, source = _run(
        stream_batch_definition, sh.unequal_chunks(SMALL), suite, result_format=BASIC
    )
    by_type = _result_by_type(result)
    assert source.iter_count == 1

    observed_mean = by_type["expect_column_mean_to_be_between"].result["observed_value"]
    observed_std = by_type["expect_column_stdev_to_be_between"].result["observed_value"]
    expected_mean = SMALL["a"].mean()
    expected_std = SMALL["a"].std()
    assert math.isclose(observed_mean, expected_mean, rel_tol=RTOL, abs_tol=ATOL)
    assert math.isclose(observed_std, expected_std, rel_tol=RTOL, abs_tol=ATOL)
    assert by_type["expect_table_row_count_to_be_between"].result["observed_value"] == len(SMALL)


def test_map_counts_and_samples_are_whole_batch(stream_batch_definition):
    suite = _suite(
        ExpectColumnValuesToBeBetween(
            column="a", min_value=0, max_value=5, result_format=SUMMARY
        )
    )
    result, _ = _run(stream_batch_definition, sh.unequal_chunks(SMALL), suite, SUMMARY)
    item = _result_by_type(result)["expect_column_values_to_be_between"]

    expected_between = ((SMALL["a"] < 0) | (SMALL["a"] > 5)) & SMALL["a"].notna()
    expected_unexpected = int(expected_between.sum())
    # element_count counts all rows; missing_count counts nulls in the domain column.
    assert item.result["element_count"] == len(SMALL)
    assert item.result["unexpected_count"] == expected_unexpected
    assert item.result["missing_count"] == int(SMALL["a"].isna().sum())
    # partial_unexpected_list holds the first N unexpected values in stream order.
    expected_values = SMALL["a"][expected_between].tolist()[:2]
    assert item.result["partial_unexpected_list"] == expected_values
    # partial_unexpected_counts must not be present in streaming SUMMARY.
    assert "partial_unexpected_counts" not in item.result
    # The result meta must state that full frequencies were not computed.
    assert item.meta["streaming"]["complete_frequencies_computed"] is False


def test_positions_and_labels_both_reported(stream_batch_definition):
    suite = _suite(
        ExpectColumnValuesToBeBetween(
            column="a", min_value=0, max_value=5, result_format=SUMMARY
        )
    )
    result, _ = _run(stream_batch_definition, sh.unequal_chunks(SMALL), suite, SUMMARY)
    item = _result_by_type(result)["expect_column_values_to_be_between"]

    unexpected_mask = ((SMALL["a"] < 0) | (SMALL["a"] > 5)) & SMALL["a"].notna()
    expected_positions = np.flatnonzero(unexpected_mask.to_numpy())[:2].tolist()
    expected_labels = SMALL.index[unexpected_mask].tolist()[:2]

    assert item.result["partial_unexpected_index_list"] == expected_labels
    assert item.result["unexpected_index_positions"] == expected_positions
    # Positions disambiguate repeated index labels.
    assert len(set(SMALL.index)) < len(SMALL.index)


def test_nonnull_counts_uphold_missing_denominator(stream_batch_definition):
    suite = _suite(ExpectColumnValuesToNotBeNull(column="b", result_format=SUMMARY))
    result, _ = _run(stream_batch_definition, sh.unequal_chunks(SMALL), suite, SUMMARY)
    item = _result_by_type(result)["expect_column_values_to_not_be_null"]
    expected_missing = int(SMALL["b"].isna().sum())
    assert item.result["element_count"] == len(SMALL)
    assert item.result["unexpected_count"] == expected_missing
    assert item.result["unexpected_percent"] == pytest.approx(expected_missing / len(SMALL) * 100)
    assert item.result["partial_unexpected_list"] == [None] * min(2, expected_missing) or (
        # NaN compares unequal to None, so check length + null-ness instead
        len(item.result["partial_unexpected_list"]) == min(2, expected_missing)
        and all(pd.isna(v) for v in item.result["partial_unexpected_list"])
    )


@pytest.mark.parametrize("chunker", [lambda df: [df], sh.unequal_chunks, sh.per_row])
def test_chunking_does_not_change_complete_report(stream_batch_definition, chunker):
    suite = _suite(
        ExpectTableRowCountToBeBetween(min_value=0, max_value=100),
        ExpectColumnMeanToBeBetween(column="a", min_value=0, max_value=100),
        ExpectColumnStdevToBeBetween(column="a", min_value=0, max_value=100),
        ExpectColumnValuesToNotBeNull(column="b", result_format=BASIC),
        ExpectColumnValuesToBeBetween(
            column="a", min_value=0, max_value=5, result_format=BASIC
        ),
    )
    result, _ = _run(stream_batch_definition, chunker(SMALL), suite, BASIC)
    by_type = _result_by_type(result)

    assert by_type["expect_table_row_count_to_be_between"].result["observed_value"] == len(SMALL)
    assert math.isclose(
        by_type["expect_column_mean_to_be_between"].result["observed_value"],
        SMALL["a"].mean(),
        rel_tol=RTOL,
        abs_tol=ATOL,
    )
    assert math.isclose(
        by_type["expect_column_stdev_to_be_between"].result["observed_value"],
        SMALL["a"].std(),
        rel_tol=RTOL,
        abs_tol=ATOL,
    )
    assert by_type["expect_column_values_to_not_be_null"].result["unexpected_count"] == int(
        SMALL["b"].isna().sum()
    )
    between = by_type["expect_column_values_to_be_between"].result
    assert between["element_count"] == len(SMALL)
    assert between["unexpected_count"] == int(
        (((SMALL["a"] < 0) | (SMALL["a"] > 5)) & SMALL["a"].notna()).sum()
    )


def test_multiple_thresholds_same_column_are_distinct(stream_batch_definition):
    suite = _suite(
        ExpectColumnValuesToBeBetween(
            column="a",
            min_value=0,
            max_value=5,
            result_format={"result_format": "BASIC", "partial_unexpected_count": 3},
        ),
        ExpectColumnValuesToBeBetween(
            column="a",
            min_value=1,
            max_value=5,
            strict_min=True,
            result_format={"result_format": "BASIC", "partial_unexpected_count": 3},
        ),
    )
    result, _ = _run(stream_batch_definition, sh.per_row(SMALL), suite, BASIC)
    between_results = [
        r for r in result.results if r.expectation_config.type == "expect_column_values_to_be_between"
    ]
    assert len(between_results) == 2
    counts = sorted(r.result["unexpected_count"] for r in between_results)
    inclusive = int((((SMALL["a"] < 0) | (SMALL["a"] > 5)) & SMALL["a"].notna()).sum())
    strict = int((((SMALL["a"] <= 1) | (SMALL["a"] > 5)) & SMALL["a"].notna()).sum())
    assert counts == sorted([inclusive, strict])


def test_mostly_uses_whole_batch_ratio(stream_batch_definition):
    suite = _suite(
        ExpectColumnValuesToNotBeNull(column="b", mostly=0.9, result_format=BASIC),
    )
    result, _ = _run(stream_batch_definition, sh.unequal_chunks(SMALL), suite, BASIC)
    item = _result_by_type(result)["expect_column_values_to_not_be_null"]
    missing = int(SMALL["b"].isna().sum())
    ratio = (len(SMALL) - missing) / len(SMALL)
    assert item.success == (ratio >= 0.9)


def test_mean_and_stdev_within_tolerance_of_whole_frame(stream_batch_definition):
    rng = np.random.default_rng(7)
    frame = pd.DataFrame({"a": rng.normal(1000.0, 50.0, 500)})
    suite = _suite(
        ExpectColumnMeanToBeBetween(column="a", min_value=-1e9, max_value=1e9),
        ExpectColumnStdevToBeBetween(column="a", min_value=-1e9, max_value=1e9),
    )
    result, _ = _run(
        stream_batch_definition,
        [frame.iloc[i : i + 37] for i in range(0, len(frame), 37)],
        suite,
        BASIC,
    )
    by_type = _result_by_type(result)
    assert math.isclose(
        by_type["expect_column_mean_to_be_between"].result["observed_value"],
        frame["a"].mean(),
        rel_tol=RTOL,
        abs_tol=ATOL,
    )
    assert math.isclose(
        by_type["expect_column_stdev_to_be_between"].result["observed_value"],
        frame["a"].std(),
        rel_tol=RTOL,
        abs_tol=ATOL,
    )


def test_positions_reported_under_basic_without_index_labels(stream_batch_definition):
    suite = _suite(
        ExpectColumnValuesToBeBetween(column="a", min_value=0, max_value=5)
    )
    result, _ = _run(stream_batch_definition, sh.unequal_chunks(SMALL), suite, BASIC)
    item = _result_by_type(result)["expect_column_values_to_be_between"]
    unexpected_mask = ((SMALL["a"] < 0) | (SMALL["a"] > 5)) & SMALL["a"].notna()
    expected_positions = np.flatnonzero(unexpected_mask.to_numpy())[:2].tolist()
    # BASIC does not request the index-list metric, but positions are still reported.
    assert "partial_unexpected_index_list" not in item.result
    assert item.result["unexpected_index_positions"] == expected_positions
