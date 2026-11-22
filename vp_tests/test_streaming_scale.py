"""Memory-scale probe: streaming must not retain the stream.

Generates 4096-row chunks on demand, validates to 100k and 1M accumulated rows, and
measures ``tracemalloc`` peaks *starting from the prepared Batch*.  The test never holds
the whole frame; the generator creates each chunk and it is discarded after being folded.
"""

from __future__ import annotations

import gc
import tracemalloc

import numpy as np
import pandas as pd
import pytest

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

CHUNK_ROWS = 4096
MIB = 1024 * 1024


def _chunk_stream(n_rows: int):
    rng = np.random.default_rng(1234)
    made = 0
    while made < n_rows:
        size = min(CHUNK_ROWS, n_rows - made)
        values = rng.normal(0, 1, size)
        values[::17] = np.nan
        other = rng.integers(0, 100, size)
        yield pd.DataFrame({"a": values, "b": other})
        made += size


def _suite() -> ExpectationSuite:
    suite = ExpectationSuite(name="scale_suite")
    suite.add_expectation(ExpectTableRowCountToBeBetween(min_value=0, max_value=10_000_000))
    suite.add_expectation(ExpectColumnMeanToBeBetween(column="a", min_value=-10, max_value=10))
    suite.add_expectation(ExpectColumnStdevToBeBetween(column="a", min_value=0, max_value=10))
    suite.add_expectation(ExpectColumnValuesToNotBeNull(column="b"))
    suite.add_expectation(
        ExpectColumnValuesToBeBetween(
            column="a", min_value=-5, max_value=5, result_format={"result_format": "BASIC"}
        )
    )
    return suite


def _measure_peak_delta(n_rows: int) -> int:
    context = gx.get_context(mode="ephemeral")
    datasource = context.data_sources.add_pandas(name=f"scale_ds_{n_rows}")
    asset = datasource.add_dataframe_stream_asset(name="scale_asset")
    bd = asset.add_batch_definition_whole_dataframe_stream(name="scale_bd")
    suite = _suite()

    batch = bd.get_batch(batch_parameters={"dataframe_chunks": _chunk_stream(n_rows)})

    # Measure only the validation phase, from the prepared Batch onward.
    gc.collect()
    tracemalloc.start()
    baseline = tracemalloc.get_traced_memory()[0]
    result = batch.validate(suite, result_format="BASIC")
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    row_count_result = next(
        r for r in result.results if r.expectation_config.type == "expect_table_row_count_to_be_between"
    )
    assert row_count_result.result["observed_value"] == n_rows
    return peak - baseline


def test_streaming_memory_is_bounded_for_one_million_rows():
    peak_100k = _measure_peak_delta(100_000)
    peak_1m = _measure_peak_delta(1_000_000)

    assert peak_1m <= 64 * MIB, f"1M-row peak {peak_1m / MIB:.1f} MiB exceeds 64 MiB"
    growth = peak_1m - peak_100k
    assert growth <= 8 * MIB, f"1M vs 100k growth {growth / MIB:.1f} MiB exceeds 8 MiB"
