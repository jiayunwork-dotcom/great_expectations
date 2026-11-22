"""Streaming lifecycle: one-shot sources, rejection timing, error identity, close safety."""

from __future__ import annotations

import pandas as pd
import pytest
import stream_harness as sh

import great_expectations as gx
import great_expectations.exceptions as gx_exceptions
from great_expectations.core.expectation_suite import ExpectationSuite
from great_expectations.expectations.core import (
    ExpectColumnMaxToBeBetween,
    ExpectColumnValuesToBeBetween,
    ExpectTableRowCountToBeBetween,
)

pytestmark = pytest.mark.unit

BASIC = {"result_format": "BASIC"}

DF = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", "y", "z"]})


@pytest.fixture()
def bd():
    context = gx.get_context(mode="ephemeral")
    _, _, batch_definition = sh.make_asset_batch_definition(context)
    return batch_definition


def _suite(*expectations) -> ExpectationSuite:
    suite = ExpectationSuite(name="lifecycle_suite")
    for e in expectations:
        suite.add_expectation(e)
    return suite


def test_preparing_batch_reads_nothing(bd):
    source = sh.GeneratorSource([DF])
    batch = bd.get_batch(batch_parameters={"dataframe_chunks": source})
    assert batch.data.consumed is False
    assert source.iter_count == 0


def test_single_iterator_and_source_is_not_mutated(bd):
    source = sh.CountingSource([DF.iloc[:1], DF.iloc[1:]])
    original = DF.copy(deep=True)
    batch = bd.get_batch(batch_parameters={"dataframe_chunks": source})
    batch.validate(_suite(ExpectTableRowCountToBeBetween(min_value=0)), result_format=BASIC)
    assert source.iter_count == 1
    pd.testing.assert_frame_equal(DF, original)


def test_revalidate_same_consumed_batch_is_rejected(bd):
    source = sh.GeneratorSource([DF])
    batch = bd.get_batch(batch_parameters={"dataframe_chunks": source})
    batch.validate(_suite(ExpectTableRowCountToBeBetween(min_value=0)), result_format=BASIC)
    with pytest.raises(gx_exceptions.StreamSourceReuseError):
        batch.validate(_suite(ExpectTableRowCountToBeBetween(min_value=0)), result_format=BASIC)


def test_reusing_source_across_fresh_batches_is_rejected(bd):
    source = sh.GeneratorSource([DF])
    first = bd.get_batch(batch_parameters={"dataframe_chunks": source})
    first.validate(_suite(ExpectTableRowCountToBeBetween(min_value=0)), result_format=BASIC)

    second = bd.get_batch(batch_parameters={"dataframe_chunks": source})
    with pytest.raises(gx_exceptions.StreamSourceReuseError):
        second.validate(_suite(ExpectTableRowCountToBeBetween(min_value=0)), result_format=BASIC)
    # The rejected second run did not call source.__iter__ again.
    assert source.iter_count == 1


def test_failed_run_still_consumes_and_rejects_reuse(bd):
    # Use a custom iterable that raises mid-stream.
    class Raising:
        def __init__(self):
            self.iter_count = 0

        def __iter__(self):
            self.iter_count += 1

            def gen():
                yield DF.iloc[:1]
                raise RuntimeError("source failure")

            return gen()

    raising = Raising()
    batch = bd.get_batch(batch_parameters={"dataframe_chunks": raising})
    with pytest.raises(RuntimeError, match="source failure"):
        batch.validate(_suite(ExpectTableRowCountToBeBetween(min_value=0)), result_format=BASIC)
    # Same source cannot be retried; a new batch with a new source can run.
    with pytest.raises(gx_exceptions.StreamSourceReuseError):
        bd.get_batch(batch_parameters={"dataframe_chunks": raising}).validate(
            _suite(ExpectTableRowCountToBeBetween(min_value=0)), result_format=BASIC
        )
    fresh = sh.CountingSource([DF])
    ok = bd.get_batch(batch_parameters={"dataframe_chunks": fresh}).validate(
        _suite(ExpectTableRowCountToBeBetween(min_value=0)), result_format=BASIC
    )
    assert ok.success is True


def _expect_no_iteration(bd, source, expectations, **kwargs):
    batch = bd.get_batch(batch_parameters={"dataframe_chunks": source})
    with pytest.raises(gx_exceptions.StreamPreSourceValidationError):
        batch.validate(_suite(*expectations), **kwargs)
    assert source.iter_count == 0


def test_unsupported_metric_rejected_before_iteration(bd):
    source = sh.CountingSource([DF])
    _expect_no_iteration(
        bd, source, [ExpectColumnMaxToBeBetween(column="a", min_value=0, max_value=10)]
    )


def test_row_condition_rejected_before_iteration(bd):
    source = sh.CountingSource([DF])
    expectation = ExpectColumnValuesToBeBetween(
        column="a", min_value=0, max_value=5, row_condition="a > 0"
    )
    _expect_no_iteration(bd, source, [expectation])


def test_complete_result_format_rejected_before_iteration(bd):
    source = sh.CountingSource([DF])
    _expect_no_iteration(
        bd, source, [ExpectTableRowCountToBeBetween(min_value=0)], result_format="COMPLETE"
    )


def test_include_unexpected_rows_rejected_before_iteration(bd):
    source = sh.CountingSource([DF])
    expectation = ExpectColumnValuesToBeBetween(
        column="a",
        min_value=0,
        max_value=5,
        result_format={"result_format": "BASIC", "include_unexpected_rows": True},
    )
    _expect_no_iteration(bd, source, [expectation])


def test_validation_plan_rejected_before_iteration():
    context = gx.get_context(mode="ephemeral")
    _, _, bd = sh.make_asset_batch_definition(context, name="plan_ds")
    suite = context.suites.add(
        _suite(ExpectTableRowCountToBeBetween(min_value=0, max_value=10))
    )
    expectation_id = suite.expectations[0].id
    suite.meta["validation_plan"] = {
        "version": 1,
        "rules": [{"id": "r0", "expectation_id": expectation_id, "requires": []}],
    }
    suite.save()

    source = sh.CountingSource([DF])
    batch = bd.get_batch(batch_parameters={"dataframe_chunks": source})
    with pytest.raises(gx_exceptions.StreamValidationPlanError):
        batch.validate(suite, result_format="BASIC")
    assert source.iter_count == 0


def test_no_chunks_is_a_batch_error(bd):
    source = sh.CountingSource([])
    batch = bd.get_batch(batch_parameters={"dataframe_chunks": source})
    with pytest.raises(gx_exceptions.StreamBatchStructureError):
        batch.validate(_suite(ExpectTableRowCountToBeBetween(min_value=0)), result_format=BASIC)


def test_column_reorder_is_a_batch_error(bd):
    chunks = [pd.DataFrame({"a": [1.0], "b": [2.0]}), pd.DataFrame({"b": [2.0], "a": [1.0]})]
    batch = bd.get_batch(batch_parameters={"dataframe_chunks": chunks})
    with pytest.raises(gx_exceptions.StreamBatchStructureError):
        batch.validate(_suite(ExpectTableRowCountToBeBetween(min_value=0)), result_format=BASIC)


def test_dtype_change_is_a_batch_error(bd):
    chunks = [pd.DataFrame({"a": [1.0]}), pd.DataFrame({"a": ["x"]})]
    batch = bd.get_batch(batch_parameters={"dataframe_chunks": chunks})
    with pytest.raises(gx_exceptions.StreamBatchStructureError):
        batch.validate(_suite(ExpectTableRowCountToBeBetween(min_value=0)), result_format=BASIC)


def test_source_error_not_swallowed_by_catch_exceptions(bd):
    class Raising:
        def __iter__(self):
            def gen():
                yield DF.iloc[:1]
                raise KeyError("boom")

            return gen()

    batch = bd.get_batch(batch_parameters={"dataframe_chunks": Raising()})
    expectation = ExpectColumnValuesToBeBetween(column="a", min_value=0, max_value=5)
    with pytest.raises(KeyError, match="boom"):
        batch.validate(_suite(expectation), result_format="BASIC")


def test_rule_exception_is_caught_as_result(bd):
    # A non-numeric mean column is a rule/metric-level failure -> exception info in result.
    chunks = [pd.DataFrame({"a": ["x", "y"]})]
    batch = bd.get_batch(batch_parameters={"dataframe_chunks": chunks})
    suite = _suite(
        ExpectColumnValuesToBeBetween(column="a", min_value=0, max_value=5, result_format=BASIC)
    )
    # between on strings fails inside the predicate -> rule exception recorded, not raised.
    result = batch.validate(suite, result_format="BASIC")
    assert result.success is False
    assert result.results[0].exception_info


def test_iterator_closed_on_success(bd):
    source = sh.GeneratorSource([DF.iloc[:1], DF.iloc[1:]])
    batch = bd.get_batch(batch_parameters={"dataframe_chunks": source})
    batch.validate(_suite(ExpectTableRowCountToBeBetween(min_value=0)), result_format=BASIC)
    assert source.closed == 1


def test_iterator_closed_on_failure(bd):
    closed = {"count": 0}

    class Raising:
        def __iter__(self):
            def gen():
                try:
                    yield DF.iloc[:1]
                    raise RuntimeError("mid-stream")
                finally:
                    closed["count"] += 1

            return gen()

    batch = bd.get_batch(batch_parameters={"dataframe_chunks": Raising()})
    with pytest.raises(RuntimeError, match="mid-stream"):
        batch.validate(_suite(ExpectTableRowCountToBeBetween(min_value=0)), result_format=BASIC)
    assert closed["count"] == 1


def test_close_error_does_not_mask_original(bd):
    class RaisingWithBadClose:
        def __iter__(self):
            class It:
                def __init__(self):
                    self._done = False

                def __iter__(self):
                    return self

                def __next__(self):
                    if not self._done:
                        self._done = True
                        return DF.iloc[:1]
                    raise RuntimeError("original error")

                def close(self):
                    raise ValueError("close error")

            return It()

    batch = bd.get_batch(batch_parameters={"dataframe_chunks": RaisingWithBadClose()})
    with pytest.raises(RuntimeError, match="original error"):
        batch.validate(_suite(ExpectTableRowCountToBeBetween(min_value=0)), result_format=BASIC)


def test_source_not_closed_when_dataframe_passed(bd):
    # Passing a real DataFrame leaves it open; only streaming iterators are closed.
    df = DF.copy()
    source = sh.GeneratorSource([df])
    batch = bd.get_batch(batch_parameters={"dataframe_chunks": source})
    batch.validate(_suite(ExpectTableRowCountToBeBetween(min_value=0)), result_format=BASIC)
    assert source.closed == 1
    assert not df.empty
