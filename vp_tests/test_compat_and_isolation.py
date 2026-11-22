"""Backward compatibility (no plan), empty suite, and isolation guarantees."""

from __future__ import annotations

import great_expectations as gx
import great_expectations.expectations as gxe
import pandas as pd
import pytest
from great_expectations.core.expectation_suite import ExpectationSuite
from great_expectations.core.expectation_validation_result import (
    expectationSuiteValidationResultSchema,
)

import vp_harness
import vp_rules


@pytest.fixture(autouse=True)
def _reset_provider_calls():
    vp_rules.reset_calls()
    yield
    vp_rules.reset_calls()


def test_no_plan_keeps_legacy_behavior_and_statistics(context=None) -> None:
    context = gx.get_context(mode="ephemeral")
    suite = ExpectationSuite(name="legacy")
    vp_harness.add_rules(
        suite,
        [
            vp_rules.ExpectVpPassing(value=1, tag="a"),
            vp_rules.ExpectVpFailing(value=2, tag="b"),
        ],
    )
    assert suite.validation_plan is None
    validator = vp_harness.build_context_validator_for_suite(
        suite, dataframe=vp_rules.sample_dataframe()
    )
    result = validator.validate(catch_exceptions=True)

    # Without a plan every declared expectation runs; no execution meta is added.
    assert len(result.results) == 2
    for item in result.results:
        assert "execution" not in item.meta
        assert item.success is not None
    stats = result.statistics
    assert stats["declared_expectations"] == 2
    assert stats["evaluated_expectations"] == 2
    assert stats["skipped_expectations"] == 0
    assert stats["unsuccessful_expectations"] == 1
    # Legacy keys remain present and unchanged.
    assert "success_percent" in stats


def test_empty_suite_succeeds_with_legacy_semantics() -> None:
    context = gx.get_context(mode="ephemeral")
    suite = ExpectationSuite(name="empty")
    validator = vp_harness.build_context_validator_for_suite(
        suite, dataframe=vp_rules.sample_dataframe()
    )
    result = validator.validate(catch_exceptions=True)
    assert result.results == []
    assert result.success is True
    assert result.statistics["evaluated_expectations"] == 0
    assert result.statistics["declared_expectations"] == 0
    assert result.statistics["skipped_expectations"] == 0
    assert result.statistics["success_percent"] is None


def test_empty_suite_with_empty_plan_round_trips() -> None:
    context = gx.get_context(mode="ephemeral")
    suite = ExpectationSuite(name="empty_plan")
    suite.meta["validation_plan"] = {"version": 1, "rules": []}
    assert suite.validation_plan.execution_layers() == []
    validator = vp_harness.build_context_validator_for_suite(
        suite, dataframe=vp_rules.sample_dataframe()
    )
    result = validator.validate(catch_exceptions=True)
    assert result.success is True
    assert result.results == []


def test_change_batch_does_not_reuse_prior_run_state() -> None:
    context = gx.get_context(mode="ephemeral")
    suite = ExpectationSuite(name="batch_change")
    vp_harness.add_rules(
        suite, [vp_rules.ExpectVpPassing(value=1, tag="a")]
    )
    ids = vp_harness.rule_refs(suite)
    vp_harness.make_plan(
        suite,
        rules=[{"id": "r0", "expectation_id": ids[0], "requires": []}],
    )
    validator = vp_harness.build_context_validator_for_suite(
        suite, dataframe=vp_rules.sample_dataframe()
    )
    first = validator.validate(catch_exceptions=True)
    first_calls = dict(vp_rules.PROVIDER_CALLS)
    assert first.results[0].success is True

    # A new run on a different batch recomputes everything; nothing is cached
    # across runs at the plan/validator orchestration level.
    second_df = pd.DataFrame({"a": [9], "b": ["z"]})
    validator2 = vp_harness.build_context_validator_for_suite(
        suite, dataframe=second_df
    )
    second = validator2.validate(catch_exceptions=True)
    assert second.results[0].success is True
    assert vp_rules.PROVIDER_CALLS["vp.counting_value"] >= first_calls.get(
        "vp.counting_value", 0
    ) + 1


def test_result_format_is_honored_for_executed_rules() -> None:
    context = gx.get_context(mode="ephemeral")
    suite = ExpectationSuite(name="rf")
    vp_harness.add_rules(
        suite,
        [
            vp_rules.ExpectVpPassing(value=1, tag="a"),
            vp_rules.ExpectVpFailing(value=2, tag="b"),
        ],
    )
    ids = vp_harness.rule_refs(suite)
    vp_harness.make_plan(
        suite,
        rules=[
            {"id": "r0", "expectation_id": ids[0], "requires": []},
            {"id": "r1", "expectation_id": ids[1], "requires": ["r0"]},
        ],
    )
    validator = vp_harness.build_context_validator_for_suite(
        suite, dataframe=vp_rules.sample_dataframe()
    )
    result = validator.validate(
        catch_exceptions=True, result_format="BOOLEAN_ONLY"
    )
    # BOOLEAN_ONLY strips result detail on the executed rule.
    executed = next(
        r for r in result.results if r.meta["execution"]["status"] == "passed"
    )
    assert executed.result == {}


def test_result_json_round_trip_without_plan() -> None:
    context = gx.get_context(mode="ephemeral")
    suite = ExpectationSuite(name="legacy_json")
    suite.add_expectation(gxe.ExpectTableColumnCountToEqual(value=2))
    validator = vp_harness.build_context_validator_for_suite(
        suite, dataframe=vp_rules.sample_dataframe()
    )
    result = validator.validate(catch_exceptions=True)
    import json

    payload = json.dumps(expectationSuiteValidationResultSchema.dump(result))
    restored = expectationSuiteValidationResultSchema.loads(payload)
    assert restored.statistics["evaluated_expectations"] == 1
    assert restored.statistics["skipped_expectations"] == 0
    assert all("execution" not in item.meta for item in restored.results)
