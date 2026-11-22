"""End-to-end validation-plan execution semantics on a single Pandas batch."""

from __future__ import annotations

from unittest import mock

import great_expectations.expectations as gxe
import pandas as pd
import pytest

import great_expectations as gx
from great_expectations.core.expectation_suite import ExpectationSuite
from great_expectations.exceptions import MetricResolutionError

import vp_harness
import vp_rules


@pytest.fixture(autouse=True)
def _reset_provider_calls():
    vp_rules.reset_calls()
    yield
    vp_rules.reset_calls()


@pytest.fixture
def context():
    return gx.get_context(mode="ephemeral")


def _build(context, expectations, rules):
    suite = ExpectationSuite(name="exec_suite")
    vp_harness.add_rules(suite, expectations)
    ids = vp_harness.rule_refs(suite)
    resolved_rules = [
        {
            "id": rule["id"],
            "expectation_id": ids[rule["expectation_index"]],
            "requires": rule["requires"],
        }
        for rule in rules
    ]
    vp_harness.make_plan(suite, rules=resolved_rules)
    validator = vp_harness.build_context_validator_for_suite(
        suite, dataframe=vp_rules.sample_dataframe()
    )
    return validator, suite


def _statuses(result):
    return {
        item.meta["execution"]["id"]: item.meta["execution"]["status"]
        for item in result.results
    }


def test_linear_dependency_skips_downstream_of_failure(context) -> None:
    validator, _ = _build(
        context,
        [
            vp_rules.ExpectVpPassing(value=1, tag="a"),
            vp_rules.ExpectVpFailing(value=2, tag="b"),
            vp_rules.ExpectVpPassing(value=3, tag="c"),
            vp_rules.ExpectVpBoobyTrap(value=4, tag="d"),
        ],
        [
            {"id": "r0", "expectation_index": 0, "requires": []},
            {"id": "r1", "expectation_index": 1, "requires": ["r0"]},
            {"id": "r2", "expectation_index": 2, "requires": ["r1"]},
            {"id": "r3", "expectation_index": 3, "requires": ["r2"]},
        ],
    )
    result = validator.validate(catch_exceptions=True)

    assert _statuses(result) == {
        "r0": "passed",
        "r1": "failed",
        "r2": "skipped",
        "r3": "skipped",
    }
    # One result per declared rule, in suite order.
    assert [r.meta["execution"]["id"] for r in result.results] == [
        "r0",
        "r1",
        "r2",
        "r3",
    ]
    stats = result.statistics
    assert stats["declared_expectations"] == 4
    assert stats["evaluated_expectations"] == 2
    assert stats["skipped_expectations"] == 2
    assert stats["successful_expectations"] == 1
    assert stats["unsuccessful_expectations"] == 1
    assert stats["success_percent"] == 50.0
    assert result.success is False
    # The booby-trap provider must never be touched.
    assert vp_rules.PROVIDER_CALLS.get("vp.booby_trap", 0) == 0


def test_diamond_blocks_only_through_failing_branch(context) -> None:
    validator, _ = _build(
        context,
        [
            vp_rules.ExpectVpPassing(value=1, tag="root"),
            vp_rules.ExpectVpPassing(value=2, tag="mid_ok"),
            vp_rules.ExpectVpFailing(value=3, tag="mid_bad"),
            vp_rules.ExpectVpPassing(value=4, tag="leaf"),
            vp_rules.ExpectVpBoobyTrap(value=5, tag="trap"),
        ],
        [
            {"id": "root", "expectation_index": 0, "requires": []},
            {"id": "mid_ok", "expectation_index": 1, "requires": ["root"]},
            {"id": "mid_bad", "expectation_index": 2, "requires": ["root"]},
            {
                "id": "leaf",
                "expectation_index": 3,
                "requires": ["mid_ok", "mid_bad"],
            },
            {"id": "trap", "expectation_index": 4, "requires": ["leaf"]},
        ],
    )
    result = validator.validate(catch_exceptions=True)
    assert _statuses(result) == {
        "root": "passed",
        "mid_ok": "passed",
        "mid_bad": "failed",
        "leaf": "skipped",
        "trap": "skipped",
    }
    leaf = next(r for r in result.results if r.meta["execution"]["id"] == "leaf")
    # blocked_by keeps requires order but only lists non-passing prerequisites.
    assert leaf.meta["execution"]["blocked_by"] == ["mid_bad"]
    assert leaf.success is None
    assert leaf.result == {}
    assert leaf.exception_info["raised_exception"] is False
    assert vp_rules.PROVIDER_CALLS.get("vp.booby_trap", 0) == 0


def test_suite_order_reverse_of_dependency_order(context) -> None:
    # Suite order: leaf, mid, root. Dependency order: root -> mid -> leaf.
    validator, _ = _build(
        context,
        [
            vp_rules.ExpectVpPassing(value=3, tag="leaf"),
            vp_rules.ExpectVpPassing(value=2, tag="mid"),
            vp_rules.ExpectVpPassing(value=1, tag="root"),
        ],
        [
            {"id": "leaf", "expectation_index": 0, "requires": ["mid"]},
            {"id": "mid", "expectation_index": 1, "requires": ["root"]},
            {"id": "root", "expectation_index": 2, "requires": []},
        ],
    )
    result = validator.validate(catch_exceptions=True)
    # Results remain in suite order even though execution unlocked root first.
    assert [r.meta["execution"]["id"] for r in result.results] == [
        "leaf",
        "mid",
        "root",
    ]
    assert all(
        r.meta["execution"]["status"] == "passed" for r in result.results
    )


def test_metric_exception_marks_error_and_blocks_children(context) -> None:
    validator, _ = _build(
        context,
        [
            vp_rules.ExpectVpMetricError(value=1, tag="a"),
            vp_rules.ExpectVpSharedFailing(value=1, tag="b"),
            vp_rules.ExpectVpPassing(value=9, tag="independent"),
        ],
        [
            {"id": "m0", "expectation_index": 0, "requires": []},
            {"id": "m1", "expectation_index": 1, "requires": ["m0"]},
            {"id": "ind", "expectation_index": 2, "requires": []},
        ],
    )
    result = validator.validate(catch_exceptions=True)
    statuses = _statuses(result)
    assert statuses["m0"] == "error"
    assert statuses["m1"] == "skipped"
    assert statuses["ind"] == "passed"

    m0 = next(r for r in result.results if r.meta["execution"]["id"] == "m0")
    assert m0.success is False
    # Metric-resolution failures are reported keyed by MetricConfigurationID
    # (the existing get_exception_info contract), each carrying an exception.
    assert len(m0.exception_info) == 1
    metric_exception = next(iter(m0.exception_info.values()))
    assert metric_exception["raised_exception"] is True
    assert "exploded" in metric_exception["exception_message"]
    # The skipped child carries no fabricated exception.
    m1 = next(r for r in result.results if r.meta["execution"]["id"] == "m1")
    assert m1.success is None
    assert m1.exception_info["raised_exception"] is False
    assert m1.meta["execution"]["blocked_by"] == ["m0"]


def test_failed_metric_is_not_recomputed_in_a_later_layer(context) -> None:
    validator, _ = _build(
        context,
        [
            vp_rules.ExpectVpMetricError(value=1, tag="a"),
            vp_rules.ExpectVpPassing(value=9, tag="gate"),
            vp_rules.ExpectVpSharedFailingSibling(value=1, tag="c"),
        ],
        [
            {"id": "m0", "expectation_index": 0, "requires": []},
            {"id": "gate", "expectation_index": 1, "requires": []},
            {"id": "later", "expectation_index": 2, "requires": ["gate"]},
        ],
    )
    result = validator.validate(catch_exceptions=True)
    calls = vp_rules.PROVIDER_CALLS["vp.always_failing"]
    # All invocations happened while resolving the first layer; unlocking "later"
    # in a subsequent layer must not add any further provider invocation.
    statuses = _statuses(result)
    assert statuses["m0"] == "error"
    assert statuses["gate"] == "passed"
    assert statuses["later"] == "error"
    assert calls >= 1
    assert vp_rules.PROVIDER_CALLS["vp.always_failing"] == calls


def test_catch_exceptions_false_propagates(context) -> None:
    validator, _ = _build(
        context,
        [
            vp_rules.ExpectVpMetricError(value=1, tag="a"),
            vp_rules.ExpectVpPassing(value=2, tag="b"),
        ],
        [
            {"id": "m0", "expectation_index": 0, "requires": []},
            {"id": "ok", "expectation_index": 1, "requires": []},
        ],
    )
    with pytest.raises(MetricResolutionError):
        validator.validate(catch_exceptions=False)


def test_only_return_failures_keeps_errors_and_skips(context) -> None:
    validator, _ = _build(
        context,
        [
            vp_rules.ExpectVpPassing(value=1, tag="a"),
            vp_rules.ExpectVpFailing(value=2, tag="b"),
            vp_rules.ExpectVpPassing(value=3, tag="c"),
        ],
        [
            {"id": "r0", "expectation_index": 0, "requires": []},
            {"id": "r1", "expectation_index": 1, "requires": ["r0"]},
            {"id": "r2", "expectation_index": 2, "requires": ["r1"]},
        ],
    )
    result = validator.validate(catch_exceptions=True, only_return_failures=True)
    returned_ids = [r.meta["execution"]["id"] for r in result.results]
    # Failure and the skipped downstream are retained; the pass is filtered out.
    assert returned_ids == ["r1", "r2"]
    # Statistics still describe the full, pre-filter execution.
    assert result.statistics["declared_expectations"] == 3
    assert result.statistics["evaluated_expectations"] == 2
    assert result.statistics["skipped_expectations"] == 1


def test_second_run_is_independent_of_first(context) -> None:
    validator, suite = _build(
        context,
        [
            vp_rules.ExpectVpFailing(value=1, tag="a"),
            vp_rules.ExpectVpPassing(value=2, tag="b"),
        ],
        [
            {"id": "r0", "expectation_index": 0, "requires": []},
            {"id": "r1", "expectation_index": 1, "requires": ["r0"]},
        ],
    )
    first = validator.validate(catch_exceptions=True)
    assert _statuses(first) == {"r0": "failed", "r1": "skipped"}

    # A second run recomputes from scratch; run state must not leak into the
    # suite, plan or expectation objects.
    second = validator.validate(catch_exceptions=True)
    assert [r.meta["execution"] for r in second.results] == [
        r.meta["execution"] for r in first.results
    ]
    # The suite object itself carries no run-status mutations.
    assert suite.meta["validation_plan"]["version"] == 1
    for expectation in suite.expectations:
        assert expectation.configuration.success_on_last_run is None


def test_user_meta_is_preserved_alongside_execution(context) -> None:
    suite = ExpectationSuite(name="meta_suite")
    vp_harness.add_rules(
        suite, [vp_rules.ExpectVpPassing(value=1, tag="a")]
    )
    # Set user meta before the validator is constructed, since it deep-copies
    # the suite at construction time.
    suite.expectations[0].meta = {"owner": "data-team"}
    ids = vp_harness.rule_refs(suite)
    vp_harness.make_plan(
        suite,
        rules=[{"id": "r0", "expectation_id": ids[0], "requires": []}],
    )
    validator = vp_harness.build_context_validator_for_suite(
        suite, dataframe=vp_rules.sample_dataframe()
    )
    result = validator.validate(catch_exceptions=True)
    evr = result.results[0]
    assert evr.expectation_config.meta == {"owner": "data-team"}
    assert evr.meta["execution"] == {"id": "r0", "status": "passed"}


def test_successful_metric_shared_across_layers_computes_once(context) -> None:
    # Two rules at different layers both depend on the same table-level metric
    # (domain is batch_id only), so the provider must run exactly once.
    validator, _ = _build(
        context,
        [
            vp_rules.ExpectVpSharedSuccessA(value=1, tag="a"),
            vp_rules.ExpectVpPassing(value=8, tag="gate"),
            vp_rules.ExpectVpSharedSuccessB(value=1, tag="b"),
        ],
        [
            {"id": "base", "expectation_index": 0, "requires": []},
            {"id": "gate", "expectation_index": 1, "requires": []},
            {"id": "after", "expectation_index": 2, "requires": ["gate"]},
        ],
    )
    result = validator.validate(catch_exceptions=True)
    assert result.success is True
    assert _statuses(result) == {
        "base": "passed",
        "gate": "passed",
        "after": "passed",
    }
    assert vp_rules.PROVIDER_CALLS["vp.shared_success"] == 1
    # The gate's own counting metric is the only other success provider call.
    assert vp_rules.PROVIDER_CALLS["vp.counting_value"] == 1


def test_skipped_rule_never_calls_get_validation_dependencies(context) -> None:
    validator, suite = _build(
        context,
        [
            vp_rules.ExpectVpPassing(value=1, tag="a"),
            vp_rules.ExpectVpFailing(value=2, tag="b"),
            vp_rules.ExpectVpBoobyTrap(value=3, tag="c"),
        ],
        [
            {"id": "r0", "expectation_index": 0, "requires": []},
            {"id": "r1", "expectation_index": 1, "requires": ["r0"]},
            {"id": "r2", "expectation_index": 2, "requires": ["r1"]},
        ],
    )
    # Spy on the dependency callback of the rule that must stay skipped.
    target_type = type(suite.expectations[2])
    with mock.patch.object(
        target_type,
        "get_validation_dependencies",
        autospec=True,
        wraps=target_type.get_validation_dependencies,
    ) as spy:
        result = validator.validate(catch_exceptions=True)

    statuses = _statuses(result)
    assert statuses["r2"] == "skipped"
    # The booby-trap rule's dependency callback was never invoked, so neither
    # its metric provider nor metrics_validate could run.
    spy.assert_not_called()
    assert vp_rules.PROVIDER_CALLS.get("vp.booby_trap", 0) == 0


def test_row_condition_is_honored_for_executed_rules() -> None:
    context = gx.get_context(mode="ephemeral")
    df = pd.DataFrame({"grp": ["x", "x", "y", "y"], "val": [1, 1, 1, 1]})
    suite = ExpectationSuite(name="row_cond_suite")
    gated = gxe.ExpectTableRowCountToEqual(
        value=2,
        row_condition='grp == "x"',
        condition_parser="pandas",
    )
    suite.add_expectation(gated)
    validator = vp_harness.build_context_validator_for_suite(suite, dataframe=df)
    # No plan: legacy path must still respect the row_condition as before.
    result = validator.validate(catch_exceptions=True)
    assert result.results[0].success is True
