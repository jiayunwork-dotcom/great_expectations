"""ValidationDefinition.run end-to-end and result serialization round trips."""

from __future__ import annotations

import json

import great_expectations as gx
from great_expectations.core.expectation_suite import ExpectationSuite
from great_expectations.core.expectation_validation_result import (
    expectationSuiteValidationResultSchema,
)
from great_expectations.core.validation_definition import ValidationDefinition

import vp_harness
import vp_rules


def _build_validation_definition(context):
    datasource = context.data_sources.add_pandas(name="vd_pandas")
    asset = datasource.add_dataframe_asset(name="vd_asset")
    batch_definition = asset.add_batch_definition_whole_dataframe(
        name="vd_batch_def"
    )

    suite = ExpectationSuite(name="vd_suite")
    vp_harness.add_rules(
        suite,
        [
            vp_rules.ExpectVpPassing(value=1, tag="a"),
            vp_rules.ExpectVpFailing(value=2, tag="b"),
            vp_rules.ExpectVpBoobyTrap(value=3, tag="c"),
        ],
    )
    # Persisting the (previously unsaved) suite assigns expectation ids, so the
    # plan must be declared against the stored suite's ids.
    suite = context.suites.add(suite)
    ids = [expectation.id for expectation in suite.expectations]
    suite.meta["validation_plan"] = {
        "version": 1,
        "rules": [
            {"id": "r0", "expectation_id": ids[0], "requires": []},
            {"id": "r1", "expectation_id": ids[1], "requires": ["r0"]},
            {"id": "r2", "expectation_id": ids[2], "requires": ["r1"]},
        ],
    }
    assert suite.validation_plan is not None
    # Re-persist so the plan-bearing suite is the stored, fresh version.
    suite.save()
    vd = context.validation_definitions.add(
        ValidationDefinition(name="vd", data=batch_definition, suite=suite)
    )
    return vd


def test_validation_definition_run_uses_suite_plan_and_persists() -> None:
    vp_rules.reset_calls()
    context = gx.get_context(mode="ephemeral")
    vd = _build_validation_definition(context)

    result = vd.run(batch_parameters={"dataframe": vp_rules.sample_dataframe()})

    statuses = {
        item.meta["execution"]["id"]: item.meta["execution"]["status"]
        for item in result.results
    }
    assert statuses == {"r0": "passed", "r1": "failed", "r2": "skipped"}
    assert result.statistics == {
        "evaluated_expectations": 2,
        "successful_expectations": 1,
        "unsuccessful_expectations": 1,
        "success_percent": 50.0,
        "declared_expectations": 3,
        "skipped_expectations": 1,
    }
    assert result.success is False
    assert vp_rules.PROVIDER_CALLS.get("vp.booby_trap", 0) == 0


def test_validation_result_json_round_trip_keeps_skip_and_stats() -> None:
    vp_rules.reset_calls()
    context = gx.get_context(mode="ephemeral")
    vd = _build_validation_definition(context)
    result = vd.run(batch_parameters={"dataframe": vp_rules.sample_dataframe()})

    payload = json.dumps(expectationSuiteValidationResultSchema.dump(result))
    restored = expectationSuiteValidationResultSchema.loads(payload)

    # success=None and blocked_by survive the full JSON round trip.
    by_id = {item.meta["execution"]["id"]: item for item in restored.results}
    assert by_id["r2"].success is None
    assert by_id["r2"].result == {}
    assert by_id["r2"].meta["execution"]["status"] == "skipped"
    assert by_id["r2"].meta["execution"]["blocked_by"] == ["r1"]
    assert by_id["r1"].success is False
    assert by_id["r0"].success is True
    assert restored.statistics["skipped_expectations"] == 1
    assert restored.statistics["declared_expectations"] == 3
    assert restored.statistics["evaluated_expectations"] == 2


def test_validation_definition_saves_and_reads_back_result() -> None:
    vp_rules.reset_calls()
    context = gx.get_context(mode="ephemeral")
    vd = _build_validation_definition(context)
    result = vd.run(batch_parameters={"dataframe": vp_rules.sample_dataframe()})
    result_id = result.meta.get("validation_id")

    # The ephemeral validation results store retained the run.
    keys = list(context.validation_results_store.list_keys())
    assert len(keys) == 1
    stored = context.validation_results_store.get(keys[0])
    statuses = {
        item.meta["execution"]["id"]: item.meta["execution"]["status"]
        for item in stored.results
    }
    assert statuses == {"r0": "passed", "r1": "failed", "r2": "skipped"}
    assert stored.statistics["skipped_expectations"] == 1
    assert result_id is not None
