"""Validation plan parsing and structural validation tests."""

from __future__ import annotations

import pytest

from great_expectations.core.validation_plan import (
    VALIDATION_PLAN_VERSION,
    ValidationPlan,
)
from great_expectations.exceptions import InvalidValidationPlanError

EXP_IDS = ["e0", "e1", "e2"]


def _valid_dict() -> dict:
    return {
        "version": 1,
        "rules": [
            {"id": "a", "expectation_id": "e0", "requires": []},
            {"id": "b", "expectation_id": "e1", "requires": ["a"]},
            {"id": "c", "expectation_id": "e2", "requires": ["a", "b"]},
        ],
    }


def test_valid_plan_parses_and_layers() -> None:
    plan = ValidationPlan.from_dict(_valid_dict(), expectation_ids=EXP_IDS)
    assert plan.version == 1
    layers = plan.execution_layers()
    assert [[r.id for r in layer] for layer in layers] == [["a"], ["b"], ["c"]]


def test_requires_order_is_preserved() -> None:
    data = _valid_dict()
    data["rules"][2]["requires"] = ["b", "a"]
    plan = ValidationPlan.from_dict(data, expectation_ids=EXP_IDS)
    assert plan.rule_by_id["c"].requires == ("b", "a")


@pytest.mark.parametrize(
    "bad_version",
    [True, False, 1.0, "1", None, 0, 2],
)
def test_version_must_be_non_bool_integer_one(bad_version) -> None:
    data = _valid_dict()
    data["version"] = bad_version
    with pytest.raises(InvalidValidationPlanError):
        ValidationPlan.from_dict(data, expectation_ids=EXP_IDS)


def test_missing_top_level_fields() -> None:
    with pytest.raises(InvalidValidationPlanError, match="version"):
        ValidationPlan.from_dict({"rules": []}, expectation_ids=[])
    with pytest.raises(InvalidValidationPlanError, match="rules"):
        ValidationPlan.from_dict({"version": 1}, expectation_ids=[])


def test_unknown_top_level_field() -> None:
    data = _valid_dict()
    data["extra"] = 1
    with pytest.raises(InvalidValidationPlanError, match="unknown field"):
        ValidationPlan.from_dict(data, expectation_ids=EXP_IDS)


def test_unknown_rule_field() -> None:
    data = _valid_dict()
    data["rules"][0]["bogus"] = 1
    with pytest.raises(InvalidValidationPlanError, match="unknown field"):
        ValidationPlan.from_dict(data, expectation_ids=EXP_IDS)


def test_missing_rule_field() -> None:
    data = _valid_dict()
    del data["rules"][0]["requires"]
    with pytest.raises(InvalidValidationPlanError, match="missing required field"):
        ValidationPlan.from_dict(data, expectation_ids=EXP_IDS)


def test_rule_id_must_be_nonempty_unique_string() -> None:
    data = _valid_dict()
    data["rules"][1]["id"] = "a"
    with pytest.raises(InvalidValidationPlanError, match="not unique"):
        ValidationPlan.from_dict(data, expectation_ids=EXP_IDS)

    data = _valid_dict()
    data["rules"][0]["id"] = ""
    with pytest.raises(InvalidValidationPlanError, match="non-empty string"):
        ValidationPlan.from_dict(data, expectation_ids=EXP_IDS)


def test_expectation_id_cannot_repeat() -> None:
    data = _valid_dict()
    data["rules"][1]["expectation_id"] = "e0"
    with pytest.raises(InvalidValidationPlanError, match="more than one rule"):
        ValidationPlan.from_dict(data, expectation_ids=EXP_IDS)


def test_duplicate_requires_entry_rejected() -> None:
    data = _valid_dict()
    data["rules"][1]["requires"] = ["a", "a"]
    with pytest.raises(InvalidValidationPlanError, match="more than once"):
        ValidationPlan.from_dict(data, expectation_ids=EXP_IDS)


def test_unknown_requires_reference() -> None:
    data = _valid_dict()
    data["rules"][0]["requires"] = ["ghost"]
    with pytest.raises(InvalidValidationPlanError, match="unknown requires reference"):
        ValidationPlan.from_dict(data, expectation_ids=EXP_IDS)


def test_self_dependency_rejected() -> None:
    data = _valid_dict()
    data["rules"][0]["requires"] = ["a"]
    with pytest.raises(InvalidValidationPlanError, match="depend on itself"):
        ValidationPlan.from_dict(data, expectation_ids=EXP_IDS)


def test_dependency_cycle_rejected() -> None:
    data = _valid_dict()
    # a -> b -> a
    data["rules"][0]["requires"] = ["b"]
    with pytest.raises(InvalidValidationPlanError, match="cycle"):
        ValidationPlan.from_dict(data, expectation_ids=EXP_IDS)


def test_coverage_must_match_suite_expectations() -> None:
    # An expectation without a rule.
    with pytest.raises(InvalidValidationPlanError, match="without a rule"):
        ValidationPlan.from_dict(_valid_dict(), expectation_ids=EXP_IDS + ["e3"])

    # A rule whose expectation no longer exists.
    with pytest.raises(InvalidValidationPlanError, match="no longer in the suite"):
        ValidationPlan.from_dict(_valid_dict(), expectation_ids=["e0", "e1"])


def test_rules_must_be_a_list_and_requires_a_list() -> None:
    data = _valid_dict()
    data["rules"] = {}
    with pytest.raises(InvalidValidationPlanError, match="'rules' must be a list"):
        ValidationPlan.from_dict(data, expectation_ids=EXP_IDS)

    data = _valid_dict()
    data["rules"][0]["requires"] = "a"
    with pytest.raises(InvalidValidationPlanError, match="'requires' must be a list"):
        ValidationPlan.from_dict(data, expectation_ids=EXP_IDS)


def test_plan_round_trips_through_dict() -> None:
    plan = ValidationPlan.from_dict(_valid_dict(), expectation_ids=EXP_IDS)
    again = ValidationPlan.from_dict(plan.to_dict(), expectation_ids=EXP_IDS)
    assert [[r.id for r in layer] for layer in again.execution_layers()] == [
        ["a"],
        ["b"],
        ["c"],
    ]


def test_version_constant() -> None:
    assert VALIDATION_PLAN_VERSION == 1
