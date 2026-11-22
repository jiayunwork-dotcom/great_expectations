"""ExpectationSuite integration of the validation_plan: persistence, copy, staleness."""

from __future__ import annotations

import copy
import json
import uuid

import pytest

import great_expectations as gx
import great_expectations.expectations as gxe
from great_expectations.core.expectation_suite import (
    ExpectationSuite,
    expectationSuiteSchema,
)
from great_expectations.exceptions import InvalidValidationPlanError


@pytest.fixture
def context():
    return gx.get_context(mode="ephemeral")


@pytest.fixture
def suite_with_plan(context) -> ExpectationSuite:
    suite = ExpectationSuite(name="plan_suite")
    e0 = gxe.ExpectTableColumnCountToEqual(value=2)
    e1 = gxe.ExpectTableColumnCountToBeBetween(min_value=1, max_value=5)
    suite.add_expectation(e0)
    suite.add_expectation(e1)
    e0.id = str(uuid.uuid4())
    e1.id = str(uuid.uuid4())
    id0, id1 = e0.id, e1.id
    suite.meta["validation_plan"] = {
        "version": 1,
        "rules": [
            {"id": "has_shape", "expectation_id": id0, "requires": []},
            {"id": "shape_range", "expectation_id": id1, "requires": ["has_shape"]},
        ],
    }
    assert suite.validation_plan is not None
    return suite


def test_plan_stored_in_meta(suite_with_plan) -> None:
    plan = suite_with_plan.validation_plan
    assert [r.id for r in plan.rules] == ["has_shape", "shape_range"]


def test_suite_json_round_trip_keeps_plan(suite_with_plan) -> None:
    raw = json.dumps(expectationSuiteSchema.dump(suite_with_plan))
    loaded_dict = expectationSuiteSchema.loads(raw)
    loaded = ExpectationSuite(**loaded_dict)
    plan = loaded.validation_plan
    assert plan is not None
    assert [r.id for r in plan.rules] == ["has_shape", "shape_range"]
    assert plan.rule_by_id["shape_range"].requires == ("has_shape",)


def test_to_json_dict_includes_plan(suite_with_plan) -> None:
    dumped = suite_with_plan.to_json_dict()
    assert dumped["meta"]["validation_plan"]["version"] == 1
    assert len(dumped["meta"]["validation_plan"]["rules"]) == 2


def test_deepcopy_carries_plan(suite_with_plan) -> None:
    copied = copy.deepcopy(suite_with_plan)
    assert copied.validation_plan.to_dict() == suite_with_plan.validation_plan.to_dict()


def test_copy_builtin_carries_plan(suite_with_plan) -> None:
    copied = copy.copy(suite_with_plan)
    # copy.copy on SerializableDictDot-based suite still exposes the same meta plan.
    assert copied.meta["validation_plan"] == suite_with_plan.meta["validation_plan"]


def test_adding_expectation_invalidates_old_plan(suite_with_plan) -> None:
    new_config = gxe.ExpectTableRowCountToEqual(value=4).configuration
    new_config.id = str(uuid.uuid4())
    suite_with_plan.add_expectation_configuration(new_config)
    with pytest.raises(InvalidValidationPlanError, match="without a rule"):
        suite_with_plan.validation_plan


def test_idless_expectation_rejected_with_plan(suite_with_plan) -> None:
    # An expectation configuration without an id cannot be covered by any rule.
    idless = gxe.ExpectTableRowCountToEqual(value=9).configuration
    idless.id = None
    suite_with_plan.add_expectation_configuration(idless)
    with pytest.raises(InvalidValidationPlanError, match="declare an id"):
        suite_with_plan.validation_plan


def test_removing_expectation_invalidates_old_plan(suite_with_plan) -> None:
    target = suite_with_plan.expectations[0]
    suite_with_plan.delete_expectation(target)
    with pytest.raises(InvalidValidationPlanError, match="no longer in the suite"):
        suite_with_plan.validation_plan


def test_invalid_plan_rejected_on_construction(context) -> None:
    suite = ExpectationSuite(name="bad")
    config = gxe.ExpectTableColumnCountToEqual(value=2).configuration
    config.id = str(uuid.uuid4())
    suite.add_expectation_configuration(config)
    suite.meta["validation_plan"] = {
        "version": 1,
        "rules": [
            {"id": "r", "expectation_id": str(uuid.uuid4()), "requires": []},
        ],
    }
    with pytest.raises(InvalidValidationPlanError):
        suite.validation_plan


def test_suite_without_plan_returns_none(context) -> None:
    suite = ExpectationSuite(name="no_plan")
    assert suite.validation_plan is None


def test_empty_suite_plan_must_have_no_rules(context) -> None:
    suite = ExpectationSuite(name="empty")
    suite.meta["validation_plan"] = {"version": 1, "rules": []}
    assert suite.validation_plan is not None
    assert suite.validation_plan.execution_layers() == []


def test_empty_suite_rejects_rules_without_expectations(context) -> None:
    suite = ExpectationSuite(name="empty_bad")
    suite.meta["validation_plan"] = {
        "version": 1,
        "rules": [{"id": "r", "expectation_id": "x", "requires": []}],
    }
    with pytest.raises(InvalidValidationPlanError):
        suite.validation_plan


def test_bool_version_rejected_at_suite_level(context) -> None:
    suite = ExpectationSuite(name="bool_ver")
    suite.meta["validation_plan"] = {"version": True, "rules": []}
    with pytest.raises(InvalidValidationPlanError):
        suite.validation_plan
