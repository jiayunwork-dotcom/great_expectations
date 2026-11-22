from __future__ import annotations

import json
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from great_expectations.exceptions import InvalidValidationPlanError

if TYPE_CHECKING:
    from great_expectations.alias_types import JSONValues

VALIDATION_PLAN_META_KEY = "validation_plan"
VALIDATION_PLAN_VERSION = 1
_PLAN_TOP_LEVEL_FIELDS = frozenset({"version", "rules"})
_RULE_FIELDS = frozenset({"id", "expectation_id", "requires"})


@dataclass(frozen=True)
class ValidationPlanRule:
    """A single rule of a :class:`ValidationPlan`.

    Args:
        id: Unique, non-empty identifier of the rule.
        expectation_id: ``Expectation.id`` of the expectation declared by this rule.
        requires: Rule ids (unique, without self reference) that must succeed before
            the expectation of this rule is evaluated.
    """

    id: str
    expectation_id: str
    requires: Tuple[str, ...]

    def to_dict(self) -> Dict[str, JSONValues]:
        return {
            "id": self.id,
            "expectation_id": self.expectation_id,
            "requires": list(self.requires),
        }


@dataclass(frozen=True)
class ValidationPlan:
    """Dependency plan describing the execution order of a suite's expectations.

    A plan is only valid for the suite it was declared against: the declared rules
    must cover every expectation of the suite exactly once, by ``Expectation.id``.

    Args:
        version: Schema version; only the non-bool integer ``1`` is accepted.
        rules: The full set of rules, one per expectation of the suite.
    """

    version: int
    rules: Tuple[ValidationPlanRule, ...]

    def to_dict(self) -> Dict[str, JSONValues]:
        return {
            "version": self.version,
            "rules": [rule.to_dict() for rule in self.rules],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @property
    def rule_by_id(self) -> Dict[str, ValidationPlanRule]:
        return {rule.id: rule for rule in self.rules}

    @property
    def rule_by_expectation_id(self) -> Dict[str, ValidationPlanRule]:
        return {rule.expectation_id: rule for rule in self.rules}

    def rule_for_expectation(self, expectation_id: Optional[str]) -> ValidationPlanRule:
        if expectation_id is None:
            raise InvalidValidationPlanError(
                "validation_plan references an expectation without an id."
            )
        try:
            return self.rule_by_expectation_id[expectation_id]
        except KeyError:
            raise InvalidValidationPlanError(
                f"validation_plan has no rule for expectation_id {expectation_id!r}."
            )

    def execution_layers(self) -> List[List[ValidationPlanRule]]:
        """Return the rules grouped into dependency layers (Kahn's algorithm).

        Rules in the same layer only depend on rules of earlier layers and may be
        evaluated together. Raises if the plan contains a cycle.
        """
        rules_by_id = self.rule_by_id
        dependents: Dict[str, List[str]] = {rule_id: [] for rule_id in rules_by_id}
        indegree: Dict[str, int] = dict.fromkeys(rules_by_id, 0)
        for rule in self.rules:
            indegree[rule.id] = len(rule.requires)
            for prerequisite_id in rule.requires:
                dependents[prerequisite_id].append(rule.id)

        layers: List[List[ValidationPlanRule]] = []
        ready = sorted(rule_id for rule_id, degree in indegree.items() if degree == 0)
        visited = 0
        while ready:
            current_layer = [rules_by_id[rule_id] for rule_id in ready]
            layers.append(current_layer)
            next_ready: List[str] = []
            for rule_id in ready:
                visited += 1
                for dependent_id in dependents[rule_id]:
                    indegree[dependent_id] -= 1
                    if indegree[dependent_id] == 0:
                        next_ready.append(dependent_id)
            ready = sorted(next_ready)

        if visited != len(self.rules):
            cycle_ids = sorted(rule_id for rule_id, degree in indegree.items() if degree > 0)
            raise InvalidValidationPlanError(
                "validation_plan contains a dependency cycle involving rule(s): "
                f"{', '.join(cycle_ids)}."
            )
        return layers

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        expectation_ids: Optional[Sequence[str]] = None,
    ) -> ValidationPlan:
        """Parse and validate a plan from its JSON-like representation.

        Args:
            data: The raw plan, e.g. ``suite.meta["validation_plan"]``.
            expectation_ids: When provided, the plan must declare exactly these
                expectation ids (the current expectations of the target suite).
        """
        if not isinstance(data, Mapping):
            raise InvalidValidationPlanError(
                f"validation_plan must be an object, received {type(data).__name__}."
            )

        unknown_fields = sorted(set(data.keys()) - _PLAN_TOP_LEVEL_FIELDS)
        if unknown_fields:
            raise InvalidValidationPlanError(
                "validation_plan contains unknown field(s): "
                f"{', '.join(unknown_fields)}."
            )

        if "version" not in data:
            raise InvalidValidationPlanError("validation_plan is missing required field 'version'.")
        version = data["version"]
        # bool is a subclass of int and must be rejected explicitly.
        if isinstance(version, bool) or not isinstance(version, int):
            raise InvalidValidationPlanError(
                "validation_plan 'version' must be a non-bool integer, "
                f"received {version!r} ({type(version).__name__})."
            )
        if version != VALIDATION_PLAN_VERSION:
            raise InvalidValidationPlanError(
                f"validation_plan 'version' must be {VALIDATION_PLAN_VERSION}, received {version}."
            )

        if "rules" not in data:
            raise InvalidValidationPlanError("validation_plan is missing required field 'rules'.")
        raw_rules = data["rules"]
        if not isinstance(raw_rules, list):
            raise InvalidValidationPlanError(
                "validation_plan 'rules' must be a list, "
                f"received {type(raw_rules).__name__}."
            )

        rules: List[ValidationPlanRule] = []
        seen_rule_ids: set[str] = set()
        seen_expectation_ids: set[str] = set()
        for index, raw_rule in enumerate(raw_rules):
            rules.append(
                _parse_rule(
                    raw_rule=raw_rule,
                    index=index,
                    seen_rule_ids=seen_rule_ids,
                    seen_expectation_ids=seen_expectation_ids,
                )
            )

        plan = cls(version=version, rules=tuple(rules))

        rule_ids = set(plan.rule_by_id.keys())
        for rule in plan.rules:
            unknown_references = sorted(set(rule.requires) - rule_ids)
            if unknown_references:
                raise InvalidValidationPlanError(
                    f"validation_plan rule {rule.id!r} has unknown requires reference(s): "
                    f"{', '.join(unknown_references)}."
                )
            if rule.id in rule.requires:
                raise InvalidValidationPlanError(
                    f"validation_plan rule {rule.id!r} cannot depend on itself."
                )

        # Runs the cycle check so an invalid plan is rejected up front, before any
        # rule callback, business-column read or metric request takes place.
        plan.execution_layers()

        if expectation_ids is not None:
            plan.validate_covers_expectations(expectation_ids=expectation_ids)

        return plan

    def validate_covers_expectations(self, expectation_ids: Sequence[str]) -> None:
        """Reject plans that no longer match the suite's current expectations."""
        declared = set(expectation_ids)
        planned = set(self.rule_by_expectation_id.keys())
        missing = sorted(declared - planned)
        stale = sorted(planned - declared)
        problems: List[str] = []
        if missing:
            problems.append(
                "expectation(s) without a rule: " + ", ".join(repr(item) for item in missing)
            )
        if stale:
            problems.append(
                "rule(s) referencing expectation_id(s) no longer in the suite: "
                + ", ".join(repr(item) for item in stale)
            )
        if problems:
            raise InvalidValidationPlanError(
                "validation_plan does not match the current expectations of the suite; "
                + "; ".join(problems) + "."
            )


def _parse_rule(
    raw_rule: Any,
    index: int,
    seen_rule_ids: set[str],
    seen_expectation_ids: set[str],
) -> ValidationPlanRule:
    label = f"validation_plan rules[{index}]"
    if not isinstance(raw_rule, Mapping):
        raise InvalidValidationPlanError(
            f"{label} must be an object, received {type(raw_rule).__name__}."
        )

    unknown_fields = sorted(set(raw_rule.keys()) - _RULE_FIELDS)
    if unknown_fields:
        raise InvalidValidationPlanError(
            f"{label} contains unknown field(s): {', '.join(unknown_fields)}."
        )

    missing_fields = sorted(field for field in _RULE_FIELDS if field not in raw_rule)
    if missing_fields:
        raise InvalidValidationPlanError(
            f"{label} is missing required field(s): {', '.join(missing_fields)}."
        )

    rule_id = raw_rule["id"]
    if not isinstance(rule_id, str) or not rule_id:
        raise InvalidValidationPlanError(
            f"{label} 'id' must be a non-empty string, received {rule_id!r}."
        )
    if rule_id in seen_rule_ids:
        raise InvalidValidationPlanError(f"validation_plan rule id {rule_id!r} is not unique.")
    seen_rule_ids.add(rule_id)

    expectation_id = raw_rule["expectation_id"]
    if not isinstance(expectation_id, str) or not expectation_id:
        raise InvalidValidationPlanError(
            f"{label} 'expectation_id' must be a non-empty string, "
            f"received {expectation_id!r}."
        )
    if expectation_id in seen_expectation_ids:
        raise InvalidValidationPlanError(
            "validation_plan expectation_id "
            f"{expectation_id!r} is declared by more than one rule."
        )
    seen_expectation_ids.add(expectation_id)

    raw_requires = raw_rule["requires"]
    if not isinstance(raw_requires, list):
        raise InvalidValidationPlanError(
            f"{label} 'requires' must be a list, received {type(raw_requires).__name__}."
        )
    requires: List[str] = []
    seen_requires: set[str] = set()
    for required_id in raw_requires:
        if not isinstance(required_id, str) or not required_id:
            raise InvalidValidationPlanError(
                f"{label} 'requires' entries must be non-empty strings, "
                f"received {required_id!r}."
            )
        if required_id in seen_requires:
            raise InvalidValidationPlanError(
                f"{label} lists requires entry {required_id!r} more than once."
            )
        seen_requires.add(required_id)
        requires.append(required_id)

    return ValidationPlanRule(
        id=rule_id,
        expectation_id=expectation_id,
        requires=tuple(requires),
    )
