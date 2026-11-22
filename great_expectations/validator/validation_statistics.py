from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from great_expectations.core.expectation_validation_result import (
        ExpectationValidationResult,
    )


class ValidationStatistics(NamedTuple):
    evaluated_expectations: int
    successful_expectations: int
    unsuccessful_expectations: int
    success_percent: float | None
    success: bool
    declared_expectations: int = 0
    skipped_expectations: int = 0


def _is_skipped(validation_result: ExpectationValidationResult) -> bool:
    execution = validation_result.meta.get("execution") if validation_result.meta else None
    return bool(execution) and execution.get("status") == "skipped"


def calc_validation_statistics(
    validation_results: list[ExpectationValidationResult],
) -> ValidationStatistics:
    """
    Calculate summary statistics for the validation results and
    return ``ExpectationStatistics``.

    Skipped expectations (declared but never executed because a prerequisite rule
    did not pass) are counted separately. They are neither a successful nor an
    unsuccessful evaluation, but their presence makes the suite-level success
    ``False``.
    """
    skipped_results = [exp for exp in validation_results if _is_skipped(exp)]
    evaluated_results = [exp for exp in validation_results if not _is_skipped(exp)]

    # calc stats
    evaluated_expectations = len(evaluated_results)
    successful_expectations = len([exp for exp in evaluated_results if exp.success])
    unsuccessful_expectations = evaluated_expectations - successful_expectations
    skipped_expectations = len(skipped_results)
    declared_expectations = len(validation_results)
    success = (
        successful_expectations == evaluated_expectations and skipped_expectations == 0
    )
    try:
        success_percent = successful_expectations / evaluated_expectations * 100
    except ZeroDivisionError:
        success_percent = None

    return ValidationStatistics(
        successful_expectations=successful_expectations,
        evaluated_expectations=evaluated_expectations,
        unsuccessful_expectations=unsuccessful_expectations,
        success=success,
        success_percent=success_percent,
        declared_expectations=declared_expectations,
        skipped_expectations=skipped_expectations,
    )
