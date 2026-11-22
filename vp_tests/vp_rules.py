"""Countable, deterministic test rules and metric providers for validation-plan tests.

These helpers are intentionally table-level so their MetricConfiguration identity
depends only on the batch id, which makes "same metric shared across dependency
layers" straightforward to demonstrate.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, Optional, Tuple

import pandas as pd

from great_expectations.core import ExpectationValidationResult
from great_expectations.exceptions import MetricResolutionError
from great_expectations.execution_engine import PandasExecutionEngine
from great_expectations.expectations.expectation import BatchExpectation
from great_expectations.expectations.metrics.metric_provider import (
    MetricProvider,
    metric_value,
)
from great_expectations.validator.metric_configuration import MetricConfiguration

# How many times each provider function actually ran.
PROVIDER_CALLS: Dict[str, int] = {}


def reset_calls() -> None:
    PROVIDER_CALLS.clear()


def _record(metric_name: str, value: Any) -> Any:
    PROVIDER_CALLS[metric_name] = PROVIDER_CALLS.get(metric_name, 0) + 1
    return value


class _CountingTableMetric(MetricProvider):
    metric_name = "vp.counting_value"
    value_keys = ("value",)

    @metric_value(engine=PandasExecutionEngine)
    def _pandas(
        cls,
        execution_engine: PandasExecutionEngine,
        metric_domain_kwargs: dict,
        metric_value_kwargs: dict,
        metrics: Dict[str, Any],
        runtime_configuration: dict,
    ):
        return _record(cls.metric_name, metric_value_kwargs["value"])


class _FailingSharedMetric(MetricProvider):
    """Fails every time it is actually computed; used to prove no recomputation."""

    metric_name = "vp.always_failing"
    value_keys = ()

    @metric_value(engine=PandasExecutionEngine)
    def _pandas(
        cls,
        execution_engine: PandasExecutionEngine,
        metric_domain_kwargs: dict,
        metric_value_kwargs: dict,
        metrics: Dict[str, Any],
        runtime_configuration: dict,
    ):
        PROVIDER_CALLS[cls.metric_name] = PROVIDER_CALLS.get(cls.metric_name, 0) + 1
        raise MetricResolutionError(
            message="vp.always_failing exploded",
            failed_metrics=[
                MetricConfiguration(
                    metric_name=cls.metric_name,
                    metric_domain_kwargs=metric_domain_kwargs,
                    metric_value_kwargs=metric_value_kwargs,
                )
            ],
        )


class _BoobyTrapMetric(MetricProvider):
    """Provider that belongs only to a rule which is expected to stay skipped."""

    metric_name = "vp.booby_trap"
    value_keys = ()

    @metric_value(engine=PandasExecutionEngine)
    def _pandas(
        cls,
        execution_engine: PandasExecutionEngine,
        metric_domain_kwargs: dict,
        metric_value_kwargs: dict,
        metrics: Dict[str, Any],
        runtime_configuration: dict,
    ):
        PROVIDER_CALLS[cls.metric_name] = PROVIDER_CALLS.get(cls.metric_name, 0) + 1
        raise AssertionError("vp.booby_trap must never be computed")


class _SharedSuccessMetric(MetricProvider):
    """Stable table-level metric shared by rules that unlock in different layers."""

    metric_name = "vp.shared_success"
    value_keys = ()

    @metric_value(engine=PandasExecutionEngine)
    def _pandas(
        cls,
        execution_engine: PandasExecutionEngine,
        metric_domain_kwargs: dict,
        metric_value_kwargs: dict,
        metrics: Dict[str, Any],
        runtime_configuration: dict,
    ):
        return _record(cls.metric_name, 42)


class _ExpectVpBase(BatchExpectation):
    """Common machinery for the countable test expectations."""

    # ``tag`` is part of the domain so distinct instances of the same expectation
    # type are not deduplicated during suite upserts.
    tag: str = ""
    domain_keys: ClassVar[Tuple[str, ...]] = ("batch_id", "tag")
    metric_dependencies = ("vp.counting_value",)
    success_keys = ("value",)
    args_keys = ("value",)

    class Config:
        arbitrary_types_allowed = True

    def _validate(
        self,
        metrics: Dict,
        runtime_configuration: Optional[dict] = None,
        execution_engine=None,
    ) -> Dict[str, Any]:
        observed = metrics.get("vp.counting_value")
        expected = self.configuration.kwargs["value"]
        return {
            "success": observed == expected,
            "result": {"observed_value": observed, "expected_value": expected},
        }


class ExpectVpPassing(_ExpectVpBase):
    # value is only used by the metric provider; success is hard-coded below.
    value: int = 0
    library_metadata: ClassVar[Dict[str, Any]] = {
        "maturity": "experimental",
        "tags": [],
        "contributors": [],
        "requirements": [],
        "has_full_test_suite": False,
        "manually_reviewed_code": True,
    }

    def _validate(self, metrics, runtime_configuration=None, execution_engine=None):
        observed = metrics.get("vp.counting_value")
        return ExpectationValidationResult(
            success=True,
            result={"observed_value": observed, "note": "always passes"},
        )


class ExpectVpFailing(_ExpectVpBase):
    value: int = 0
    library_metadata: ClassVar[Dict[str, Any]] = {
        "maturity": "experimental",
        "tags": [],
        "contributors": [],
        "requirements": [],
        "has_full_test_suite": False,
        "manually_reviewed_code": True,
    }

    def _validate(
        self,
        metrics: Dict,
        runtime_configuration: Optional[dict] = None,
        execution_engine=None,
    ) -> Dict[str, Any]:
        observed = metrics.get("vp.counting_value")
        return {
            "success": False,
            "result": {"observed_value": observed, "note": "always fails"},
        }


class ExpectVpMetricError(BatchExpectation):
    """Depends on a metric whose provider raises MetricResolutionError."""

    value: int = 0
    tag: str = ""
    # Table-level domain keeps the failing metric identity shared across rules.
    domain_keys: ClassVar[Tuple[str, ...]] = ("batch_id",)
    metric_dependencies = ("vp.always_failing",)
    success_keys = ("value",)
    args_keys = ("value",)
    library_metadata: ClassVar[Dict[str, Any]] = {
        "maturity": "experimental",
        "tags": [],
        "contributors": [],
        "requirements": [],
        "has_full_test_suite": False,
        "manually_reviewed_code": True,
    }

    class Config:
        arbitrary_types_allowed = True

    def _validate(self, metrics, runtime_configuration=None, execution_engine=None):
        return {"success": True, "result": {}}


class ExpectVpBoobyTrap(BatchExpectation):
    """If this rule is not skipped, its sole metric raises AssertionError."""

    value: int = 0
    tag: str = ""
    domain_keys: ClassVar[Tuple[str, ...]] = ("batch_id",)
    metric_dependencies = ("vp.booby_trap",)
    success_keys = ("value",)
    args_keys = ("value",)
    library_metadata: ClassVar[Dict[str, Any]] = {
        "maturity": "experimental",
        "tags": [],
        "contributors": [],
        "requirements": [],
        "has_full_test_suite": False,
        "manually_reviewed_code": True,
    }

    class Config:
        arbitrary_types_allowed = True

    def _validate(self, metrics, runtime_configuration=None, execution_engine=None):
        return {"success": True, "result": {}}


class ExpectVpSharedFailing(BatchExpectation):
    """Shares the failing metric with ExpectVpMetricError."""

    value: int = 0
    tag: str = ""
    # Table-level domain keeps the failing metric identity shared across rules.
    domain_keys: ClassVar[Tuple[str, ...]] = ("batch_id",)
    metric_dependencies = ("vp.always_failing",)
    success_keys = ("value",)
    args_keys = ("value",)
    library_metadata: ClassVar[Dict[str, Any]] = {
        "maturity": "experimental",
        "tags": [],
        "contributors": [],
        "requirements": [],
        "has_full_test_suite": False,
        "manually_reviewed_code": True,
    }

    class Config:
        arbitrary_types_allowed = True

    def _validate(self, metrics, runtime_configuration=None, execution_engine=None):
        return {"success": True, "result": {}}


class ExpectVpSharedFailingSibling(ExpectVpSharedFailing):
    """Distinct expectation type resolving the same shared failing metric."""

    pass


class _ExpectVpSharedSuccessBase(BatchExpectation):
    value: int = 0
    tag: str = ""
    # Table-level domain => identical MetricConfiguration across these rules.
    domain_keys: ClassVar[Tuple[str, ...]] = ("batch_id",)
    metric_dependencies = ("vp.shared_success",)
    success_keys = ("value",)
    args_keys = ("value",)
    library_metadata: ClassVar[Dict[str, Any]] = {
        "maturity": "experimental",
        "tags": [],
        "contributors": [],
        "requirements": [],
        "has_full_test_suite": False,
        "manually_reviewed_code": True,
    }

    class Config:
        arbitrary_types_allowed = True


class ExpectVpSharedSuccessA(_ExpectVpSharedSuccessBase):
    def _validate(self, metrics, runtime_configuration=None, execution_engine=None):
        return {"success": metrics.get("vp.shared_success") == 42, "result": {}}


class ExpectVpSharedSuccessB(_ExpectVpSharedSuccessBase):
    def _validate(self, metrics, runtime_configuration=None, execution_engine=None):
        return {"success": metrics.get("vp.shared_success") == 42, "result": {}}


def sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame({"a": [1, 2, 3, 4], "b": ["x", "y", "z", "w"]})
