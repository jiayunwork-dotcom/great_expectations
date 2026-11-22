from __future__ import annotations

from copy import copy
from functools import cached_property
from typing import TYPE_CHECKING, Optional

from great_expectations import __version__ as ge_version
from great_expectations.core.expectation_validation_result import (
    ExpectationSuiteValidationResult,
    ExpectationValidationResult,
)
from great_expectations.core.result_format import (
    DEFAULT_RESULT_FORMAT,
    ResultFormat,
)
from great_expectations.data_context.data_context.context_factory import project_manager
from great_expectations.util import convert_to_json_serializable  # noqa: TID251 # FIXME CoP
from great_expectations.validator.validator import Validator as OldValidator
from great_expectations.validator.validator import calc_validation_statistics

if TYPE_CHECKING:
    from great_expectations.core import ExpectationSuite
    from great_expectations.core.batch_definition import BatchDefinition
    from great_expectations.core.result_format import ResultFormatUnion
    from great_expectations.core.suite_parameters import SuiteParameterDict
    from great_expectations.datasource.fluent.batch_request import BatchParameters
    from great_expectations.expectations.expectation import (
        Expectation,
        ExpectationConfiguration,
    )


class Validator:
    """Validator.

    Responsible for running expectations on a batch definition.
    """

    def __init__(
        self,
        batch_definition: BatchDefinition,
        result_format: ResultFormatUnion = DEFAULT_RESULT_FORMAT,
        batch_parameters: Optional[BatchParameters] = None,
    ) -> None:
        self._batch_definition = batch_definition
        self._batch_parameters = batch_parameters
        self.result_format = result_format

        self._get_validator = project_manager.get_validator

    def validate_expectation(
        self,
        expectation: Expectation,
        expectation_parameters: Optional[SuiteParameterDict] = None,
    ) -> ExpectationValidationResult:
        """Run a single expectation against the batch definition"""
        streaming = self._streaming_batch_data()
        if streaming is not None:
            from great_expectations.execution_engine.streaming.engine import (
                run_streaming_validation,
            )

            runtime_configuration = self._runtime_configuration()
            processed = self._wrapped_validator.process_expectations_for_validation(
                [expectation.configuration], expectation_parameters
            )
            results, _streaming_meta = run_streaming_validation(
                validator=self._wrapped_validator,
                streaming_batch_data=streaming,
                expectation_configurations=processed,
                runtime_configuration=runtime_configuration,
                catch_exceptions=bool(runtime_configuration.get("catch_exceptions", True)),
                validation_plan=None,
            )
            assert len(results) == 1
            return results[0]

        results = self._validate_expectation_configs(
            expectation_configs=[expectation.configuration],
            expectation_parameters=expectation_parameters,
        )

        assert len(results) == 1
        return results[0]

    def validate_expectation_suite(
        self,
        expectation_suite: ExpectationSuite,
        expectation_parameters: Optional[SuiteParameterDict] = None,
    ) -> ExpectationSuiteValidationResult:
        """Run an expectation suite against the batch definition"""
        streaming = self._streaming_batch_data()
        if streaming is not None:
            return self._validate_streaming_suite(
                expectation_suite=expectation_suite,
                expectation_parameters=expectation_parameters,
                streaming_batch_data=streaming,
            )

        results = self._validate_expectation_configs(
            expectation_configs=expectation_suite.expectation_configurations,
            expectation_parameters=expectation_parameters,
            validation_plan=expectation_suite.validation_plan,
        )
        statistics = calc_validation_statistics(results)

        return ExpectationSuiteValidationResult(
            results=results,
            success=statistics.success,
            suite_name=expectation_suite.name,
            statistics={
                "evaluated_expectations": statistics.evaluated_expectations,
                "successful_expectations": statistics.successful_expectations,
                "unsuccessful_expectations": statistics.unsuccessful_expectations,
                "success_percent": statistics.success_percent,
                "declared_expectations": statistics.declared_expectations,
                "skipped_expectations": statistics.skipped_expectations,
            },
            meta={
                # run_id, validation_time, and fkeys are added to this dict
                # in ValidationDefinition.run
                "great_expectations_version": ge_version,
                "batch_spec": convert_to_json_serializable(
                    self._wrapped_validator.active_batch_spec
                ),
                "batch_markers": self._wrapped_validator.active_batch_markers,
                "active_batch_definition": convert_to_json_serializable(
                    self._wrapped_validator.active_batch_definition
                ),
            },
            batch_id=self.active_batch_id,
        )

    # ------------------------------------------------------------------ streaming
    def _streaming_batch_data(self):
        """Return the loaded StreamingBatchData for this validation, or ``None``.

        The wrapped validator is created from the batch request exactly as the ordinary
        path does; for a streaming batch it holds a ``StreamingBatchData`` placeholder that
        has not read the source yet.
        """
        from great_expectations.execution_engine.streaming.batch_data import (
            StreamingBatchData,
        )

        batch_manager = self._wrapped_validator.execution_engine.batch_manager
        active_id = batch_manager.active_batch_data_id
        if active_id is None:
            return None
        active = batch_manager.batch_data_cache.get(active_id)
        if isinstance(active, StreamingBatchData):
            return active
        return None

    def _validate_streaming_suite(
        self,
        *,
        expectation_suite: ExpectationSuite,
        expectation_parameters: Optional[SuiteParameterDict],
        streaming_batch_data,
    ) -> ExpectationSuiteValidationResult:
        from great_expectations.execution_engine.streaming.engine import (
            run_streaming_validation,
        )

        runtime_configuration = self._runtime_configuration()

        # Suite parameters are substituted before graph construction, exactly as the
        # ordinary path does, so thresholds are finalized before pre-source checks.
        processed = self._wrapped_validator.process_expectations_for_validation(
            expectation_suite.expectation_configurations,
            expectation_parameters,
        )

        results, streaming_meta = run_streaming_validation(
            validator=self._wrapped_validator,
            streaming_batch_data=streaming_batch_data,
            expectation_configurations=processed,
            runtime_configuration=runtime_configuration,
            catch_exceptions=bool(runtime_configuration.get("catch_exceptions", True)),
            validation_plan=expectation_suite.validation_plan,
        )

        statistics = calc_validation_statistics(results)

        batch_spec = self._wrapped_validator.active_batch_spec
        return ExpectationSuiteValidationResult(
            results=results,
            success=statistics.success,
            suite_name=expectation_suite.name,
            statistics={
                "evaluated_expectations": statistics.evaluated_expectations,
                "successful_expectations": statistics.successful_expectations,
                "unsuccessful_expectations": statistics.unsuccessful_expectations,
                "success_percent": statistics.success_percent,
                "declared_expectations": statistics.declared_expectations,
                "skipped_expectations": statistics.skipped_expectations,
            },
            meta={
                "great_expectations_version": ge_version,
                "batch_spec": convert_to_json_serializable(batch_spec),
                "batch_markers": self._wrapped_validator.active_batch_markers,
                "active_batch_definition": convert_to_json_serializable(
                    self._wrapped_validator.active_batch_definition
                ),
                "streaming": streaming_meta,
            },
            batch_id=self.active_batch_id,
        )

    def _runtime_configuration(self) -> dict:
        if isinstance(self.result_format, ResultFormat):
            return {"result_format": copy(self.result_format.value)}
        return {"result_format": copy(self.result_format)}

    @property
    def active_batch_id(self) -> Optional[str]:
        return self._wrapped_validator.active_batch_id

    @property
    def _include_rendered_content(self) -> bool:
        return project_manager.is_using_cloud()

    @cached_property
    def _wrapped_validator(self) -> OldValidator:
        batch_request = self._batch_definition.build_batch_request(
            batch_parameters=self._batch_parameters
        )
        return self._get_validator(batch_request=batch_request)

    def _validate_expectation_configs(
        self,
        expectation_configs: list[ExpectationConfiguration],
        expectation_parameters: Optional[SuiteParameterDict] = None,
        validation_plan=None,
    ) -> list[ExpectationValidationResult]:
        """Run a list of expectation configurations against the batch definition"""
        processed_expectation_configs = self._wrapped_validator.process_expectations_for_validation(
            expectation_configs, expectation_parameters
        )

        runtime_configuration: dict
        if isinstance(self.result_format, ResultFormat):
            runtime_configuration = {"result_format": copy(self.result_format.value)}
        else:
            runtime_configuration = {"result_format": copy(self.result_format)}

        results = self._wrapped_validator.graph_validate(
            configurations=processed_expectation_configs,
            runtime_configuration=runtime_configuration,
            validation_plan=validation_plan,
        )

        if self._include_rendered_content:
            for result in results:
                result.render()

        return results
