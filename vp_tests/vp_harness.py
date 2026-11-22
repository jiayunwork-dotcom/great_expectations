"""Pandas-backed validation harness builders for the validation-plan tests."""

from __future__ import annotations

import uuid
from typing import List, Optional, Tuple

import pandas as pd

import great_expectations as gx
from great_expectations.core.expectation_suite import ExpectationSuite
from great_expectations.expectations.expectation import Expectation
from great_expectations.validator.validator import Validator


def build_validator(
    dataframe: Optional[pd.DataFrame] = None,
    suite: Optional[ExpectationSuite] = None,
) -> Tuple[Validator, ExpectationSuite]:
    context = gx.get_context(mode="ephemeral")
    datasource = context.data_sources.add_pandas(name="vp_pandas")
    asset = datasource.add_dataframe_asset(name="vp_asset")
    batch_definition = asset.add_batch_definition_whole_dataframe(name="vp_batch_def")
    batch_request = batch_definition.build_batch_request(
        batch_parameters={"dataframe": dataframe}
    )

    if suite is None:
        suite = ExpectationSuite(name="vp_suite")
    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite=suite,
    )
    return validator, suite


def build_context_validator_for_suite(
    suite: ExpectationSuite,
    dataframe: Optional[pd.DataFrame] = None,
) -> Validator:
    """Build a validator bound to an already-populated suite.

    The calling test must have activated an ephemeral context before populating
    the suite, because adding expectations consults the project store.
    """
    context = gx.get_context(mode="ephemeral")
    datasource = context.data_sources.add_pandas(name="vp_pandas")
    asset = datasource.add_dataframe_asset(name="vp_asset")
    batch_definition = asset.add_batch_definition_whole_dataframe(name="vp_batch_def")
    batch_request = batch_definition.build_batch_request(
        batch_parameters={"dataframe": dataframe}
    )
    return context.get_validator(
        batch_request=batch_request,
        expectation_suite=suite,
    )


def add_rules(
    suite: ExpectationSuite, expectations: List[Expectation]
) -> List[Expectation]:
    for expectation in expectations:
        # add_expectation rejects expectations that already carry an id.
        suite.add_expectation(expectation)
        if expectation.id is None:
            expectation.id = str(uuid.uuid4())
    return suite.expectations


def rule_refs(suite: ExpectationSuite) -> dict:
    """Map suite order index to expectation id for plan construction."""
    return {index: exp.id for index, exp in enumerate(suite.expectations)}


def make_plan(suite: ExpectationSuite, rules: List[dict]) -> ExpectationSuite:
    """Attach a well-formed plan; rules reference expectations by their generated ids."""
    suite.meta["validation_plan"] = {"version": 1, "rules": rules}
    # Touch the property so malformed plans fail during test setup, not mid-run.
    assert suite.validation_plan is not None
    return suite
