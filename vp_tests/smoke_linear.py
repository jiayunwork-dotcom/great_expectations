"""Manual smoke run for the validation-plan implementation (not part of pytest)."""

from __future__ import annotations

import sys

sys.path.insert(0, "/workspace/vp_tests")

import great_expectations as gx
from great_expectations.core.expectation_suite import ExpectationSuite

import vp_harness
import vp_rules

vp_rules.reset_calls()

# Activate the ephemeral project before expectations are added to the suite.
context = gx.get_context(mode="ephemeral")
suite = ExpectationSuite(name="vp_suite")
rules = vp_harness.add_rules(
    suite,
    [
        vp_rules.ExpectVpPassing(value=1, tag="t0"),
        vp_rules.ExpectVpFailing(value=2, tag="t1"),
        vp_rules.ExpectVpPassing(value=3, tag="t2"),
        vp_rules.ExpectVpBoobyTrap(value=4, tag="t3"),
    ],
)
ids = vp_harness.rule_refs(suite)

# Linear: r0 (pass) -> r1 (fail) -> r2 (should skip) -> r3 (booby trap, should skip)
vp_harness.make_plan(
    suite,
    rules=[
        {"id": "r0", "expectation_id": ids[0], "requires": []},
        {"id": "r1", "expectation_id": ids[1], "requires": ["r0"]},
        {"id": "r2", "expectation_id": ids[2], "requires": ["r1"]},
        {"id": "r3", "expectation_id": ids[3], "requires": ["r2"]},
    ],
)

validator = vp_harness.build_context_validator_for_suite(
    suite, dataframe=vp_rules.sample_dataframe()
)
result = validator.validate(catch_exceptions=True)
for item in result.results:
    print(item.meta.get("execution"), "success=", item.success, "result=", item.result)
print("statistics:", result.statistics)
print("provider calls:", vp_rules.PROVIDER_CALLS)
assert vp_rules.PROVIDER_CALLS.get("vp.booby_trap", 0) == 0
print("booby trap untouched OK")
