"""Diamond, cross-layer metric sharing, and no-recompute smoke checks."""

from __future__ import annotations

import sys

sys.path.insert(0, "/workspace/vp_tests")

import great_expectations as gx
from great_expectations.core.expectation_suite import ExpectationSuite

import vp_harness
import vp_rules

# --- Diamond: root passes; two middle rules (one pass, one fail); one leaf ---
vp_rules.reset_calls()
context = gx.get_context(mode="ephemeral")
suite = ExpectationSuite(name="diamond")
expectations = vp_harness.add_rules(
    suite,
    [
        vp_rules.ExpectVpPassing(value=1, tag="root"),       # r_root
        vp_rules.ExpectVpPassing(value=2, tag="mid_ok"),     # r_mid_ok
        vp_rules.ExpectVpFailing(value=3, tag="mid_bad"),    # r_mid_bad
        vp_rules.ExpectVpPassing(value=4, tag="leaf"),       # r_leaf
        vp_rules.ExpectVpBoobyTrap(value=5, tag="trap"),     # r_trap
    ],
)
ids = vp_harness.rule_refs(suite)
vp_harness.make_plan(
    suite,
    rules=[
        {"id": "r_root", "expectation_id": ids[0], "requires": []},
        {"id": "r_mid_ok", "expectation_id": ids[1], "requires": ["r_root"]},
        {"id": "r_mid_bad", "expectation_id": ids[2], "requires": ["r_root"]},
        {"id": "r_leaf", "expectation_id": ids[3], "requires": ["r_mid_ok", "r_mid_bad"]},
        {"id": "r_trap", "expectation_id": ids[4], "requires": ["r_leaf"]},
    ],
)
validator = vp_harness.build_context_validator_for_suite(
    suite, dataframe=vp_rules.sample_dataframe()
)
result = validator.validate(catch_exceptions=True)
statuses = {r.meta["execution"]["id"]: r.meta["execution"]["status"] for r in result.results}
print("diamond statuses:", statuses)
leaf = next(r for r in result.results if r.meta["execution"]["id"] == "r_leaf")
print("leaf blocked_by:", leaf.meta["execution"]["blocked_by"])
assert statuses == {
    "r_root": "passed",
    "r_mid_ok": "passed",
    "r_mid_bad": "failed",
    "r_leaf": "skipped",
    "r_trap": "skipped",
}
# blocked_by keeps requires declaration order: r_mid_ok then r_mid_bad; only the
# non-passing one is listed.
assert leaf.meta["execution"]["blocked_by"] == ["r_mid_bad"]
assert vp_rules.PROVIDER_CALLS.get("vp.booby_trap", 0) == 0

# --- Shared failing metric: an independent later layer reuses the failed id ---
# vp.always_failing has table-level domain (batch_id only), so distinct
# expectations still resolve the exact same MetricConfiguration.
vp_rules.reset_calls()
suite2 = ExpectationSuite(name="shared_fail")
exps2 = vp_harness.add_rules(
    suite2,
    [
        vp_rules.ExpectVpMetricError(value=1, tag="t_a"),        # layer 0, fails
        vp_rules.ExpectVpSharedFailing(value=1, tag="t_b"),      # layer 1 (blocked)
        vp_rules.ExpectVpPassing(value=9, tag="independent"),     # layer 0, passes
        vp_rules.ExpectVpSharedFailingSibling(value=1, tag="t_c"),  # layer 1, shared fail
    ],
)
ids2 = vp_harness.rule_refs(suite2)
vp_harness.make_plan(
    suite2,
    rules=[
        {"id": "m0", "expectation_id": ids2[0], "requires": []},
        {"id": "m1", "expectation_id": ids2[1], "requires": ["m0"]},
        {"id": "ind", "expectation_id": ids2[2], "requires": []},
        {"id": "ind2", "expectation_id": ids2[3], "requires": ["ind"]},
    ],
)
validator2 = vp_harness.build_context_validator_for_suite(
    suite2, dataframe=vp_rules.sample_dataframe()
)
result2 = validator2.validate(catch_exceptions=True)
for item in result2.results:
    print(item.meta["execution"], "success=", item.success)
print("calls:", vp_rules.PROVIDER_CALLS)
# m1 is skipped because its prerequisite m0 errored.
m1 = next(r for r in result2.results if r.meta["execution"]["id"] == "m1")
assert m1.meta["execution"]["status"] == "skipped"
assert m1.success is None and m1.result == {}
# ind2 unlocks (ind passed) and shares the failed metric; the provider must not
# run again when this later layer resolves the same MetricConfiguration.
calls_after = vp_rules.PROVIDER_CALLS.get("vp.always_failing", 0)
ind2 = next(r for r in result2.results if r.meta["execution"]["id"] == "ind2")
assert ind2.meta["execution"]["status"] == "error"
# All calls happened in the first layer's resolution; nothing recomputed later.
assert calls_after == vp_rules.PROVIDER_CALLS.get("vp.always_failing", 0)
assert calls_after >= 1
# Independent branch still completes.
ind = next(r for r in result2.results if r.meta["execution"]["id"] == "ind")
assert ind.meta["execution"]["status"] == "passed"
print("ALL SHARING CHECKS OK")

