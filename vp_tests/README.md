# Validation Plan — design notes and rerun guide

## What was built

A suite can declare an execution plan in `suite.meta["validation_plan"]`:

```python
suite.meta["validation_plan"] = {
    "version": 1,
    "rules": [
        {"id": "has_amount", "expectation_id": "<Expectation.id>", "requires": []},
        {"id": "amount_positive",
         "expectation_id": "<another Expectation.id>",
         "requires": ["has_amount"]},
    ],
}
```

Without that key every existing code path behaves exactly as before.

## Coordination between rule scheduling and metric resolution

### Layered unlocking

`ValidationPlan.execution_layers()` runs Kahn's algorithm and groups rules into
dependency layers. `Validator._graph_validate_with_plan` walks the layers:

- A rule whose `requires` all have status `passed` is **unlocked** and executes.
- Otherwise it is reported as **skipped**. Its `blocked_by` lists the direct
  prerequisites that did not pass, kept in `requires` declaration order.
- `failed`, `error` and `skipped` all propagate: any downstream rule stays
  blocked. Independent branches in the same/later layers still run.

A skipped rule never reaches `_generate_metric_dependency_subgraphs...`, so its
`get_validation_dependencies` callback, metric providers and `metrics_validate`
are never invoked. The booby-trap provider in `vp_rules.py` proves this.

### Metric identity reuse across layers

All layer sub-graphs are merged into one suite-level `ValidationGraph`. Each
layer resolution is an *incremental* call that receives the previous layers'
`resolved_metrics` and `failed_metric_info`:

- A `MetricConfiguration` already present in `resolved_metrics` is simply reused
  by the graph's `_parse`, so an identity shared by rules that unlock in
  different layers is computed at most once.
- A failed metric is seeded into the next resolution's `aborted_metrics_info`,
  so it is never recomputed when a later, independent rule needs the same
  identity. The existing within-run `MAX_METRIC_COMPUTATION_RETRIES` budget is
  unchanged.
- Failure attribution stays local: aborted info is intersected with each rule's
  own sub-graph (`ExpectationValidationGraph.get_exception_info`), so a failed
  metric only marks rules that actually depend on it. Independent branches get
  their own results.

### Why not-yet-unlocked rules never compute metrics

The graph for a rule is only built after the rule unlocks, and only unlocked
rules' edges enter the suite graph. A locked rule's exclusive metrics therefore
have no edge in the graph being resolved, and no provider can be called for
them. Shared metrics are only requested through edges of unlocked rules.

### Exceptions

- `catch_exceptions=False`: an exception propagates immediately; remaining
  branches are not finished, matching the requested semantics.
- `catch_exceptions=True`: the rule that actually used the failing metric gets
  an `error` result carrying the real exception info. Its descendants are merely
  skipped (`success=None`, empty `result`, no fabricated exception).

### State lifetime

`resolved_metrics`, `failed_metric_info`, `rule_statuses`, per-run graphs and
result maps are all locals of a single `graph_validate` call and are released
when it returns. The suite, plan and Expectation objects are never mutated with
run status. A later run, different batch or changed parameters start fresh.
(The ExecutionEngine's pre-existing metric cache is untouched and still assumes
an immutable batch, as in upstream behavior.)

## Reporting and statistics

Each declared rule yields exactly one `ExpectationValidationResult`, emitted in
suite order. Executed results keep their original metrics, exception info and
user `meta`; `meta["execution"] = {"id", "status"}` is added. Skipped results
add `blocked_by`, use `success=None` and `result={}`.

`calc_validation_statistics` now reports:

- `evaluated_expectations` — executed rules only
- `successful_expectations` / `unsuccessful_expectations` / `success_percent` —
  over executed rules
- `declared_expectations` = evaluated + skipped
- `skipped_expectations`

The suite succeeds only with no failure, error or skip. An empty suite still
succeeds. `only_return_failures` keeps failures, errors and skips while the
statistics describe the full pre-filter run.

`ValidationDefinition.run` uses the same suite plan via the v1 `Validator`,
persists the result, and the result survives a JSON round trip including
`success=None`, `blocked_by` and the new statistics.

## Rerun

```bash
/workspace/vp_tests/run_all.sh
```

It runs the 57 feature tests plus the touched GE regression suites. The wrapper
pins an isolated pytest config because this container lacks the en_US locale,
sqlalchemy and spark assumed by the repository's top-level config.
