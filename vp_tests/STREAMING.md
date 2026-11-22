# Dataframe-chunk streaming validation — design notes and rerun guide

## What it is

A Pandas datasource asset can expose a **one-shot synchronous iterable of DataFrame
chunks** as a *single logical batch*:

```python
datasource = context.data_sources.add_pandas("pandas")
asset = datasource.add_dataframe_stream_asset("events")
batch_definition = asset.add_batch_definition_whole_dataframe_stream("whole_stream")

def source():                      # a one-shot iterable of frames
    for i in range(0, N, 8192):
        yield frame.iloc[i : i + 8192]

# Option A — fluent run (results are persisted and can be read back)
validation = context.validation_definitions.add(
    ValidationDefinition(name="vd", data=batch_definition, suite=suite)
)
result = validation.run(batch_parameters={"dataframe_chunks": source()})

# Option B — take a Batch and validate directly
batch = batch_definition.get_batch(batch_parameters={"dataframe_chunks": source()})
result = batch.validate(suite)
```

Every chunk is a fragment of the batch. Row counts, missing rates and numeric
statistics are interpreted over the whole logical batch, never per chunk, and the
source is **never concatenated back into one DataFrame**.

The source object is **never serialized**. The asset/BatchDefinition stores only its
identity; the batch spec carries a `<DATAFRAME_CHUNKS>` placeholder in result metadata.

## Supported rules

The five rules below stream natively. The streaming adapters reuse the **existing
Pandas providers** (`ColumnValuesBetween._pandas`, `ColumnValuesNonNull._pandas`,
`ColumnMean._pandas`, `ColumnStandardDeviation._pandas`); no predicate is re-implemented.

| Rule | Metrics streamed |
| --- | --- |
| `ExpectTableRowCountToBeBetween` | `table.row_count` |
| `ExpectColumnMeanToBeBetween` | `column.mean` |
| `ExpectColumnStdevToBeBetween` | `column.standard_deviation` |
| `ExpectColumnValuesToNotBeNull` | `column_values.nonnull.{condition,unexpected_count,unexpected_values,unexpected_index_list}` |
| `ExpectColumnValuesToBeBetween` | `column_values.between.{...}` |

Numeric aggregates (`mean`, `standard_deviation`) require a real numeric column
containing finite numbers and missing values only. Booleans, complex/datetime/object
columns and non-finite floats raise a `StreamingMetricDataError` naming the
`batch_id`, `column` and `chunk_index`.

## Provider capability declaration and state protocol

A provider opts in by declaring `streaming_supported = True` and implementing three
classmethods (base class `great_expectations.execution_engine.streaming.StreamingMetricProvider`):

- `create_streaming_state(metric, execution_engine)` — build a bounded accumulator; **must not read data**.
- `accumulate_streaming_state(state, metric, dataframe, execution_engine, chunk_index, row_offset, scratch)` — fold one chunk; receives chunks in source order and a per-chunk `scratch` cache.
- `finalize_streaming_state(state, metric, execution_engine)` — turn the accumulator into the whole-batch value.

State is created at most once per `MetricConfiguration.id` per run and is discarded
when the run ends; it is never stored on the `BatchDefinition`, suite, asset or context.

A local custom provider registered this way (or via `register_streaming_adapter(name,
adapter)`) participates in the **same single scan** as the built-in rules; the
`Validator` never dispatches on expectation type. `vp_tests/stream_custom_provider.py`
demonstrates this and records which chunks each provider consumed.

### How multiple metrics share one scan

1. The suite-level dependency graph is built from the expectations' own
   `get_validation_dependencies` (through the existing sub-graph builder), so the metric
   set is exactly what ordinary validation would compute. Graph construction reads no data.
2. `MetricConfiguration.id` distinguishes and de-duplicates the metrics required this run;
   each identity is accumulated at most once.
3. A single pass folds every chunk into every identity. Derived map metrics for the same
   `(column, thresholds, condition)` reuse one boolean evaluation per chunk through the
   `scratch` cache; unrelated conditions on different columns have distinct cache keys.
4. Finalization runs after the pass, so the report explains the **whole stream**, not the
   last chunk.

### What happens when one metric errors

A metric-level exception during accumulation/finalization is recorded against that
identity and **does not stop the scan**; its state is dropped. Final results attribute the
failure only to expectations whose sub-graph contains that identity
(`ExpectationValidationGraph.get_exception_info` semantics). Independent expectations whose
metrics succeeded still report normally. With `catch_exceptions=False` the first such
failure is re-raised after the scan.

`catch_exceptions` never swallows **source** errors or batch-structure errors — those
propagate unchanged, carrying their original exception identity.

### Reuse and release

Shared metric identities are reused within a run. When the run returns, all accumulators,
per-chunk scratch caches, the resolved metrics and failure maps are released; nothing
survives on the `Definition` or suite. A later validation (on a new source) starts fresh.

## Pre-source rejection

Before the source is touched (i.e. `source.__iter__` is never called) the run rejects:

- metrics without a streaming adapter → `StreamUnsupportedMetricError`
- any `row_condition` → `StreamRowConditionError`
- a non-Pandas engine → `StreamEngineError`
- a non-empty `validation_plan` → `StreamValidationPlanError`
- `COMPLETE`, `include_unexpected_rows` or `return_unexpected_index_query` →
  `StreamResultFormatError`

## Batch structure contract

- The source must yield **at least one chunk**. The first chunk may have zero rows; it
  declares the column names and dtypes.
- Every later chunk must have the same column **names and order** and the same dtypes.
  Pandas index labels may repeat.
- No chunks, reordered/renamed columns or changed dtypes → `StreamBatchStructureError`.
- A **single zero-row chunk** is a legitimate empty table; an **empty stream** is not.

## One-shot semantics

- Preparing a Batch (`get_batch`) never reads the source.
- A validation obtains exactly one iterator and consumes it once. A second validation over
  the same source object — same Batch or a freshly-fetched Batch — raises
  `StreamSourceReuseError` (never a spurious zero-row success). A new Batch with a new
  source runs normally.
- On normal completion and on failure the iterator is closed if it supports `close`; a
  close error never masks the original error. A caller-supplied DataFrame is never closed.

## Result reporting

- Existing result objects are reused; thresholds, `strict_min`/`strict_max`, `mostly`, the
  missing-value denominator and empty-data semantics are exactly the rules' own.
- Map results report whole-batch `element_count`, `missing_count`, `unexpected_count` and
  percentages.
- `partial_unexpected_list` / `partial_unexpected_index_list` are taken in **full-stream
  order**, length bounded by `partial_unexpected_count`. Original index labels are
  preserved; the matching **0-based full-stream positions** are reported separately as
  `result["unexpected_index_positions"]` (BASIC and SUMMARY), disambiguating repeated
  labels.
- Streaming SUMMARY does **not** emit `partial_unexpected_counts`; each map result records
  `meta["streaming"]["complete_frequencies_computed"] = False` so a truncated sample is
  never presented as whole-batch frequencies.
- `BOOLEAN_ONLY` keeps its usual meaning (empty `result`, real `success`).
- The suite result meta carries `meta["streaming"] = {chunk_count, row_count, columns,
  complete_frequencies_computed}`.

## Rerun

```bash
/workspace/vp_tests/run_all.sh
```

or, for just the streaming work:

```bash
PYTHONPATH=/workspace/vp_tests python3 -m pytest /workspace/vp_tests -q \
  -o filterwarnings= -o addopts= -p no:cacheprovider -p locale_patch
```

The memory-scale probe (`test_streaming_scale.py`) streams 4096-row generated chunks to
100k and 1M accumulated rows without holding the whole frame, measuring `tracemalloc` peaks
from the prepared Batch onward. Observed peaks are well under the 64 MiB / 8 MiB-growth
budgets.

## Boundaries

- No prefetching, async execution or checkpoint/resume.
- No `row_condition`, non-Pandas engines, or validation plans on the streaming path.
- COMPLETE, full `unexpected_rows` and full frequency detail are rejected rather than
  truncated.
- Within-chunk predicate evaluation uses the registered Pandas kernel, so GPU/out-of-core
  frame types are out of scope.
