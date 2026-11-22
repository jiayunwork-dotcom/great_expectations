#!/usr/bin/env bash
# Single entry point to rerun the validation-plan + streaming work.
#
# It runs:
#   1. The feature test package (plan parsing, suite integration, execution
#      semantics, ValidationDefinition.run, compat/isolation) and the
#      dataframe-chunk streaming tests (semantics, lifecycle, fluent assets,
#      custom providers, result contracts, and the memory-scale probe).
#   2. The touched Great Expectations regression tests (validator graph,
#      statistics, expectation suite, validation definition, pandas assets).
#
# The isolated config avoids the repo-level pytest settings that assume
# en_US.UTF-8, sqlalchemy and spark, none of which are installed here.
set -euo pipefail

# Default to the interpreter that has Great Expectations installed; allow an
# override so the suite can also be run from a virtualenv.
PYTHON="${PYTHON:-python3}"
VP_TESTS="/workspace/vp_tests"
REPO_TESTS="/workspace/tests"

BASE=(
  -q
  -o filterwarnings=
  -o addopts=
  -p no:cacheprovider
  -p locale_patch
)
# Backend-exclusion flags only exist on the repository's own conftest.
REPO_ONLY=(--no-sqlalchemy --no-spark)

echo "== validation-plan + streaming feature tests =="
PYTHONPATH="${VP_TESTS}" "${PYTHON}" -m pytest "${VP_TESTS}" "${BASE[@]}"

echo "== touched GE regression tests =="
PYTHONPATH="${VP_TESTS}" "${PYTHON}" -m pytest \
  "${REPO_TESTS}/validator" \
  "${REPO_TESTS}/core/test_expectation_suite.py" \
  "${REPO_TESTS}/core/test_validation_definition.py" \
  "${REPO_TESTS}/core/test_expectation_validation_result.py" \
  "${REPO_TESTS}/datasource/fluent/data_asset/test_pandas_asset.py" \
  "${REPO_TESTS}/render/test_default_jinja_view.py" \
  "${BASE[@]}" \
  "${REPO_ONLY[@]}" \
  -k "not postgres"

echo "ALL CHECKS PASSED"
