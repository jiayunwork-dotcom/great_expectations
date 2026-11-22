#!/usr/bin/env bash
# Single entry point to rerun the validation-plan work.
#
# It runs:
#   1. The new validation-plan test package (plan parsing, suite integration,
#      execution semantics, ValidationDefinition.run, compat/isolation).
#   2. The touched Great Expectations regression tests (validator graph,
#      statistics, expectation suite, validation definition).
#
# The isolated config avoids the repo-level pytest settings that assume
# en_US.UTF-8, sqlalchemy and spark, none of which are installed here.
set -euo pipefail

VENV_PYTHON="/workspace/.venv/bin/python"
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

echo "== validation-plan feature tests =="
PYTHONPATH="${VP_TESTS}" "${VENV_PYTHON}" -m pytest "${VP_TESTS}" "${BASE[@]}"

echo "== touched GE regression tests =="
PYTHONPATH="${VP_TESTS}" "${VENV_PYTHON}" -m pytest \
  "${REPO_TESTS}/validator" \
  "${REPO_TESTS}/core/test_expectation_suite.py" \
  "${REPO_TESTS}/core/test_validation_definition.py" \
  "${REPO_TESTS}/core/test_expectation_validation_result.py" \
  "${REPO_TESTS}/render/test_default_jinja_view.py" \
  "${BASE[@]}" \
  "${REPO_ONLY[@]}" \
  -k "not postgres"

echo "ALL CHECKS PASSED"
