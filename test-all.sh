#!/usr/bin/env bash
# Run all unit tests for the Canon in D curve generator project.
set -euo pipefail

cd "$(dirname "$0")"
pytest curve_generator_tests.py "$@"
