#!/usr/bin/env bash
# Simple wrapper that calls the Python pipeline with defaults

set -euo pipefail

python3 "$(dirname "$0")/run_pipeline.py" "$@"
