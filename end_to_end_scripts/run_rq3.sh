#!/usr/bin/env bash
# Run the complete RQ3 sample-size equivalence experiment.
#
# The underlying Python CLI options are forwarded unchanged. For example:
#
#   bash end_to_end_scripts/run_rq3.sh
#   bash end_to_end_scripts/run_rq3.sh --sample-fraction 0.1
#   bash end_to_end_scripts/run_rq3.sh --bootstrap-repetitions 100 --workers 4
#
# Outputs:
#   analysis/results/rq3/sample_equivalence/results.json
#   analysis/results/rq3/sample_equivalence/report.md
#   analysis/results/rq3/sample_equivalence/plots/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

exec python3 analysis/utils/calculate_sample_equivalence.py "$@"
