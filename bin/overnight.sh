#!/usr/bin/env bash
# Detector overnight harness — fully non-interactive, log-to-disk.
# Usage:
#   bin/overnight.sh                    # full suite
#   bin/overnight.sh --canary-only      # fast canary only
#   nohup bin/overnight.sh & disown     # fire and forget
#   tmux new -s overnight 'bin/overnight.sh'
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Non-interactive mode: no prompts, unbuffered output, no HF noise
export DETECTOR_NONINTERACTIVE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_PROGRESS_BARS=1

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOGDIR="$ROOT/logs"
REPORTDIR="$ROOT/reports"
mkdir -p "$LOGDIR" "$REPORTDIR"

LOG="$LOGDIR/overnight-$STAMP.log"
VENV="$ROOT/.venv/bin/python"

echo "=== Detector overnight: $STAMP ===" | tee "$LOG"
echo "Log: $LOG" | tee -a "$LOG"

# Pass through any CLI args (--canary-only, --sweep-only, --corpus, etc.)
"$VENV" scripts/overnight.py "$@" 2>&1 | tee -a "$LOG"

EXIT_CODE=${PIPESTATUS[0]}

echo "" | tee -a "$LOG"
echo "=== Finished: $(date -u +%Y%m%dT%H%M%SZ) (exit $EXIT_CODE) ===" | tee -a "$LOG"
exit $EXIT_CODE
