#!/usr/bin/env bash
# Overnight drift check + resolver audit — non-interactive, log-to-disk.
# Usage:
#   bin/overnight_drift.sh                     # full suite (3 models × 3 seeds)
#   bin/overnight_drift.sh --models qwen3b     # single model
#   bin/overnight_drift.sh --skip-retry        # skip retry enforcement
#   nohup bin/overnight_drift.sh & disown      # fire and forget
#   tmux new -s drift 'bin/overnight_drift.sh'
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
mkdir -p "$LOGDIR"

LOG="$LOGDIR/overnight-drift-$STAMP.log"
VENV="$ROOT/.venv/bin/python"

echo "=== Overnight drift check: $STAMP ===" | tee "$LOG"
echo "Log: $LOG" | tee -a "$LOG"

# Pass through any CLI args
"$VENV" scripts/overnight_drift.py "$@" 2>&1 | tee -a "$LOG"

EXIT_CODE=${PIPESTATUS[0]}

echo "" | tee -a "$LOG"
echo "=== Finished: $(date -u +%Y%m%dT%H%M%SZ) (exit $EXIT_CODE) ===" | tee -a "$LOG"
exit $EXIT_CODE
