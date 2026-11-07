#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCOPE="${MODEL_SCOPE:-full}"
SUBJECT="${SUBJECT:-}"
args=(--model_scope "$SCOPE")
if [[ "$SCOPE" == interaction ]]; then
  [[ -n "$SUBJECT" ]] || { echo "Set SUBJECT to an exact REST3D object mask name." >&2; exit 2; }
  args+=(--subject "$SUBJECT")
fi
exec "${PYTHON:-python}" "$SCRIPT_DIR/run_full_pipeline.py" "${args[@]}" "$@"
