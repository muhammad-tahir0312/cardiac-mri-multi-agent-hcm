#!/usr/bin/env bash
# The models live under a DIFFERENT macOS account on this machine
# (/Users/it-macmini-1/, not /Users/it-macmini-1-tahir/). A bare `ollama serve`
# started from this shell looks in ~/.ollama, finds nothing, and every request
# 404s with "model not found". Point it at the real store.
#
# On UM HPC there is no such split — just `ollama serve &` inside the job script,
# or set OLLAMA_MODELS to wherever you pulled the weights on /scratch.
set -euo pipefail
export OLLAMA_MODELS="${OLLAMA_MODELS:-/Users/it-macmini-1/.ollama/models}"
pkill -f "ollama serve" 2>/dev/null || true
sleep 1
nohup ollama serve > /tmp/ollama.log 2>&1 &
sleep 4
ollama list
