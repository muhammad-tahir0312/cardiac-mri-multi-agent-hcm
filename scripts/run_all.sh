#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Reproduce the entire thesis, end to end, from a clean checkout.
#
#   ./scripts/run_all.sh                 # this Mac (MPS)
#   CFG=configs/hpc.yaml ./scripts/run_all.sh    # UM HPC (CUDA)
#
# Stages are independent and resumable — every one skips work already on disk, so
# you can kill this and restart it without losing anything.
#
# WALL-CLOCK, measured on the M4 (16 GB, MPS):
#   manifest + ground truth   ~6 min      no GPU, no network, no LLM
#   segmentation, per dataset ~2 h        CineMA 3-seed ensemble, ~45 s/subject
#   corpus + index            ~4 min
#   one LLM config (830 subj) ~17 h       qwen2.5:14b at ~73 s/subject
#   the full 10-config grid   ~7 days     <- DO NOT run this on the Mac. See below.
#
# On a single HPC GPU node the LLM stage drops to well under an hour per config,
# which is what makes the ablation grid tractable. Run stages 1-3 locally, then
# move `artifacts/` to /scratch and run stage 5 there.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")/.."

CFG="${CFG:-configs/default.yaml}"
PY="${PY:-./.venv/bin/python}"
CMR="$PY -m cmr.cli --config-file $CFG"
CKPT="${CKPT:-acdc_sax}"

echo "═══ 0. environment ═══"
$CMR doctor

echo; echo "═══ 1. data spine — 830 subjects, canonical label order VERIFIED ═══"
$CMR manifest

echo; echo "═══ 2. ground truth — the denominator of every MAE in the thesis ═══"
# Also calibrates the plausibility gate: it must accept >=98% of expert annotation.
$CMR quantify --gt

echo; echo "═══ 3. guideline corpus -> chunks -> FAISS ═══"
$CMR corpus

echo; echo "═══ 4. segmentation (Agent 1) ═══"
# acdc_sax -> ACDC   is a SUPERVISED baseline  (the checkpoint was fine-tuned on ACDC)
# acdc_sax -> M&Ms   is TRUE cross-dataset ZERO-SHOT. That run IS RQ3.
# mnms_sax -> M&Ms   is the in-domain ceiling the zero-shot run is falling short of.
for DS in ACDC MnMs MnMs2; do
  $CMR segment --dataset "$DS" --ckpt "$CKPT"
done

echo; echo "═══ 5. the experiment grid ═══"
# Every row of configs/experiments.yaml. Each writes to its own artifacts/runs/{config_id}/,
# so no ablation can overwrite another.
#
#   H1  grounding helps ................ full vs no_rag
#   H2  boundary-aware recomputation ... full vs no_feedback  (GT LVEF in [35,55] only)
#   H3  decomposition exposes failure .. full vs no_gate
for EXP in full no_rag dense_only bm25_only biobert no_gate no_feedback small_llm template; do
  echo "--- $EXP"
  $CMR run  --config "$EXP"
  $CMR eval --config "$EXP"
done

echo; echo "═══ 6. figures and tables — generated, never typed ═══"
$CMR figures --config full

echo; echo "═══ 7. the checks that must hold ═══"
$PY -m pytest -q

echo
echo "Done. Everything is under artifacts/."
echo "If a number is going into the thesis and you cannot point at the file under"
echo "artifacts/ that produced it, it does not go into the thesis."
