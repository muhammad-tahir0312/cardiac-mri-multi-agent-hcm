#!/usr/bin/env bash
# nnU-Net v2 baseline — runs on this Mac (smoke) and on UM HPC (real), same script.
#
# WE DO NOT SPECIFY SPACING OR PATCH SIZE. That is the entire reason the proposal chose
# nnU-Net: it derives them from the dataset fingerprint. (§3.3.3's hand-specified
# 1.5x1.5x8 mm / 192x192 both contradicts §3.3.2 and is wrong — on ACDC nnU-Net derives
# 1.5625x1.5625 mm / 256x224 for 2d, and 5.0x1.5625x1.5625 mm / 20x256x224 for
# 3d_fullres. Report what it derived; do not dictate it.)
#
# Mac smoke run (this is the exact config that produced our smoke Dice):
#   EPOCHS=1 CONFIG=2d FOLDS=0 BATCH=2 DA_WORKERS=0 DEVICE=cpu ./baselines/train_nnunet.sh
# UM HPC, the real run:
#   EPOCHS=1000 CONFIG=3d_fullres FOLDS="0 1 2 3 4" ./baselines/train_nnunet.sh
#
# BATCH / DA_WORKERS / DEVICE are Mac-only crutches for 16 GB of shared unified memory.
# LEAVE THEM UNSET ON HPC — there, nnU-Net's own derived plan is the whole point.
#
# ─── UM HPC (SLURM): uncomment this block, then `sbatch baselines/train_nnunet.sh` ───
# #SBATCH --job-name=nnunet-acdc
# #SBATCH --partition=gpu
# #SBATCH --gres=gpu:1
# #SBATCH --cpus-per-task=8
# #SBATCH --mem=48G
# #SBATCH --time=24:00:00
# #SBATCH --output=logs/nnunet-%j.out
# module load CUDA
# source /path/to/cmr/.venv/bin/activate
# ────────────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# nnU-Net reads these three from the environment and refuses to start without them.
export nnUNet_raw="${nnUNet_raw:-$REPO/artifacts/nnunet/raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-$REPO/artifacts/nnunet/preprocessed}"
export nnUNet_results="${nnUNet_results:-$REPO/artifacts/nnunet/results}"
mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"

DATASET_ID="${DATASET_ID:-27}"          # 27=ACDC, 28=M&Ms (see nnunet_convert.py)
DATASET="${DATASET:-acdc}"
CONFIG="${CONFIG:-2d}"                  # 2d | 3d_fullres
FOLDS="${FOLDS:-0}"                     # "0" for a smoke run; "0 1 2 3 4" for the real one
EPOCHS="${EPOCHS:-5}"                   # 5 = smoke. 1000 = nnU-Net's full schedule.
CONVERT="${CONVERT:-1}"                 # 0 to skip the raw conversion (already done)

# BATCH: leave EMPTY on HPC — use nnU-Net's derived batch size, untouched.
# On this Mac you MUST set it. nnU-Net plans for a dedicated CUDA GPU and derives
# batch_size=56 for ACDC 2d. That does not fit in 16 GB of *shared* unified memory:
# measured 21.8 GB of swap and 466k pageouts, and epoch 0 never completed. Setting
# BATCH rewrites ONLY batch_size into a side plans file; spacing and patch size stay
# exactly as derived, so the reported configuration is still nnU-Net's, not ours.
BATCH="${BATCH:-}"

# Data-augmentation worker processes. nnU-Net defaults to 12; each holds its own
# augmentation buffers, and on a 16 GB Mac that alone got the trainer SIGKILLed by the
# OS twice (no traceback, just leaked semaphores — that is what an OS reap looks like).
# 0 selects nnU-Net's SingleThreadedAugmenter: no worker processes at all. On HPC leave
# it unset and let nnU-Net use all the cores you asked SLURM for.
if [ -n "${DA_WORKERS:-}" ]; then export nnUNet_n_proc_DA="$DA_WORKERS"; fi

PY="${PY:-$REPO/.venv/bin/python}"
BIN="$(dirname "$PY")"

# Device: cuda on HPC, else mps on the Mac, else cpu. Same resolution order as cmr.config.
DEVICE="${DEVICE:-$("$PY" - <<'EOF'
import torch
print("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
EOF
)}"

# nnU-Net picks the trainer by CLASS NAME, so the epoch count is selected by choosing a
# trainer variant rather than by a flag. The shipped variants are 1/5/10/20/50/100/250/
# 500/750/2000/4000/8000 epochs; anything else falls back to the 1000-epoch default.
if [ "$EPOCHS" = "1000" ]; then
  TRAINER="nnUNetTrainer"
else
  TRAINER="nnUNetTrainer_${EPOCHS}epochs"
fi

echo "── nnU-Net baseline ─────────────────────────────────────────────"
echo "   dataset   : Dataset$(printf '%03d' "$DATASET_ID") ($DATASET)"
echo "   config    : $CONFIG      folds: $FOLDS"
echo "   trainer   : $TRAINER     device: $DEVICE"
echo "   raw       : $nnUNet_raw"
echo "─────────────────────────────────────────────────────────────────"

if [ "$CONVERT" = "1" ]; then
  "$PY" -m baselines.nnunet_convert --dataset "$DATASET" --dataset-id "$DATASET_ID"
fi

# Fingerprint -> plan -> preprocess. This is where spacing and patch size are DERIVED.
"$BIN/nnUNetv2_plan_and_preprocess" -d "$DATASET_ID" -c "$CONFIG" --verify_dataset_integrity

# Optional batch-size-only override for memory-constrained machines. See BATCH above.
PLANS_ARGS=()
if [ -n "$BATCH" ]; then
  "$PY" - "$DATASET_ID" "$DATASET" "$CONFIG" "$BATCH" <<'EOF'
import json, os, sys
did, ds, cfg, batch = int(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4])
d = os.path.join(os.environ["nnUNet_preprocessed"],
                 f"Dataset{did:03d}_" + {"acdc": "ACDC", "mnms": "MnMs"}[ds])
p = json.load(open(os.path.join(d, "nnUNetPlans.json")))
p["plans_name"] = "nnUNetPlans_smoke"
old = p["configurations"][cfg]["batch_size"]
p["configurations"][cfg]["batch_size"] = batch          # the ONLY field we touch.
json.dump(p, open(os.path.join(d, "nnUNetPlans_smoke.json"), "w"), indent=2)
print(f"  BATCH override: {cfg} batch_size {old} -> {batch}; spacing/patch size unchanged")
EOF
  PLANS_ARGS=(-p nnUNetPlans_smoke)
fi

for FOLD in $FOLDS; do
  echo "── training fold $FOLD ──"
  # ${a[@]+"${a[@]}"} — expanding an EMPTY array under `set -u` is an error in bash 3.2,
  # which is what stock macOS ships. This form is the portable no-op-when-empty spelling.
  "$BIN/nnUNetv2_train" "$DATASET_ID" "$CONFIG" "$FOLD" -tr "$TRAINER" \
      ${PLANS_ARGS[@]+"${PLANS_ARGS[@]}"} -device "$DEVICE"
done

echo
echo "Derived configuration (report this; do not hand-specify it):"
"$PY" - <<EOF
import json, os
p = os.path.join(os.environ["nnUNet_preprocessed"], f"Dataset{$DATASET_ID:03d}_" +
                 {"acdc": "ACDC", "mnms": "MnMs"}["$DATASET"], "nnUNetPlans.json")
c = json.load(open(p))["configurations"]["$CONFIG"]
print(f"  target spacing : {c['spacing']}")
print(f"  patch size     : {c['patch_size']}")
print(f"  batch size     : {c['batch_size']}")
EOF
echo "Results (incl. validation Dice): $nnUNet_results"
