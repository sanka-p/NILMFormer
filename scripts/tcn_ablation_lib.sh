#!/bin/bash
#
# Shared helpers for the TCN ablation runners (run_tcn_ablation.sh, run_tcn_kettle.sh).
# Source it, do not execute it.
#
# Provides:
#   PY                     interpreter, defaulted and overridable
#   tcn_preflight          fail fast if PY cannot reach a GPU
#   tcn_is_complete FILE   true only for a finished run, not a mid-training checkpoint

# Interpreter choice is a portability AND a comparability matter:
#   env/bin/python    torch 2.12.0+cu130 -- needs a CUDA 13 driver (>= 580). Fails on the
#                     RTX 3090 node, whose driver reports CUDA 12.2.
#   .venv/bin/python  torch 2.6.0+cu124  -- runs on any CUDA 12.x driver (>= 525) and on
#                     580 as well, so it works on every node here.
# Default to .venv so all arms use ONE torch build wherever they land; mixing builds
# across arms would confound the very comparison the ablation is making. Verified that
# the two envs build identical data despite pandas 3.0 vs 2.2 (Kettle ws=256: 7914 test /
# 63673 train windows, same aggregate sum to the last decimal).
PY="${PY:-.venv/bin/python}"

tcn_cuda_ok() {
    [ -x "$1" ] || return 1
    "$1" - >/dev/null 2>&1 <<'PYCHK'
import torch
torch.zeros(1).cuda()
PYCHK
}

# Check the GPU in seconds rather than discovering it after minutes of data preparation.
tcn_preflight() {
    local script_name="${1:-$0}"
    if tcn_cuda_ok "$PY"; then
        return 0
    fi
    {
        echo "ERROR: ${PY} cannot initialise CUDA on this node."
        "$PY" -c "import torch; print('  torch', torch.__version__, '| cuda build', torch.version.cuda)" 2>/dev/null | tail -1
        nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -2 | sed 's/^/  gpu: /'
        echo "  A cu130 torch build needs driver >= 580; a cu124 build runs on any 12.x driver."
        local alt
        for alt in .venv/bin/python env/bin/python; do
            if [ "$alt" != "$PY" ] && tcn_cuda_ok "$alt"; then
                echo "  Working alternative on this node:  PY=${alt} bash ${script_name}"
                return 1
            fi
        done
        echo "  No interpreter here can reach a GPU -- check nvidia-smi and the node you are on."
    } >&2
    return 1
}

# SeqToSeqTrainer.save() writes the SAME .pt on every best-loss epoch during training
# (src/helpers/trainer.py:200) and again after evaluation, so the file existing does not
# mean the run finished. Only the post-eval write carries test_metrics_timestamp, which is
# also the key scripts/make_table.py reads -- use that as the completion marker.
tcn_is_complete() {
    "$PY" - "$1" >/dev/null 2>&1 <<'PYCHK'
import sys
import torch
try:
    log = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
except Exception:
    sys.exit(1)
sys.exit(0 if isinstance(log, dict) and "test_metrics_timestamp" in log else 1)
PYCHK
}
