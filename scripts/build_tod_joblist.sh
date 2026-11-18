#!/bin/bash
# Emit the prior-information study's job list in scripts/sweep_parallel.sh's format:
#   dataset appliance window_size model seed root
#
# File order IS priority order: a worker walks the list top to bottom and claims the
# first unlocked, unfinished cell, so the most informative runs are taken first and the
# optional ones are only reached when everything above them is done or claimed.
#
# Windows come from results/window_selection.json (scripts/pick_window.py), so the
# per-appliance rule and the sweep cannot drift apart.
set -u

ROOT=${ROOT:-result-redd-tod}
OUT=${OUT:-.sweep-tod/joblist.txt}
PY=${PY:-./.venv/bin/python}
WS_JSON=${WS_JSON:-results/window_selection.json}
SEEDS=${SEEDS:-"0 1 2"}
GRID=${GRID:-"256 512 128"}
# Set to 0 to leave the fixed-grid sensitivity block out of the list entirely.
WITH_PHASE2=${WITH_PHASE2:-1}

# Worst-first: TCN+KL scores F1 31.9 / 41.5 / 49.2 on WM / DW / Microwave vs Fridge 87.4.
APPLIANCES=${APPLIANCES:-"WashingMachine Dishwasher Microwave Fridge"}

# Priority 0: the class-weighted loss. Promoted to the front because it addresses the
# dominant measured defect rather than a refinement. Every arm on this grid fires far
# more often than its appliance is on (3.0x reference, up to 31x), the excess scales
# with how rare the appliance is, and threshold calibration recovers 4-12 F1 points
# from it after the fact -- which says the models are miscalibrated, not weaker. These
# arms attack that at the source, in the loss. See src/helpers/losses.py.
CW_MODELS=${CW_MODELS:-"TCN_KL_cw3 TCN_KL_cw1_5 TCN_KL_act_cw3 TCN_KL_cw10"}

# Priority 1: the prior-information ladder. Reference first, then the arm that is one
# change from it, then the time-of-day arm with the control that makes it readable.
PRIOR_MODELS=${PRIOR_MODELS:-"TCN_KL_scratch TCN_KL_stat TCN_KL_act_tod TCN_KL_act TCN_KL_act_tod_stat TCN_KL_act_stat TCN_KL_act_tod_film"}
# Priority 2: KL basis geometry. TCN_KLrand is the control that decides whether the KL
# structure matters at all, so it is not last.
# Ordered by value, because under contention the tail of this list may not finish:
#   1 KLperapp   most likely to IMPROVE accuracy (per-target basis, not the pooled mix)
#   2 KLdct      control: is FITTING the basis worth anything over a fixed DCT?
#   3 KL4        is the low-variance tail worth 6 of 11 channels?
#   4 KLwave64_16 + 5 KL64_16 -- the matched wavelet-vs-KL pair, ADJACENT on purpose:
#                 arm 4 is uninterpretable without arm 5 as its counterpart.
# The rest are refinements and secondary controls.
KL_MODELS=${KL_MODELS:-"TCN_KLperapp TCN_KLdct TCN_KL4 TCN_KLwave64_16 TCN_KL64_16 TCN_KLwhiten TCN_KLlearn TCN_KLrand TCN_KL64_13 TCN_KLwave64"}

if [ ! -f "$WS_JSON" ]; then
  echo "missing $WS_JSON -- run: $PY scripts/pick_window.py" >&2; exit 1
fi

mkdir -p "$(dirname "$OUT")"
# Build into a temp file and rename at the end. Workers hold $OUT open on fd 9 and
# reopen it each pass, so an atomic rename lets them finish the current pass on the old
# list and pick up the new one next pass. Truncating in place would corrupt their reads.
TMP="$OUT.tmp$$"
trap 'rm -f "$TMP"' EXIT
: > "$TMP"

declare -A WS
for app in $APPLIANCES; do
  WS[$app]=$($PY -c "import json,sys;print(json.load(open('$WS_JSON'))['appliances'][sys.argv[1]]['window'])" "$app")
done

emit() { echo "REDD $1 $2 $3 $4 $ROOT" >> "$TMP"; }

# Which block goes first. File order is priority order, so this is the one knob that
# decides what a free worker picks up next. Default keeps the documented priority
# (class-weighted loss, then the prior ladder, then KL geometry); override it to pull a
# block forward without editing the model lists, e.g. BLOCK_ORDER="CW KL PRIOR".
BLOCK_ORDER=${BLOCK_ORDER:-"CW PRIOR KL"}

emit_block() {
  local models
  case "$1" in
    CW)    models=$CW_MODELS ;;
    PRIOR) models=$PRIOR_MODELS ;;
    KL)    models=$KL_MODELS ;;
    *) echo "unknown block '$1' in BLOCK_ORDER (want CW, PRIOR or KL)" >&2; exit 1 ;;
  esac
  for m in $models; do
    for app in $APPLIANCES; do
      for s in $SEEDS; do emit "$app" "${WS[$app]}" "$m" "$s"; done
    done
  done
}

# ---- blocks at appliance-dependent windows, in BLOCK_ORDER ----
for b in $BLOCK_ORDER; do emit_block "$b"; done

# ---- block 3: fixed-grid sensitivity, last ----
if [ "$WITH_PHASE2" = "1" ]; then
  for ws in $GRID; do
    for m in $CW_MODELS $PRIOR_MODELS $KL_MODELS; do
      for app in $APPLIANCES; do
        for s in $SEEDS; do
          # skip the cell already covered by its appliance-dependent window
          [ "$ws" = "${WS[$app]}" ] && continue
          emit "$app" "$ws" "$m" "$s"
        done
      done
    done
  done
fi

mv -f "$TMP" "$OUT"
trap - EXIT
echo "wrote $OUT: $(wc -l < "$OUT") cells"
awk '{print $4}' "$OUT" | sort | uniq -c | sort -rn | head -25
