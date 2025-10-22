#!/usr/bin/env bash
# One-command study. Resumable: re-run the same command after any interruption.
# Each arm stops cleanly at its USD cap (--cap); the caps are ceilings, well above the cost of the reported study (see README).
set -euo pipefail
: "${OPENAI_API_KEY:?set OPENAI_API_KEY first}"
# Keep the machine awake for the duration (macOS). Linux: run under `systemd-inhibit`; Windows: see README.
if [[ -z "${CAFFEINATED:-}" ]] && command -v caffeinate >/dev/null 2>&1; then
  export CAFFEINATED=1
  exec caffeinate -i -s "$0" "$@"
fi
MAIN_MODEL=${MAIN_MODEL:-gpt-4.1-mini}
REP_MODEL=${REP_MODEL:-gpt-4o-mini}
FILTER_MODEL=${FILTER_MODEL:-gpt-4.1-nano}
RUNS=${RUNS:-runs}
WORKERS=${WORKERS:-4}

echo "== smoke: 4 sessions per architecture, then a cost projection (stop here if the projection is over budget)"
python -m harness.run --out $RUNS/smoke --model $MAIN_MODEL --filter-model $FILTER_MODEL --smoke 4 --cap 3 --workers $WORKERS
read -r -p "Continue with the full study? [y/N] " ans; [[ "$ans" == "y" ]] || exit 0

echo "== A: main (enforced, standard attacks, $MAIN_MODEL)"
python -m harness.run --out $RUNS/main --model $MAIN_MODEL --filter-model $FILTER_MODEL --cap 28 --workers $WORKERS

echo "== B: observe-only, injected sessions only, paired seeds, A's thresholds"
python -m harness.run --out $RUNS/observe --model $MAIN_MODEL --filter-model $FILTER_MODEL --cap 14 --workers $WORKERS \
  --observe-only --only-injected --thresholds $RUNS/main/thresholds.json

echo "== C: detector-aware attacks, observe-only, A's thresholds"
python -m harness.run --out $RUNS/adaptive --model $MAIN_MODEL --filter-model $FILTER_MODEL --cap 16 --workers $WORKERS \
  --adaptive --observe-only --thresholds $RUNS/main/thresholds.json --eval 400 --attack-fraction 0.75

echo "== D: replication on $REP_MODEL (own calibration)"
python -m harness.run --out $RUNS/replicate --model $REP_MODEL --filter-model $FILTER_MODEL --cap 12 --workers $WORKERS

echo "== analysis"
python -m harness.analyze --main $RUNS/main --observe $RUNS/observe --adaptive $RUNS/adaptive \
  --replicate $RUNS/replicate --out $RUNS/analysis
echo "done. results: $RUNS/analysis (results.json, tables.tex, summary.md, fig_*.pdf). Post-hoc analyses: python -m harness.posthoc --main $RUNS/main --observe $RUNS/observe --adaptive $RUNS/adaptive --replicate $RUNS/replicate"
