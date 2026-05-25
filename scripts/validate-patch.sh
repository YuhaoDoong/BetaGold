#!/usr/bin/env bash
# v3.7.248: per-tag pytest validator with normalized output.
#
# Usage:
#   scripts/validate-patch.sh <tag>                       # specific tag
#   scripts/validate-patch.sh all                         # all tagged rounds
#
# Re-runs the pytest subset associated with each v3.7.* tag, normalizes
# non-deterministic output (durations, absolute paths), and writes the
# normalized result to
#   data/backtest_history/<tag>/VALIDATION.md
#
# Bit-identical reproducibility across runs is enforced via the
# normalization helper; pytest's "passed in N.Ns" line is regex-replaced.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <tag>|all" >&2
  exit 2
fi

TAG="$1"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Tag → pytest subset dispatch. Pure POSIX-compatible case statement so the
# script runs on macOS default bash (no associative arrays needed).
tests_for_tag() {
  case "$1" in
    v3.7.233) echo "tests/test_regime_no_lookahead.py" ;;
    v3.7.234) echo "tests/test_per_asset_cfg.py" ;;
    v3.7.235) echo "tests/test_per_asset_cfg.py" ;;
    v3.7.236) echo "tests/test_data_freshness.py" ;;
    v3.7.237) echo "tests/test_data_freshness.py" ;;
    v3.7.238) echo "tests/test_per_asset_cfg.py" ;;
    v3.7.239) echo "tests/test_expiry_intrinsic.py" ;;
    v3.7.240) echo "tests/test_cross_asset_selector.py" ;;
    v3.7.241) echo "tests/test_max_move_window.py" ;;
    v3.7.242) echo "tests/test_layer2_disposition.py" ;;
    v3.7.243) echo "tests/test_calibration_audit.py" ;;
    v3.7.244) echo "tests/test_calibration_scaler.py" ;;
    v3.7.245) echo "tests/test_calibration_retrain.py" ;;
    v3.7.246) echo "tests/test_calibration_per_regime.py" ;;
    v3.7.247) echo "tests/test_calibration_gate.py" ;;
    *) echo "" ;;
  esac
}

ALL_TAGS="v3.7.233 v3.7.234 v3.7.235 v3.7.236 v3.7.237 v3.7.238 v3.7.239 \
          v3.7.240 v3.7.241 v3.7.242 v3.7.243 v3.7.244 v3.7.245 v3.7.246 v3.7.247"

normalize() {
  # stdin → stdout normalized:
  #   - strip "passed in N.Ns" / "failed in N.Ns" duration
  #   - rewrite the repo root to <REPO>
  #   - rewrite arbitrary ~/.pytest_cache paths
  sed -E \
      -e 's/passed in [0-9]+\.[0-9]+s/passed in <NORMALIZED>s/g' \
      -e 's/failed in [0-9]+\.[0-9]+s/failed in <NORMALIZED>s/g' \
      -e "s|${REPO_ROOT}|<REPO>|g" \
      -e 's|/Users/[^ ]*/\.pytest_cache|<HOME>/.pytest_cache|g'
}

run_one_tag() {
  local tag="$1"
  local tests
  tests="$(tests_for_tag "$tag")"
  if [[ -z "$tests" ]]; then
    echo "No test subset registered for tag $tag" >&2
    return 3
  fi
  local outdir="/Users/yhdong/Gold/data/backtest_history/${tag}"
  mkdir -p "$outdir"
  local out_md="${outdir}/VALIDATION.md"
  echo "[validate-patch] $tag → $tests"
  {
    echo "# $tag VALIDATION"
    echo
    echo "Tests: \`$tests\`"
    echo
    echo '```'
    conda run -n gold python -m pytest "$tests" -v 2>&1 | normalize
    echo '```'
  } > "$out_md"
  echo "[validate-patch] wrote $out_md"
}

if [[ "$TAG" == "all" ]]; then
  for tag in $ALL_TAGS; do
    run_one_tag "$tag" || true
  done
else
  run_one_tag "$TAG"
fi
