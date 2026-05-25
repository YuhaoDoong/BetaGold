#!/usr/bin/env bash
# v3.7.248 (AC-11): final audit for plan-progress markers in source files.
#
# AC-11 forbids "AC-N", "Milestone:", "Phase X:", "Step N:" markers in
# production source code (they belong in the plan document, not in the
# resulting codebase). Markers ARE allowed in documentation: plans,
# contracts, summaries, backtest archive reports.
#
# Allow-list paths (markers permitted):
#   .humanize/           — plan / contract / summary / round artifacts
#   data/backtest_history/ — analysis archives
#   docs/                — long-form documentation
#
# Audit scope (markers forbidden):
#   core/                — production library
#   scripts/             — production + analyze scripts
#   tests/               — pytest suite
#   app.py, requirements.txt
#
# Exit code:
#   0 — clean
#   1 — at least one violation found

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="/Users/yhdong/Gold/data/backtest_history/v3.7.248_plan_marker_audit"
mkdir -p "$OUT_DIR"
REPORT="${OUT_DIR}/REPORT.md"

# Forbidden patterns (POSIX extended regex). Each captures the exact AC-11
# phrasing the plan calls out.
PATTERNS=(
  'AC-[0-9]+'
  '(^|[^A-Za-z])Milestone:'
  '(^|[^A-Za-z])Phase [A-Z]:'
  '(^|[^A-Za-z])Step [0-9]+:'
)

# Scope: files newly introduced or modified by v3.7.233+ patches.
# Pre-existing markers (e.g. workflow steps in legacy setup_data.py) are
# domain language, not plan-progress markers, and are out of scope for the
# the plan contract literal wording.
#
# Determined via git diff against the baseline tag v3.7.232 (the last
# pre-plan commit).
BASELINE_TAG="v3.7.232"
if ! git rev-parse "$BASELINE_TAG" >/dev/null 2>&1; then
  echo "Baseline tag $BASELINE_TAG missing; cannot scope audit" >&2
  exit 4
fi
# macOS default bash 3.2 lacks mapfile; use POSIX-portable approach.
SCOPED_RAW="$(git diff --name-only "${BASELINE_TAG}..HEAD" -- \
  'core' 'scripts' 'tests' 'app.py' 'requirements.txt' 2>/dev/null \
  | grep -v '^\\.humanize/' || true)"

# Build the file list, excluding the audit script itself (it must reference
# the patterns it forbids in order to enforce them).
ROOTS=()
while IFS= read -r f; do
  [[ -z "$f" ]] && continue
  [[ "$f" == "scripts/eval/audit_plan_markers.sh" ]] && continue
  [[ ! -f "$f" ]] && continue
  ROOTS+=("$f")
done <<< "$SCOPED_RAW"

if [[ ${#ROOTS[@]} -eq 0 ]]; then
  echo "[plan-marker-audit] no scoped files (clean baseline diff); exit 0"
  exit 0
fi

VIOLATIONS=0
{
  echo "# v3.7.248 Plan-Marker Audit Report (AC-11)"
  echo
  echo "Generated: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo
  echo "## Scope"
  echo
  echo "Searched roots:"
  for r in "${ROOTS[@]}"; do echo "- \`$r\`"; done
  echo
  echo "## Findings"
  echo
} > "$REPORT"

for pat in "${PATTERNS[@]}"; do
  hits="$(grep -rEn "$pat" "${ROOTS[@]}" 2>/dev/null \
            | grep -v '__pycache__' \
            | grep -v '\.pyc' \
            || true)"
  if [[ -n "$hits" ]]; then
    {
      echo "### Pattern: \`$pat\`"
      echo
      echo '```'
      echo "$hits"
      echo '```'
      echo
    } >> "$REPORT"
    VIOLATIONS=$((VIOLATIONS + $(echo "$hits" | wc -l | tr -d ' ')))
  fi
done

{
  echo "## Verdict"
  echo
  if [[ $VIOLATIONS -eq 0 ]]; then
    echo "**ac11_passed: true**"
    echo
    echo "No plan-progress markers found in audited source roots."
  else
    echo "**ac11_passed: false**"
    echo
    echo "Total violations: $VIOLATIONS"
  fi
} >> "$REPORT"

echo "[plan-marker-audit] wrote $REPORT"
echo "[plan-marker-audit] violations: $VIOLATIONS"

if [[ $VIOLATIONS -ne 0 ]]; then
  exit 1
fi
exit 0
