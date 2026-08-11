#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

BRANCH="${BRANCH:-3dh-query-stage1-radar-candidate-recall}"
OUTPUT_DIR="${1:-$REPO_ROOT/outputs/nano_q1_offline_transfer}"

if ! git show-ref --verify --quiet "refs/heads/$BRANCH"; then
  echo "local branch not found: $BRANCH" >&2
  exit 1
fi
CURRENT_BRANCH="$(git branch --show-current)"
if [[ "$CURRENT_BRANCH" != "$BRANCH" ]]; then
  echo "check out $BRANCH before packaging (current: $CURRENT_BRANCH)" >&2
  exit 1
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "tracked working-tree or staged changes exist; commit them first" >&2
  exit 1
fi

COMMIT="$(git rev-parse "refs/heads/$BRANCH^{commit}")"
SHORT_COMMIT="$(git rev-parse --short=12 "$COMMIT")"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(realpath "$OUTPUT_DIR")"
BASENAME="3dh_query_q1_code_${SHORT_COMMIT}.bundle"
BUNDLE="$OUTPUT_DIR/$BASENAME"
CHECKSUM="$BUNDLE.sha256"
MANIFEST="$OUTPUT_DIR/3dh_query_q1_code_${SHORT_COMMIT}_manifest.txt"

for path in "$BUNDLE" "$CHECKSUM" "$MANIFEST"; do
  if [[ -e "$path" ]]; then
    echo "refusing to overwrite existing package file: $path" >&2
    exit 1
  fi
done

git bundle create "$BUNDLE" HEAD "refs/heads/$BRANCH"
git bundle verify "$BUNDLE"
(
  cd "$OUTPUT_DIR"
  sha256sum "$BASENAME" > "$(basename "$CHECKSUM")"
)

{
  echo "type: offline Git bundle for Q1 Nano"
  echo "created at: $(date --iso-8601=seconds)"
  echo "source repository: $REPO_ROOT"
  echo "branch: $BRANCH"
  echo "commit: $COMMIT"
  echo "bundle: $BASENAME"
  echo "bundle sha256: $(sha256sum "$BUNDLE" | awk '{print $1}')"
} > "$MANIFEST"

echo "Q1 Nano offline code package created:"
echo "  $BUNDLE"
echo "  $CHECKSUM"
echo "  $MANIFEST"
