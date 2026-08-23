#!/usr/bin/env bash
# Commit and push using the local git identity (not an assistant author).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <serial-number>" >&2
  exit 2
fi

SERIAL="$1"
if ! [[ "$SERIAL" =~ ^[0-9]+$ ]]; then
  echo "serial number must be a positive integer" >&2
  exit 2
fi

git add -A
if git diff --cached --quiet; then
  echo "No staged changes for #${SERIAL} Commit"
  exit 0
fi

git commit -m "#${SERIAL} Commit"
git push -u origin HEAD
