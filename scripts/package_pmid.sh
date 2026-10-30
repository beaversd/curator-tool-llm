#!/usr/bin/env bash
# Build a deployable tarball of the per-PMID pipeline (no secrets, no data, no results).
#
#   scripts/package_pmid.sh            # -> dist/curator-pmid-<git sha>.tar.gz
#
# On the server:
#   mkdir -p /opt/curator-pmid && tar -xzf curator-pmid-*.tar.gz -C /opt/curator-pmid
#   cd /opt/curator-pmid && python3 -m venv .venv && .venv/bin/pip install -r requirements-pmid.txt
#   cp .env.pmid.example .env && chmod 600 .env   # then fill in the keys
#   scripts/run_pmid.sh 24751536 PINK1
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

sha="$(git rev-parse --short HEAD 2>/dev/null || echo local)"
[[ -n "$(git status --porcelain 2>/dev/null)" ]] && sha="${sha}-dirty"
mkdir -p dist
out="dist/curator-pmid-${sha}.tar.gz"

# COPYFILE_DISABLE keeps macOS tar from adding ._* AppleDouble files.
COPYFILE_DISABLE=1 tar -czf "$out" \
  --exclude='__pycache__' --exclude='*.pyc' --exclude='*.log' --exclude='.DS_Store' \
  run_extraction.py run_merge.py run_review.py \
  reactome_llm curator_llm \
  requirements-pmid.txt .env.pmid.example \
  scripts/run_pmid.sh \
  LICENSE

echo "wrote $out ($(du -h "$out" | cut -f1))"
