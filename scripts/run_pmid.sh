#!/usr/bin/env bash
# Run the full-text pipeline for ONE PMID: extract -> merge -> review.
# Same chain as curator_llm.services.extraction.SubprocessExtractor, from the command line.
# Needs ANTHROPIC_API_KEY, OPENAI_API_KEY (review) and PUBMED_API_KEY in .env.
# Neo4j settings (REACTOME_NEO4J_*) are for the merging step; MongoDB is not used.
#
#   scripts/run_pmid.sh 24751536              # no gene -> results/pmid24751536_*
#   scripts/run_pmid.sh 24751536 PINK1        # gene label -> results/pink1_pmid24751536_*
#   scripts/run_pmid.sh 24751536 PINK1 --no-review
#
# Outputs (in results/): <stem>_extraction.json, <stem>_merged.json, <stem>_review.md
# Exit codes: 0 ok, 1 usage/env error, 2 no PMC full text / no reactions, 3 a step failed.
#
# Environment (optional):
#   PYTHON (default: .venv/bin/python if present, else python3)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

review=1
pos=()
for arg in "$@"; do
  case "$arg" in
    --no-review) review=0 ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    -*) echo "unknown option: $arg" >&2; exit 1 ;;
    *) pos+=("$arg") ;;
  esac
done
if [[ ${#pos[@]} -lt 1 || ${#pos[@]} -gt 2 ]]; then
  echo "usage: $0 PMID [GENE] [--no-review]" >&2; exit 1
fi
PMID="${pos[0]}"
GENE="${pos[1]:-}"
if ! [[ "$PMID" =~ ^[0-9]{4,9}$ ]]; then
  echo "error: '$PMID' is not a PMID (4-9 digits)" >&2; exit 1
fi
[[ -f .env ]] || { echo "error: $ROOT/.env not found (copy .env.pmid.example)" >&2; exit 1; }

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x .venv/bin/python ]]; then PYTHON=.venv/bin/python; else PYTHON=python3; fi
fi

# Same stem run_extraction.py uses, so the merge/review inputs line up.
STEM="$("$PYTHON" -c 'import sys; sys.path.insert(0, "reactome_llm")
from PubMedFetcher import output_stem; print(output_stem(sys.argv[1], gene=sys.argv[2] or None))' "$PMID" "$GENE")"
EXTRACTION="results/${STEM}_extraction.json"
MERGED="results/${STEM}_merged.json"

gene_args=()
[[ -n "$GENE" ]] && gene_args=(--gene "$GENE")

echo "=== [1/3] extraction: PMID $PMID${GENE:+ (gene $GENE)}"
"$PYTHON" run_extraction.py "$PMID" ${gene_args[@]+"${gene_args[@]}"} --overwrite || { echo "extraction failed" >&2; exit 3; }
if [[ ! -s "$EXTRACTION" ]] || [[ "$("$PYTHON" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$EXTRACTION")" == 0 ]]; then
  echo "no reactions extracted for PMID $PMID (no usable PMC full text?)" >&2; exit 2
fi

echo "=== [2/3] merge: $EXTRACTION"
"$PYTHON" run_merge.py "$EXTRACTION" || { echo "merge failed" >&2; exit 3; }

if [[ $review -eq 1 ]]; then
  echo "=== [3/3] review: $MERGED"
  "$PYTHON" run_review.py "$MERGED" || { echo "review failed" >&2; exit 3; }
else
  echo "=== [3/3] review skipped"
fi

echo "=== done: results/${STEM}_*"
