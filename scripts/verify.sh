#!/usr/bin/env bash
# One command per verification area, so "which suites do I run" is never
# a memory exercise. Wraps the regression table in CONTRIBUTING.md.
#
# Usage:
#   bash scripts/verify.sh chat      # ingest, themes, scope, turns, summary, when
#   bash scripts/verify.sh all      # every CPU suite + eval smoke + docs
#   SALT_ENV=salt-next bash scripts/verify.sh all    # the same, in another conda env
#   SALT_PY=/path/to/python bash scripts/verify.sh all  # or under any interpreter
#   SMOKE_BASE=runs/base bash scripts/verify.sh smoke   # compare with a reference run
#
# Areas: chat, engine, dedup, keys, evict, incr, tail, text, scope, turns, summary, when,
# agents, mcp, models, pdf, docs, smoke, vllm, serve, all. `all` covers everything that
# runs on CPU with no server (vllm and serve need a GPU or a running server, run
# them explicitly).
#
# Suites run under the `salt` conda environment when it exists (they need
# its dependencies, e.g. pypdf), else under the current python. SALT_ENV
# names another conda environment and SALT_PY any interpreter, and the
# smoke run follows the same choice. With SMOKE_BASE the smoke output must
# match that run byte for byte, or under the cross-environment rule with
# SMOKE_ACROSS=1.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SALT_ENV="${SALT_ENV:-salt}"
PY=(python)
if [ -n "${SALT_PY:-}" ]; then
  if [ ! -x "$SALT_PY" ]; then
    echo "SALT_PY is not an executable python: $SALT_PY" >&2
    exit 2
  fi
  PY=("$SALT_PY")
  PATH="$(dirname "$SALT_PY"):$PATH"
elif command -v conda >/dev/null 2>&1 \
    && conda env list 2>/dev/null | awk '{print $1}' | grep -qx "$SALT_ENV"; then
  PY=(conda run --no-capture-output -n "$SALT_ENV" python)
  SALT_PY="$(conda env list | awk -v n="$SALT_ENV" '$1 == n {print $NF}')/bin/python"
elif [ "$SALT_ENV" != salt ]; then
  echo "no conda environment named $SALT_ENV" >&2
  exit 2
fi
export PATH
[ -n "${SALT_PY:-}" ] && export SALT_PY

FAILED=()

run() {
  local label="$1"; shift
  echo "== $label"
  if ! "$@"; then
    FAILED+=("$label")
    echo "-- FAILED: $label"
  fi
}

docs_suite() {
  # the python suites run under the salt env, and mkdocs is not one of its
  # dependencies, so a bare `mkdocs` resolves to nothing whenever that env
  # is the active one. Try the module too before giving up, and say what to
  # do rather than printing "command not found"
  if command -v mkdocs >/dev/null 2>&1; then
    mkdocs build --strict
  elif python -m mkdocs --version >/dev/null 2>&1; then
    python -m mkdocs build --strict
  else
    echo "mkdocs not found. Install it (pip install mkdocs-material) or run"
    echo "this area from an environment that has it."
    return 1
  fi
}

pdf_suite() {
  # pypdf is nondeterministic: believe a failure only when it repeats
  local n
  for n in 1 2 3; do
    if "${PY[@]}" scripts/chat_pdf_regression.py; then return 0; fi
    echo "-- pdf suite attempt $n/3 failed (pypdf nondeterminism, retrying)"
  done
  return 1
}

smoke_suite() {
  local out="$REPO_ROOT/runs/run_$(date +%Y%m%d_%H%M%S)"
  MAX_SAMPLES=5 RUN_EVAL=0 OUT_DIR="$out" bash scripts/run_datasets.sh || return 1
  if [ -n "${SMOKE_BASE:-}" ]; then
    "${PY[@]}" scripts/smoke_compare.py "$SMOKE_BASE" "$out" \
      ${SMOKE_ACROSS:+--across}
  fi
}

area_chat()   { run "chat ingest"     "${PY[@]}" scripts/chat_ingest_regression.py
                run "chat themes"     "${PY[@]}" scripts/chat_theme_regression.py
                area_scope; area_turns; area_summary; area_when; }
area_dedup()  { run "near-dup gate"   "${PY[@]}" scripts/chat_dedup_regression.py; }
area_keys()   { run "coverage keys"   "${PY[@]}" scripts/chat_keystab_regression.py; }
area_evict()  { run "session cap"     "${PY[@]}" scripts/chat_evict_regression.py; }
area_incr()   { run "carried caches"  "${PY[@]}" scripts/chat_incremental_regression.py; }
area_tail()   { run "tail exclusion"  "${PY[@]}" scripts/chat_tail_regression.py; }
area_text()   { run "chat text"       "${PY[@]}" scripts/chat_textclean_regression.py; }
area_scope()  { run "branch scope"    "${PY[@]}" scripts/chat_scope_regression.py; }
area_turns()  { run "scripted turns"  "${PY[@]}" scripts/chat_turns_regression.py; }
area_summary() { run "summary coverage" "${PY[@]}" scripts/chat_summary_regression.py; }
area_when()   { run "time windows"    "${PY[@]}" scripts/chat_when_regression.py; }
area_agents() { run "agent layer"     "${PY[@]}" scripts/chat_agents_regression.py; }
area_mcp()    { run "mcp server"      "${PY[@]}" scripts/chat_mcp_regression.py; }
area_models() { run "model loading"   "${PY[@]}" scripts/chat_models_regression.py; }
area_pdf()    { run "pdf ingestion"   pdf_suite; }
area_smoke()  { run "eval smoke"      smoke_suite; }
area_docs()   { run "mkdocs strict"   docs_suite; }
area_engine() { area_chat; area_smoke; }
area_vllm()   { run "vllm backend"    "${PY[@]}" scripts/chat_vllm_regression.py; }
area_serve()  { run "serving"         "${PY[@]}" scripts/chat_serve_regression.py; }
area_all()    { area_text; area_keys; area_chat; area_dedup; area_evict
                area_incr; area_tail; area_agents; area_mcp; area_models; area_pdf
                area_smoke; area_docs; }

case "${1:-}" in
  chat|engine|dedup|keys|evict|incr|tail|text|scope|turns|summary|when|agents|mcp|models|pdf|docs|smoke|vllm|serve|all)
    "area_$1" ;;
  *)
    sed -n '2,23p' "$0" | sed 's/^# \{0,1\}//'
    exit 2 ;;
esac

echo
if [ "${#FAILED[@]}" -eq 0 ]; then
  echo "verify: all green"
else
  echo "verify: FAILED - ${FAILED[*]}"
  exit 1
fi
