# Contributing to SALT

Thanks for wanting to help. Contributions of all sizes are welcome: typo fixes,
bug reports, new dataset adapters, or new features.

## Getting started

```bash
git clone https://github.com/<your-username>/SALT.git
cd SALT
git remote add upstream https://github.com/oteomamo/SALT.git

conda env create -f environment.yml
conda activate salt
pip install -e .
```

`scripts/setup_env.sh` runs the environment steps in one shot (see the README
Installation section). The vLLM eval backend is optional. `eval.py --backend hf`
needs no extra install.

## Making a change

1. Branch off `main`: `git checkout -b feat/my-thing`
2. Make your change, matching the style of the surrounding code (see below).
3. Verify it: run the regression scripts for the area you touched (table below)
   and a small end-to-end check.
4. Commit in small steps with clear messages (see "Commit messages").
5. Push to your fork and open a PR against `main`. The PR form asks what
   changed, why, and how you verified it. Please fill in every section.

## Organizing your change

Big unexplained diffs are hard to review and hard to trust, so the rules here
are about making changes easy to follow:

- **One concern per PR.** A feature and an unrelated cleanup are two PRs.
- **Split large changes into small commits.** Each commit should build, pass
  the relevant regression scripts, and make sense on its own. A reviewer should
  be able to read your PR commit by commit.
- **Explain, don't just show.** Every PR says what changed, why it was needed,
  and how it was verified (which commands you ran and what they reported).
- **Update the docs in the same PR** when behavior changes: the README, the
  files under `docs/`, and any `--help` text your change touches.

Regression scripts to run, by area:

| If you touched | Run |
|---|---|
| `salt/chat/ingest.py`, `salt/chat/cli.py`, `salt/engine/session_trie.py` | `python scripts/chat_ingest_regression.py` and `python scripts/chat_theme_regression.py` |
| The near-duplicate gate (`--dedup-cos` paths) | `python scripts/chat_dedup_regression.py` |
| `salt/chat/pdfio.py` (PDF or text ingestion) | `python scripts/chat_pdf_regression.py` |
| The vLLM backend (`--backend vllm`) | `python scripts/chat_vllm_regression.py` |
| The selection engine (`salt/engine/`) | `python scripts/chat_theme_regression.py` plus a smoke run: `MAX_SAMPLES=5 RUN_EVAL=0 bash scripts/run_datasets.sh` |

All of them run on CPU. If a script needs a model, it downloads to your HF
cache on first use.

## Commit messages

A short imperative subject, then a body of two to six lines covering what
changed, why, and how it was verified:

```
saltChat: skip empty lines in tail compaction

Blank assistant replies were creating empty tail entries and breaking
strict chat templates that require alternating roles. Compaction now
drops them before the pair check. Verified with
scripts/chat_ingest_regression.py (all groups pass) and a manual
20-turn REPL session.
```

A [conventional-commit](https://www.conventionalcommits.org/) prefix
(`feat:`, `fix:`, `docs:`, `perf:`) is appreciated but not required.

## Using AI to contribute

AI-assisted contributions are welcome. The maintainers use AI tools too. The
same expectations apply to everyone, including us:

- **You are the author of record.** Review every line before pushing. If you
  cannot explain a change, do not submit it.
- **Disclose substantial AI involvement** in the PR description. A sentence is
  enough. Maintainer changes pushed without a PR are covered by the standing
  note above: the maintainers use AI tools.
- **AI output follows the same organization rules.** Small commits, clear
  explanations, verification you actually ran. AI makes it easy to produce a
  large diff quickly, which is exactly why the explanation matters more, not
  less.
- **Only measured numbers.** Any performance or accuracy claim in a PR must
  come from a run you did, never from an estimate.

## Code style

- **Naming**: `snake_case` for functions/variables, `PascalCase` for classes.
- **Comments are kept minimal.** The code should read on its own. A one-line
  comment is fine when it states a constraint the code cannot show, anything
  longer belongs in the PR description.
- **Docstrings**: short (one to three lines) on modules and public functions.
- **Selectors stay separate**: the connector (`salt/compress.py`) injects the
  selector. Engine modules never import a selector directly.
- **Dependencies**: keep them minimal, and discuss before adding a new one.

## Git identity

Before pushing, make sure Git is set to an email GitHub can link to your
account:

```bash
git config user.name
git config user.email
```

Avoid placeholder values such as `you@example.com`. Unresolved author emails
create avoidable provenance friction for downstream users.

## License

SALT is released under the [MIT License](LICENSE). By contributing, you agree
that your contributions will be licensed under the same terms.
