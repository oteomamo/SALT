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

One command runs the right suites for an area:

```bash
bash scripts/verify.sh chat      # after touching the chat loop or session trie
bash scripts/verify.sh all      # every CPU suite plus the eval smoke run
```

The script picks the `salt` conda environment automatically when it
exists (the suites need its dependencies, `pypdf` among them) and
retries the PDF suite before believing a failure, since `pypdf` is not
deterministic. The table it wraps, for reference:

| If you touched | Run |
|---|---|
| `salt/chat/ingest.py`, `salt/chat/cli.py`, `salt/engine/session_trie.py` | `python scripts/chat_ingest_regression.py` and `python scripts/chat_theme_regression.py` |
| `salt/engine/chat_text.py` (chat ingest cleaning) | `python scripts/chat_textclean_regression.py` |
| Cross-turn coverage keys (`salt/engine/session_trie.py`, `salt/engine/celf.py`) | `python scripts/chat_keystab_regression.py` |
| The near-duplicate gate (`--dedup-cos` paths) | `python scripts/chat_dedup_regression.py` |
| The session cap (`--max-sentences` paths) | `python scripts/chat_evict_regression.py` |
| `salt/chat/pdfio.py` (PDF or text ingestion) | `python scripts/chat_pdf_regression.py` |
| The vLLM backend (`--backend vllm`) | `python scripts/chat_vllm_regression.py` |
| Persistent serving (`saltServe`, `--backend vllm-serve`) | `python scripts/chat_serve_regression.py` |
| The selection engine (`salt/engine/`) | `python scripts/chat_theme_regression.py` plus a smoke run: `MAX_SAMPLES=5 RUN_EVAL=0 bash scripts/run_datasets.sh` |

All of them run on CPU. If a script needs a model, it downloads to your HF
cache on first use.

## The frozen core

SALT's published results come from the one-shot compression path, and
every past and future result must stay reproducible. These files are
therefore frozen: `salt/compress.py` and `salt/engine/celf.py`,
`trie_core.py`, `embedder.py`, `dataset_modes.py`, `sentence_filter.py`,
`retrieval.py`.

Frozen does not mean untouchable. It means a change there must be
strictly additive: a new parameter whose default reproduces today's
behavior exactly, so a run with no flags selects byte-identical output.
PRs that reorder, re-tune, or "clean up" these files will be asked to
restructure. PRs that add an opt-in seam with the default proven
unchanged are welcome, and the chat layer adds them regularly.

To prove a default is unchanged, run the eval smoke on `main` and on
your branch with fixed output dirs and diff the results:

```bash
MAX_SAMPLES=5 RUN_EVAL=0 OUT_DIR=runs/base bash scripts/run_datasets.sh
MAX_SAMPLES=5 RUN_EVAL=0 OUT_DIR=runs/mine bash scripts/run_datasets.sh
```

Timing fields may differ. The compressed text may not.

Everyday feature work does not need any of this: the conversation layer
(`salt/engine/session_trie.py` and everything under `salt/chat/`) plus
`docs/` is where features live, and it is not frozen.

## Adding a saltChat option, the golden path

The most common change in this repo. The flag itself is the smallest
part. A finished option touches these places, in this order:

1. The flag in `build_parser()` (`salt/chat/cli.py`), defaulting to
   today's behavior.
2. `ChatState.__init__` stores it.
3. The call site passes it (usually the `compress()` call in
   `chat_turn`).
4. The behavior reports itself in the returned `stats`, so it can be
   judged before it is trusted.
5. The `/stats` handler prints one line when the option is active.
6. A one or two sentence concept bullet on `docs/chatbot.md`.
7. A one line row on `docs/options.md` (what it does, when to turn it
   on).
8. A changelog bullet under the current minor, when behavior changes.
9. A regression check that pins the new behavior on AND pins the
   default identical to before.

Then `bash scripts/verify.sh chat`, all in the same PR. An option
nobody can discover, judge, or trust is not finished.

## Adding a regression check

The harnesses under `scripts/` share idioms. Match them:

- Assert-based, refusing `python -O` (see the `__debug__` guard at the
  top of any harness), one printed line per check group, `PASS` at the
  end.
- CPU-only and deterministic: fixed transcripts and fixtures, no clocks
  and no randomness.
- Pin both sides of a change: the new behavior with its option on, and
  the old behavior byte-identical with it off.
- A new script gets a row in the table above and a line in
  `scripts/README.md`. A new group in an existing harness extends that
  harness's numbered docstring list.

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
