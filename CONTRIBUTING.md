# Contributing to SALT

Thanks for wanting to help. Contributions of all sizes are welcome — typo fixes,
bug reports, new dataset adapters, or new features.

## Getting started

```bash
# Fork on GitHub, then clone your fork
git clone https://github.com/<your-username>/SALT.git
cd SALT
git remote add upstream https://github.com/oteomamo/SALT.git

# Create the environment and install SALT in editable mode
conda env create -f environment.yml
conda activate salt
pip install -e .
```

`scripts/setup_env.sh` runs the environment steps in one shot (see the README
Installation section). The vLLM eval backend is optional — `eval.py --backend hf`
needs no extra install.

## Making a change

1. Branch off `main`: `git checkout -b feat/my-thing`
2. Make your change, matching the style of the surrounding code (see below).
3. Sanity-check that it runs — e.g. a small compression:

   ```bash
   python compress.py --data salt/datasets/longbench/data/hotpotqa.jsonl \
     --output /tmp/out.jsonl --device cuda --token-budget-pct 0.20 --max-samples 3
   ```

4. Commit with a clear message. A [conventional-commit](https://www.conventionalcommits.org/)
   prefix is appreciated but not required: `feat:`, `fix:`, `docs:`, `perf:`.
5. Push to your fork and open a PR against `main`.

## Code style

- **Naming**: `snake_case` for functions/variables, `PascalCase` for classes.
- **Docstrings**: on modules and public functions — match the existing tone.
- **Selectors stay separate**: the connector (`compress.py`) injects the selector;
  engine modules never import a selector directly.
- **Dependencies**: keep them minimal, and discuss before adding a new one.

## Git identity

Before pushing, make sure Git is set to an email GitHub can link to your account —
agentic coding tools and automation don't always inherit your shell config:

```bash
git config user.name
git config user.email
```

Avoid placeholder values such as `you@example.com`; unresolved author emails
create avoidable provenance friction for downstream users.

## License

The project license is being finalized (see the README). By contributing, you
agree your work will be released under that license.
