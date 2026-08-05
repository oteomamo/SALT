# 📦 Installation

Requires Python 3.10 and a CUDA GPU (CPU works for compression, just slower).

**1. Clone the repository**

```bash
git clone https://github.com/oteomamo/SALT.git
cd SALT
```

**2. Create the environment**

With conda:

```bash
conda env create -f environment.yml
conda activate salt
```

Or with venv:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**3. Install SALT in editable mode**

```bash
pip install -e .
```

This also installs the two console commands: `salt` (one-shot compression,
see [Usage](usage.md)) and `saltChat` (interactive chat, see
[Chatbot mode](chatbot.md)).

**4. Authenticate with Hugging Face** - the eval model
(`meta-llama/Llama-3.1-8B-Instruct`) is gated:

```bash
hf auth login
```

Or skip the CLI and export the token directly: `export HF_TOKEN=hf_...`.

**5. (Optional) vLLM backend.** `eval.py` defaults to vLLM,
`saltChat --backend vllm` uses it for prefix caching, and `saltServe`
launches a persistent model server with it. Install it into the `salt`
env:

```bash
pip install "vllm==0.11.0" "prometheus-fastapi-instrumentator>=8.0.1"
```

The second pin keeps the server's routes healthy next to newer fastapi
releases. Skip this and run `eval.py --backend hf` for a portable run
that needs no vLLM. `saltChat` already defaults to its HF backend.
`saltServe` can also run a vLLM installed in a separate environment
through `--vllm-bin`.

**6. (Optional) MCP server.** `salt-mcp` puts SALT behind the Model
Context Protocol, so an editor or an agent runtime can compress text
and keep conversation memory through it. Install the extra:

```bash
pip install "salt[mcp]"
```

Then point a client at the `salt-mcp` command, see
[MCP server](mcp.md). It needs no GPU.

> `bash scripts/setup_env.sh` does steps 2–3 in one shot (add `WITH_VLLM=1` to
> include vLLM).
