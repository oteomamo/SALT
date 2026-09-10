# -*- coding: utf-8 -*-
"""Regression harness for scripted runs (--turns).

Drives a small scripted file through the real chat path against a fake
runner that answers from a script and keeps every prompt, with the
sessions kept under a temporary directory. Groups:

  A. PARSING - --turns-mode is off the parser with its two values and
     its default, and an item id is made safe and unique as a session
     name.
  B. CONVERSATION MODE - the default: every item enters the one launch
     session, the memory builds across them (a later turn's prompt
     carries an earlier item's words), and the rows carry the shipped
     keys and no session field.
  C. INDEPENDENT MODE - every item runs in a fresh session named after
     the launch id and its own, no later prompt carries an earlier
     item's words, every row names its session, a document is attached
     to the session of the item after it and to no other, a repeated id
     never resumes another item's memory, and each session persists on
     disk under its own id.
  D. LONG RUNS - every turn prints its time and an ETA, a failing turn
     writes its row with the error and the run goes on, --turns-resume
     keeps the file, skips the items it already answers, retries a
     failed one and never skips a document, and --turns-timeout sets
     the served client's stall timeout or says it does not apply.

Needs the BGE encoder (downloaded to the HF cache on first use). CPU is
the default device; the run takes about a minute.

Usage:
    python scripts/chat_turns_regression.py [--device cpu]
"""

import argparse
import io
import json
import shutil
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from salt.engine.session_trie import SessionTrie

if not __debug__:
    sys.exit("run without -O: this harness is assert-based")

BGE_MODEL = "BAAI/bge-small-en-v1.5"
BASE = "turns_base"
ITEMS = [
    {"id": "zeppelin", "puzzle": "The zeppelin mooring mast in Lakehurst "
                                  "needs a new hydrogen manifold gasket. "
                                  "Which material suits it?"},
    {"id": "sourdough/loaf 2", "puzzle": "My sourdough starter doubled in "
                                          "six hours. Is it ready to bake?"},
    {"id": "zeppelin", "puzzle": "How long does a nitrile gasket last on "
                                  "a mooring mast?"},
]
DOC_TEXT = ("The garden irrigation system uses a drip line on each "
            "vegetable bed. A timer valve opens the drip line for twenty "
            "minutes at dawn. Rain sensors pause the schedule after heavy "
            "rainfall. The pump pressure stays near two bar.")


class _FakeRunner:
    kind = "fake"

    def __init__(self, tokenizer, fail_on=None):
        self.tokenizer = tokenizer
        self.alias = "fake"
        self.cfg = {"alias": "fake", "hf_id": "test/fake", "path": "-"}
        self.max_input_len = 4096
        self.last_prompt_tokens = None
        self.last_engine_stats = None
        self.prompts = []
        self.fail_on = fail_on

    def input_budget(self, max_new_tokens=None):
        return self.max_input_len

    def stream_chat(self, messages, **overrides):
        self.prompts.append(json.loads(json.dumps(messages)))
        if self.fail_on == len(self.prompts):
            raise RuntimeError("the server went quiet")
        yield f"noted point {len(self.prompts)}."

    def unload(self):
        pass


def make_state(tmp, tok, mdl, device, flags=(), fail_on=None):
    from salt.chat import cli
    args = cli.build_parser().parse_args(
        ["--device", device, "--sync-ingest", "--conversation-id", BASE,
         *flags])
    trie = SessionTrie(BASE, cache_dir=tmp, model_name=BGE_MODEL,
                       budget_pct_default=args.budget_pct)
    return cli.ChatState(args, tok, mdl, _FakeRunner(tok, fail_on), trie)


def write_items(tmp, items, name="turns.json"):
    p = tmp / name
    p.write_text(json.dumps(items), encoding="utf-8")
    return p


def prompt_text(messages):
    return "\n".join(m["content"] for m in messages)


def check_parsing():
    from salt.chat import cli
    assert cli.TURNS_MODES == ("conversation", "independent")
    p = cli.build_parser()
    assert p.parse_args([]).turns_mode == "conversation"
    assert p.parse_args(["--turns-mode", "independent"]).turns_mode == "independent"
    taken = set()
    assert cli.item_session_id("run", "zeppelin", taken) == "run-zeppelin"
    assert cli.item_session_id("run", "sourdough/loaf 2", taken) == "run-sourdough_loaf_2"
    assert cli.item_session_id("run", "zeppelin", taken) == "run-zeppelin-2"
    assert cli.item_session_id("run", 7, taken) == "run-7"
    assert cli.item_session_id("run", "...", taken) == "run-item"
    for cid in taken:
        assert cli.valid_session_id(cid), cid
    print("A. parsing: the mode and its default are on the parser, and an "
          "item id becomes a safe, unique session name")


def run(tmp, tok, mdl, device, items, mode="conversation", label="run"):
    from salt.chat import cli
    root = tmp / label
    root.mkdir()
    sessions, cli.SESSIONS_DIR = cli.SESSIONS_DIR, root
    try:
        state = make_state(root, tok, mdl, device, ["--turns-mode", mode])
        out = root / "out.jsonl"
        with redirect_stdout(io.StringIO()):
            cli.run_turns(state, cli.load_turns(write_items(root, items)),
                          str(out), mode=mode)
        rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
        return state, rows, root
    finally:
        cli.SESSIONS_DIR = sessions


def check_conversation(tmp, tok, mdl, device):
    state, rows, root = run(tmp, tok, mdl, device, ITEMS, label="conv")
    assert [set(r) for r in rows] == [{"id", "turn", "question", "answer"}] * 3, rows
    assert state.trie.conversation_id == BASE
    assert state.trie.n_turns == 6, state.trie.n_turns
    third = prompt_text(state.runner.prompts[2])
    assert "hydrogen manifold gasket" in third, (
        "the third turn's prompt carries no memory of the first item")
    assert "SALT memory" in third
    assert sorted(p.name for p in root.iterdir() if p.is_dir()) == [BASE], (
        sorted(p.name for p in root.iterdir()))
    print("B. conversation mode: one session, memory builds across the "
          "items, rows carry the shipped keys and no session field")


def check_independent(tmp, tok, mdl, device):
    state, rows, root = run(tmp, tok, mdl, device, ITEMS, "independent",
                            label="indep")
    assert [r["conversation"] for r in rows] == [
        f"{BASE}-zeppelin", f"{BASE}-sourdough_loaf_2", f"{BASE}-zeppelin-2"], rows
    assert all(set(r) == {"id", "turn", "question", "answer", "conversation"}
               for r in rows), rows
    for msgs in state.runner.prompts[1:]:
        text = prompt_text(msgs)
        assert "hydrogen manifold gasket" not in text, (
            "a later item's prompt carried the first item's words")
    assert state.trie.conversation_id == f"{BASE}-zeppelin-2"
    assert state.trie.n_turns == 2, state.trie.n_turns
    dirs = sorted(p.name for p in root.iterdir() if p.is_dir())
    for cid in (f"{BASE}-zeppelin", f"{BASE}-sourdough_loaf_2",
                f"{BASE}-zeppelin-2"):
        assert cid in dirs, dirs
    # a document rides into the item after it and no further
    doc = root / "notes.txt"
    doc.write_text(DOC_TEXT, encoding="utf-8")
    items = [{"id": "a", "doc": str(doc)},
             {"id": "b", "puzzle": "When does the timer valve open the drip line?"},
             {"id": "c", "puzzle": "What is a good hydration for a first dough?"}]
    state, rows, root = run(tmp, tok, mdl, device, items, "independent",
                            label="indep_doc")
    assert [r.get("kind", "chat") for r in rows] == ["doc", "chat", "chat"], rows
    assert rows[0]["conversation"] == rows[1]["conversation"] == f"{BASE}-a", rows
    assert rows[2]["conversation"] == f"{BASE}-c", rows
    first, second = state.runner.prompts
    assert "drip line" in prompt_text(first), "the document did not reach the item after it"
    assert "drip line" not in prompt_text(second), "the document leaked into the next item"
    print("C. independent mode: a fresh session per item named after the "
          "launch and item ids, no memory across items, rows name their "
          "session, a document rides into the item after it only, a "
          "repeated id never resumes another item, sessions persist")


def check_long_runs(tmp, tok, mdl, device):
    from salt.chat import cli
    root = tmp / "long"
    root.mkdir()
    sessions, cli.SESSIONS_DIR = cli.SESSIONS_DIR, root
    try:
        out = root / "out.jsonl"
        turns = cli.load_turns(write_items(root, ITEMS))
        # a failing second turn: its row carries the error, the run goes on
        state = make_state(root, tok, mdl, device, fail_on=2)
        buf = io.StringIO()
        with redirect_stdout(buf):
            cli.run_turns(state, turns, str(out))
        rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
        assert [r["answer"] is None for r in rows] == [False, True, False], rows
        assert rows[1]["error"].startswith("RuntimeError: the server went quiet"), rows[1]
        assert "error" not in rows[0] and "error" not in rows[2], rows
        text = buf.getvalue()
        assert text.count("ETA") == 3 and "elapsed" in text, text
        state.trie.save()
        # resume: the answered items are skipped, the failed one runs again
        state = make_state(root, tok, mdl, device)
        with redirect_stdout(io.StringIO()) as buf:
            cli.run_turns(state, turns, str(out), resume=True)
        rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 4 and len(state.runner.prompts) == 1, (rows, state.runner.prompts)
        assert rows[3]["id"] == "sourdough/loaf 2" and rows[3]["answer"], rows[3]
        assert buf.getvalue().count("skipped, answered in an earlier run") == 2
        assert cli.answered_turns(str(out)) == {("id", "zeppelin"), ("id", "sourdough/loaf 2")}
        # without --turns-resume the file starts over
        state = make_state(root, tok, mdl, device)
        with redirect_stdout(io.StringIO()):
            cli.run_turns(state, turns, str(out))
        rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 3 and len(state.runner.prompts) == 3
        # a document is never skipped on resume
        doc = root / "notes.txt"
        doc.write_text(DOC_TEXT, encoding="utf-8")
        items = [{"id": "a", "doc": str(doc)}, {"id": "b", "puzzle": "When does the timer valve open?"}]
        out2 = root / "out2.jsonl"
        state = make_state(root, tok, mdl, device)
        with redirect_stdout(io.StringIO()):
            cli.run_turns(state, cli.load_turns(write_items(root, items, "t2.json")), str(out2))
        state = make_state(root, tok, mdl, device)
        with redirect_stdout(io.StringIO()) as buf:
            cli.run_turns(state, cli.load_turns(write_items(root, items, "t2.json")), str(out2), resume=True)
        assert "attach>" in buf.getvalue() and buf.getvalue().count("skipped") == 1
        # the stall timeout reaches a served client and is declined elsewhere
        state = make_state(root, tok, mdl, device)
        state.runner.read_timeout = None
        with redirect_stdout(io.StringIO()):
            cli.run_turns(state, turns[:1], None, timeout=7.5)
        assert state.runner.read_timeout == 7.5
        state = make_state(root, tok, mdl, device)
        with redirect_stdout(io.StringIO()) as buf:
            cli.run_turns(state, turns[:1], None, timeout=7.5)
        assert "does not apply" in buf.getvalue(), buf.getvalue()
        assert cli.clock(59) == "59s" and cli.clock(61) == "1m 01s" and cli.clock(3725) == "1h 02m"
        p = cli.build_parser()
        assert p.parse_args([]).turns_timeout is None and not p.parse_args([]).turns_resume
        assert p.parse_args(["--turns-timeout", "30", "--turns-resume"]).turns_resume
        assert cli.main(["--turns-timeout", "-1"]) == 1
        assert cli.main(["--turns-resume"]) == 1
    finally:
        cli.SESSIONS_DIR = sessions
    print("D. long runs: every turn reports time and ETA, a failing turn "
          "keeps its row with the error and the run goes on, resume skips "
          "answered items and retries the failed one and never a document, "
          "the stall timeout reaches a served client and is declined "
          "elsewhere, and the launch checks refuse a bad timeout or a "
          "resume without an output file")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    check_parsing()
    from salt.engine.compressor import load_bge
    print(f"Loading BGE encoder {BGE_MODEL} on {args.device} ...")
    tok, mdl = load_bge(BGE_MODEL, args.device)
    tmp = Path(tempfile.mkdtemp(prefix="salt_turns_regression_"))
    try:
        check_conversation(tmp, tok, mdl, args.device)
        check_independent(tmp, tok, mdl, args.device)
        check_long_runs(tmp, tok, mdl, args.device)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("PASS")


if __name__ == "__main__":
    main()
