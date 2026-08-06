# -*- coding: utf-8 -*-
"""Regression harness for the MCP server (salt-mcp).

Drives the real server as a subprocess over a stdio pipe, the way a
client does, and covers the surface a client depends on:

  1. Handshake: the server names itself and reports the version this
     checkout carries.
  2. Tool list: an EXACT pin on the tool names and on each one's
     arguments, since a renamed tool or a renamed argument breaks every
     client silently.
  3. salt_compress: a round trip on a real text, with the reply
     schema-validated against what the tool declares.
  4. Refusals: an empty text and an impossible budget come back as
     errors rather than as an empty success.
  5. Sessions: create, resume, list and stats round-trip through a
     scratch sessions folder, an id the REPL would refuse is refused
     here too, and the least recently used session is evicted with its
     unsaved rows written first. The stats snapshot and the switch
     inventory are pinned by shape, since a rule written against either
     breaks silently when one moves.
  6. Turns and memory: a message is remembered, an exchange goes in as
     one call, a turn submitted a moment earlier is in the memory the
     next call reads, and the block carries the same labels a chat turn
     is given.
  7. Documents: a file and a bare text each land under their own
     source name, a name that looks like a path keeps only its last
     part, and an unreadable file is refused rather than half read.
  8. Bad calls: a malformed argument, an oversized text, an unknown
     conversation and an unknown tool each come back as a typed refusal
     naming its kind, never as a traceback, and the server answers the
     next call as if nothing happened. Garbage on the pipe does not end
     it either.
  9. Read-only: a second server on the same folder reads everything
     and refuses every write with one stable, recognisable error, and
     the conversation it read is byte for byte as it was left.
 10. Off-path: importing saltChat still imports neither the server nor
     the MCP SDK.
 11. Delegation: a task handed to a stub worker comes back with the
     conversation's memory selected for it, leaves a ledger line under
     the session, remembers the answer as a worker row when asked to,
     and commits nothing to the conversation either way.
 12. Hardening: a session evicted while its ingest queue is still full
     loses nothing, a conversation damaged between two file writes opens
     repaired and says so, and a server killed mid-session closes down
     rather than being killed.

Skips with exit 0 when the mcp extra is not installed, so a plain
install stays green. CPU only, no GPU and no model.
Assert-based: refuses to run under python -O.
"""

import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

if not __debug__:
    sys.exit("this harness is assert-based - run it without python -O")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
BGE_MODEL = "BAAI/bge-small-en-v1.5"

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
except ImportError:
    print("SKIP: the mcp extra is not installed "
          "(pip install 'salt[mcp]') - nothing to drive")
    sys.exit(0)

from _agent_stub import Stub                                    # noqa: E402
from salt.agents.snapshot import KEYS as SNAPSHOT_KEYS           # noqa: E402
from salt.agents.snapshot import SCHEMA as SNAPSHOT_SCHEMA       # noqa: E402
from salt.mcp.server import salt_version                        # noqa: E402

# eight sentences, one of which is the only place December is discussed:
# a budget this tight has to drop most of it, and a query has to reach
# the one sentence that answers it
TEXT = (
    "The house has 9 kW of solar panels and a 5 kW inverter. "
    "Winter evenings are the worst case, with about four hours of draw. "
    "The roof faces south and was replaced in 2019. "
    "In December the panels produce almost nothing, so the battery carries "
    "the evening on its own. "
    "The installer quoted two options, one of them with a second inverter. "
    "The cheaper option reuses the existing inverter and adds a battery. "
    "A dog named Rex lives in the house and has nothing to do with the "
    "electrical system. "
    "The utility charges more for power between five and nine in the evening."
)

TOOLS = {"salt_compress": ["budget_pct", "query", "text"],
         "session_create": ["conversation_id"],
         "session_resume": ["conversation_id"],
         "session_list": [],
         "session_stats": ["conversation_id"],
         "session_add_turn": ["conversation_id", "exchange", "role", "sync",
                              "text"],
         "session_memory": ["budget_pct", "conversation_id", "query"],
         "salt_ingest_document": ["conversation_id", "path", "source_name",
                                  "text"],
         "roster_list": ["probe"],
         "salt_switches": [],
         "salt_delegate": ["budget_pct", "context_query", "conversation_id",
                           "ingest", "target", "task"]}
STAT_KEYS = {"orig_words", "kept_words", "n_sentences", "n_sentences_raw",
             "n_selected", "word_budget", "actual_tokens",
             "theme_coverage_pct", "compression_ratio", "budget_pct",
             "queried"}


def payload(result):
    """What the tool returned, from either half of the reply. The text
    block is the contract for a client that reads no structured output,
    so both are checked to carry the same thing."""
    structured = result.structured_content or {}
    structured = structured.get("result", structured)
    text = json.loads(result.content[0].text)
    assert text == structured, (
        f"the text block and the structured reply disagree:\n"
        f"{text}\n{structured}")
    return structured


def check_reply(out, budget, queried):
    assert set(out) == {"compressed", "stats"}, sorted(out)
    stats = out["stats"]
    assert set(stats) == STAT_KEYS, (
        f"the stats block changed shape: {sorted(stats)}")
    assert stats["budget_pct"] == budget, stats
    assert stats["queried"] is queried, stats
    assert stats["orig_words"] > 0 and stats["kept_words"] > 0, stats
    assert stats["kept_words"] <= stats["word_budget"], (
        f"the reply kept {stats['kept_words']} words against a budget of "
        f"{stats['word_budget']}")
    assert stats["n_selected"] <= stats["n_sentences"], stats
    kept = out["compressed"]
    assert kept.strip(), "a successful compression returned no text"
    assert len(kept.split()) == stats["kept_words"], (
        f"the word count does not describe the text it came with: "
        f"{len(kept.split())} vs {stats['kept_words']}")
    return stats


DOC = (
    "The roof faces south and was replaced in 2019. "
    "The utility charges more for power between five and nine. "
    "A battery of nine kilowatt hours covers the winter evening draw. "
    "The panels are rated at nine kilowatts in full sun."
)


async def drive(sessions):
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "salt.mcp.server", "--device", "cpu",
              "--sessions-dir", str(sessions), "--max-open-sessions", "2",
              "--max-ingest-chars", "4000"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            info = init.server_info
            assert info.name == "salt", f"the server named itself {info.name!r}"
            assert info.version == salt_version(), (
                f"the handshake reports {info.version!r}, the package is "
                f"{salt_version()!r}")
            print(f"1. handshake: the server answers as {info.name!r} "
                  f"carrying salt {info.version}")

            listed = await session.list_tools()
            names = {t.name: sorted(t.input_schema["properties"])
                     for t in listed.tools}
            assert names == TOOLS, f"the tool surface moved: {names}"
            required = {t.name: set(t.input_schema.get("required", []))
                        for t in listed.tools}
            assert required["salt_compress"] == {"text"}, required
            print(f"2. tool list: {len(names)} tools pinned by name and "
                  f"arguments, with only the text required of a "
                  f"compression")

            res = await session.call_tool(
                "salt_compress",
                {"text": TEXT, "budget_pct": 0.25,
                 "query": "what happens to the panels in December"})
            assert not res.is_error, res.content
            out = payload(res)
            stats = check_reply(out, 0.25, True)
            assert "December" in out["compressed"], (
                f"the query did not reach the sentence that answers it: "
                f"{out['compressed']!r}")

            plain = payload(await session.call_tool(
                "salt_compress", {"text": TEXT, "budget_pct": 0.25}))
            check_reply(plain, 0.25, False)
            assert plain["compressed"] != out["compressed"], (
                "the query changed nothing about what was kept")

            whole = payload(await session.call_tool(
                "salt_compress", {"text": TEXT, "budget_pct": 1.0}))
            assert whole["stats"]["kept_words"] >= stats["kept_words"], (
                "a full budget kept less than a quarter of one")
            print(f"3. salt_compress: {stats['orig_words']} words to "
                  f"{stats['kept_words']} under a quarter budget, the "
                  f"queried sentence among them, and every declared stat "
                  f"present and consistent")

            for bad, why in (({"text": "   "}, "an empty text"),
                             ({"text": TEXT, "budget_pct": 0}, "a zero budget"),
                             ({"text": TEXT, "budget_pct": 3}, "an over budget")):
                failed = await session.call_tool("salt_compress", bad)
                assert failed.is_error, f"{why} was accepted"
            print("4. refusals: an empty text, a zero budget and a budget "
                  "over 1 each come back as errors")

            made = payload(await session.call_tool(
                "session_create", {"conversation_id": "mcp-one"}))
            assert made["created"] and made["n_turns"] == 0, made
            assert Path(made["path"]).is_dir(), made
            again = await session.call_tool("session_create",
                                            {"conversation_id": "mcp-one"})
            auto = payload(await session.call_tool("session_create", {}))
            assert auto["conversation_id"].startswith("chat-"), auto
            for cid in ("mcp two", "../escape", ""):
                bad = await session.call_tool("session_resume",
                                              {"conversation_id": cid})
                assert bad.is_error, f"{cid!r} was accepted as a session id"

            listed = payload(await session.call_tool("session_list", {}))
            names = [s["conversation_id"] for s in listed["sessions"]]
            assert "mcp-one" in names, names
            assert listed["n"] == len(listed["sessions"]), listed
            assert set(listed["open"]) <= set(names) | {
                auto["conversation_id"]}, listed

            stats = payload(await session.call_tool(
                "session_stats", {"conversation_id": "mcp-one"}))
            assert stats["conversation_id"] == "mcp-one", stats
            assert stats["open_sessions"] <= 2, (
                f"the pool holds {stats['open_sessions']} sessions past a "
                f"cap of 2")
            assert stats["budget_pct"] == 0.2, stats
            assert stats["snapshot_schema"] == SNAPSHOT_SCHEMA, stats
            assert list(stats["snapshot"]) == list(SNAPSHOT_KEYS), (
                f"the snapshot changed shape: {list(stats['snapshot'])}")
            snap = stats["snapshot"]
            assert snap["n_sentences"] == 0 and snap["n_turns"] == 0, snap
            # a session with no chat model and no verbatim tail says so
            # rather than reporting a zero it cannot know
            assert snap["model_window"] is None, snap
            assert snap["tail_occupancy"] is None, snap
            assert snap["drift_cos"] is None and snap["orphan_keys"] is None, (
                f"a conversation nobody has read from reported a "
                f"compression: {snap}")

            switches = payload(await session.call_tool("salt_switches", {}))
            assert switches["writable"] is False, switches
            assert switches["snapshot_keys"] == list(SNAPSHOT_KEYS), switches
            names_seen = [s["name"] for s in switches["switches"]]
            assert len(names_seen) == len(set(names_seen)) >= 10, names_seen
            for sw in switches["switches"]:
                assert set(sw) == {"name", "flag", "value", "default",
                                   "stats_key", "what", "changed"}, sw
                assert sw["value"] == sw["default"] and not sw["changed"], (
                    f"the server reports {sw['name']} away from its "
                    f"shipped value: {sw}")
                assert sw["flag"].startswith("--") and sw["what"], sw
            missing = await session.call_tool(
                "session_stats", {"conversation_id": "never-existed"})
            assert missing.is_error, "stats answered for a session nobody made"

            # the cap is 2, so opening a third closes the oldest. Its
            # config has to be on disk after that, which is the whole
            # point of draining and saving before letting go
            payload(await session.call_tool("session_create",
                                            {"conversation_id": "mcp-two"}))
            payload(await session.call_tool("session_create",
                                            {"conversation_id": "mcp-three"}))
            open_now = payload(await session.call_tool(
                "session_list", {}))["open"]
            assert len(open_now) <= 2, open_now
            assert "mcp-one" not in open_now, (
                f"the least recently used session stayed open: {open_now}")
            assert (sessions / "mcp-one" / "config.json").is_file(), (
                "an evicted session was dropped without being written")
            resumed = payload(await session.call_tool(
                "session_resume", {"conversation_id": "mcp-one"}))
            assert not resumed["created"], resumed
            print(f"5. sessions: create, resume, list and stats round-trip, "
                  f"3 malformed ids refused, a cap of 2 evicted the oldest "
                  f"with its state written first, and the "
                  f"{len(SNAPSHOT_KEYS)}-signal snapshot and "
                  f"{len(names_seen)}-switch inventory came back pinned")

            payload(await session.call_tool("session_create",
                                            {"conversation_id": "mcp-turns"}))
            one = payload(await session.call_tool("session_add_turn", {
                "conversation_id": "mcp-turns", "role": "user",
                "text": "We are sizing a home battery for a house with "
                        "9 kW of panels.", "sync": True}))
            assert one["added"] == 1 and one["new_sentences"] == 1, one
            batch = payload(await session.call_tool("session_add_turn", {
                "conversation_id": "mcp-turns", "sync": True,
                "exchange": [
                    {"role": "assistant",
                     "text": "The inverter is rated at 5 kW continuous, so "
                             "the evening draw matters most."},
                    {"role": "user",
                     "text": "Winter evenings are the worst case, roughly "
                             "four hours of draw each night."},
                    {"role": "assistant",
                     "text": "In December the panels produce almost "
                             "nothing, so the battery carries the "
                             "evening."}]}))
            assert batch["added"] == 3, batch
            assert batch["n_sentences"] == 4, batch
            for bad in ({"conversation_id": "mcp-turns"},
                        {"conversation_id": "mcp-turns", "text": "  "},
                        {"conversation_id": "mcp-turns", "text": "hi",
                         "role": "worker"}):
                refused = await session.call_tool("session_add_turn", bad)
                assert refused.is_error, f"{bad} was accepted"

            # submitted without sync, then read: the read drains first, so
            # the sentence is in the memory rather than still in flight
            payload(await session.call_tool("session_add_turn", {
                "conversation_id": "mcp-turns", "role": "user",
                "text": "The installer quoted two options, and the cheaper "
                        "one reuses the existing inverter."}))
            mem = payload(await session.call_tool("session_memory", {
                "conversation_id": "mcp-turns",
                "query": "what happens to the panels in December",
                "budget_pct": 0.5}))
            assert mem["stats"]["n_sentences"] == 5, (
                f"a turn submitted without sync was missed by the read: "
                f"{mem['stats']}")
            assert "SALT memory" in mem["memory"], mem["memory"][:200]
            assert "earlier conversation" in mem["memory"], mem["memory"][:200]
            assert "December" in mem["memory"], mem["memory"]
            assert mem["stats"]["n_selected"] >= 1, mem["stats"]
            assert mem["stats"]["budget_pct"] == 0.5, mem["stats"]
            empty = payload(await session.call_tool("session_memory", {
                "conversation_id": "mcp-two", "query": "anything"}))
            assert empty["memory"] == "" and empty["stats"][
                "n_selected"] == 0, empty
            no_query = await session.call_tool("session_memory", {
                "conversation_id": "mcp-turns", "query": "   "})
            assert no_query.is_error, "a memory read with no query was taken"
            print(f"6. turns and memory: 1 message and a 3 turn exchange "
                  f"remembered, 3 malformed rows refused, and a read after "
                  f"an unsynced write saw all {mem['stats']['n_sentences']} "
                  f"sentences in a labeled block")

            doc = sessions / "notes.txt"
            doc.write_text(DOC, encoding="utf-8")
            filed = payload(await session.call_tool("salt_ingest_document", {
                "conversation_id": "mcp-turns", "path": str(doc)}))
            assert filed["source"] == "notes.txt", filed
            assert filed["added"] >= 3 and not filed["merged_into_existing"], (
                filed)
            assert "notes.txt" in filed["attachments"], filed
            typed = payload(await session.call_tool("salt_ingest_document", {
                "conversation_id": "mcp-turns",
                "text": "The second inverter would cost more than the "
                        "battery it supports.",
                "source_name": "../../etc/passwd"}))
            assert typed["source"] == "passwd", (
                f"a source name kept a path: {typed['source']}")
            assert typed["added"] == 1, typed
            for bad, why in (
                    ({"conversation_id": "mcp-turns"}, "neither a path nor "
                                                      "a text"),
                    ({"conversation_id": "mcp-turns", "path": str(doc),
                      "text": DOC}, "both a path and a text"),
                    ({"conversation_id": "mcp-turns",
                      "path": str(sessions / "absent.txt")}, "a missing file"),
                    ({"conversation_id": "mcp-turns", "text": "   "},
                     "an empty text")):
                refused = await session.call_tool("salt_ingest_document", bad)
                assert refused.is_error, f"{why} was accepted"
            after = payload(await session.call_tool("session_memory", {
                "conversation_id": "mcp-turns",
                "query": "when was the roof replaced", "budget_pct": 0.4}))
            assert "notes.txt" in after["memory"], (
                f"the document is not labeled with its source: "
                f"{after['memory'][:300]}")

            # a conversation that has been read from can describe itself:
            # the signals a decision reads are numbers now, not None
            lived = payload(await session.call_tool(
                "session_stats", {"conversation_id": "mcp-turns"}))["snapshot"]
            assert lived["n_sentences"] == lived["n_alive"] > 5, lived
            assert lived["n_turns"] >= 4 and lived["live_words"] > 0, lived
            assert lived["n_attachments"] == 2, lived
            assert 0 < lived["attachment_words"] < lived["live_words"], lived
            assert lived["coverage_keys"] > 0 and lived["masked"] == 0, lived
            assert lived["orphan_keys"] is not None, (
                f"a conversation that was just read from reports no "
                f"compression: {lived}")
            assert lived["session_age_s"] is not None, lived
            print(f"7. documents: a file under its own name, a typed text "
                  f"under a name stripped to {typed['source']!r}, 4 bad "
                  f"calls refused, and the excerpts labeled with the file "
                  f"they came from")

            await check_bad_calls(session, sessions)


BAD_CALLS = (
    ("session_memory", {"conversation_id": "mcp-turns", "query": "  "},
     "invalid_argument", "a blank query"),
    ("session_memory", {"conversation_id": "mcp-turns", "query": "x",
                        "budget_pct": 9}, "invalid_argument",
     "a budget over one"),
    ("session_memory", {"conversation_id": "no-such-thing", "query": "x"},
     "not_found", "a conversation nobody made"),
    ("session_stats", {"conversation_id": "no/such/thing"},
     "invalid_session", "an id the REPL would refuse"),
    ("session_add_turn", {"conversation_id": "mcp-turns",
                          "exchange": [{"role": "nobody", "text": "hi"}]},
     "invalid_argument", "a role nobody speaks"),
    ("session_add_turn", {"conversation_id": "mcp-turns",
                          "text": "x" * 4001}, "too_large",
     "a message past the character bound"),
    ("salt_compress", {"text": "x" * 4001}, "too_large",
     "a text past the character bound"),
    ("salt_delegate", {"task": "think about it"}, "no_roster",
     "a delegation with no roster loaded"),
    ("salt_ingest_document", {"conversation_id": "mcp-turns",
                              "path": "/nowhere/at/all.txt"}, "not_found",
     "a file that is not there"),
)


async def check_bad_calls(session, sessions):
    """Every way a call can be wrong, and the server still standing."""
    from salt.mcp.errors import PREFIXES
    for tool, args, code, why in BAD_CALLS:
        refused = await session.call_tool(tool, args)
        assert refused.is_error, f"{why} was accepted by {tool}"
        said = refused.content[0].text
        assert PREFIXES[code] in said, (
            f"{why} came back as {said!r}, which is not a {code} refusal")
        assert "Traceback" not in said and "File \"" not in said, (
            f"{tool} sent a traceback over the wire: {said}")
    unknown = await session.call_tool("no_such_tool", {})
    assert unknown.is_error, "an unknown tool was accepted"
    wrong_type = await session.call_tool("session_stats",
                                         {"conversation_id": 17})
    assert wrong_type.is_error, "session_stats accepted a number as an id"

    # the whole point: after all of that, the next call works
    alive = payload(await session.call_tool(
        "session_stats", {"conversation_id": "mcp-turns"}))
    assert alive["n_sentences"] > 0, alive
    check_pipe_garbage(sessions)
    print(f"8. bad calls: {len(BAD_CALLS)} refusals each typed by kind, an "
          f"unknown tool and a wrongly typed argument refused, no traceback "
          f"on the wire, a server that answered the next call, and another "
          f"that survived garbage on its pipe")


def break_session(root, cid):
    """Damage one conversation the way a crash between two file writes
    does: the corpus holds a sentence the embedding matrix does not."""
    import numpy as np
    path = Path(root) / cid / "embeddings.npy"
    rows = np.load(path)
    np.save(path, rows[:-1])
    return rows.shape[0]


def check_hardening(root):
    """Closing under load, opening after damage, and dying on a signal."""
    import select
    from salt.mcp.pool import SessionPool
    from salt.mcp.server import Engine, add_turns, session_payload

    engine = Engine("cpu").ready()
    pool = SessionPool(root, capacity=1)
    # a session closed with its queue still full: eviction has to drain
    # before it drops, or the rows nobody waited for are simply lost
    add_turns(engine, pool, "mcp-evicted",
              [{"role": r, "text": t} for r, t in TALK], sync=False)
    pending = pool.open["mcp-evicted"].ingest.pending
    pool.get("mcp-other")
    assert "mcp-evicted" not in pool.open, "the cap did not evict"
    reopened = SessionPool(root).get("mcp-evicted")
    assert reopened.trie.n_sentences >= 3, (
        f"an eviction under load lost rows: "
        f"{reopened.trie.n_sentences} sentences survived")
    assert not reopened.warnings, reopened.warnings
    pool.close_all()

    # a conversation damaged between two file writes opens, repairs
    # itself, and says so
    rows = break_session(root, "mcp-evicted")
    damaged = SessionPool(root)
    payload_out = session_payload(damaged, "mcp-evicted")
    notes = payload_out.get("warnings") or []
    assert any("repaired as it opened" in n for n in notes), (
        f"a repaired conversation reported nothing: {payload_out}")
    assert damaged.open["mcp-evicted"].trie.n_sentences == rows - 1, (
        "the repair did not roll the conversation back")
    damaged.close_all()

    env = dict(os.environ)
    env.pop("MKL_THREADING_LAYER", None)
    proc = subprocess.Popen(
        [sys.executable, "-m", "salt.mcp.server", "--device", "cpu",
         "--sessions-dir", str(root)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, text=True, cwd=REPO, env=env)
    proc.stdin.write(json.dumps({
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                   "clientInfo": {"name": "term", "version": "0"}}}) + "\n")
    proc.stdin.flush()
    assert select.select([proc.stdout], [], [], 60)[0], "no handshake"
    proc.stdout.readline()
    proc.terminate()
    code = proc.wait(timeout=30)
    assert code == 0, (
        f"a server killed mid-session left with status {code} rather than "
        f"closing down")
    print(f"12. hardening: a session evicted with {pending} jobs still "
          f"queued kept every row, a conversation damaged between two "
          f"writes opened repaired and said so, and a server told to stop "
          f"mid-session shut down cleanly")


def check_pipe_garbage(sessions):
    """Nonsense on the pipe, then a real request. A server that dies on
    the first malformed line is a server one bad client can end."""
    import select
    env = dict(os.environ)
    env.pop("MKL_THREADING_LAYER", None)
    proc = subprocess.Popen(
        [sys.executable, "-m", "salt.mcp.server", "--device", "cpu",
         "--sessions-dir", str(sessions)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, text=True, cwd=REPO, env=env)
    try:
        proc.stdin.write("this is not json at all\n")
        proc.stdin.write("{\"jsonrpc\": \"2.0\", \"id\": 1}\n")
        proc.stdin.write(json.dumps({
            "jsonrpc": "2.0", "id": 2, "method": "initialize",
            "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                       "clientInfo": {"name": "fuzz", "version": "0"}}}) + "\n")
        proc.stdin.flush()
        ready = select.select([proc.stdout], [], [], 60)[0]
        assert ready, "the server never answered after garbage on the pipe"
        line = proc.stdout.readline()
        answer = json.loads(line)
        assert answer.get("id") == 2, (
            f"the server answered something other than the real request: "
            f"{line.strip()}")
        assert proc.poll() is None, "the server exited on a malformed line"
    finally:
        proc.stdin.close()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
    return True


def digest(root):
    """Every byte of every session file, so a read-only run can be shown
    to have changed nothing at all."""
    import hashlib
    out = {}
    for f in sorted(Path(root).rglob("*")):
        if f.is_file():
            out[str(f.relative_to(root))] = hashlib.sha256(
                f.read_bytes()).hexdigest()
    return out


async def drive_read_only(sessions, before):
    """A second server over the same folder, started --read-only."""
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "salt.mcp.server", "--device", "cpu",
              "--sessions-dir", str(sessions), "--read-only"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            assert "read-only" in (init.instructions or ""), (
                f"the handshake does not say the server is read-only: "
                f"{init.instructions!r}")
            listed = await session.list_tools()
            assert {t.name for t in listed.tools} == set(TOOLS), (
                "a read-only server offers a different set of tools, so a "
                "client cannot rely on the surface")

            # reads all answer
            listing = payload(await session.call_tool("session_list", {}))
            assert listing["n"] >= 1, listing
            stats = payload(await session.call_tool(
                "session_stats", {"conversation_id": "mcp-turns"}))
            assert stats["n_sentences"] >= 5, stats
            mem = payload(await session.call_tool("session_memory", {
                "conversation_id": "mcp-turns",
                "query": "what happens to the panels in December",
                "budget_pct": 0.5}))
            assert mem["memory"], "a read-only server returned no memory"
            assert mem["stats"]["committed"] is False, (
                f"a read-only read committed the turn: {mem['stats']}")
            compressed = payload(await session.call_tool(
                "salt_compress", {"text": TEXT, "budget_pct": 0.25}))
            assert compressed["compressed"], compressed

            # writes all refuse, with the one wording
            writes = (
                ("session_create", {"conversation_id": "mcp-new-one"}),
                ("session_add_turn", {"conversation_id": "mcp-turns",
                                      "text": "one more thing"}),
                ("salt_ingest_document", {"conversation_id": "mcp-turns",
                                          "text": "a document"}),
            )
            for tool, args in writes:
                refused = await session.call_tool(tool, args)
                assert refused.is_error, f"{tool} wrote on a read-only server"
                said = refused.content[0].text
                assert "read-only server:" in said, (
                    f"{tool} refused without the stable wording: {said}")
                assert tool in said, f"{tool} is not named in its own "\
                                     f"refusal: {said}"
            assert not (sessions / "mcp-new-one").exists(), (
                "a refused session_create still made a directory")
    after = digest(sessions)
    changed = [k for k in set(before) | set(after)
               if before.get(k) != after.get(k)]
    assert not changed, f"a read-only server changed {changed}"
    print(f"9. read-only: reads answer and the memory block comes back "
          f"uncommitted, 3 writes refuse with one stable error, and all "
          f"{len(after)} session files are byte for byte unchanged")


def check_off_path():
    code = ("import salt.chat.cli, sys; "
            "print([m for m in ('salt.mcp', 'salt.mcp.server', 'mcp') "
            "if m in sys.modules])")
    # the delegation group loads the encoder here, and importing torch pins
    # MKL_THREADING_LAYER for the whole process: a child that inherits it
    # dies against libgomp before it can import anything
    env = dict(os.environ)
    env.pop("MKL_THREADING_LAYER", None)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, cwd=REPO, env=env)
    assert out.returncode == 0, out.stderr[-400:]
    assert out.stdout.strip() == "[]", (
        f"saltChat now imports the MCP layer: {out.stdout.strip()}")
    print("10. off-path: importing saltChat pulls in neither the server "
          "nor the MCP SDK")


TALK = [("user", "The house has 9 kW of solar panels and a 5 kW inverter."),
        ("assistant", "In December the panels produce almost nothing, so "
                      "the battery carries the evening on its own."),
        ("user", "The utility charges more for power between five and nine "
                 "in the evening.")]
ANSWER = ("The panels are idle in December", ", so the battery pays for "
                                             "the evening peak.")


def worker_roster(url):
    """One attached worker, resolved by hand. A roster file would have to
    resolve its alias against the model registry, and this harness runs
    where nothing is registered."""
    from salt.agents.roster import Roster, RosterEntry
    return Roster(path="<test>", entries=(
        RosterEntry(name="w", alias="stub", role="worker", server_url=url,
                    model={"alias": "stub", "hf_id": "some/model",
                           "path": BGE_MODEL}, timeout_s=10),))


def trie_digest(session_dir):
    """Everything the conversation saved about itself. The delegation
    ledger is left out on purpose: it is a record of what happened, not
    part of what the conversation remembers."""
    return {k: v for k, v in digest(session_dir).items()
            if k != "delegations.jsonl"}


def sent_prompt(runtime, stub):
    """What the worker was actually asked, read back through its own
    tokenizer: the serve client sends token ids, not text."""
    tokenizer = runtime.worker("w").runner.tokenizer
    return tokenizer.decode(stub.httpd.last_payload["prompt"]).lower()


def check_delegation(root):
    from salt.agents.roster import RosterError
    from salt.mcp.agents import AgentRuntime, roster_payload, run_delegation
    from salt.mcp.pool import SessionPool
    from salt.mcp.server import Engine, add_turns

    engine = Engine("cpu").ready()
    pool = SessionPool(root)
    bare = AgentRuntime(engine, pool=pool)
    empty = roster_payload(bare)
    assert empty["workers"] == [] and "--roster" in empty["note"], empty
    try:
        run_delegation(bare, "anything at all")
        raise AssertionError("a server with no roster delegated")
    except RosterError as exc:
        assert "--roster" in str(exc), exc

    with Stub(cards=[{"id": "some/model", "max_model_len": 4096}],
              pieces=ANSWER) as stub:
        runtime = AgentRuntime(engine, pool=pool, roster=worker_roster(stub.url))
        table = roster_payload(runtime)
        row = table["workers"][0]
        assert (row["name"], row["role"], row["mode"]) == ("w", "worker",
                                                           "attach"), row
        assert row["endpoint"] == stub.url and row["state"] == "DECLARED", row

        cid = "mcp-delegate"
        add_turns(engine, pool, cid,
                  [{"role": r, "text": t} for r, t in TALK], sync=True)
        session = pool.get(cid)
        session.trie.save()
        before = trie_digest(session.trie.cache_dir)
        sentences = session.trie.n_sentences

        out = run_delegation(runtime, "what carries the evening in winter",
                             conversation_id=cid, budget_pct=0.6)
        assert out["status"] == "ok", out
        assert out["answer"] == "".join(ANSWER), out
        assert out["target"] == "w" and out["id"] == 1, out
        assert out["context"]["n_selected"] > 0, (
            f"the worker was sent no memory at all: {out}")
        assert out["recorded"] and not out["remembered"], out
        sent = sent_prompt(runtime, stub)
        assert "december" in sent and "what carries the evening" in sent, (
            f"the worker did not get the conversation under the task: "
            f"{sent!r}")

        # a delegation is a read the conversation never learns about
        if session.trie.dirty:
            session.trie.save()
        assert trie_digest(session.trie.cache_dir) == before, (
            "a delegation changed what the conversation remembers")
        assert session.trie.n_sentences == sentences, session.trie.n_sentences

        lines = [json.loads(x) for x in
                 (session.trie.cache_dir / "delegations.jsonl")
                 .read_text().splitlines() if x.strip()]
        assert len(lines) == 1, lines
        rec = lines[0]
        assert rec["target"] == "w" and rec["status"] == "ok", rec
        assert rec["ingest"] is False and rec["id"] == 1, rec
        assert rec["context_stats"]["n_selected"] == out["context"][
            "n_selected"], rec

        kept = run_delegation(runtime, "say that again",
                              conversation_id=cid, ingest=True)
        assert kept["remembered"] and kept["id"] == 2, kept
        session.drain()
        assert session.trie.n_sentences > sentences, (
            "an ingested answer added nothing to the conversation")
        assert session.trie.roles[-1] == "worker", session.trie.roles[-3:]
        assert session.trie.origins[-1] == "w", session.trie.origins[-3:]

        # context-free: the worker gets the task and nothing else
        alone = run_delegation(runtime, "name three colours")
        assert alone["status"] == "ok" and alone["context"][
            "n_selected"] == 0, alone
        assert not alone["recorded"] and alone["conversation_id"] == "", alone
        assert "december" not in sent_prompt(runtime, stub), (
            "a context-free delegation carried a conversation with it")
        pool.close_all()
        runtime.close()
    print("11. delegation: a task goes to the worker under the "
          "conversation's own memory, the ledger records it, an ingested "
          "answer lands as a worker row, and the conversation itself is "
          "byte for byte unchanged by either")


def main():
    sessions = Path(tempfile.mkdtemp(prefix="salt_mcp_regression_"))
    try:
        asyncio.run(drive(sessions))
        asyncio.run(drive_read_only(sessions, digest(sessions)))
        # before the delegation group, which loads the encoder into this
        # process and makes the off-path claim harder to state honestly
        check_off_path()
        check_delegation(sessions)
        check_hardening(sessions)
    finally:
        shutil.rmtree(sessions, ignore_errors=True)
    print("PASS")


if __name__ == "__main__":
    main()
