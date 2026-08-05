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
     unsaved rows written first.
  6. Turns and memory: a message is remembered, an exchange goes in as
     one call, a turn submitted a moment earlier is in the memory the
     next call reads, and the block carries the same labels a chat turn
     is given.
  7. Documents: a file and a bare text each land under their own
     source name, a name that looks like a path keeps only its last
     part, and an unreadable file is refused rather than half read.
  8. Read-only: a second server on the same folder reads everything
     and refuses every write with one stable, recognisable error, and
     the conversation it read is byte for byte as it was left.
  9. Off-path: importing saltChat still imports neither the server nor
     the MCP SDK.

Skips with exit 0 when the mcp extra is not installed, so a plain
install stays green. CPU only, no GPU and no model.
Assert-based: refuses to run under python -O.
"""

import asyncio
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

if not __debug__:
    sys.exit("this harness is assert-based - run it without python -O")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
except ImportError:
    print("SKIP: the mcp extra is not installed "
          "(pip install 'salt[mcp]') - nothing to drive")
    sys.exit(0)

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
                                  "text"]}
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
              "--sessions-dir", str(sessions), "--max-open-sessions", "2"])
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
                  f"3 malformed ids refused, and a cap of 2 evicted the "
                  f"oldest with its state written first")

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
            print(f"7. documents: a file under its own name, a typed text "
                  f"under a name stripped to {typed['source']!r}, 4 bad "
                  f"calls refused, and the excerpts labeled with the file "
                  f"they came from")


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
    print(f"8. read-only: reads answer and the memory block comes back "
          f"uncommitted, 3 writes refuse with one stable error, and all "
          f"{len(after)} session files are byte for byte unchanged")


def check_off_path():
    code = ("import salt.chat.cli, sys; "
            "print([m for m in ('salt.mcp', 'salt.mcp.server', 'mcp') "
            "if m in sys.modules])")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, cwd=REPO)
    assert out.returncode == 0, out.stderr[-400:]
    assert out.stdout.strip() == "[]", (
        f"saltChat now imports the MCP layer: {out.stdout.strip()}")
    print("9. off-path: importing saltChat pulls in neither the server "
          "nor the MCP SDK")


def main():
    sessions = Path(tempfile.mkdtemp(prefix="salt_mcp_regression_"))
    try:
        asyncio.run(drive(sessions))
        asyncio.run(drive_read_only(sessions, digest(sessions)))
    finally:
        shutil.rmtree(sessions, ignore_errors=True)
    check_off_path()
    print("PASS")


if __name__ == "__main__":
    main()
