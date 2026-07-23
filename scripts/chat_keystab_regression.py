# -*- coding: utf-8 -*-
"""Regression harness for cross-turn coverage-key stability.

Replays a transcript built to force df-rank movement - a first topic,
then a second topic whose keywords overtake the first in document
frequency, then a probe back to topic one - and checks the coverage-seed
accounting compress() reports every turn:

  1. coverage_seed_matched + coverage_orphan_keys == coverage_seed_keys
     on every turn (the accounting never loses a key).
  2. coverage_keys == len(trie.coverage) (the stat tells the truth).
  3. Orphan counts stay >= 0 and are PRINTED per turn as a measurement.
     Orphaning is the defect under study, not an error: today's default
     path is allowed to orphan, and the stable-keys work drives the
     printed numbers toward zero.
  4. Deferred-turn atomicity: a compress with defer_commit=True whose
     commit callable is never invoked leaves ALL SIX persisted fields -
     coverage, coverage_turn, _n_compress, drift_ema, kw_order and
     theme_admitted - exactly as they were, and invoking the callable
     applies them. kw_order and theme_admitted used to be written inside
     compress, outside the commit, so a turn saltChat dropped on a failed
     generation widened the append-only order for good. Runs under
     stable_keys regardless of --stable, the only path those two move on.

With --stable the same replay runs under stable_keys=True and must end
with zero orphans once the session's keyword order has warmed up.

Needs the BGE encoder (downloaded to the HF cache on first use). CPU is
the default device; the run takes well under a minute.

Usage:
    python scripts/chat_keystab_regression.py [--device cpu] [--stable]
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if not __debug__:
    sys.exit("this harness is assert-based - run it without python -O")

from salt.engine.compressor import load_bge
from salt.engine.session_trie import SessionTrie

BGE_MODEL = "BAAI/bge-small-en-v1.5"

TOPIC_A = [
    ("The reactor coolant loop showed a slow pressure drift overnight.",
     "A slow coolant drift usually means the loop needs a valve check."),
    ("Could the coolant drift come from the reactor recirculation pump?",
     "The recirculation pump can cause loop drift when its seal wears."),
    ("Schedule the coolant valve check and the pump seal inspection.",
     "Both the valve check and the seal inspection are on the board."),
    ("Log the reactor loop pressure hourly until the inspection.",
     "Hourly loop pressure logging is enabled for the reactor."),
]

TOPIC_B = [
    ("New topic: my sourdough starter smells odd after feeding.",
     "An odd starter smell after feeding often means overfermentation."),
    ("How much flour should the sourdough starter get per feeding?",
     "Equal weights of flour and water per feeding keeps a starter stable."),
    ("The dough tore during shaping, was the dough underproofed?",
     "Tearing dough during shaping points at underproofed dough."),
    ("What hydration should the next dough batch use?",
     "Try seventy percent hydration for the next dough batch."),
    ("My loaf crust came out pale, does the dough need more steam?",
     "A pale crust wants more steam early in the bake, not more dough."),
    ("Can the starter live in the fridge between dough batches?",
     "A fridge slows the starter nicely between dough batches."),
    ("The crumb was dense again, is my starter too weak for this dough?",
     "A dense crumb with a weak starter means feed it before the dough."),
    ("Give me a simple weekly routine for the starter and the dough.",
     "Feed the starter midweek and mix the dough the night before baking."),
]

PROBE = ("Back to the earlier equipment issue, what did we schedule for "
         "the reactor?")


def persisted_state(trie):
    """The six fields a turn commits, snapshotted so a discarded
    defer_commit turn can be proven a no-op against them."""
    return {"coverage": dict(trie.coverage),
            "coverage_turn": dict(trie.coverage_turn),
            "n_compress": trie._n_compress,
            "drift_ema": trie.drift_ema,
            "kw_order": list(trie.kw_order),
            "theme_admitted": set(trie.theme_admitted)}


def changed_fields(before, after):
    return [k for k in before if before[k] != after[k]]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--budget", type=float, default=0.25)
    ap.add_argument("--stable", action="store_true",
                    help="run the replay under stable_keys=True")
    ap.add_argument("--keep", action="store_true",
                    help="keep the temp session dir for inspection")
    args = ap.parse_args()

    print(f"Loading BGE encoder {BGE_MODEL} on {args.device} ...")
    tok, mdl = load_bge(BGE_MODEL, args.device)
    ckw = dict(tokenizer=tok, model=mdl, device=args.device)
    extra = {"stable_keys": True} if args.stable else {}

    tmp = Path(tempfile.mkdtemp(prefix="salt_keystab_regression_"))
    try:
        trie = SessionTrie("keystab", cache_dir=tmp, model_name=BGE_MODEL)
        orphan_trail = []
        for xi, (user, assistant) in enumerate(TOPIC_A + TOPIC_B):
            comp = trie.compress(query=user, budget_pct=args.budget,
                                 **ckw, **extra)
            s = comp.get("stats", {})
            if s:
                assert (s["coverage_seed_matched"] + s["coverage_orphan_keys"]
                        == s["coverage_seed_keys"]), (
                    f"exchange {xi + 1}: seed accounting lost a key: {s}")
                assert s["coverage_orphan_keys"] >= 0
                assert s["coverage_keys"] == len(trie.coverage), (
                    f"exchange {xi + 1}: coverage_keys stat disagrees with "
                    f"the persisted dict")
                orphan_trail.append(s["coverage_orphan_keys"])
            trie.add_turn(user, "user", **ckw)
            trie.add_turn(assistant, "assistant", **ckw)
        comp = trie.compress(query=PROBE, budget_pct=args.budget,
                             **ckw, **extra)
        s = comp["stats"]
        assert (s["coverage_seed_matched"] + s["coverage_orphan_keys"]
                == s["coverage_seed_keys"]), f"probe: {s}"
        assert s["coverage_keys"] == len(trie.coverage)
        orphan_trail.append(s["coverage_orphan_keys"])

        mode = "stable" if args.stable else "default"
        print(f"[{mode}] per-turn orphaned coverage keys: {orphan_trail}")
        print(f"[{mode}] probe turn: {s['coverage_seed_matched']} of "
              f"{s['coverage_seed_keys']} seed keys matched, "
              f"{s['coverage_orphan_keys']} orphaned "
              f"(mass {s['coverage_orphan_mass']})")
        if args.stable:
            assert sum(orphan_trail) == 0, (
                "stable mode orphaned keys somewhere in the replay - "
                "the frozen order plus sticky membership is not holding: "
                f"{orphan_trail}")

        # 4. deferred-turn atomicity: a discarded defer_commit turn must
        # leave ALL SIX persisted fields untouched. Runs under stable_keys
        # whatever --stable is, the only path kw_order/theme_admitted move.
        d = SessionTrie("keystab_defer", cache_dir=tmp, model_name=BGE_MODEL)
        for user, assistant in TOPIC_A:
            d.compress(query=user, budget_pct=args.budget,
                       stable_keys=True, **ckw)
            d.add_turn(user, "user", **ckw)
            d.add_turn(assistant, "assistant", **ckw)
        # fresh-topic keywords enter the corpus WITHOUT a committed compress
        # folding them into kw_order, so the next compress has real order
        # growth to either apply or discard
        for user, assistant in TOPIC_B:
            d.add_turn(user, "user", **ckw)
            d.add_turn(assistant, "assistant", **ckw)
        before = persisted_state(d)
        comp = d.compress(query=PROBE, budget_pct=args.budget,
                          stable_keys=True, defer_commit=True, **ckw)
        assert comp["commit"] is not None, "deferred turn returned no commit"
        assert persisted_state(d) == before, (
            "a discarded deferred turn changed a persisted field: "
            f"{changed_fields(before, persisted_state(d))}")
        # committing the same callable must move the frozen order, proving
        # the discard withheld a real change rather than a no-op
        comp["commit"](save=False)
        after = persisted_state(d)
        assert after != before, "the deferred commit was a no-op"
        assert len(d.kw_order) > len(before["kw_order"]), (
            "committing the fresh-topic turn did not extend the frozen "
            f"order ({len(before['kw_order'])} -> {len(d.kw_order)})")
        print(f"deferred atomicity: discard left all six fields, commit "
              f"extended kw_order {len(before['kw_order'])} -> "
              f"{len(d.kw_order)}")
        print("PASS")
    finally:
        if args.keep:
            print(f"session dir kept under {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
