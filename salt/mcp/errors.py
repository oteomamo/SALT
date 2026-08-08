# -*- coding: utf-8 -*-
"""What a refused tool call looks like on the wire.

Every refusal carries a code, and every code renders as a fixed opening
phrase, so a client can tell one kind of refusal from another by reading
the front of the message and a person can read the whole of it. The
phrases are the contract: they are matched, so they change only the way
a tool name changes, which is to say never.

Nothing else reaches a client. An unexpected failure inside a tool is
reported as a refusal with its type named, because a traceback over a
protocol is an internal detail escaping into somebody else's editor.
"""

import threading

# one tool call at a time, however many the SDK dispatches at once. The
# whole layer is built on the session pool, the ingest workers and the
# stdout redirect being touched by one call at a time, and the SDK runs
# every sync tool on a thread of its own - a client that pipelines two
# calls would otherwise race the pool, evict a session out from under a
# call still using it, and leave sys.stdout pointing at stderr for the
# life of the process. Serial is also what the module docstrings have
# promised all along; this lock is just that promise made true
SERIAL = threading.Lock()

# code -> the phrase every message under it begins with
PREFIXES = {
    "invalid_argument": "invalid argument:",
    "invalid_session": "invalid session id:",
    "not_found": "no such conversation:",
    "read_only": "read-only server:",
    "too_large": "too large:",
    "no_roster": "no roster:",
    "worker_failed": "worker failed:",
    "failed": "the call failed:",
}
CODES = tuple(PREFIXES)
# a text arg longer than this is refused rather than encoded. Generous on
# purpose: this is a guard against a client sending a whole disk, not a
# document-size policy
DEFAULT_MAX_CHARS = 400_000


class ToolError(ValueError):
    """A refusal a client can act on: a code, and a sentence saying why."""

    def __init__(self, code, message):
        self.code = code if code in PREFIXES else "failed"
        self.message = str(message)
        super().__init__(f"{PREFIXES[self.code]} {self.message}")


def guarded(fn, *args, **kwargs):
    """Run one tool body, and let nothing out of it untyped.

    The order matters: anything that already knows what it is passes
    through, the agent layer's own failures are about the roster rather
    than about the call, and everything left is either a bad argument or
    a fault, never a traceback.
    """
    from salt.agents.delegate import DelegationError
    from salt.agents.roster import RosterError
    from salt.mcp.pool import SessionError
    try:
        with SERIAL:
            return fn(*args, **kwargs)
    except ToolError:
        raise
    except RosterError as exc:
        raise ToolError("no_roster", exc) from exc
    except DelegationError as exc:
        raise ToolError("worker_failed", exc) from exc
    except SessionError as exc:
        raise ToolError("invalid_session", exc) from exc
    except Exception as exc:
        # a bare ValueError lands here too, on purpose: by this point
        # every argument has been validated by the helpers above, so a
        # ValueError out of the engine is the server's fault, and calling
        # it an invalid argument would send a client off to fix a call
        # that was fine
        raise ToolError("failed", f"{type(exc).__name__}: {exc}") from exc


def need_text(name, value, max_chars=DEFAULT_MAX_CHARS):
    """One text argument, present and within bounds. Returns it."""
    if not isinstance(value, str) or not value.strip():
        raise ToolError("invalid_argument", f"{name} needs a non-empty text")
    if max_chars and len(value) > max_chars:
        raise ToolError("too_large",
                        f"{name} is {len(value)} characters and this server "
                        f"accepts {max_chars} (--max-ingest-chars)")
    return value


def need_budget(value, default=None):
    """A share of the words to keep, over 0 and at most 1."""
    if value is None:
        return default
    try:
        budget = float(value)
    except (TypeError, ValueError):
        raise ToolError("invalid_argument",
                        f"budget_pct must be a number, got {value!r}") from None
    if not 0 < budget <= 1:
        raise ToolError("invalid_argument",
                        f"budget_pct must be over 0 and at most 1, got "
                        f"{value!r}")
    return budget
