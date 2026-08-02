# -*- coding: utf-8 -*-
"""Agent layer for saltChat: model roster, offload, orchestration.

Grows over the 2.10.z line. Nothing heavy is imported here - the package
must stay importable on any install, with or without a GPU."""

from salt.agents.roster import Roster, RosterEntry, RosterError, load_roster
from salt.agents.worker import WorkerError, WorkerHandle

__all__ = ["Roster", "RosterEntry", "RosterError", "WorkerError",
           "WorkerHandle", "load_roster"]
