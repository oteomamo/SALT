# -*- coding: utf-8 -*-
"""MCP server for SALT: compression and conversation memory as tools.

An MCP client (an editor, an agent runtime) speaks JSON-RPC over stdio to
``salt-mcp`` and reaches the same compression and session memory saltChat
uses. Installed with the optional extra: ``pip install salt[mcp]``.

Nothing is imported here. The server pulls in the SDK and the engine only
when it runs, so an install without the extra still imports this package,
and saltChat never loads a line of it.
"""
