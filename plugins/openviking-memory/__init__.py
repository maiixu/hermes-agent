"""OpenViking memory plugin — MemoryProvider interface.

Read-only semantic search over a self-hosted OpenViking knowledge server.
Supports search (fast/deep/auto), URI-based content reading, and
filesystem-style browsing.

Original PR #3369 by Mibayy, adapted to MemoryProvider ABC.

Config via environment variables (profile-scoped via each profile's .env):
  OPENVIKING_ENDPOINT  — Server URL (default: http://127.0.0.1:1933)
  OPENVIKING_API_KEY   — Optional API key

Profile isolation: read-only with no local storage. Different profiles
can point to different servers via their own .env file.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

SEARCH_SCHEMA = {
    "name": "viking_search",
    "description": (
        "Semantic search over OpenViking knowledge base. "
        "Returns ranked results with URIs for deeper reading."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query."},
            "mode": {
                "type": "string", "enum": ["auto", "fast", "deep"],
                "description": "Search depth (default: auto).",
            },
            "scope": {"type": "string", "description": "URI prefix to scope search."},
            "limit": {"type": "integer", "description": "Max results (default: 10)."},
        },
        "required": ["query"],
    },
}

READ_SCHEMA = {
    "name": "viking_read",
    "description": (
        "Read content at a viking:// URI. Supports three detail levels: "
        "abstract (summary), overview (key points), read (full content)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "uri": {"type": "string", "description": "viking:// URI to read."},
            "level": {
                "type": "string", "enum": ["abstract", "overview", "read"],
                "description": "Detail level (default: overview).",
            },
        },
        "required": ["uri"],
    },
}

BROWSE_SCHEMA = {
    "name": "viking_browse",
    "description": (
        "Browse the OpenViking knowledge store like a filesystem. "
        "Supports tree (hierarchy), list (directory), and stat (metadata)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string", "enum": ["tree", "list", "stat"],
                "description": "Browse action.",
            },
            "path": {"type": "string", "description": "Path to browse (default: root)."},
        },
        "required": ["action"],
    },
}


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class OpenVikingMemoryProvider(MemoryProvider):
    """Read-only memory via OpenViking self-hosted knowledge server."""

    def __init__(self):
        self._endpoint = ""
        self._api_key = ""

    @property
    def name(self) -> str:
        return "openviking"

    def get_config_schema(self):
        return [
            {"key": "endpoint", "description": "OpenViking server URL", "required": True, "default": "http://127.0.0.1:1933"},
            {"key": "api_key", "description": "OpenViking API key (if server requires auth)", "secret": True, "env_var": "OPENVIKING_API_KEY"},
        ]

    def is_available(self) -> bool:
        endpoint = os.environ.get("OPENVIKING_ENDPOINT", "")
        if not endpoint:
            return False
        # Quick health check
        try:
            import httpx
            resp = httpx.get(f"{endpoint}/health", timeout=3.0)
            return resp.status_code == 200
        except Exception:
            return False

    def initialize(self, session_id: str, **kwargs) -> None:
        self._endpoint = os.environ.get("OPENVIKING_ENDPOINT", "http://127.0.0.1:1933")
        self._api_key = os.environ.get("OPENVIKING_API_KEY", "")

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self._api_key:
            h["X-API-Key"] = self._api_key
        return h

    def system_prompt_block(self) -> str:
        return (
            "# OpenViking Knowledge Base\n"
            f"Active. Endpoint: {self._endpoint}\n"
            "Use viking_search to find information, viking_read for details, "
            "viking_browse to explore the knowledge tree."
        )

    def prefetch(self, query: str) -> str:
        """OpenViking is tool-driven, no automatic prefetch."""
        return ""

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [SEARCH_SCHEMA, READ_SCHEMA, BROWSE_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        try:
            import httpx
        except ImportError:
            return json.dumps({"error": "httpx not installed"})

        try:
            if tool_name == "viking_search":
                return self._search(httpx, args)
            elif tool_name == "viking_read":
                return self._read(httpx, args)
            elif tool_name == "viking_browse":
                return self._browse(httpx, args)
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def _search(self, httpx, args: dict) -> str:
        query = args.get("query", "")
        if not query:
            return json.dumps({"error": "query is required"})
        payload = {"query": query, "mode": args.get("mode", "auto")}
        if args.get("scope"):
            payload["scope"] = args["scope"]
        if args.get("limit"):
            payload["limit"] = args["limit"]
        resp = httpx.post(
            f"{self._endpoint}/v1/search",
            json=payload, headers=self._headers(), timeout=30.0,
        )
        return resp.text

    def _read(self, httpx, args: dict) -> str:
        uri = args.get("uri", "")
        if not uri:
            return json.dumps({"error": "uri is required"})
        level = args.get("level", "overview")
        resp = httpx.post(
            f"{self._endpoint}/v1/read",
            json={"uri": uri, "level": level},
            headers=self._headers(), timeout=30.0,
        )
        return resp.text

    def _browse(self, httpx, args: dict) -> str:
        action = args.get("action", "tree")
        path = args.get("path", "/")
        resp = httpx.post(
            f"{self._endpoint}/v1/browse",
            json={"action": action, "path": path},
            headers=self._headers(), timeout=30.0,
        )
        return resp.text


def register(ctx) -> None:
    """Register OpenViking as a memory provider plugin."""
    ctx.register_memory_provider(OpenVikingMemoryProvider())
