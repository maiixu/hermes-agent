"""Cognitive memory plugin — MemoryProvider interface.

Semantic memory with vector embeddings (via litellm), auto-classification,
contradiction detection, importance decay, and time-based forgetting.
Local SQLite storage with binary-packed float32 embeddings.

Original PR #727 by 0xbyt4, adapted to MemoryProvider ABC.

Requires: litellm (for embeddings via any provider — OpenAI, Cohere, etc.)
Config via environment: uses litellm's standard env vars (OPENAI_API_KEY, etc.)
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import sqlite3
import struct
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)

from hermes_constants import get_hermes_home

_DB_DIR = get_hermes_home() / "cognitive_memory"
_EMBEDDING_DIM = 1536  # text-embedding-3-small default
_SIMILARITY_DEDUP_THRESHOLD = 0.95


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------

def _get_embedding(text: str) -> Optional[List[float]]:
    """Get embedding via litellm."""
    try:
        import litellm
        resp = litellm.embedding(model="text-embedding-3-small", input=[text])
        return resp.data[0]["embedding"]
    except Exception as e:
        logger.debug("Embedding failed: %s", e)
        return None


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _pack_embedding(emb: List[float]) -> bytes:
    return struct.pack(f"{len(emb)}f", *emb)


def _unpack_embedding(data: bytes) -> List[float]:
    n = len(data) // 4
    return list(struct.unpack(f"{n}f", data))


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

_CATEGORY_PATTERNS = {
    "preference": [r"\b(?:prefer|like|love|hate|dislike|favorite)\b"],
    "correction": [r"\b(?:actually|no,|wrong|incorrect|not right)\b"],
    "fact": [r"\b(?:is|are|was|were|has|have)\b"],
    "procedure": [r"\b(?:first|then|step|always|never|usually)\b"],
    "environment": [r"\b(?:running|using|installed|version|os|platform)\b"],
}


def _classify(content: str) -> str:
    content_lower = content.lower()
    for category, patterns in _CATEGORY_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, content_lower):
                return category
    return "general"


def _estimate_importance(content: str, category: str) -> float:
    base = {"correction": 0.9, "preference": 0.7, "procedure": 0.6}.get(category, 0.5)
    # Longer content slightly more important
    length_bonus = min(len(content) / 500, 0.2)
    return min(base + length_bonus, 1.0)


# ---------------------------------------------------------------------------
# Tool schema
# ---------------------------------------------------------------------------

COGNITIVE_RECALL_SCHEMA = {
    "name": "cognitive_recall",
    "description": (
        "Semantic memory with automatic classification and importance scoring. "
        "Actions: recall (search by meaning), store (add a fact), "
        "forget (remove by ID), status (memory stats)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["recall", "store", "forget", "status"],
                "description": "Action to perform.",
            },
            "query": {"type": "string", "description": "Search query (for 'recall')."},
            "content": {"type": "string", "description": "Fact to store (for 'store')."},
            "category": {
                "type": "string",
                "enum": ["preference", "fact", "procedure", "environment", "correction", "general"],
                "description": "Category (auto-detected if omitted).",
            },
            "memory_id": {"type": "integer", "description": "Memory ID (for 'forget')."},
        },
        "required": ["action"],
    },
}


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class CognitiveMemoryProvider(MemoryProvider):
    """Semantic memory with embeddings, classification, and forgetting."""

    def __init__(self):
        self._conn = None
        self._decay_half_life = 30  # days
        self._last_decay = 0

    @property
    def name(self) -> str:
        return "cognitive"

    def get_config_schema(self):
        return [
            {"key": "embedding_model", "description": "Embedding model (litellm format)", "default": "text-embedding-3-small"},
            {"key": "decay_half_life", "description": "Importance decay half-life in days (0=disabled)", "default": "30"},
        ]

    def is_available(self) -> bool:
        try:
            import litellm  # noqa: F401
            return True
        except ImportError:
            return False

    def initialize(self, session_id: str, **kwargs) -> None:
        _DB_DIR.mkdir(parents=True, exist_ok=True)
        db_path = _DB_DIR / "cognitive.db"
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                category TEXT DEFAULT 'general',
                importance REAL DEFAULT 0.5,
                embedding BLOB,
                retrieval_count INTEGER DEFAULT 0,
                helpful_count INTEGER DEFAULT 0,
                created_at REAL,
                updated_at REAL,
                deleted INTEGER DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_mem_importance ON memories(importance DESC);
            CREATE INDEX IF NOT EXISTS idx_mem_category ON memories(category);
        """)
        self._conn.commit()

    def system_prompt_block(self) -> str:
        if not self._conn:
            return ""
        try:
            count = self._conn.execute(
                "SELECT COUNT(*) FROM memories WHERE deleted = 0"
            ).fetchone()[0]
        except Exception:
            count = 0
        if count == 0:
            return ""
        return (
            f"# Cognitive Memory\n"
            f"Active. {count} memories with semantic recall and importance scoring.\n"
            f"Use cognitive_recall to search, store facts, or check status.\n"
            f"Memories decay over time — frequently used facts persist, unused ones fade."
        )

    def prefetch(self, query: str) -> str:
        if not self._conn or not query:
            return ""
        emb = _get_embedding(query)
        if not emb:
            return ""
        try:
            rows = self._conn.execute(
                "SELECT id, content, importance, embedding FROM memories "
                "WHERE deleted = 0 AND embedding IS NOT NULL "
                "ORDER BY importance DESC LIMIT 50"
            ).fetchall()
            scored = []
            now = time.time()
            for row in rows:
                mem_emb = _unpack_embedding(row[3])
                sim = _cosine_similarity(emb, mem_emb)
                importance = row[2]
                score = 0.5 * sim + 0.3 * importance + 0.2 * max(0, 1 - (now - (row[0] * 86400)) / (30 * 86400))
                if sim > 0.3:
                    scored.append((score, row[1]))
            scored.sort(reverse=True)
            if not scored:
                return ""
            lines = [f"- {content}" for _, content in scored[:5]]
            return "## Cognitive Memory\n" + "\n".join(lines)
        except Exception as e:
            logger.debug("Cognitive prefetch failed: %s", e)
            return ""

    def sync_turn(self, user_content: str, assistant_content: str) -> None:
        # Run decay cycle periodically
        self._maybe_decay()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [COGNITIVE_RECALL_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        if tool_name != "cognitive_recall":
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

        action = args.get("action", "")

        if action == "store":
            return self._store(args)
        elif action == "recall":
            return self._recall(args)
        elif action == "forget":
            return self._forget(args)
        elif action == "status":
            return self._status()
        return json.dumps({"error": f"Unknown action: {action}"})

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        if action == "add" and self._conn and content:
            category = "preference" if target == "user" else _classify(content)
            importance = _estimate_importance(content, category)
            emb = _get_embedding(content)
            now = time.time()
            self._conn.execute(
                "INSERT INTO memories (content, category, importance, embedding, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (content, category, importance, _pack_embedding(emb) if emb else None, now, now),
            )
            self._conn.commit()

    def shutdown(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # -- Internal methods ----------------------------------------------------

    def _store(self, args: dict) -> str:
        content = args.get("content", "")
        if not content:
            return json.dumps({"error": "content is required"})

        category = args.get("category") or _classify(content)
        importance = _estimate_importance(content, category)
        emb = _get_embedding(content)

        # Dedup check
        if emb:
            rows = self._conn.execute(
                "SELECT id, embedding FROM memories WHERE deleted = 0 AND embedding IS NOT NULL"
            ).fetchall()
            for row in rows:
                existing_emb = _unpack_embedding(row[1])
                if _cosine_similarity(emb, existing_emb) > _SIMILARITY_DEDUP_THRESHOLD:
                    return json.dumps({"error": "Very similar memory already exists", "existing_id": row[0]})

        now = time.time()
        cur = self._conn.execute(
            "INSERT INTO memories (content, category, importance, embedding, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (content, category, importance, _pack_embedding(emb) if emb else None, now, now),
        )
        self._conn.commit()
        return json.dumps({"id": cur.lastrowid, "category": category, "importance": round(importance, 2)})

    def _recall(self, args: dict) -> str:
        query = args.get("query", "")
        if not query:
            return json.dumps({"error": "query is required"})

        emb = _get_embedding(query)
        if not emb:
            return json.dumps({"error": "Embedding generation failed"})

        rows = self._conn.execute(
            "SELECT id, content, category, importance, embedding, created_at FROM memories "
            "WHERE deleted = 0 AND embedding IS NOT NULL "
            "ORDER BY importance DESC LIMIT 50"
        ).fetchall()

        now = time.time()
        results = []
        for row in rows:
            mem_emb = _unpack_embedding(row[4])
            sim = _cosine_similarity(emb, mem_emb)
            days_old = (now - (row[5] or now)) / 86400
            recency = max(0, 1 - days_old / 90)
            score = 0.5 * sim + 0.3 * row[3] + 0.2 * recency
            if sim > 0.2:
                results.append({
                    "id": row[0], "content": row[1], "category": row[2],
                    "score": round(score, 3), "similarity": round(sim, 3),
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        # Bump retrieval counts
        for r in results[:10]:
            self._conn.execute(
                "UPDATE memories SET retrieval_count = retrieval_count + 1 WHERE id = ?",
                (r["id"],),
            )
        self._conn.commit()
        return json.dumps({"results": results[:10], "count": len(results[:10])})

    def _forget(self, args: dict) -> str:
        memory_id = args.get("memory_id")
        if memory_id is None:
            return json.dumps({"error": "memory_id is required"})
        self._conn.execute("UPDATE memories SET deleted = 1 WHERE id = ?", (int(memory_id),))
        self._conn.commit()
        return json.dumps({"forgotten": True, "id": memory_id})

    def _status(self) -> str:
        total = self._conn.execute("SELECT COUNT(*) FROM memories WHERE deleted = 0").fetchone()[0]
        by_cat = self._conn.execute(
            "SELECT category, COUNT(*) FROM memories WHERE deleted = 0 GROUP BY category"
        ).fetchall()
        return json.dumps({
            "total": total,
            "by_category": {row[0]: row[1] for row in by_cat},
            "decay_half_life_days": self._decay_half_life,
        })

    def _maybe_decay(self) -> None:
        """Run importance decay every ~1 hour."""
        now = time.time()
        if now - self._last_decay < 3600:
            return
        self._last_decay = now
        if not self._conn or self._decay_half_life <= 0:
            return
        try:
            factor = 0.5 ** (1.0 / self._decay_half_life)
            self._conn.execute(
                "UPDATE memories SET importance = importance * ? WHERE deleted = 0",
                (factor,),
            )
            # Prune very low importance
            self._conn.execute(
                "UPDATE memories SET deleted = 1 WHERE deleted = 0 AND importance < 0.05"
            )
            self._conn.commit()
        except Exception as e:
            logger.debug("Cognitive decay failed: %s", e)


def register(ctx) -> None:
    """Register cognitive memory as a memory provider plugin."""
    ctx.register_memory_provider(CognitiveMemoryProvider())
