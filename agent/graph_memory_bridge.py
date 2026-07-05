"""Bridge between Hermes agent loop and Jarvis v2 graph memory pipeline.

This module is the thin glue that connects the jarvis-core retrieval pipeline
to the Hermes run_agent.py loop. It:

1. Imports jarvis-core components (gracefully degrades if not installed)
2. Constructs a MessageContext from the agent's runtime metadata
3. Runs the RetrievalPipeline for each user turn
4. Returns a compact prompt block for injection into the system prompt

Design decisions:
- Per-turn retrieval: graph context is ephemeral, not session-cached, because
  different user messages need different retrieved context.
- Graceful degradation: if jarvis-core is not installed or the graph memory
  server is down, the bridge silently returns empty string.
- No mutation: the bridge never writes to the graph; it only reads.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy-loaded singleton to avoid import cost on non-Jarvis deployments.
_pipeline = None
_import_error: Optional[str] = None


def _load_pipeline():
    """Attempt to build the RetrievalPipeline from jarvis-core config.

    Returns None (and logs why) if anything is unavailable.
    """
    global _pipeline, _import_error

    if _pipeline is not None:
        return _pipeline
    if _import_error is not None:
        return None

    try:
        from jarvis_v2.channel_policy import ChannelPolicyResolver
        from jarvis_v2.config_loader import load_channel_policy
        from jarvis_v2.graph_adapter import GraphMemoryAdapter
        from jarvis_v2.retrieval_pipeline import RetrievalPipeline

        # Find config examples relative to jarvis-core package.
        # Editable installs place the .py under src/, so we walk up past src/.
        # Regular installs place them under the site-packages dir.
        import jarvis_v2
        pkg_file = Path(jarvis_v2.__file__).resolve()
        # Walk up from jarvis_v2/__init__.py to find config/examples/
        # Editable layout: jarvis-core/src/jarvis_v2/__init__.py -> jarvis-core/config/
        candidate_roots = [pkg_file.parent.parent, pkg_file.parent.parent.parent]
        config_dir = None
        for root in candidate_roots:
            candidate = root / "config" / "examples"
            if candidate.is_dir():
                config_dir = candidate
                break

        if not config_dir or not config_dir.is_dir():
            _import_error = f"config dir not found at {config_dir}"
            logger.warning("graph_memory_bridge: %s", _import_error)
            return None

        # Load channel policies
        policies = []
        for fname in ("channel-policy.discord.private.json",
                       "channel-policy.discord.json",
                       "channel-policy.discord.public.json"):
            fpath = config_dir / fname
            if fpath.is_file():
                try:
                    policies.append(load_channel_policy(fpath))
                except Exception as exc:
                    logger.warning("graph_memory_bridge: failed to load %s: %s", fname, exc)

        if not policies:
            _import_error = "no channel policies loaded"
            logger.warning("graph_memory_bridge: %s", _import_error)
            return None

        resolver = ChannelPolicyResolver(policies)
        graph_adapter = GraphMemoryAdapter(timeout_s=5.0)

        _pipeline = RetrievalPipeline(
            resolver=resolver,
            graph_adapter=graph_adapter,
        )
        logger.info(
            "graph_memory_bridge: pipeline initialized with %d channel policies",
            len(policies),
        )
        return _pipeline

    except ImportError as exc:
        _import_error = f"jarvis-core not installed: {exc}"
        logger.debug("graph_memory_bridge: %s", _import_error)
        return None
    except Exception as exc:
        _import_error = f"init failed: {exc}"
        logger.warning("graph_memory_bridge: %s", _import_error)
        return None


def build_graph_context(
    *,
    user_message: str,
    platform: str = "cli",
    channel_id: str = "",
    channel_name: str = "",
    channel_type: str = "dm",
    user_id: str = "",
    user_name: str = "",
    session_id: str = "",
    thread_id: str = "",
) -> str:
    """Run the retrieval pipeline and return a compact prompt block.

    Returns empty string if the pipeline is unavailable, the graph is down,
    or no relevant context was found. This is safe to call every turn.
    """
    pipeline = _load_pipeline()
    if pipeline is None:
        return ""

    if not user_message or not user_message.strip():
        return ""

    try:
        from jarvis_v2.models import MessageContext
        from jarvis_v2.prompt_render import render_prompt_context

        msg_ctx = MessageContext(
            text=user_message,
            platform=platform or "cli",
            channel_id=channel_id or "",
            channel_name=channel_name or "",
            channel_type=channel_type or "dm",
            user_id=user_id or "",
            user_name=user_name or "",
            session_id=session_id or None,
            thread_id=thread_id or None,
        )

        trace = pipeline.run(msg_ctx)

        if trace.graph_error:
            logger.debug(
                "graph_memory_bridge: graph query error for turn: %s",
                trace.graph_error,
            )

        if not trace.prompt_context.retrieved and not trace.prompt_context.always_on:
            return ""

        rendered = render_prompt_context(trace.prompt_context)
        if not rendered:
            return ""

        logger.info(
            "graph_memory_bridge: injected %d chars (%d nodes, %d omitted) "
            "intent=%s trust=%s",
            trace.prompt_context.chars_used,
            len(trace.prompt_context.retrieved) + len(trace.prompt_context.always_on),
            len(trace.prompt_context.omitted_reasons),
            trace.analysis.primary_intent,
            trace.analysis.channel_trust,
        )
        return rendered

    except Exception as exc:
        logger.warning("graph_memory_bridge: retrieval failed: %s", exc)
        return ""


def health_check() -> dict:
    """Check graph memory bridge health. Returns status dict."""
    result = {"bridge": "unavailable", "graph": "unknown", "error": None}

    pipeline = _load_pipeline()
    if pipeline is None:
        result["error"] = _import_error or "pipeline not initialized"
        return result

    result["bridge"] = "ok"

    # Check graph server health
    try:
        import urllib.request
        import json
        r = urllib.request.urlopen("http://127.0.0.1:7476/health", timeout=3)
        data = json.loads(r.read())
        result["graph"] = "ok" if data.get("status") == "ok" else "degraded"
        result["neo4j"] = data.get("neo4j", False)
        result["uptime_s"] = data.get("uptime_s", 0)
    except Exception as exc:
        result["graph"] = "unreachable"
        result["error"] = str(exc)

    return result
