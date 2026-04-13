"""Generic webhook platform adapter.

Runs an aiohttp HTTP server that receives webhook POSTs from external
services (GitHub, GitLab, JIRA, Stripe, etc.), validates HMAC signatures,
transforms payloads into agent prompts, and routes responses back to the
source or to another configured platform.

Configuration lives in config.yaml under platforms.webhook.extra.routes.
Each route defines:
  - events: which event types to accept (header-based filtering)
  - secret: HMAC secret for signature validation (REQUIRED)
  - prompt: template string formatted with the webhook payload
  - skills: optional list of skills to load for the agent
  - deliver: where to send the response (github_comment, telegram, etc.)
  - deliver_extra: additional delivery config (repo, pr_number, chat_id)

Security:
  - HMAC secret is required per route (validated at startup)
  - Rate limiting per route (fixed-window, configurable)
  - Idempotency cache prevents duplicate agent runs on webhook retries
  - Body size limits checked before reading payload
  - Set secret to "INSECURE_NO_AUTH" to skip validation (testing only)
"""

import asyncio
import hashlib
import hmac
import json
import logging
import os
import re
import subprocess
import shlex
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from aiohttp import web

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    web = None  # type: ignore[assignment]

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)

logger = logging.getLogger(__name__)

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8644
_INSECURE_NO_AUTH = "INSECURE_NO_AUTH"
_DYNAMIC_ROUTES_FILENAME = "webhook_subscriptions.json"


def check_webhook_requirements() -> bool:
    """Check if webhook adapter dependencies are available."""
    return AIOHTTP_AVAILABLE


class WebhookAdapter(BasePlatformAdapter):
    """Generic webhook receiver that triggers agent runs from HTTP POSTs."""

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.WEBHOOK)
        self._host: str = config.extra.get("host", DEFAULT_HOST)
        self._port: int = int(config.extra.get("port", DEFAULT_PORT))
        self._global_secret: str = self._resolve_secret(config.extra.get("secret", ""), config.extra.get("secret_env", ""))
        self._static_routes: Dict[str, dict] = config.extra.get("routes", {})
        self._dynamic_routes: Dict[str, dict] = {}
        self._dynamic_routes_mtime: float = 0.0
        self._routes: Dict[str, dict] = dict(self._static_routes)
        self._runner = None

        # Delivery info keyed by session chat_id.
        #
        # Read by every send() invocation for the chat_id (status messages
        # AND the final response).  Cleaned up via TTL on each POST so the
        # dict stays bounded — see _prune_delivery_info().  Do NOT pop on
        # send(), or interim status messages (e.g. fallback notifications,
        # context-pressure warnings) will consume the entry before the
        # final response arrives, causing the response to silently fall
        # back to the "log" deliver type.
        self._delivery_info: Dict[str, dict] = {}
        self._delivery_info_created: Dict[str, float] = {}

        # Reference to gateway runner for cross-platform delivery (set externally)
        self.gateway_runner = None

        # Idempotency: TTL cache of recently processed delivery IDs.
        # Prevents duplicate agent runs when webhook providers retry.
        self._seen_deliveries: Dict[str, float] = {}
        self._idempotency_ttl: int = 3600  # 1 hour

        # Rate limiting: per-route timestamps in a fixed window.
        self._rate_counts: Dict[str, List[float]] = {}
        self._rate_limit: int = int(config.extra.get("rate_limit", 30))  # per minute

        # Body size limit (auth-before-body pattern)
        self._max_body_bytes: int = int(
            config.extra.get("max_body_bytes", 1_048_576)
        )  # 1MB

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_secret(secret: str, secret_env: str) -> str:
        """Return secret value, falling back to an env var.

        If ``secret`` is non-empty, use it directly.
        If ``secret_env`` names an env var, read it from the environment.
        """
        if secret:
            return secret
        if secret_env:
            return os.environ.get(secret_env, "")
        return ""

    async def connect(self) -> bool:
        # Load agent-created subscriptions before validating
        self._reload_dynamic_routes()

        # Validate routes at startup — secret is required per route
        for name, route in self._routes.items():
            secret = self._resolve_secret(route.get("secret", ""), route.get("secret_env", "")) or self._global_secret
            if not secret:
                raise ValueError(
                    f"[webhook] Route '{name}' has no HMAC secret. "
                    f"Set 'secret' on the route or globally. "
                    f"For testing without auth, set secret to '{_INSECURE_NO_AUTH}'."
                )

        app = web.Application()
        app.router.add_get("/health", self._handle_health)
        app.router.add_post("/webhooks/{route_name}", self._handle_webhook)

        # Port conflict detection — fail fast if port is already in use
        import socket as _socket
        try:
            with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as _s:
                _s.settimeout(1)
                _s.connect(('127.0.0.1', self._port))
            logger.error('[webhook] Port %d already in use. Set a different port in config.yaml: platforms.webhook.port', self._port)
            return False
        except (ConnectionRefusedError, OSError):
            pass  # port is free

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self._host, self._port)
        await site.start()
        self._mark_connected()

        route_names = ", ".join(self._routes.keys()) or "(none configured)"
        logger.info(
            "[webhook] Listening on %s:%d — routes: %s",
            self._host,
            self._port,
            route_names,
        )
        return True

    async def disconnect(self) -> None:
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
        self._mark_disconnected()
        logger.info("[webhook] Disconnected")

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Deliver the agent's response to the configured destination.

        chat_id is ``webhook:{route}:{delivery_id}``.  The delivery info
        stored during webhook receipt is read with ``.get()`` (not popped)
        so that interim status messages emitted before the final response
        — fallback-model notifications, context-pressure warnings, etc. —
        do not consume the entry and silently downgrade the final response
        to the ``log`` deliver type.  TTL cleanup happens on POST.
        """
        delivery = self._delivery_info.get(chat_id, {})
        deliver_type = delivery.get("deliver", "log")

        if deliver_type == "log":
            logger.info("[webhook] Response for %s: %s", chat_id, content[:200])
            return SendResult(success=True)

        if deliver_type == "github_comment":
            return await self._deliver_github_comment(content, delivery)

        if deliver_type == "github_action":
            return await self._deliver_github_action(content, delivery)

        # Cross-platform delivery — any platform with a gateway adapter
        if self.gateway_runner and deliver_type in (
            "telegram",
            "discord",
            "slack",
            "signal",
            "sms",
            "whatsapp",
            "matrix",
            "mattermost",
            "homeassistant",
            "email",
            "dingtalk",
            "feishu",
            "wecom",
            "wecom_callback",
            "weixin",
            "bluebubbles",
        ):
            return await self._deliver_cross_platform(
                deliver_type, content, delivery
            )

        logger.warning("[webhook] Unknown deliver type: %s", deliver_type)
        return SendResult(
            success=False, error=f"Unknown deliver type: {deliver_type}"
        )

    def _prune_delivery_info(self, now: float) -> None:
        """Drop delivery_info entries older than the idempotency TTL.

        Mirrors the cleanup pattern used for ``_seen_deliveries``.  Called
        on each POST so the dict size is bounded by ``rate_limit * TTL``
        even if many webhooks fire and never receive a final response.
        """
        cutoff = now - self._idempotency_ttl
        stale = [
            k
            for k, t in self._delivery_info_created.items()
            if t < cutoff
        ]
        for k in stale:
            self._delivery_info.pop(k, None)
            self._delivery_info_created.pop(k, None)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": chat_id, "type": "webhook"}

    # ------------------------------------------------------------------
    # HTTP handlers
    # ------------------------------------------------------------------

    async def _handle_health(self, request: "web.Request") -> "web.Response":
        """GET /health — simple health check."""
        return web.json_response({"status": "ok", "platform": "webhook"})

    def _reload_dynamic_routes(self) -> None:
        """Reload agent-created subscriptions from disk if the file changed."""
        from hermes_constants import get_hermes_home
        hermes_home = get_hermes_home()
        subs_path = hermes_home / _DYNAMIC_ROUTES_FILENAME
        if not subs_path.exists():
            if self._dynamic_routes:
                self._dynamic_routes = {}
                self._routes = dict(self._static_routes)
                logger.debug("[webhook] Dynamic subscriptions file removed, cleared dynamic routes")
            return
        try:
            mtime = subs_path.stat().st_mtime
            if mtime <= self._dynamic_routes_mtime:
                return  # No change
            data = json.loads(subs_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return
            # Merge: static routes take precedence over dynamic ones
            self._dynamic_routes = {
                k: v for k, v in data.items()
                if k not in self._static_routes
            }
            self._routes = {**self._dynamic_routes, **self._static_routes}
            self._dynamic_routes_mtime = mtime
            logger.info(
                "[webhook] Reloaded %d dynamic route(s): %s",
                len(self._dynamic_routes),
                ", ".join(self._dynamic_routes.keys()) or "(none)",
            )
        except Exception as e:
            logger.error("[webhook] Failed to reload dynamic routes: %s", e)

    async def _handle_webhook(self, request: "web.Request") -> "web.Response":
        """POST /webhooks/{route_name} — receive and process a webhook event."""
        # Hot-reload dynamic subscriptions on each request (mtime-gated, cheap)
        self._reload_dynamic_routes()

        route_name = request.match_info.get("route_name", "")
        route_config = self._routes.get(route_name)

        if not route_config:
            return web.json_response(
                {"error": f"Unknown route: {route_name}"}, status=404
            )

        # ── Auth-before-body ─────────────────────────────────────
        # Check Content-Length before reading the full payload.
        content_length = request.content_length or 0
        if content_length > self._max_body_bytes:
            return web.json_response(
                {"error": "Payload too large"}, status=413
            )

        # ── Rate limiting ────────────────────────────────────────
        now = time.time()
        window = self._rate_counts.setdefault(route_name, [])
        window[:] = [t for t in window if now - t < 60]
        if len(window) >= self._rate_limit:
            return web.json_response(
                {"error": "Rate limit exceeded"}, status=429
            )
        window.append(now)

        # Read body
        try:
            raw_body = await request.read()
        except Exception as e:
            logger.error("[webhook] Failed to read body: %s", e)
            return web.json_response({"error": "Bad request"}, status=400)

        # Validate HMAC signature (skip for INSECURE_NO_AUTH testing mode)
        secret = self._resolve_secret(route_config.get("secret", ""), route_config.get("secret_env", "")) or self._global_secret
        if secret and secret != _INSECURE_NO_AUTH:
            if not self._validate_signature(request, raw_body, secret):
                logger.warning(
                    "[webhook] Invalid signature for route %s", route_name
                )
                return web.json_response(
                    {"error": "Invalid signature"}, status=401
                )

        # Parse payload
        try:
            payload = json.loads(raw_body)
        except json.JSONDecodeError:
            # Try form-encoded as fallback
            try:
                import urllib.parse

                payload = dict(
                    urllib.parse.parse_qsl(raw_body.decode("utf-8"))
                )
            except Exception:
                return web.json_response(
                    {"error": "Cannot parse body"}, status=400
                )

        # Check event type filter
        event_type = (
            request.headers.get("X-GitHub-Event", "")
            or request.headers.get("X-GitLab-Event", "")
            or payload.get("event_type", "")
            or "unknown"
        )
        allowed_events = route_config.get("events", [])
        logger.info("[webhook] received event_type=%s route=%s", event_type, route_name)
        if allowed_events and event_type not in allowed_events:
            logger.info(
                "[webhook] Ignoring event %s for route %s (allowed: %s)",
                event_type,
                route_name,
                allowed_events,
            )
            return web.json_response(
                {"status": "ignored", "event": event_type}
            )

        # Check sender allowlist — drop events from unauthorized senders
        # before running the agent (security + cost control).
        sender_allowlist = route_config.get("sender_allowlist", [])
        if sender_allowlist:
            sender_login = (
                payload.get("sender", {}).get("login", "")
                if isinstance(payload.get("sender"), dict)
                else ""
            )
            if sender_login not in sender_allowlist:
                logger.info(
                    "[webhook] Ignoring event from sender '%s' (not in allowlist) route=%s",
                    sender_login,
                    route_name,
                )
                return web.json_response(
                    {"status": "ignored", "reason": "sender_not_allowed"}
                )

        # Format prompt from template
        # Inject event_type/route into payload so {event_type} substitutes correctly
        payload = dict(payload)
        payload.setdefault("event_type", event_type)
        payload.setdefault("route_name", route_name)
        prompt_template = route_config.get("prompt", "")
        prompt = self._render_prompt(
            prompt_template, payload, event_type, route_name
        )

        # Inject skill content if configured.
        # We call build_skill_invocation_message() directly rather than
        # using /skill-name slash commands — the gateway's command parser
        # would intercept those and break the flow.
        skills = route_config.get("skills", [])
        if skills:
            try:
                from agent.skill_commands import (
                    build_skill_invocation_message,
                    get_skill_commands,
                )

                skill_cmds = get_skill_commands()
                for skill_name in skills:
                    cmd_key = f"/{skill_name}"
                    if cmd_key in skill_cmds:
                        skill_content = build_skill_invocation_message(
                            cmd_key, user_instruction=prompt
                        )
                        if skill_content:
                            prompt = skill_content
                            break  # Load the first matching skill
                    else:
                        logger.warning(
                            "[webhook] Skill '%s' not found", skill_name
                        )
            except Exception as e:
                logger.warning("[webhook] Skill loading failed: %s", e)

        # Build a unique delivery ID
        delivery_id = request.headers.get(
            "X-GitHub-Delivery",
            request.headers.get("X-Request-ID", str(int(time.time() * 1000))),
        )

        # ── Idempotency ─────────────────────────────────────────
        # Skip duplicate deliveries (webhook retries).
        now = time.time()
        # Prune expired entries
        self._seen_deliveries = {
            k: v
            for k, v in self._seen_deliveries.items()
            if now - v < self._idempotency_ttl
        }
        if delivery_id in self._seen_deliveries:
            logger.info(
                "[webhook] Skipping duplicate delivery %s", delivery_id
            )
            return web.json_response(
                {"status": "duplicate", "delivery_id": delivery_id},
                status=200,
            )
        self._seen_deliveries[delivery_id] = now

        # Use delivery_id in session key so concurrent webhooks on the
        # same route get independent agent runs (not queued/interrupted).
        session_chat_id = f"webhook:{route_name}:{delivery_id}"

        # Store delivery info for send().  Read by every send() invocation
        # for this chat_id (interim status messages and the final response),
        # so we do NOT pop on send.  TTL-based cleanup keeps the dict bounded.
        deliver_config = {
            "deliver": route_config.get("deliver", "log"),
            "deliver_extra": self._render_delivery_extra(
                route_config.get("deliver_extra", {}), payload
            ),
            "payload": payload,
        }
        self._delivery_info[session_chat_id] = deliver_config
        self._delivery_info_created[session_chat_id] = now
        self._prune_delivery_info(now)

        # Build source and event
        source = self.build_source(
            chat_id=session_chat_id,
            chat_name=f"webhook/{route_name}",
            chat_type="webhook",
            user_id=f"webhook:{route_name}",
            user_name=route_name,
        )
        event = MessageEvent(
            text=prompt,
            message_type=MessageType.TEXT,
            source=source,
            raw_message=payload,
            message_id=delivery_id,
        )

        logger.info(
            "[webhook] %s event=%s route=%s prompt_len=%d delivery=%s",
            request.method,
            event_type,
            route_name,
            len(prompt),
            delivery_id,
        )

        # Non-blocking — return 202 Accepted immediately
        task = asyncio.create_task(self.handle_message(event))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

        return web.json_response(
            {
                "status": "accepted",
                "route": route_name,
                "event": event_type,
                "delivery_id": delivery_id,
            },
            status=202,
        )

    # ------------------------------------------------------------------
    # Signature validation
    # ------------------------------------------------------------------

    def _validate_signature(
        self, request: "web.Request", body: bytes, secret: str
    ) -> bool:
        """Validate webhook signature (GitHub, GitLab, generic HMAC-SHA256)."""
        # GitHub: X-Hub-Signature-256 = sha256=<hex>
        gh_sig = request.headers.get("X-Hub-Signature-256", "")
        if gh_sig:
            expected = "sha256=" + hmac.new(
                secret.encode(), body, hashlib.sha256
            ).hexdigest()
            return hmac.compare_digest(gh_sig, expected)

        # GitLab: X-Gitlab-Token = <plain secret>
        gl_token = request.headers.get("X-Gitlab-Token", "")
        if gl_token:
            return hmac.compare_digest(gl_token, secret)

        # Generic: X-Webhook-Signature = <hex HMAC-SHA256>
        generic_sig = request.headers.get("X-Webhook-Signature", "")
        if generic_sig:
            expected = hmac.new(
                secret.encode(), body, hashlib.sha256
            ).hexdigest()
            return hmac.compare_digest(generic_sig, expected)

        # No recognised signature header but secret is configured → reject
        logger.debug(
            "[webhook] Secret configured but no signature header found"
        )
        return False

    # ------------------------------------------------------------------
    # Prompt rendering
    # ------------------------------------------------------------------

    def _render_prompt(
        self,
        template: str,
        payload: dict,
        event_type: str,
        route_name: str,
    ) -> str:
        """Render a prompt template with the webhook payload.

        Supports dot-notation access into nested dicts:
        ``{pull_request.title}`` → ``payload["pull_request"]["title"]``

        Special token ``{__raw__}`` dumps the entire payload as indented
        JSON (truncated to 4000 chars).  Useful for monitoring alerts or
        any webhook where the agent needs to see the full payload.
        """
        if not template:
            truncated = json.dumps(payload, indent=2)[:4000]
            return (
                f"Webhook event '{event_type}' on route "
                f"'{route_name}':\n\n```json\n{truncated}\n```"
            )

        def _resolve(match: re.Match) -> str:
            key = match.group(1)
            # Special token: dump the entire payload as JSON
            if key == "__raw__":
                return json.dumps(payload, indent=2)[:4000]
            value: Any = payload
            for part in key.split("."):
                if isinstance(value, dict):
                    value = value.get(part, f"{{{key}}}")
                else:
                    return f"{{{key}}}"
            if isinstance(value, (dict, list)):
                return json.dumps(value, indent=2)[:2000]
            return str(value)

        return re.sub(r"\{([a-zA-Z0-9_.]+)\}", _resolve, template)

    def _render_delivery_extra(
        self, extra: dict, payload: dict
    ) -> dict:
        """Render delivery_extra template values with payload data."""
        rendered: Dict[str, Any] = {}
        for key, value in extra.items():
            if isinstance(value, str):
                rendered[key] = self._render_prompt(value, payload, "", "")
            else:
                rendered[key] = value
        return rendered

    # ------------------------------------------------------------------
    # Response delivery
    # ------------------------------------------------------------------

    async def _deliver_github_comment(
        self, content: str, delivery: dict
    ) -> SendResult:
        """Post agent response as a GitHub PR/issue comment via ``gh`` CLI."""
        extra = delivery.get("deliver_extra", {})
        repo = extra.get("repo", "")
        pr_number = extra.get("pr_number", "")

        if not repo or not pr_number:
            logger.error(
                "[webhook] github_comment delivery missing repo or pr_number"
            )
            return SendResult(
                success=False, error="Missing repo or pr_number"
            )

        try:
            result = subprocess.run(
                [
                    "gh",
                    "pr",
                    "comment",
                    str(pr_number),
                    "--repo",
                    repo,
                    "--body",
                    content,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                logger.info(
                    "[webhook] Posted comment on %s#%s", repo, pr_number
                )
                return SendResult(success=True)
            else:
                logger.error(
                    "[webhook] gh pr comment failed: %s", result.stderr
                )
                return SendResult(success=False, error=result.stderr)
        except FileNotFoundError:
            logger.error(
                "[webhook] 'gh' CLI not found — install GitHub CLI for "
                "github_comment delivery"
            )
            return SendResult(
                success=False, error="gh CLI not installed"
            )
        except Exception as e:
            logger.error("[webhook] github_comment delivery error: %s", e)
            return SendResult(success=False, error=str(e))

    async def _deliver_cross_platform(
        self, platform_name: str, content: str, delivery: dict
    ) -> SendResult:
        """Route response to another platform (telegram, discord, etc.)."""
        if not self.gateway_runner:
            return SendResult(
                success=False,
                error="No gateway runner for cross-platform delivery",
            )

        try:
            target_platform = Platform(platform_name)
        except ValueError:
            return SendResult(
                success=False, error=f"Unknown platform: {platform_name}"
            )

        adapter = self.gateway_runner.adapters.get(target_platform)
        if not adapter:
            return SendResult(
                success=False,
                error=f"Platform {platform_name} not connected",
            )

        # Use home channel if no specific chat_id in deliver_extra
        extra = delivery.get("deliver_extra", {})
        chat_id = extra.get("chat_id", "")
        if not chat_id:
            home = self.gateway_runner.config.get_home_channel(target_platform)
            if home:
                chat_id = home.chat_id
            else:
                return SendResult(
                    success=False,
                    error=f"No chat_id or home channel for {platform_name}",
                )

        # Pass thread_id from deliver_extra so Telegram forum topics work
        metadata = None
        thread_id = extra.get("message_thread_id") or extra.get("thread_id")
        if thread_id:
            metadata = {"thread_id": thread_id}

        return await adapter.send(chat_id, content, metadata=metadata)

    # ------------------------------------------------------------------
    # GitHub Action delivery
    # ------------------------------------------------------------------

    async def _deliver_github_action(
        self, content: str, delivery: dict
    ) -> SendResult:
        """Parse ACTION directive from agent response and execute it."""
        match = re.search(
            r"ACTION:\s*(\w+)((?:\s+\w+=\S+)+)", content
        )
        if not match:
            logger.info(
                "[webhook_action] No ACTION found in response — treating as ignored"
            )
            return SendResult(success=True)

        action_type = match.group(1)
        params_raw = match.group(2).strip()
        params = dict(re.findall(r"(\w+)=(\S+)", params_raw))
        repo = params.get("repo", "")
        logger.info(
            "[webhook_action] Parsed action=%s params=%s", action_type, params
        )

        try:
            if action_type == "merge_pr":
                pr_num = int(params.get("pr", 0))
                return await self._action_merge_pr(repo, pr_num)
            elif action_type == "address_comments":
                pr_num = int(params.get("pr", 0))
                return await self._action_address_comments(repo, pr_num, merge_after=False)
            elif action_type == "address_and_merge":
                pr_num = int(params.get("pr", 0))
                return await self._action_address_comments(repo, pr_num, merge_after=True)
            elif action_type == "reply_issue":
                issue_num = int(params.get("issue", 0))
                return await self._action_reply_issue(repo, issue_num)
            else:
                logger.warning("[webhook_action] Unknown action type: %s", action_type)
                return SendResult(success=False, error=f"Unknown action: {action_type}")
        except Exception as e:
            logger.error("[webhook_action] Action failed: %s", e)
            return SendResult(success=False, error=str(e))

    async def _fetch_pr_context(self, repo: str, pr_num: int) -> dict:
        """Fetch PR title, body, diff, review comments, and linked issue."""
        import shlex

        bot_env = "source ~/code/personal-intelligence/scripts/setup-bot-env.sh hermes 2>/dev/null"

        async def run_gh(cmd: str) -> subprocess.CompletedProcess:
            full = f"bash -c {shlex.quote(bot_env + ' && ' + cmd)}"
            return await asyncio.to_thread(
                subprocess.run,
                full,
                shell=True,
                capture_output=True,
                text=True,
                timeout=60,
            )

        # PR title, body, branch
        pr_result = await asyncio.to_thread(
            subprocess.run,
            ["bash", "-c",
             f"{bot_env} && gh pr view {pr_num} --repo {shlex.quote(repo)} "
             f"--json title,body,headRefName"],
            capture_output=True, text=True, timeout=60,
        )
        pr_data = {}
        if pr_result.returncode == 0:
            try:
                pr_data = json.loads(pr_result.stdout)
            except json.JSONDecodeError:
                pass

        # Diff
        diff_result = await asyncio.to_thread(
            subprocess.run,
            ["bash", "-c",
             f"{bot_env} && gh pr diff {pr_num} --repo {shlex.quote(repo)}"],
            capture_output=True, text=True, timeout=60,
        )
        diff_text = diff_result.stdout[:8000] if diff_result.returncode == 0 else ""

        # Review comments (inline)
        comments_result = await asyncio.to_thread(
            subprocess.run,
            ["bash", "-c",
             f"{bot_env} && gh api repos/{repo}/pulls/{pr_num}/comments"],
            capture_output=True, text=True, timeout=60,
        )
        comments = []
        if comments_result.returncode == 0:
            try:
                comments = json.loads(comments_result.stdout)
            except json.JSONDecodeError:
                pass

        # Reviews (review-level bodies)
        reviews_result = await asyncio.to_thread(
            subprocess.run,
            ["bash", "-c",
             f"{bot_env} && gh api repos/{repo}/pulls/{pr_num}/reviews"],
            capture_output=True, text=True, timeout=60,
        )
        reviews = []
        if reviews_result.returncode == 0:
            try:
                reviews = json.loads(reviews_result.stdout)
            except json.JSONDecodeError:
                pass

        # Linked issue from PR body
        linked_issue = None
        pr_body = pr_data.get("body", "") or ""
        issue_match = re.search(r"(?:Closes|Fixes|Resolves)\s+#(\d+)", pr_body, re.IGNORECASE)
        if issue_match:
            issue_num = issue_match.group(1)
            issue_result = await asyncio.to_thread(
                subprocess.run,
                ["bash", "-c",
                 f"{bot_env} && gh issue view {issue_num} --repo {shlex.quote(repo)} "
                 f"--json title,body"],
                capture_output=True, text=True, timeout=60,
            )
            if issue_result.returncode == 0:
                try:
                    linked_issue = json.loads(issue_result.stdout)
                    linked_issue["number"] = issue_num
                except json.JSONDecodeError:
                    pass

        return {
            "title": pr_data.get("title", ""),
            "body": pr_body,
            "branch": pr_data.get("headRefName", ""),
            "diff": diff_text,
            "comments": comments,
            "reviews": reviews,
            "linked_issue": linked_issue,
        }

    async def _spawn_cc(self, prompt: str, cwd: str, timeout: int = 1800) -> dict:
        """Spawn a Claude Code headless session and return parsed output."""
        import shlex

        cmd = (
            "source ~/code/personal-intelligence/scripts/setup-bot-env.sh claude && "
            "AWS_PROFILE=bedrock-claude AWS_REGION=us-west-2 CLAUDE_CODE_USE_BEDROCK=1 "
            f"/opt/homebrew/bin/claude -p {shlex.quote(prompt)} "
            "--output-format json --no-session-persistence --dangerously-skip-permissions"
        )
        result = await asyncio.to_thread(
            subprocess.run,
            ["bash", "-c", cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        logger.info(
            "[webhook_action] CC exit_code=%d stdout_len=%d",
            result.returncode,
            len(result.stdout),
        )

        output_text = ""
        blocked_reason = None

        if result.stdout:
            try:
                parsed = json.loads(result.stdout)
                output_text = parsed.get("result", result.stdout)
            except json.JSONDecodeError:
                output_text = result.stdout

        combined = output_text + "\n" + result.stderr
        blocked_match = re.search(r"BLOCKED:\s*(.+)", combined)
        if blocked_match:
            blocked_reason = blocked_match.group(1).strip()

        return {
            "output": output_text,
            "stderr": result.stderr,
            "exit_code": result.returncode,
            "blocked": blocked_reason,
        }

    async def _post_pr_comment(self, repo: str, pr_num: int, body: str) -> None:
        """Post a comment on a PR as hermes-bot."""
        import shlex

        cmd = (
            f"source ~/code/personal-intelligence/scripts/setup-bot-env.sh hermes && "
            f"gh pr comment {pr_num} --repo {shlex.quote(repo)} --body {shlex.quote(body)}"
        )
        result = await asyncio.to_thread(
            subprocess.run,
            ["bash", "-c", cmd],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            logger.error(
                "[webhook_action] Failed to post PR comment: %s", result.stderr
            )

    async def _action_merge_pr(self, repo: str, pr_num: int) -> SendResult:
        """Merge a PR using hermes-bot identity."""
        import shlex

        logger.info("[webhook_action] Merging PR %s#%d", repo, pr_num)
        cmd = (
            f"source ~/code/personal-intelligence/scripts/setup-bot-env.sh hermes && "
            f"gh pr merge {pr_num} --repo {shlex.quote(repo)} --merge --delete-branch"
        )
        result = await asyncio.to_thread(
            subprocess.run,
            ["bash", "-c", cmd],
            capture_output=True, text=True, timeout=60,
        )
        stdout = result.stdout or ""
        if result.returncode == 0:
            logger.info("[webhook_action] Merged PR %s#%d", repo, pr_num)
            await self._post_pr_comment(repo, pr_num, "✅ Merged and branch deleted.")
            return SendResult(success=True)

        stderr = result.stderr.lower()
        if any(kw in stderr.lower() or kw in stdout.lower() for kw in ("conflict", "not mergeable", "cannot be cleanly", "unmergeable", "merge_conflict")):
            logger.info(
                "[webhook_action] Conflict detected on %s#%d, attempting CC resolution",
                repo, pr_num,
            )
            resolved = await self._resolve_conflict_and_merge(repo, pr_num)
            if resolved:
                return SendResult(success=True)

        error_msg = result.stderr.strip()
        logger.error("[webhook_action] Merge failed for %s#%d: %s", repo, pr_num, error_msg)
        await self._post_pr_comment(repo, pr_num, f"❌ Merge failed: {error_msg}")
        return SendResult(success=False, error=error_msg)

    async def _resolve_conflict_and_merge(self, repo: str, pr_num: int) -> bool:
        """Spawn CC to resolve merge conflicts, then retry merge."""
        import shlex
        import tempfile

        ctx = await self._fetch_pr_context(repo, pr_num)
        pr_branch = ctx.get("branch", "")
        if not pr_branch:
            logger.error("[webhook_action] Cannot resolve conflict: unknown branch")
            return False

        with tempfile.TemporaryDirectory() as tmpdir:
            clone_cmd = (
                f"source ~/code/personal-intelligence/scripts/setup-bot-env.sh hermes && "
                f"gh repo clone {shlex.quote(repo)} {shlex.quote(tmpdir)} -- "
                f"--depth=1 --branch={shlex.quote(pr_branch)}"
            )
            clone_result = await asyncio.to_thread(
                subprocess.run,
                ["bash", "-c", clone_cmd],
                capture_output=True, text=True, timeout=120,
            )
            if clone_result.returncode != 0:
                logger.error(
                    "[webhook_action] Clone failed: %s", clone_result.stderr
                )
                return False

            prompt = (
                f"You are resolving merge conflicts in {repo} PR #{pr_num}.\n"
                f"Branch: {pr_branch}\n\n"
                "Resolve all merge conflicts, commit the resolution using the bot identity "
                "already set in env vars, and push to the remote branch.\n"
                "If you cannot resolve cleanly, output: BLOCKED: <reason>\n"
                "When done, output: DONE"
            )
            cc_result = await self._spawn_cc(prompt, tmpdir)
            if cc_result["blocked"]:
                await self._post_pr_comment(
                    repo, pr_num,
                    f"❌ Conflict resolution blocked: {cc_result['blocked']}"
                )
                return False

        # Retry merge after conflict resolution
        import shlex as _shlex
        retry_cmd = (
            f"source ~/code/personal-intelligence/scripts/setup-bot-env.sh hermes && "
            f"gh pr merge {pr_num} --repo {_shlex.quote(repo)} --merge --delete-branch"
        )
        retry_result = await asyncio.to_thread(
            subprocess.run,
            ["bash", "-c", retry_cmd],
            capture_output=True, text=True, timeout=60,
        )
        if retry_result.returncode == 0:
            await self._post_pr_comment(repo, pr_num, "✅ Conflicts resolved and merged.")
            return True
        return False

    async def _action_address_comments(
        self, repo: str, pr_num: int, merge_after: bool
    ) -> SendResult:
        """Spawn CC to address PR review comments."""
        import shlex
        import tempfile
        from pathlib import Path

        logger.info(
            "[webhook_action] Addressing comments on %s#%d merge_after=%s",
            repo, pr_num, merge_after,
        )

        ctx = await self._fetch_pr_context(repo, pr_num)

        soul_content = ""
        try:
            soul_content = Path("~/.hermes/SOUL.md").expanduser().read_text()
        except Exception as e:
            logger.warning("[webhook_action] Could not read SOUL.md: %s", e)

        # Format unresolved review comments
        comments_parts = []
        for c in ctx.get("comments", []):
            if c.get("position") is not None:  # unresolved inline comment
                path = c.get("path", "")
                body = c.get("body", "")
                comments_parts.append(f"- `{path}`: {body}")

        # Also include review-level comments
        for r in ctx.get("reviews", []):
            body = r.get("body", "").strip()
            if body and r.get("state") in ("CHANGES_REQUESTED", "COMMENTED"):
                reviewer = r.get("user", {}).get("login", "reviewer")
                comments_parts.append(f"- [{reviewer} review]: {body}")

        comments_text = "\n".join(comments_parts) if comments_parts else "(no inline comments)"

        linked_issue_section = ""
        if ctx.get("linked_issue"):
            issue = ctx["linked_issue"]
            linked_issue_section = (
                f"## Linked Issue #{issue.get('number', '')}\n"
                f"**{issue.get('title', '')}**\n\n{issue.get('body', '')}"
            )

        prompt = f"""You are hermes-bot-maiixu[bot], a GitHub automation bot.

## Identity
Git identity and GH_TOKEN are already set in your environment variables.
Always commit as hermes-bot-maiixu[bot].

## SOUL guidelines
{soul_content}

## Task
Repository: {repo}
PR #{pr_num}: {ctx.get('title', '')}

{linked_issue_section}

## PR Description
{ctx.get('body', '')}

## Review comments to address
{comments_text}

## Instructions
1. Address every review comment listed above with the minimum necessary change
2. Commit your changes using the bot identity already set in env vars
3. Push to the remote branch
4. Only modify files mentioned in the review comments
5. If you encounter something you cannot resolve, output: BLOCKED: <reason>
6. When complete, output: DONE"""

        pr_branch = ctx.get("branch", "")
        if not pr_branch:
            logger.error("[webhook_action] Cannot address comments: unknown branch")
            return SendResult(success=False, error="Unknown PR branch")

        with tempfile.TemporaryDirectory() as tmpdir:
            clone_cmd = (
                f"source ~/code/personal-intelligence/scripts/setup-bot-env.sh hermes && "
                f"gh repo clone {shlex.quote(repo)} {shlex.quote(tmpdir)} -- "
                f"--depth=1 --branch={shlex.quote(pr_branch)}"
            )
            clone_result = await asyncio.to_thread(
                subprocess.run,
                ["bash", "-c", clone_cmd],
                capture_output=True, text=True, timeout=120,
            )
            if clone_result.returncode != 0:
                logger.error(
                    "[webhook_action] Clone failed: %s", clone_result.stderr
                )
                return SendResult(success=False, error=clone_result.stderr)

            cc_result = await self._spawn_cc(prompt, tmpdir)

        if cc_result["blocked"]:
            await self._post_pr_comment(
                repo, pr_num,
                f"🚫 Could not address comments: {cc_result['blocked']}"
            )
            return SendResult(success=False, error=cc_result["blocked"])

        if merge_after:
            return await self._action_merge_pr(repo, pr_num)
        else:
            await self._post_pr_comment(
                repo, pr_num,
                "📝 Addressed review comments. Re-requesting review."
            )
            re_review_cmd = (
                f"source ~/code/personal-intelligence/scripts/setup-bot-env.sh hermes && "
                f"gh pr review {pr_num} --repo {shlex.quote(repo)} --request-review"
            )
            await asyncio.to_thread(
                subprocess.run,
                ["bash", "-c", re_review_cmd],
                capture_output=True, text=True, timeout=30,
            )
            return SendResult(success=True)

    async def _action_reply_issue(self, repo: str, issue_num: int) -> SendResult:
        """Spawn CC to reply to a GitHub issue."""
        import shlex
        import tempfile
        from pathlib import Path

        logger.info("[webhook_action] Replying to issue %s#%d", repo, issue_num)
        bot_env = "source ~/code/personal-intelligence/scripts/setup-bot-env.sh hermes 2>/dev/null"

        issue_result = await asyncio.to_thread(
            subprocess.run,
            ["bash", "-c",
             f"{bot_env} && gh issue view {issue_num} --repo {shlex.quote(repo)} "
             f"--json title,body,comments"],
            capture_output=True, text=True, timeout=60,
        )
        issue_data = {}
        if issue_result.returncode == 0:
            try:
                issue_data = json.loads(issue_result.stdout)
            except json.JSONDecodeError:
                pass

        soul_content = ""
        try:
            soul_content = Path("~/.hermes/SOUL.md").expanduser().read_text()
        except Exception as e:
            logger.warning("[webhook_action] Could not read SOUL.md: %s", e)

        comments_text = ""
        for c in issue_data.get("comments", []):
            author = c.get("author", {}).get("login", "user")
            body = c.get("body", "")
            comments_text += f"\n**{author}**: {body}\n"

        prompt = f"""You are hermes-bot-maiixu[bot], a GitHub automation bot.

## Identity
Git identity and GH_TOKEN are already set in your environment variables.

## SOUL guidelines
{soul_content}

## Task
Reply to GitHub issue {repo}#{issue_num}.

## Issue: {issue_data.get('title', '')}
{issue_data.get('body', '')}

## Existing comments
{comments_text or '(none)'}

## Instructions
1. Post a thoughtful, helpful reply to the issue using:
   gh issue comment {issue_num} --repo {shlex.quote(repo)} --body '<your reply>'
2. Be concise and address the issue directly
3. When complete, output: DONE"""

        with tempfile.TemporaryDirectory() as tmpdir:
            cc_result = await self._spawn_cc(prompt, tmpdir)

        if cc_result["blocked"]:
            logger.error(
                "[webhook_action] reply_issue blocked: %s", cc_result["blocked"]
            )
            return SendResult(success=False, error=cc_result["blocked"])

        return SendResult(success=True)

