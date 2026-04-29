"""Built-in slash command handlers."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import sys
from datetime import datetime

from nanobot import __version__
from nanobot.bus.events import OutboundMessage
from nanobot.command.router import CommandContext, CommandRouter
from nanobot.utils.helpers import build_status_content
from nanobot.utils.restart import set_restart_notice_to_env


async def cmd_stop(ctx: CommandContext) -> OutboundMessage:
    """Cancel all active tasks and subagents for the session."""
    loop = ctx.loop
    msg = ctx.msg
    total = await loop._cancel_active_tasks(msg.session_key)
    content = f"Stopped {total} task(s)." if total else "No active task to stop."
    return OutboundMessage(
        channel=msg.channel, chat_id=msg.chat_id, content=content,
        metadata=dict(msg.metadata or {})
    )


async def cmd_restart(ctx: CommandContext) -> OutboundMessage:
    """Restart the process in-place via os.execv."""
    msg = ctx.msg
    set_restart_notice_to_env(
        channel=msg.channel,
        chat_id=msg.chat_id,
        metadata=dict(msg.metadata or {}),
    )

    async def _do_restart():
        await asyncio.sleep(1)
        os.execv(sys.executable, [sys.executable, "-m", "nanobot"] + sys.argv[1:])

    asyncio.create_task(_do_restart())
    return OutboundMessage(
        channel=msg.channel, chat_id=msg.chat_id, content="Restarting...",
        metadata=dict(msg.metadata or {})
    )


async def cmd_status(ctx: CommandContext) -> OutboundMessage:
    """Build an outbound status message for a session."""
    loop = ctx.loop
    session = ctx.session or loop.sessions.get_or_create(ctx.key)
    ctx_est = 0
    try:
        ctx_est, _ = loop.consolidator.estimate_session_prompt_tokens(session)
    except Exception:
        pass
    if ctx_est <= 0:
        ctx_est = loop._last_usage.get("prompt_tokens", 0)

    # Fetch web search provider usage (best-effort, never blocks the response)
    search_usage_text: str | None = None
    try:
        from nanobot.utils.searchusage import fetch_search_usage
        web_cfg = getattr(loop, "web_config", None)
        search_cfg = getattr(web_cfg, "search", None) if web_cfg else None
        if search_cfg is not None:
            provider = getattr(search_cfg, "provider", "duckduckgo")
            api_key = getattr(search_cfg, "api_key", "") or None
            usage = await fetch_search_usage(provider=provider, api_key=api_key)
            search_usage_text = usage.format()
    except Exception:
        pass  # Never let usage fetch break /status
    active_tasks = loop._active_tasks.get(ctx.key, [])
    task_count = sum(1 for t in active_tasks if not t.done())
    try:
        task_count += loop.subagents.get_running_count_by_session(ctx.key)
    except Exception:
        pass
    return OutboundMessage(
        channel=ctx.msg.channel,
        chat_id=ctx.msg.chat_id,
        content=build_status_content(
            version=__version__, model=loop.model,
            start_time=loop._start_time, last_usage=loop._last_usage,
            context_window_tokens=loop.context_window_tokens,
            session_msg_count=len(session.get_history(max_messages=0)),
            context_tokens_estimate=ctx_est,
            search_usage_text=search_usage_text,
            active_task_count=task_count,
            max_completion_tokens=getattr(
                getattr(loop.provider, "generation", None), "max_tokens", 8192
            ),
        ),
        metadata={**dict(ctx.msg.metadata or {}), "render_as": "text"},
    )


async def cmd_new(ctx: CommandContext) -> OutboundMessage:
    """Stop active task and start a fresh session."""
    loop = ctx.loop
    await loop._cancel_active_tasks(ctx.key)
    session = ctx.session or loop.sessions.get_or_create(ctx.key)
    snapshot = session.messages[session.last_consolidated:]
    session.clear()
    loop.sessions.save(session)
    loop.sessions.invalidate(session.key)
    if snapshot:
        loop._schedule_background(loop.consolidator.archive(snapshot))
    return OutboundMessage(
        channel=ctx.msg.channel, chat_id=ctx.msg.chat_id,
        content="New session started.",
        metadata=dict(ctx.msg.metadata or {})
    )


async def cmd_dream(ctx: CommandContext) -> OutboundMessage:
    """Manually trigger a Dream consolidation run."""
    import time

    loop = ctx.loop
    msg = ctx.msg

    async def _run_dream():
        t0 = time.monotonic()
        try:
            did_work = await loop.dream.run()
            elapsed = time.monotonic() - t0
            if did_work:
                content = f"Dream completed in {elapsed:.1f}s."
            else:
                content = "Dream: nothing to process."
        except Exception as e:
            elapsed = time.monotonic() - t0
            content = f"Dream failed after {elapsed:.1f}s: {e}"
        await loop.bus.publish_outbound(OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=content,
        ))

    asyncio.create_task(_run_dream())
    return OutboundMessage(
        channel=msg.channel, chat_id=msg.chat_id, content="Dreaming...",
    )


def _extract_changed_files(diff: str) -> list[str]:
    """Extract changed file paths from a unified diff."""
    files: list[str] = []
    seen: set[str] = set()
    for line in diff.splitlines():
        if not line.startswith("diff --git "):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        path = parts[3]
        if path.startswith("b/"):
            path = path[2:]
        if path in seen:
            continue
        seen.add(path)
        files.append(path)
    return files


def _format_changed_files(diff: str) -> str:
    files = _extract_changed_files(diff)
    if not files:
        return "No tracked memory files changed."
    return ", ".join(f"`{path}`" for path in files)


def _format_dream_log_content(commit, diff: str, *, requested_sha: str | None = None) -> str:
    files_line = _format_changed_files(diff)
    lines = [
        "## Dream Update",
        "",
        "Here is the selected Dream memory change." if requested_sha else "Here is the latest Dream memory change.",
        "",
        f"- Commit: `{commit.sha}`",
        f"- Time: {commit.timestamp}",
        f"- Changed files: {files_line}",
    ]
    if diff:
        lines.extend([
            "",
            f"Use `/dream-restore {commit.sha}` to undo this change.",
            "",
            "```diff",
            diff.rstrip(),
            "```",
        ])
    else:
        lines.extend([
            "",
            "Dream recorded this version, but there is no file diff to display.",
        ])
    return "\n".join(lines)


def _format_dream_restore_list(commits: list) -> str:
    lines = [
        "## Dream Restore",
        "",
        "Choose a Dream memory version to restore. Latest first:",
        "",
    ]
    for c in commits:
        lines.append(f"- `{c.sha}` {c.timestamp} - {c.message.splitlines()[0]}")
    lines.extend([
        "",
        "Preview a version with `/dream-log <sha>` before restoring it.",
        "Restore a version with `/dream-restore <sha>`.",
    ])
    return "\n".join(lines)


async def cmd_dream_log(ctx: CommandContext) -> OutboundMessage:
    """Show what the last Dream changed.

    Default: diff of the latest commit (HEAD~1 vs HEAD).
    With /dream-log <sha>: diff of that specific commit.
    """
    store = ctx.loop.consolidator.store
    git = store.git

    if not git.is_initialized():
        if store.get_last_dream_cursor() == 0:
            msg = "Dream has not run yet. Run `/dream`, or wait for the next scheduled Dream cycle."
        else:
            msg = "Dream history is not available because memory versioning is not initialized."
        return OutboundMessage(
            channel=ctx.msg.channel, chat_id=ctx.msg.chat_id,
            content=msg, metadata={"render_as": "text"},
        )

    args = ctx.args.strip()

    if args:
        # Show diff of a specific commit
        sha = args.split()[0]
        result = git.show_commit_diff(sha)
        if not result:
            content = (
                f"Couldn't find Dream change `{sha}`.\n\n"
                "Use `/dream-restore` to list recent versions, "
                "or `/dream-log` to inspect the latest one."
            )
        else:
            commit, diff = result
            content = _format_dream_log_content(commit, diff, requested_sha=sha)
    else:
        # Default: show the latest commit's diff
        commits = git.log(max_entries=1)
        result = git.show_commit_diff(commits[0].sha) if commits else None
        if result:
            commit, diff = result
            content = _format_dream_log_content(commit, diff)
        else:
            content = "Dream memory has no saved versions yet."

    return OutboundMessage(
        channel=ctx.msg.channel, chat_id=ctx.msg.chat_id,
        content=content, metadata={"render_as": "text"},
    )


async def cmd_dream_restore(ctx: CommandContext) -> OutboundMessage:
    """Restore memory files from a previous dream commit.

    Usage:
        /dream-restore          — list recent commits
        /dream-restore <sha>    — revert a specific commit
    """
    store = ctx.loop.consolidator.store
    git = store.git
    if not git.is_initialized():
        return OutboundMessage(
            channel=ctx.msg.channel, chat_id=ctx.msg.chat_id,
            content="Dream history is not available because memory versioning is not initialized.",
        )

    args = ctx.args.strip()
    if not args:
        # Show recent commits for the user to pick
        commits = git.log(max_entries=10)
        if not commits:
            content = "Dream memory has no saved versions to restore yet."
        else:
            content = _format_dream_restore_list(commits)
    else:
        sha = args.split()[0]
        result = git.show_commit_diff(sha)
        changed_files = _format_changed_files(result[1]) if result else "the tracked memory files"
        new_sha = git.revert(sha)
        if new_sha:
            content = (
                f"Restored Dream memory to the state before `{sha}`.\n\n"
                f"- New safety commit: `{new_sha}`\n"
                f"- Restored files: {changed_files}\n\n"
                f"Use `/dream-log {new_sha}` to inspect the restore diff."
            )
        else:
            content = (
                f"Couldn't restore Dream change `{sha}`.\n\n"
                "It may not exist, or it may be the first saved version with no earlier state to restore."
            )
    return OutboundMessage(
        channel=ctx.msg.channel, chat_id=ctx.msg.chat_id,
        content=content, metadata={"render_as": "text"},
    )


_HISTORY_DEFAULT_COUNT = 10
_HISTORY_MAX_COUNT = 50
_HISTORY_MAX_CONTENT_CHARS = 200


def _format_history_message(msg: dict) -> str | None:
    """Format a single history message for display. Returns None to skip."""
    role = msg.get("role")
    if role not in ("user", "assistant"):
        return None
    content = msg.get("content") or ""
    if isinstance(content, list):
        parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
        content = " ".join(parts)
    content = str(content).strip()
    if not content:
        return None
    if len(content) > _HISTORY_MAX_CONTENT_CHARS:
        content = content[:_HISTORY_MAX_CONTENT_CHARS] + "…"
    label = "👤 You" if role == "user" else "🤖 Bot"
    return f"{label}: {content}"


async def cmd_history(ctx: CommandContext) -> OutboundMessage:
    """Show the last N messages of the current session (default 10, max 50).

    Usage: /history [count]
    """
    count = _HISTORY_DEFAULT_COUNT
    if ctx.args.strip():
        try:
            count = max(1, min(int(ctx.args.strip()), _HISTORY_MAX_COUNT))
        except ValueError:
            return OutboundMessage(
                channel=ctx.msg.channel, chat_id=ctx.msg.chat_id,
                content="Usage: /history [count] — e.g. /history 5 (default: 10, max: 50)",
                metadata=dict(ctx.msg.metadata or {}),
            )

    session = ctx.session or ctx.loop.sessions.get_or_create(ctx.key)
    history = session.get_history(max_messages=0)
    visible = [_format_history_message(m) for m in history]
    visible = [m for m in visible if m is not None]
    recent = visible[-count:]

    if not recent:
        return OutboundMessage(
            channel=ctx.msg.channel, chat_id=ctx.msg.chat_id,
            content="No conversation history yet.",
            metadata=dict(ctx.msg.metadata or {}),
        )

    header = f"Last {len(recent)} message(s):\n"
    return OutboundMessage(
        channel=ctx.msg.channel, chat_id=ctx.msg.chat_id,
        content=header + "\n".join(recent),
        metadata={**dict(ctx.msg.metadata or {}), "render_as": "text"},
    )


_CHECKNOW_ROUNDS = 5
_CHECKNOW_MAX_CONTENT_CHARS = 1200


def _latest_turn_window(messages: list[dict], *, rounds: int = _CHECKNOW_ROUNDS) -> list[tuple[int, dict]]:
    visible = [
        (idx, msg)
        for idx, msg in enumerate(messages)
        if msg.get("role") in {"user", "assistant"}
    ]
    if not visible:
        return []

    user_turns = 0
    start = 0
    for pos in range(len(visible) - 1, -1, -1):
        if visible[pos][1].get("role") == "user":
            user_turns += 1
            if user_turns >= rounds:
                start = pos
                break
    return visible[start:]


def _message_text(msg: dict) -> str:
    content = msg.get("content") or ""
    if isinstance(content, list):
        parts = [
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        return "\n".join(part for part in parts if part)
    return str(content)


def _truncate_checknow_text(text: str) -> str:
    if len(text) <= _CHECKNOW_MAX_CONTENT_CHARS:
        return text
    return text[:_CHECKNOW_MAX_CONTENT_CHARS] + "\n...[truncated]"


def _strong_garbled_reason(text: str) -> str | None:
    stripped = text.strip()
    if len(stripped) < 30:
        return None
    if re.search(r"([!！?？。.,，;；:：_\-=*#~`|/\\])\1{20,}", stripped):
        return "long repeated punctuation/symbol run"
    if len(stripped) >= 80 and re.search(r"(.{1,8})\1{12,}", stripped):
        return "repeated short fragment"
    if stripped.count("\ufffd") >= 3:
        return "replacement-character corruption"
    code_fragments = re.findall(r"[A-Za-z_][A-Za-z0-9_]{0,24}[().{}\[\]_/:=-]", stripped)
    if len(stripped) >= 120 and len(code_fragments) >= 10:
        return "random code-like fragments"
    return None


def _weak_garbled_reason(text: str) -> str | None:
    strong = _strong_garbled_reason(text)
    if strong:
        return strong
    stripped = text.strip()
    if len(stripped) < 80:
        return None
    code_fragments = re.findall(r"[A-Za-z_][A-Za-z0-9_]{0,24}[().{}\[\]_/:=-]", stripped)
    if len(code_fragments) >= 5 and re.search(r"[\u4e00-\u9fff]", stripped):
        return "mixed natural language with random code-like fragments"
    symbol_count = len(re.findall(r"[^\w\s\u4e00-\u9fff]", stripped, flags=re.UNICODE))
    if symbol_count / max(1, len(stripped)) > 0.45:
        return "very high symbol density"
    return None


def _extract_json_object(text: str) -> dict:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:].strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        raw = raw[start:end + 1]
    try:
        parsed = json.loads(raw)
    except Exception:
        try:
            import json_repair

            parsed = json_repair.loads(raw)
        except Exception:
            return {}
    return parsed if isinstance(parsed, dict) else {}


def _parse_checknow_indices(text: str) -> set[int]:
    data = _extract_json_object(text)
    raw_indices = (
        data.get("delete_indices")
        or data.get("garbled_indices")
        or data.get("indices")
        or []
    )
    if not isinstance(raw_indices, list):
        return set()
    indices: set[int] = set()
    for value in raw_indices:
        if isinstance(value, bool):
            continue
        try:
            indices.add(int(value))
        except (TypeError, ValueError):
            continue
    return indices


async def _model_checknow_indices(ctx: CommandContext, candidates: list[dict]) -> set[int]:
    prompt = (
        "You are a conservative cleanup classifier for a chat log.\n"
        "Return ONLY JSON in this exact shape: "
        "{\"delete_indices\":[0],\"reasons\":{\"0\":\"reason\"}}.\n"
        "Mark only assistant messages that are obvious garbled output: long repeated "
        "punctuation, corrupted random fragments, incoherent mixed-language/code-token "
        "junk, or severe repetition. Do not mark a message just because it is awkward, "
        "incorrect, short, or unhelpful. Never mark user messages.\n"
    )
    response = await ctx.loop.provider.chat_with_retry(
        model=ctx.loop.model,
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": json.dumps(
                    {"messages": candidates},
                    ensure_ascii=False,
                ),
            },
        ],
        tools=None,
        tool_choice=None,
        max_tokens=256,
        temperature=0,
    )
    if response.finish_reason == "error":
        return set()
    return _parse_checknow_indices(response.content or "")


async def cmd_checknow(ctx: CommandContext) -> OutboundMessage:
    """Clean obviously garbled assistant messages from the latest five turns."""
    session = ctx.session or ctx.loop.sessions.get_or_create(ctx.key)
    window = _latest_turn_window(session.messages)
    if not window:
        return OutboundMessage(
            channel=ctx.msg.channel,
            chat_id=ctx.msg.chat_id,
            content="No recent conversation to check.",
            metadata=dict(ctx.msg.metadata or {}),
        )

    candidates: list[dict] = []
    local_to_original: dict[int, int] = {}
    heuristic_reasons: dict[int, str] = {}
    for local_idx, (original_idx, msg) in enumerate(window):
        text = _message_text(msg)
        candidates.append({
            "index": local_idx,
            "role": msg.get("role"),
            "content": _truncate_checknow_text(text),
        })
        local_to_original[local_idx] = original_idx
        if msg.get("role") == "assistant":
            reason = _strong_garbled_reason(text)
            if reason:
                heuristic_reasons[local_idx] = reason

    model_indices = await _model_checknow_indices(ctx, candidates)

    delete_original: dict[int, str] = {}
    for local_idx, reason in heuristic_reasons.items():
        delete_original[local_to_original[local_idx]] = reason
    for local_idx in model_indices:
        original_idx = local_to_original.get(local_idx)
        if original_idx is None:
            continue
        msg = session.messages[original_idx]
        if msg.get("role") != "assistant":
            continue
        weak_reason = _weak_garbled_reason(_message_text(msg))
        if weak_reason:
            delete_original[original_idx] = f"model flagged; {weak_reason}"

    if not delete_original:
        return OutboundMessage(
            channel=ctx.msg.channel,
            chat_id=ctx.msg.chat_id,
            content=f"Checknow finished: checked the latest {_CHECKNOW_ROUNDS} turn(s), no obvious garbled assistant messages found.",
            metadata={**dict(ctx.msg.metadata or {}), "render_as": "text"},
        )

    backup = None
    try:
        path = ctx.loop.sessions._get_session_path(session.key)
        if path.exists():
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            backup = path.with_name(f"{path.stem}.checknow-{stamp}{path.suffix}.bak")
            shutil.copy2(path, backup)
    except Exception:
        backup = None

    removed_before_cursor = sum(1 for idx in delete_original if idx < session.last_consolidated)
    for original_idx in sorted(delete_original, reverse=True):
        del session.messages[original_idx]
    session.last_consolidated = max(0, session.last_consolidated - removed_before_cursor)
    session.metadata.pop("_last_summary", None)
    ctx.loop.sessions.save(session, fsync=True)

    reasons = [
        f"- message #{idx}: {reason}"
        for idx, reason in sorted(delete_original.items())
    ]
    backup_line = f"\nBackup: `{backup}`" if backup else "\nBackup: not available."
    content = (
        f"Checknow removed {len(delete_original)} obvious garbled assistant message(s) "
        f"from the latest {_CHECKNOW_ROUNDS} turn(s).\n"
        + "\n".join(reasons)
        + backup_line
    )
    return OutboundMessage(
        channel=ctx.msg.channel,
        chat_id=ctx.msg.chat_id,
        content=content,
        metadata={**dict(ctx.msg.metadata or {}), "render_as": "text"},
    )


async def cmd_help(ctx: CommandContext) -> OutboundMessage:
    """Return available slash commands."""
    return OutboundMessage(
        channel=ctx.msg.channel,
        chat_id=ctx.msg.chat_id,
        content=build_help_text(),
        metadata={**dict(ctx.msg.metadata or {}), "render_as": "text"},
    )


def build_help_text() -> str:
    """Build canonical help text shared across channels."""
    lines = [
        "🐈 nanobot commands:",
        "/new — Stop current task and start a new conversation",
        "/stop — Stop the current task",
        "/restart — Restart the bot",
        "/status — Show bot status",
        "/history [n] — Show the last N conversation messages (default 10)",
        "/checknow — Clean obvious garbled assistant messages from recent history",
        "/dream — Manually trigger Dream consolidation",
        "/dream-log — Show what the last Dream changed",
        "/dream-restore — Revert memory to a previous state",
        "/help — Show available commands",
    ]
    return "\n".join(lines)


def register_builtin_commands(router: CommandRouter) -> None:
    """Register the default set of slash commands."""
    router.priority("/stop", cmd_stop)
    router.priority("/restart", cmd_restart)
    router.priority("/status", cmd_status)
    router.exact("/new", cmd_new)
    router.exact("/status", cmd_status)
    router.exact("/history", cmd_history)
    router.prefix("/history ", cmd_history)
    router.exact("/checknow", cmd_checknow)
    router.exact("/dream", cmd_dream)
    router.exact("/dream-log", cmd_dream_log)
    router.prefix("/dream-log ", cmd_dream_log)
    router.exact("/dream-restore", cmd_dream_restore)
    router.prefix("/dream-restore ", cmd_dream_restore)
    router.exact("/help", cmd_help)
