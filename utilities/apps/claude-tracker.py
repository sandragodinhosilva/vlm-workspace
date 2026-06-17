#!/usr/bin/env python3
"""Claude Code Spend Tracker — local dashboard for token usage and cost estimation."""

import argparse
import glob
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

LONDON_TZ = ZoneInfo("Europe/London")
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

CLAUDE_DIR = Path.home() / ".claude"
STATS_CACHE = CLAUDE_DIR / "stats-cache.json"
PROJECTS_DIR = CLAUDE_DIR / "projects"

# Anthropic pricing per 1M tokens (as of 2026-03)
PRICING = {
    "claude-opus-4-6": {
        "input": 15.0, "output": 75.0,
        "cache_read": 1.875, "cache_write": 18.75,
    },
    "claude-opus-4-5-20251101": {
        "input": 15.0, "output": 75.0,
        "cache_read": 1.875, "cache_write": 18.75,
    },
    "claude-sonnet-4-6": {
        "input": 3.0, "output": 15.0,
        "cache_read": 0.30, "cache_write": 3.75,
    },
    "claude-sonnet-4-5-20250929": {
        "input": 3.0, "output": 15.0,
        "cache_read": 0.30, "cache_write": 3.75,
    },
}
# Fallback for unknown models — use Sonnet pricing (cheapest)
DEFAULT_PRICING = {"input": 3.0, "output": 15.0, "cache_read": 0.30, "cache_write": 3.75}

# Data cache
_cache = {"data": None, "ts": 0}
CACHE_TTL = 10  # seconds


def compute_cost(model, input_t, output_t, cache_read_t, cache_write_t):
    p = PRICING.get(model, DEFAULT_PRICING)
    return (
        input_t * p["input"] / 1_000_000
        + output_t * p["output"] / 1_000_000
        + cache_read_t * p["cache_read"] / 1_000_000
        + cache_write_t * p["cache_write"] / 1_000_000
    )


def _in_range(date_str, start_dt, end_dt):
    """Check if a YYYY-MM-DD string falls in [start_dt, end_dt)."""
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return start_dt <= d < end_dt
    except ValueError:
        return False


def load_stats_cache():
    """Load aggregate data from stats-cache.json."""
    if not STATS_CACHE.exists():
        return None
    with open(STATS_CACHE) as f:
        return json.load(f)


def scan_session_files(after_date=None):
    """Scan session JSONL files for per-message usage data.

    If after_date is given (YYYY-MM-DD string), only scan files modified after that date.
    Returns: model_totals, daily_usage, daily_cost, total_messages, hourly_entries
    """
    model_totals = {}
    daily_usage = {}  # date -> {model -> token counts}
    daily_cost = {}   # date -> cost
    total_messages = 0
    # Store individual entries with timestamps for rolling-window calculations
    hourly_entries = []  # list of (iso_timestamp, model, inp, out, cr, cw)
    user_turn_timestamps = []  # timestamps of human user turns (for message counting)

    if after_date:
        cutoff_ts = datetime.strptime(after_date, "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        ).timestamp()
    else:
        cutoff_ts = 0

    session_files = []
    for project_dir in PROJECTS_DIR.iterdir():
        if not project_dir.is_dir():
            continue
        for f in project_dir.glob("*.jsonl"):
            if f.stat().st_mtime > cutoff_ts:
                session_files.append(f)

    for fpath in session_files:
        try:
            with open(fpath) as f:
                for line in f:
                    line = line.strip()
                    if not line or '"usage"' not in line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Track user turns for conversational message counts
                    if entry.get("type") == "user":
                        ts_u = entry.get("timestamp", "")
                        if ts_u and not entry.get("isSidechain"):
                            user_turn_timestamps.append(ts_u)
                        continue

                    msg = entry.get("message")
                    if not msg or not isinstance(msg, dict):
                        continue
                    usage = msg.get("usage")
                    if not usage:
                        continue

                    model = msg.get("model", "unknown")
                    if not model or model.startswith("<") or model == "unknown":
                        continue
                    inp = usage.get("input_tokens", 0)
                    out = usage.get("output_tokens", 0)
                    cr = usage.get("cache_read_input_tokens", 0)
                    cw = usage.get("cache_creation_input_tokens", 0)

                    if model not in model_totals:
                        model_totals[model] = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
                    model_totals[model]["input"] += inp
                    model_totals[model]["output"] += out
                    model_totals[model]["cache_read"] += cr
                    model_totals[model]["cache_write"] += cw
                    total_messages += 1

                    # Timestamp
                    ts = entry.get("timestamp", "")
                    date = ts[:10] if ts else "unknown"

                    # Store for rolling-window calculations
                    if ts:
                        hourly_entries.append((ts, model, inp, out, cr, cw))

                    # Daily grouping
                    if date not in daily_usage:
                        daily_usage[date] = {}
                    if model not in daily_usage[date]:
                        daily_usage[date][model] = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
                    daily_usage[date][model]["input"] += inp
                    daily_usage[date][model]["output"] += out
                    daily_usage[date][model]["cache_read"] += cr
                    daily_usage[date][model]["cache_write"] += cw
        except (PermissionError, OSError):
            continue

    # Compute daily costs
    for date, models in daily_usage.items():
        cost = 0
        for model, t in models.items():
            cost += compute_cost(model, t["input"], t["output"], t["cache_read"], t["cache_write"])
        daily_cost[date] = round(cost, 2)

    return model_totals, daily_usage, daily_cost, total_messages, hourly_entries, user_turn_timestamps


_sessions_cache = {"data": None, "ts": 0}


def scan_sessions():
    """Scan all session JSONL files and return per-session summaries, newest first."""
    now = time.time()
    if _sessions_cache["data"] and (now - _sessions_cache["ts"]) < CACHE_TTL:
        return _sessions_cache["data"]

    sessions = []
    for project_dir in PROJECTS_DIR.iterdir():
        if not project_dir.is_dir():
            continue
        project_name = project_dir.name
        for fpath in project_dir.glob("*.jsonl"):
            session_id = fpath.stem
            # per-model token accumulators — avoids double-counting when multiple models used
            per_model = {}  # model -> {inp, out, cr, cw}
            tool_calls = 0
            user_msgs = 0
            asst_msgs = 0
            first_ts = last_ts = None
            has_compaction = False

            try:
                with open(fpath) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        etype = entry.get("type", "")
                        ts_raw = entry.get("timestamp", "")
                        if ts_raw:
                            try:
                                ts_dt = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
                                if first_ts is None or ts_dt < first_ts:
                                    first_ts = ts_dt
                                if last_ts is None or ts_dt > last_ts:
                                    last_ts = ts_dt
                            except ValueError:
                                pass

                        if etype == "user":
                            user_msgs += 1
                        elif etype == "assistant":
                            asst_msgs += 1
                            msg = entry.get("message", {})
                            if isinstance(msg, dict):
                                model = msg.get("model", "")
                                if model and not model.startswith("<"):
                                    if model not in per_model:
                                        per_model[model] = {"inp": 0, "out": 0, "cr": 0, "cw": 0}
                                    usage = msg.get("usage")
                                    if usage:
                                        per_model[model]["inp"] += usage.get("input_tokens", 0)
                                        per_model[model]["out"] += usage.get("output_tokens", 0)
                                        per_model[model]["cr"] += usage.get("cache_read_input_tokens", 0)
                                        per_model[model]["cw"] += usage.get("cache_creation_input_tokens", 0)
                                # count tool_use blocks
                                for block in msg.get("content", []) if isinstance(msg.get("content"), list) else []:
                                    if isinstance(block, dict) and block.get("type") == "tool_use":
                                        tool_calls += 1
                        elif etype == "summary":
                            has_compaction = True

            except OSError:
                continue

            if first_ts is None or not per_model:
                continue

            cost = sum(
                compute_cost(model, t["inp"], t["out"], t["cr"], t["cw"])
                for model, t in per_model.items()
            )
            inp_total = sum(t["inp"] for t in per_model.values())
            out_total = sum(t["out"] for t in per_model.values())
            cr_total  = sum(t["cr"]  for t in per_model.values())
            cw_total  = sum(t["cw"]  for t in per_model.values())

            duration_s = int((last_ts - first_ts).total_seconds()) if last_ts and first_ts else 0

            sessions.append({
                "id": session_id,
                "project": project_name,
                "startTime": first_ts.isoformat(),
                "endTime": last_ts.isoformat() if last_ts else first_ts.isoformat(),
                "duration": duration_s,
                "userMessages": user_msgs,
                "assistantMessages": asst_msgs,
                "toolCalls": tool_calls,
                "tokens": inp_total + out_total,
                "input": inp_total,
                "output": out_total,
                "cacheRead": cr_total,
                "cacheWrite": cw_total,
                "cost": round(cost, 4),
                "models": sorted(per_model.keys()),
                "hasCompaction": has_compaction,
                "gitBranch": entry.get("gitBranch", "") if 'entry' in dir() else "",
            })

    sessions.sort(key=lambda s: s["endTime"], reverse=True)
    _sessions_cache["data"] = sessions
    _sessions_cache["ts"] = now
    return sessions


def load_session_messages(session_id):
    """Return ordered list of {role, text, ts} for a session, across all project dirs."""
    for project_dir in PROJECTS_DIR.iterdir():
        if not project_dir.is_dir():
            continue
        fpath = project_dir / f"{session_id}.jsonl"
        if not fpath.exists():
            continue
        messages = []
        try:
            with open(fpath) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    etype = entry.get("type", "")
                    ts = entry.get("timestamp", "")
                    msg = entry.get("message", {})
                    if not isinstance(msg, dict):
                        try:
                            msg = json.loads(msg)
                        except Exception:
                            msg = {}
                    if etype == "user":
                        content = msg.get("content", "")
                        if isinstance(content, list):
                            # extract text parts only, skip images
                            parts = [b.get("text", "") for b in content
                                     if isinstance(b, dict) and b.get("type") == "text"]
                            text = "\n".join(p for p in parts if p)
                        else:
                            text = str(content) if content else ""
                        if text:
                            messages.append({"role": "user", "text": text, "ts": ts})
                    elif etype == "assistant":
                        content = msg.get("content", [])
                        if isinstance(content, list):
                            parts = []
                            for block in content:
                                if not isinstance(block, dict):
                                    continue
                                btype = block.get("type", "")
                                if btype == "text":
                                    parts.append(block.get("text", ""))
                                elif btype == "tool_use":
                                    parts.append(f"[Tool: {block.get('name', 'unknown')}]")
                            text = "\n".join(p for p in parts if p)
                        else:
                            text = str(content) if content else ""
                        if text:
                            messages.append({"role": "assistant", "text": text, "ts": ts})
        except OSError:
            pass
        return messages
    return []


def _aggregate_monthly(all_daily):
    """Sum daily costs into a month→cost dict."""
    monthly = {}
    for d, c in all_daily.items():
        if d == "unknown":
            continue
        month = d[:7]
        monthly[month] = round(monthly.get(month, 0) + c, 2)
    return monthly


def get_data():
    """Build the full data payload, with caching."""
    now = time.time()
    if _cache["data"] and (now - _cache["ts"]) < CACHE_TTL:
        return _cache["data"]

    stats = load_stats_cache()
    last_computed = stats.get("lastComputedDate", "1970-01-01") if stats else "1970-01-01"

    # Aggregate model usage from stats-cache
    cached_models = {}
    if stats and "modelUsage" in stats:
        for model, u in stats["modelUsage"].items():
            cached_models[model] = {
                "input": u.get("inputTokens", 0),
                "output": u.get("outputTokens", 0),
                "cache_read": u.get("cacheReadInputTokens", 0),
                "cache_write": u.get("cacheCreationInputTokens", 0),
            }

    # Daily data from stats-cache
    cached_daily_cost = {}
    if stats and "dailyModelTokens" in stats:
        for day in stats["dailyModelTokens"]:
            date = day["date"]
            cost = 0
            for model, tokens in day.get("tokensByModel", {}).items():
                # dailyModelTokens only has total tokens, not split — estimate as input
                p = PRICING.get(model, DEFAULT_PRICING)
                cost += tokens * p["input"] / 1_000_000
            cached_daily_cost[date] = round(cost, 2)

    # Scan recent sessions
    recent_models, recent_daily, recent_daily_cost, recent_msgs, hourly_entries, user_turn_timestamps = scan_session_files(
        after_date=last_computed
    )

    # Merge model totals
    merged = {}
    for model, t in cached_models.items():
        merged[model] = dict(t)
    for model, t in recent_models.items():
        if model in merged:
            for k in ("input", "output", "cache_read", "cache_write"):
                merged[model][k] += t[k]
        else:
            merged[model] = dict(t)

    # Compute per-model costs
    model_rows = []
    total_cost = 0
    for model in sorted(merged.keys()):
        t = merged[model]
        cost = compute_cost(model, t["input"], t["output"], t["cache_read"], t["cache_write"])
        total_cost += cost
        model_rows.append({
            "model": model,
            "input": t["input"],
            "output": t["output"],
            "cache_read": t["cache_read"],
            "cache_write": t["cache_write"],
            "cost": round(cost, 2),
        })

    # Merge daily costs
    all_daily = dict(cached_daily_cost)
    for date, cost in recent_daily_cost.items():
        all_daily[date] = all_daily.get(date, 0) + cost

    # Today's cost
    now_utc = datetime.now(timezone.utc)
    today = now_utc.strftime("%Y-%m-%d")
    today_cost = all_daily.get(today, 0)

    # Summary
    total_sessions = stats.get("totalSessions", 0) if stats else 0
    total_messages = stats.get("totalMessages", 0) if stats else 0
    total_messages += recent_msgs
    first_date = stats.get("firstSessionDate", "")[:10] if stats else ""

    # Pre-parse user turn timestamps for human-turn counting
    def _parse_user_turns(after_dt):
        count = 0
        for ts_str in user_turn_timestamps:
            try:
                ts_dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                if ts_dt >= after_dt:
                    count += 1
            except ValueError:
                pass
        return count

    # --- Fixed 5-hour window (Claude Enterprise rate limit window) ---
    # Algorithm: collect all usage timestamps today (London), sorted ascending.
    # Walk through them: whenever a timestamp falls outside the current window,
    # start a new window anchored at floor_to_hour(that timestamp).
    # This handles multiple windows per day automatically.
    now_london = now_utc.astimezone(LONDON_TZ)
    today_london = now_london.replace(hour=0, minute=0, second=0, microsecond=0)

    today_ts_london = sorted(
        ts_dt
        for ts_str, *_ in hourly_entries
        for ts_dt in [
            datetime.fromisoformat(ts_str.replace("Z", "+00:00")).astimezone(LONDON_TZ)
        ]
        if ts_dt >= today_london
    )

    if today_ts_london:
        window_start = today_ts_london[0].replace(minute=0, second=0, microsecond=0)
        for ts_dt in today_ts_london[1:]:
            if ts_dt >= window_start + timedelta(hours=5):
                # This message falls after the current window — start a new one
                window_start = ts_dt.replace(minute=0, second=0, microsecond=0)
        window_end = window_start + timedelta(hours=5)
        window_start_utc = window_start.astimezone(timezone.utc)
        window_end_utc = window_end.astimezone(timezone.utc)
        five_h_start = window_start.strftime("%H:%M")
        five_h_end = window_end.strftime("%H:%M")
        # Show reset time only if window hasn't expired yet
        five_h_reset_at = window_end.strftime("%H:%M") if now_london < window_end else None
    else:
        # No usage today — show a hypothetical window from the current hour
        window_start_utc = now_utc.replace(minute=0, second=0, microsecond=0)
        window_end_utc = window_start_utc + timedelta(hours=5)
        five_h_start = now_london.replace(minute=0, second=0, microsecond=0).strftime("%H:%M")
        five_h_end = (now_london.replace(minute=0, second=0, microsecond=0) + timedelta(hours=5)).strftime("%H:%M")
        five_h_reset_at = None

    five_h_cost = 0
    five_h_tokens = 0
    five_h_by_model = {}
    for ts_str, model, inp, out, cr, cw in hourly_entries:
        try:
            ts_dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except ValueError:
            continue
        if window_start_utc <= ts_dt < window_end_utc:
            c = compute_cost(model, inp, out, cr, cw)
            five_h_cost += c
            five_h_tokens += inp + out
            if model not in five_h_by_model:
                five_h_by_model[model] = {"cost": 0, "messages": 0}
            five_h_by_model[model]["cost"] += c
            five_h_by_model[model]["messages"] += 1
    five_h_messages = _parse_user_turns(window_start_utc)

    # --- Today's detailed stats ---
    today_start = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    today_tokens = 0
    today_by_model = {}
    for ts_str, model, inp, out, cr, cw in hourly_entries:
        try:
            ts_dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except ValueError:
            continue
        if ts_dt >= today_start:
            today_tokens += inp + out
            if model not in today_by_model:
                today_by_model[model] = {"cost": 0, "messages": 0}
            today_by_model[model]["cost"] += compute_cost(model, inp, out, cr, cw)
            today_by_model[model]["messages"] += 1
    today_messages = _parse_user_turns(today_start)

    # --- Weekly stats (Monday-Sunday) ---
    # Current week
    days_since_monday = now_utc.weekday()  # 0=Monday
    week_start = (now_utc - timedelta(days=days_since_monday)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    week_cost = 0
    week_tokens = 0
    week_daily = {}  # day_name -> cost
    for ts_str, model, inp, out, cr, cw in hourly_entries:
        try:
            ts_dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except ValueError:
            continue
        if ts_dt >= week_start:
            c = compute_cost(model, inp, out, cr, cw)
            week_cost += c
            week_tokens += inp + out
            day_name = ts_dt.strftime("%a")  # Mon, Tue, ...
            week_daily[day_name] = week_daily.get(day_name, 0) + c
    week_messages = _parse_user_turns(week_start)

    # Also add daily costs from stats-cache for this week (dates before last_computed)
    for d, c in cached_daily_cost.items():
        try:
            d_dt = datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if d_dt >= week_start:
            week_cost += c
            day_name = d_dt.strftime("%a")
            week_daily[day_name] = week_daily.get(day_name, 0) + c

    # Build ordered week_daily list (Mon-Sun)
    week_daily_ordered = []
    for day_name in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]:
        week_daily_ordered.append({"day": day_name, "cost": round(week_daily.get(day_name, 0), 2)})

    # Previous week comparison — homologous: same day-of-week + time-of-day
    prev_week_start = week_start - timedelta(days=7)
    prev_week_cutoff = now_utc - timedelta(days=7)  # same weekday + hour as now

    # Daily-granularity costs for full days before today's equivalent last week
    prev_week_full_day_cutoff = prev_week_cutoff.replace(hour=0, minute=0, second=0, microsecond=0)
    prev_week_cost = sum(
        c for d, c in all_daily.items()
        if d != "unknown" and _in_range(d, prev_week_start, prev_week_full_day_cutoff)
    )
    # Add partial day from hourly_entries for the homologous day itself
    for ts_str, model, inp, out, cr, cw in hourly_entries:
        try:
            ts_dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except ValueError:
            continue
        if prev_week_full_day_cutoff <= ts_dt < prev_week_cutoff:
            prev_week_cost += compute_cost(model, inp, out, cr, cw)

    data = {
        "models": model_rows,
        "totalCost": round(total_cost, 2),
        "todayCost": round(today_cost, 2),
        "totalSessions": total_sessions,
        "totalMessages": total_messages,
        "firstDate": first_date,
        "today": today,
        "lastComputed": last_computed,
        "dailyCost": [
            {"date": d, "cost": c}
            for d, c in sorted(all_daily.items())
            if d != "unknown"
        ],
        # Monthly cost map: {"2026-03": 19.6, ...}
        "monthlyCost": _aggregate_monthly(all_daily),
        # Rolling 5-hour window
        "fiveHour": {
            "cost": round(five_h_cost, 2),
            "messages": five_h_messages,
            "tokens": five_h_tokens,
            "conversations": sum(
                1 for s in scan_sessions()
                if s.get("endTime", s["startTime"]) >= window_start_utc.isoformat()
            ),
            "window": f"{five_h_start} – {five_h_end} London",
            "resetAt": five_h_reset_at,  # HH:MM London time, or null if no messages
            "byModel": [
                {"model": m, "cost": round(v["cost"], 2), "messages": v["messages"]}
                for m, v in sorted(five_h_by_model.items())
            ],
        },
        # Today detail
        "todayDetail": {
            "cost": round(today_cost, 2),
            "messages": today_messages,
            "tokens": today_tokens,
            "conversations": sum(
                1 for s in scan_sessions()
                if s.get("endTime", s["startTime"]) >= today_start.isoformat()
            ),
            "byModel": [
                {"model": m, "cost": round(v["cost"], 2), "messages": v["messages"]}
                for m, v in sorted(today_by_model.items())
            ],
        },
        # This week
        "week": {
            "cost": round(week_cost, 2),
            "messages": week_messages,
            "tokens": week_tokens,
            "daily": week_daily_ordered,
            "weekStart": week_start.strftime("%Y-%m-%d"),
            "prevWeekCost": round(prev_week_cost, 2),
        },
    }

    _cache["data"] = data
    _cache["ts"] = now
    return data


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Claude Usage Tracker</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
  /* ── SIDEBAR LAYOUT ── */
  .layout { display: flex; min-height: 100vh; }
  .sidebar {
    width: 220px; min-width: 220px; background: var(--white);
    border-right: 1px solid var(--border); display: flex; flex-direction: column;
    padding: 28px 0 20px; position: fixed; top: 0; left: 0; bottom: 0;
    z-index: 100; box-shadow: 2px 0 12px rgba(0,0,0,0.04);
  }
  .sidebar-logo {
    display: flex; align-items: center; gap: 10px;
    padding: 0 20px 28px; border-bottom: 1px solid var(--border);
    margin-bottom: 12px;
  }
  .logo-mark {
    width: 34px; height: 34px; border-radius: 10px;
    background: var(--gradient-main);
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 4px 12px rgba(99,102,241,0.3); flex-shrink: 0;
  }
  .logo-mark svg { width: 18px; height: 18px; }
  .sidebar-title { font-size: 14px; font-weight: 800; letter-spacing: -0.3px; line-height: 1.2; }
  .sidebar-title span { display: block; font-size: 11px; font-weight: 500; color: var(--text-tertiary); }
  .nav-group { padding: 0 10px; margin-bottom: 4px; }
  .nav-label {
    font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px;
    color: var(--text-tertiary); padding: 6px 10px 4px;
  }
  .nav-item {
    display: flex; align-items: center; gap: 10px;
    padding: 9px 12px; border-radius: 10px; cursor: pointer;
    font-size: 13px; font-weight: 500; color: var(--text-secondary);
    transition: all 0.15s; user-select: none; border: none; background: none;
    width: 100%; text-align: left;
  }
  .nav-item svg { width: 16px; height: 16px; flex-shrink: 0; opacity: 0.6; }
  .nav-item:hover { background: #F1F5F9; color: var(--text); }
  .nav-item:hover svg { opacity: 1; }
  .nav-item.active {
    background: linear-gradient(135deg, rgba(99,102,241,0.1), rgba(139,92,246,0.08));
    color: var(--indigo); font-weight: 700;
  }
  .nav-item.active svg { opacity: 1; }
  .sidebar-footer {
    margin-top: auto; padding: 16px 20px 0;
    border-top: 1px solid var(--border); font-size: 11px;
    color: var(--text-tertiary); font-weight: 500; line-height: 1.5;
  }
  .main-content {
    margin-left: 220px; flex: 1; min-width: 0;
  }
  .tab-panel { display: none; }
  .tab-panel.active { display: block; }
  @media (max-width: 700px) {
    .sidebar { display: none; }
    .main-content { margin-left: 0; }
  }

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg: #F8F9FC;
    --white: #FFFFFF;
    --text: #0F172A;
    --text-secondary: #475569;
    --text-tertiary: #94A3B8;
    --border: rgba(0,0,0,0.06);
    --border-strong: rgba(0,0,0,0.1);

    --indigo: #6366F1;
    --violet: #8B5CF6;
    --teal: #14B8A6;
    --emerald: #10B981;
    --amber: #F59E0B;
    --rose: #F43F5E;

    --gradient-main: linear-gradient(135deg, #6366F1, #8B5CF6, #A855F7);
    --gradient-teal: linear-gradient(135deg, #14B8A6, #06B6D4);
    --gradient-warm: linear-gradient(135deg, #F59E0B, #F97316);
    --gradient-rose: linear-gradient(135deg, #F43F5E, #EC4899);

    --shadow-sm: 0 1px 2px rgba(0,0,0,0.04);
    --shadow-md: 0 4px 16px rgba(0,0,0,0.06);
    --shadow-lg: 0 8px 32px rgba(0,0,0,0.08);

    --radius: 16px;
    --radius-sm: 10px;
    --font: 'Inter', -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
    --mono: 'SF Mono', 'Cascadia Code', 'Fira Code', 'JetBrains Mono', monospace;
  }

  body {
    font-family: var(--font);
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
    min-height: 100vh;
    overflow-x: hidden;
  }

  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
  }
  .animate { animation: fadeUp 0.4s ease-out both; }
  .delay-1 { animation-delay: 0.05s; }
  .delay-2 { animation-delay: 0.1s; }
  .delay-3 { animation-delay: 0.15s; }
  .delay-4 { animation-delay: 0.2s; }
  .delay-5 { animation-delay: 0.25s; }

  /* ── PAGE HEADER (inside main content) ── */
  .page-header {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 32px; flex-wrap: wrap; gap: 12px;
  }
  .page-title { font-size: 20px; font-weight: 800; letter-spacing: -0.4px; }
  .page-header-right { display: flex; align-items: center; gap: 10px; }
  .date-badge {
    color: var(--text-tertiary); font-size: 12px; font-weight: 500;
    background: var(--white); padding: 5px 12px; border-radius: 16px;
    border: 1px solid var(--border);
  }
  .refresh-btn {
    background: var(--white); border: 1px solid var(--border);
    color: var(--text-secondary); padding: 6px 14px; border-radius: 16px;
    cursor: pointer; font-size: 12px; font-weight: 500; font-family: var(--font);
    transition: all 0.2s; display: flex; align-items: center; gap: 5px;
  }
  .refresh-btn:hover { border-color: var(--indigo); color: var(--indigo); }
  .refresh-btn svg { width: 13px; height: 13px; }
  .refresh-btn.spinning svg { animation: spin 0.7s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* ── CONTAINER ── */
  .container { padding: 32px 32px 60px; max-width: 1100px; }

  /* ── STAT CARDS ── */
  .stats-row {
    display: grid; grid-template-columns: repeat(4, 1fr);
    gap: 14px; margin-bottom: 28px;
  }
  @media (max-width: 780px) { .stats-row { grid-template-columns: repeat(2, 1fr); } }

  .stat-card {
    background: var(--white); border-radius: var(--radius);
    padding: 20px; position: relative;
    border: 1px solid var(--border);
    box-shadow: var(--shadow-sm);
    transition: box-shadow 0.2s, transform 0.2s;
    overflow: hidden;
  }
  .stat-card:hover { box-shadow: var(--shadow-md); transform: translateY(-1px); }
  .stat-card .accent-bar {
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    border-radius: var(--radius) var(--radius) 0 0;
  }
  .stat-label {
    font-size: 11px; font-weight: 600; color: var(--text-tertiary);
    text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 8px;
    display: flex; align-items: center; justify-content: space-between;
  }
  .stat-value {
    font-size: 30px; font-weight: 800; letter-spacing: -1.5px;
    line-height: 1; margin-bottom: 5px;
  }
  .stat-sub { font-size: 12px; color: var(--text-tertiary); font-weight: 500; }

  /* month picker inside stat card */
  .month-select {
    font-size: 11px; font-weight: 600; color: var(--indigo);
    background: #EEF2FF; border: none; border-radius: 8px;
    padding: 2px 6px; cursor: pointer; font-family: var(--font);
    outline: none; max-width: 110px;
  }

  /* ── SECTION HEADERS ── */
  .section-header {
    display: flex; align-items: center; gap: 10px;
    margin-bottom: 14px;
  }
  .section-icon {
    width: 26px; height: 26px; border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
  }
  .section-icon svg { width: 14px; height: 14px; }
  .section-title { font-size: 15px; font-weight: 700; letter-spacing: -0.3px; }

  /* ── CHARTS GRID ── */
  .charts-grid {
    display: grid; grid-template-columns: 5fr 2fr;
    gap: 14px; margin-bottom: 28px;
  }
  @media (max-width: 800px) { .charts-grid { grid-template-columns: 1fr; } }

  .chart-card {
    background: var(--white); border-radius: var(--radius);
    padding: 22px; border: 1px solid var(--border);
    box-shadow: var(--shadow-sm);
  }
  .chart-card h3 {
    font-size: 12px; font-weight: 700; color: var(--text-tertiary);
    text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 18px;
  }
  canvas { display: block; width: 100% !important; }

  .legend {
    display: flex; gap: 16px; margin-top: 14px;
    font-size: 12px; font-weight: 500; color: var(--text-secondary);
    flex-wrap: wrap;
  }
  .legend-item { display: flex; align-items: center; gap: 7px; }
  .legend-dot { width: 10px; height: 10px; border-radius: 4px; flex-shrink: 0; }

  /* ── MODEL TABLE ── */
  .table-card {
    background: var(--white); border-radius: var(--radius);
    border: 1px solid var(--border); box-shadow: var(--shadow-sm);
    overflow: hidden; margin-bottom: 28px;
  }
  .table-inner { overflow-x: auto; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th {
    text-align: left; padding: 11px 14px;
    font-size: 10px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.6px; color: var(--text-tertiary);
    border-bottom: 1px solid var(--border);
    background: #FAFBFC; white-space: nowrap;
  }
  td { padding: 13px 14px; border-bottom: 1px solid var(--border); vertical-align: middle; }
  tbody tr { transition: background 0.1s; }
  tbody tr:hover td { background: #FAFBFF; }
  tbody tr:last-child td { border-bottom: none; }
  .num {
    text-align: right; font-variant-numeric: tabular-nums;
    font-family: var(--mono); font-size: 12px; font-weight: 600;
    letter-spacing: -0.3px;
  }

  .model-badge {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 3px 10px; border-radius: 20px;
    font-size: 12px; font-weight: 600;
  }
  .model-dot { width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; }
  .model-opus { background: #EEF2FF; color: #3730A3; }
  .model-opus .model-dot { background: #4F46E5; }
  .model-sonnet { background: #ECFDF5; color: #047857; }
  .model-sonnet .model-dot { background: #10B981; }
  .model-haiku { background: #FFF7ED; color: #C2410C; }
  .model-haiku .model-dot { background: #F97316; }

  .total-row td { font-weight: 700; border-top: 2px solid var(--border-strong); background: #FAFBFC; }

  .token-bar-wrap {
    display: flex; gap: 1px; height: 4px; margin-top: 5px;
    border-radius: 4px; overflow: hidden;
  }
  .token-bar-in { background: var(--indigo); height: 100%; }
  .token-bar-cache { background: var(--amber); height: 100%; }
  .token-bar-out { background: var(--teal); height: 100%; }

  /* ── PERIOD CARDS ── */
  .period-grid {
    display: grid; grid-template-columns: repeat(3, 1fr);
    gap: 14px; margin-bottom: 28px;
  }
  @media (max-width: 900px) { .period-grid { grid-template-columns: 1fr; } }

  .period-card {
    background: var(--white); border-radius: var(--radius);
    padding: 20px; border: 1px solid var(--border);
    box-shadow: var(--shadow-sm); position: relative; overflow: hidden;
  }
  .period-card .period-accent {
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    border-radius: var(--radius) var(--radius) 0 0;
  }
  .period-header { display: flex; align-items: center; gap: 10px; margin-bottom: 14px; }
  .period-icon {
    width: 30px; height: 30px; border-radius: 9px;
    display: flex; align-items: center; justify-content: center; flex-shrink: 0;
  }
  .period-icon svg { width: 16px; height: 16px; }
  .period-title { font-size: 13px; font-weight: 700; }
  .period-subtitle { font-size: 11px; color: var(--text-tertiary); font-weight: 500; }
  .period-cost { font-size: 26px; font-weight: 800; letter-spacing: -1.5px; line-height: 1; margin-bottom: 4px; }
  .period-meta { font-size: 11px; color: var(--text-tertiary); font-weight: 500; margin-bottom: 12px; }
  .period-stats { display: flex; gap: 14px; font-size: 12px; color: var(--text-secondary); font-weight: 500; flex-wrap: wrap; }
  .period-stat { display: flex; align-items: center; gap: 4px; }
  .period-stat strong { font-weight: 700; color: var(--text); }
  .period-delta {
    display: inline-flex; align-items: center; gap: 3px;
    font-size: 11px; font-weight: 600; padding: 2px 7px;
    border-radius: 10px; margin-left: 6px;
  }
  .period-delta.up { background: #FEF2F2; color: #DC2626; }
  .period-delta.down { background: #F0FDF4; color: #16A34A; }
  .period-delta.same { background: #F8FAFC; color: var(--text-tertiary); }

  /* reset pill */
  .reset-pill {
    display: inline-flex; align-items: center; gap: 4px;
    background: #FEF2F2; color: #DC2626; border-radius: 10px;
    font-size: 11px; font-weight: 600; padding: 2px 8px; margin-top: 6px;
  }

  /* mini week bars */
  .week-bars { display: flex; align-items: flex-end; gap: 6px; margin-top: 16px; }
  .week-bar-col { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: flex-end; }
  .week-bar-area { width: 100%; height: 120px; display: flex; align-items: flex-end; }
  .week-bar { width: 100%; border-radius: 4px 4px 0 0; min-height: 2px; }
  .week-bar-label { font-size: 9px; font-weight: 600; color: var(--text-tertiary); margin-top: 6px; text-transform: uppercase; }
  .week-bar-cost { font-size: 9px; font-weight: 600; color: var(--text-secondary); margin-bottom: 3px; }

  /* today model list */
  .today-models { margin-top: 10px; }
  .today-model-row {
    display: flex; align-items: center; justify-content: space-between;
    padding: 5px 0; border-bottom: 1px solid var(--border); font-size: 12px;
  }
  .today-model-row:last-child { border-bottom: none; }
  .today-model-cost { font-family: var(--mono); font-weight: 700; font-size: 12px; }

  /* ── PRICING TABLE ── */
  .pricing-card {
    background: var(--white); border-radius: var(--radius);
    border: 1px solid var(--border); box-shadow: var(--shadow-sm);
    overflow: hidden; margin-bottom: 28px;
  }
  .pricing-inner { overflow-x: auto; }
  .pricing-inner table { font-size: 12px; }
  .pricing-inner th { font-size: 10px; }
  .pricing-note {
    padding: 10px 14px; font-size: 11px; color: var(--text-tertiary);
    border-top: 1px solid var(--border); font-weight: 500;
  }

  /* ── SESSION LIST ── */
  .session-card {
    background: var(--white); border-radius: var(--radius);
    border: 1px solid var(--border); box-shadow: var(--shadow-sm);
    padding: 16px 20px; margin-bottom: 10px;
    display: grid; grid-template-columns: 1fr auto;
    gap: 10px 20px; align-items: start;
    transition: box-shadow 0.15s, transform 0.15s;
  }
  .session-card:hover { box-shadow: var(--shadow-md); transform: translateY(-1px); }
  .session-main { min-width: 0; }
  .session-top {
    display: flex; align-items: center; gap: 8px; margin-bottom: 6px; flex-wrap: wrap;
  }
  .session-time { font-size: 13px; font-weight: 700; color: var(--text); }
  .session-id { font-size: 11px; color: var(--text-tertiary); font-family: var(--mono); }
  .session-branch {
    font-size: 11px; color: var(--text-tertiary); font-weight: 500;
    display: inline-flex; align-items: center; gap: 3px;
  }
  .compaction-badge {
    font-size: 10px; font-weight: 700; background: #FFF7ED; color: #C2410C;
    border-radius: 8px; padding: 1px 7px;
  }
  .session-meta {
    display: flex; gap: 16px; font-size: 12px; color: var(--text-secondary);
    font-weight: 500; flex-wrap: wrap; align-items: center;
  }
  .session-meta-item { display: flex; align-items: center; gap: 4px; }
  .session-meta-item svg { width: 12px; height: 12px; opacity: 0.5; flex-shrink: 0; }
  .session-models { display: flex; gap: 4px; flex-wrap: wrap; }
  .session-cost {
    text-align: right; font-size: 20px; font-weight: 800;
    letter-spacing: -1px; color: var(--indigo); white-space: nowrap;
  }
  .session-cost-sub { font-size: 11px; color: var(--text-tertiary); font-weight: 500; text-align: right; }

  /* ── SESSION VIEWER DRAWER ── */
  .drawer-overlay {
    display: none; position: fixed; inset: 0; background: rgba(15,23,42,0.35);
    z-index: 200; backdrop-filter: blur(2px);
  }
  .drawer-overlay.open { display: block; }
  .drawer {
    position: fixed; top: 0; right: -680px; width: 660px; max-width: 100vw;
    height: 100vh; background: var(--white); z-index: 201;
    box-shadow: -8px 0 40px rgba(15,23,42,0.15);
    display: flex; flex-direction: column;
    transition: right 0.25s cubic-bezier(.4,0,.2,1);
  }
  .drawer.open { right: 0; }
  .drawer-header {
    padding: 20px 24px 16px; border-bottom: 1px solid var(--border);
    display: flex; align-items: flex-start; justify-content: space-between; gap: 12px;
    flex-shrink: 0;
  }
  .drawer-title { font-size: 14px; font-weight: 700; color: var(--text); }
  .drawer-meta { font-size: 11px; color: var(--text-tertiary); margin-top: 3px; }
  .drawer-close {
    background: none; border: none; cursor: pointer; color: var(--text-tertiary);
    padding: 4px; border-radius: 6px; display: flex; align-items: center;
    flex-shrink: 0;
  }
  .drawer-close:hover { background: var(--bg); color: var(--text); }
  .drawer-body { flex: 1; overflow-y: auto; padding: 20px 24px; }
  .drawer-loading {
    color: var(--text-tertiary); font-size: 13px; text-align: center; padding: 60px 0;
  }
  .msg-bubble {
    margin-bottom: 20px; max-width: 90%;
  }
  .msg-bubble.user { margin-left: auto; }
  .msg-bubble.assistant { margin-right: auto; }
  .msg-role {
    font-size: 10px; font-weight: 700; letter-spacing: 0.5px; text-transform: uppercase;
    margin-bottom: 4px; color: var(--text-tertiary);
  }
  .msg-bubble.user .msg-role { text-align: right; color: var(--indigo); }
  .msg-bubble.assistant .msg-role { color: var(--teal); }
  .msg-text {
    font-size: 13px; line-height: 1.6; color: var(--text);
    background: var(--bg); border-radius: 12px; padding: 12px 16px;
    white-space: pre-wrap; word-break: break-word;
    border: 1px solid var(--border);
  }
  .msg-bubble.user .msg-text {
    background: #EEF2FF; border-color: #C7D2FE;
  }
  .msg-tool {
    font-size: 11px; color: var(--text-tertiary); font-style: italic;
    background: #F8FAFC; border: 1px solid var(--border);
    border-radius: 8px; padding: 4px 10px; display: inline-block; margin: 2px 0;
  }
  .session-card { cursor: pointer; }

  /* ── FOOTER ── */
  .footer {
    text-align: center; padding: 20px 0;
    color: var(--text-tertiary); font-size: 11px; font-weight: 500;
  }

  /* ── LOADING ── */
  .loading {
    display: flex; align-items: center; justify-content: center;
    min-height: 100vh; flex-direction: column; gap: 20px;
  }
  .spinner {
    width: 32px; height: 32px;
    border: 3px solid var(--border); border-top-color: var(--indigo);
    border-radius: 50%; animation: spin 0.7s linear infinite;
  }
  .loading-text { color: var(--text-secondary); font-size: 14px; font-weight: 500; }

  @media (max-width: 600px) {
    .container { padding: 20px 16px 40px; }
    .stats-row { grid-template-columns: 1fr 1fr; }
    .stat-value { font-size: 24px; }
  }
</style>
</head>
<body>

<div id="loadingState" class="loading">
  <div class="spinner"></div>
  <div class="loading-text">Scanning Claude Code sessions...</div>
</div>

<div id="appRoot" style="display:none;">

<div class="layout">
  <!-- ── SIDEBAR ── -->
  <nav class="sidebar">
    <div class="sidebar-logo">
      <div class="logo-mark">
        <svg viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
          <path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/>
        </svg>
      </div>
      <div class="sidebar-title">Claude Usage Tracker<span>Cost &amp; Token Monitor</span></div>
    </div>
    <div class="nav-group">
      <div class="nav-label">Dashboard</div>
      <button class="nav-item active" onclick="showTab('windows')" id="nav-windows">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
        Usage
      </button>
      <button class="nav-item" onclick="showTab('overview')" id="nav-overview">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg>
        Overview
      </button>
      <button class="nav-item" onclick="showTab('sessions')" id="nav-sessions">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
        Sessions
      </button>
      <button class="nav-item" onclick="showTab('pricing')" id="nav-pricing">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z"/><path d="M12 6v6l4 2"/></svg>
        Pricing
      </button>
    </div>
    <div class="sidebar-footer">
      Auto-refreshes every 30s<br>
      Last update: <span id="lastUpdate">—</span>
    </div>
  </nav>

  <!-- ── MAIN CONTENT ── -->
  <div class="main-content">

    <!-- TAB: Overview -->
    <div id="tab-overview" class="tab-panel">
      <div class="container animate">
        <div class="page-header">
          <div class="page-title">Overview</div>
          <div class="page-header-right">
            <span class="date-badge" id="dateRange"></span>
            <button class="refresh-btn" onclick="doRefresh()" id="refreshBtn">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/>
              </svg>
              Refresh
            </button>
          </div>
        </div>

        <div class="stats-row">
          <div class="stat-card">
            <div class="accent-bar" style="background: var(--gradient-main);"></div>
            <div class="stat-label">
              Monthly Cost
              <select class="month-select" id="monthSelect" onchange="updateMonthCost()"></select>
            </div>
            <div class="stat-value" id="totalCost" style="color: var(--indigo);">—</div>
            <div class="stat-sub" id="totalCostSub"></div>
          </div>
          <div class="stat-card">
            <div class="accent-bar" style="background: var(--gradient-teal);"></div>
            <div class="stat-label">Today's Cost</div>
            <div class="stat-value" id="todayCost" style="color: var(--teal);">—</div>
            <div class="stat-sub" id="todayDate"></div>
          </div>
          <div class="stat-card">
            <div class="accent-bar" style="background: var(--gradient-warm);"></div>
            <div class="stat-label">Total Sessions</div>
            <div class="stat-value" id="totalSessions" style="color: var(--amber);">—</div>
            <div class="stat-sub">conversations</div>
          </div>
          <div class="stat-card">
            <div class="accent-bar" style="background: var(--gradient-rose);"></div>
            <div class="stat-label">Total Messages</div>
            <div class="stat-value" id="totalMessages" style="color: var(--rose);">—</div>
            <div class="stat-sub">API calls</div>
          </div>
        </div>

        <!-- Summary donut + period cards side-by-side -->
        <div class="charts-grid">
          <div class="chart-card">
            <h3>Daily Estimated Cost (last 30 days)</h3>
            <canvas id="dailyChart" height="180"></canvas>
            <div class="legend">
              <div class="legend-item"><span class="legend-dot" style="background:var(--indigo);"></span> Daily est. cost</div>
            </div>
          </div>
          <div class="chart-card">
            <h3>Cost by Model</h3>
            <canvas id="modelDonut" height="200"></canvas>
            <div id="donutLegend" class="legend" style="flex-direction:column; gap:8px; margin-top:16px;"></div>
          </div>
        </div>

        <!-- Token breakdown table -->
        <div class="table-card" style="margin-bottom:28px;">
          <h3 style="font-size:13px;font-weight:700;color:var(--text);margin:0 0 16px;">Token Breakdown by Model (all time)</h3>
          <div class="table-inner">
            <table>
              <thead>
                <tr>
                  <th>Model</th>
                  <th class="num">Input</th>
                  <th class="num">Output</th>
                  <th class="num">Cache Read</th>
                  <th class="num">Cache Write</th>
                  <th class="num">Est. Cost</th>
                  <th style="width:100px;">Mix</th>
                </tr>
              </thead>
              <tbody id="modelTable"></tbody>
            </table>
          </div>
        </div>

        <div class="footer">Auto-refreshes every 30s</div>
      </div>
    </div>

    <!-- TAB: Usage Windows -->
    <div id="tab-windows" class="tab-panel active">
      <div class="container animate">
        <div class="page-header">
          <div class="page-title">Usage</div>
        </div>

        <div class="period-grid">
          <!-- 5-hour rolling -->
          <div class="period-card">
            <div class="period-accent" style="background: var(--gradient-rose);"></div>
            <div class="period-header">
              <div class="period-icon" style="background: linear-gradient(135deg, #FFE4E6, #FECDD3);">
                <svg viewBox="0 0 24 24" fill="none" stroke="#E11D48" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                  <circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>
                </svg>
              </div>
              <div>
                <div class="period-title">5-Hour Window</div>
                <div class="period-subtitle" id="fiveHWindow">—</div>
              </div>
            </div>
            <div class="period-cost" id="fiveHCost" style="color: var(--rose);">—</div>
            <div class="period-meta">Enterprise rate limit window</div>
            <div id="fiveHReset"></div>
            <div class="period-stats" style="margin-top:10px;">
              <div class="period-stat"><strong id="fiveHConvs">0</strong> conversations</div>
              <div class="period-stat"><strong id="fiveHMsgs">0</strong> messages</div>
              <div class="period-stat"><strong id="fiveHTok">0</strong> in+out tokens</div>
            </div>
            <div class="today-models" id="fiveHModels"></div>
          </div>

          <!-- Today -->
          <div class="period-card">
            <div class="period-accent" style="background: var(--gradient-teal);"></div>
            <div class="period-header">
              <div class="period-icon" style="background: linear-gradient(135deg, #CCFBF1, #99F6E4);">
                <svg viewBox="0 0 24 24" fill="none" stroke="#0D9488" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                  <rect x="3" y="4" width="18" height="18" rx="2"/><path d="M16 2v4"/><path d="M8 2v4"/><path d="M3 10h18"/>
                </svg>
              </div>
              <div>
                <div class="period-title">Today</div>
                <div class="period-subtitle" id="todayLabel"></div>
              </div>
            </div>
            <div class="period-cost" id="todayDetailCost" style="color: var(--teal);">—</div>
            <div class="period-meta" id="todayDetailMeta"></div>
            <div class="period-stats" style="margin-top:10px;">
              <div class="period-stat"><strong id="todayConvs">0</strong> conversations</div>
              <div class="period-stat"><strong id="todayMsgs">0</strong> messages</div>
              <div class="period-stat"><strong id="todayTok">0</strong> in+out tokens</div>
            </div>
            <div class="today-models" id="todayModels"></div>
          </div>

          <!-- This week -->
          <div class="period-card">
            <div class="period-accent" style="background: var(--gradient-main);"></div>
            <div class="period-header">
              <div class="period-icon" style="background: linear-gradient(135deg, #E0E7FF, #C7D2FE);">
                <svg viewBox="0 0 24 24" fill="none" stroke="#4F46E5" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                  <rect x="3" y="4" width="18" height="18" rx="2"/><path d="M16 2v4"/><path d="M8 2v4"/><path d="M3 10h18"/>
                  <path d="M8 14h.01"/><path d="M12 14h.01"/><path d="M16 14h.01"/><path d="M8 18h.01"/><path d="M12 18h.01"/>
                </svg>
              </div>
              <div>
                <div class="period-title">This Week</div>
                <div class="period-subtitle" id="weekLabel"></div>
              </div>
            </div>
            <div class="period-cost" id="weekCost" style="color: var(--indigo);">—</div>
            <div class="period-meta" id="weekMeta"></div>
            <div class="week-bars" id="weekBars"></div>
          </div>
        </div>
        <div class="footer"></div>
      </div>
    </div>

    <!-- TAB: Sessions -->
    <div id="tab-sessions" class="tab-panel">
      <div class="container animate">
        <div class="page-header">
          <div class="page-title">Sessions</div>
          <div class="page-header-right">
            <input type="text" id="sessionSearch" placeholder="Search sessions…"
              oninput="filterSessions()"
              style="font-size:12px;font-family:var(--font);padding:6px 14px;border-radius:16px;border:1px solid var(--border);outline:none;background:var(--white);color:var(--text);width:220px;">
          </div>
        </div>
        <div id="sessionsList">
          <div style="color:var(--text-tertiary);font-size:13px;padding:40px 0;text-align:center;">Loading sessions…</div>
        </div>
        <div class="footer"></div>
      </div>
    </div>

    <!-- TAB: Pricing -->
    <div id="tab-pricing" class="tab-panel">
      <div class="container animate">
        <div class="page-header">
          <div class="page-title">Pricing Reference</div>
        </div>
        <div class="pricing-card">
          <div class="pricing-inner">
            <table>
              <thead>
                <tr>
                  <th>Model</th>
                  <th class="num">Input / 1M</th>
                  <th class="num">Output / 1M</th>
                  <th class="num">Cache Read / 1M</th>
                  <th class="num">Cache Write / 1M</th>
                </tr>
              </thead>
              <tbody id="pricingTable"></tbody>
            </table>
          </div>
          <div class="pricing-note">Prices in USD. Cache write refers to prompt cache creation tokens. Rates as of March 2026.</div>
        </div>
        <div class="footer"></div>
      </div>
    </div>

  </div><!-- /main-content -->
</div><!-- /layout -->
</div><!-- /appRoot -->

<script>
// ── Per-model distinct colors (not grouped by type) ──
const MODEL_PALETTE = [
  '#6366F1', '#10B981', '#F97316', '#06B6D4', '#A855F7',
  '#F43F5E', '#14B8A6', '#F59E0B', '#3B82F6', '#84CC16',
];
const _modelColorMap = {};
let _modelColorIdx = 0;
function modelColor(m) {
  if (!_modelColorMap[m]) {
    _modelColorMap[m] = MODEL_PALETTE[_modelColorIdx % MODEL_PALETTE.length];
    _modelColorIdx++;
  }
  return _modelColorMap[m];
}

// Pricing table data (mirrors backend PRICING dict)
const PRICING_DATA = [
  { model: 'claude-opus-4-6',           input: 15.0,  output: 75.0, cache_read: 1.875, cache_write: 18.75 },
  { model: 'claude-opus-4-5-20251101',  input: 15.0,  output: 75.0, cache_read: 1.875, cache_write: 18.75 },
  { model: 'claude-sonnet-4-6',         input: 3.0,   output: 15.0, cache_read: 0.30,  cache_write: 3.75  },
  { model: 'claude-sonnet-4-5-20250929',input: 3.0,   output: 15.0, cache_read: 0.30,  cache_write: 3.75  },
  { model: 'claude-haiku-4-5-20251001', input: 0.8,   output: 4.0,  cache_read: 0.08,  cache_write: 1.0   },
];

function fmt(n) {
  if (n >= 1e9) return (n / 1e9).toFixed(1) + 'B';
  if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return n.toLocaleString();
}

function fmtCost(n) {
  if (n >= 1000) return '$' + n.toLocaleString('en-US', {minimumFractionDigits: 0, maximumFractionDigits: 0});
  return '$' + n.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2});
}

function modelType(m) {
  if (m.includes('opus')) return 'opus';
  if (m.includes('sonnet')) return 'sonnet';
  if (m.includes('haiku')) return 'haiku';
  return 'opus';
}

function shortModel(m) {
  return m.replace('claude-', '').replace(/-\d{8,}$/, '');
}

function modelBadge(m) {
  const type = modelType(m);
  return '<span class="model-badge model-' + type + '">' +
    '<span class="model-dot"></span>' + shortModel(m) + '</span>';
}

// ── TABS ──
function showTab(name) {
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  document.getElementById('nav-' + name).classList.add('active');
  // redraw charts if switching to those tabs
  if (name === 'overview' && _lastData) {
    drawBarChart('dailyChart', _lastData.dailyCost, false);
    drawDonut('modelDonut', _lastData.models);
  }
  if (name === 'sessions') loadSessions();
}

let _lastData = null;
let _monthlyCost = {};

function updateMonthCost() {
  const sel = document.getElementById('monthSelect');
  const month = sel.value;
  const cost = _monthlyCost[month] || 0;
  document.getElementById('totalCost').textContent = fmtCost(cost);
  document.getElementById('totalCostSub').textContent = month;
}

// Shared bar chart tooltip
const _chartTooltip = (() => {
  const el = document.createElement('div');
  el.style.cssText = 'position:fixed;pointer-events:none;display:none;background:#1E293B;color:#F8FAFC;' +
    'font:600 12px Inter,system-ui,sans-serif;padding:6px 12px;border-radius:8px;' +
    'box-shadow:0 4px 16px rgba(0,0,0,0.25);white-space:nowrap;z-index:999;';
  document.body.appendChild(el);
  return el;
})();

function drawBarChart(canvasId, daily, allTime) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;

  const rect = canvas.parentElement.getBoundingClientRect();
  const w = rect.width - 48;
  const h = allTime ? 220 : 180;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  canvas.style.width = w + 'px';
  canvas.style.height = h + 'px';
  ctx.scale(dpr, dpr);

  const data = allTime ? daily : daily.slice(-30);
  if (!data.length) return;

  const maxCost = Math.max(...data.map(x => x.cost), 1);
  const barW = Math.max(3, Math.min(20, (w - 40) / data.length - 2));
  const gap = 2;
  const chartH = h - 30;
  const startX = 40;

  // Store bar hit regions for tooltip
  const bars = [];

  ctx.fillStyle = '#94A3B8';
  ctx.font = '10px Inter, system-ui, sans-serif';
  ctx.textAlign = 'right';
  for (let i = 0; i <= 4; i++) {
    const y = chartH - (i / 4) * (chartH - 10);
    const val = maxCost * i / 4;
    ctx.fillText('$' + (val >= 1000 ? Math.round(val/1000) + 'k' : Math.round(val)), startX - 6, y + 3);
    ctx.strokeStyle = 'rgba(0,0,0,0.04)';
    ctx.beginPath(); ctx.moveTo(startX, y); ctx.lineTo(w, y); ctx.stroke();
  }

  data.forEach((d, i) => {
    const x = startX + i * (barW + gap);
    const barH2 = Math.max(1, (d.cost / maxCost) * (chartH - 10));
    const y = chartH - barH2;
    const grad = ctx.createLinearGradient(x, y, x, chartH);
    grad.addColorStop(0, '#6366F1');
    grad.addColorStop(1, '#8B5CF6');
    ctx.fillStyle = grad;
    ctx.beginPath();
    const r = Math.min(3, barW / 2);
    ctx.moveTo(x + r, y); ctx.lineTo(x + barW - r, y);
    ctx.quadraticCurveTo(x + barW, y, x + barW, y + r);
    ctx.lineTo(x + barW, chartH); ctx.lineTo(x, chartH); ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y); ctx.fill();
    bars.push({ x, y, w: barW, h: barH2, date: d.date, cost: d.cost });
    const showLabel = data.length <= 15 || i % Math.ceil(data.length / 10) === 0;
    if (showLabel) {
      ctx.fillStyle = '#94A3B8';
      ctx.font = '9px Inter, system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(d.date.slice(5), x + barW / 2, chartH + 14);
    }
  });

  // Attach/replace mousemove handler
  canvas._bars = bars;
  canvas.onmousemove = (e) => {
    const cr = canvas.getBoundingClientRect();
    const mx = (e.clientX - cr.left) * (canvas.width / cr.width) / dpr;
    const my = (e.clientY - cr.top) * (canvas.height / cr.height) / dpr;
    const hit = canvas._bars.find(b => mx >= b.x && mx <= b.x + b.w && my >= b.y && my <= b.y + b.h);
    if (hit) {
      _chartTooltip.textContent = hit.date + '  ' + fmtCost(hit.cost);
      _chartTooltip.style.display = 'block';
      _chartTooltip.style.left = (e.clientX + 12) + 'px';
      _chartTooltip.style.top  = (e.clientY - 32) + 'px';
      canvas.style.cursor = 'crosshair';
    } else {
      _chartTooltip.style.display = 'none';
      canvas.style.cursor = '';
    }
  };
  canvas.onmouseleave = () => { _chartTooltip.style.display = 'none'; };
}

function drawDonut(canvasId, models) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const size = 200;
  canvas.width = size * dpr;
  canvas.height = size * dpr;
  canvas.style.width = size + 'px';
  canvas.style.height = size + 'px';
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, size, size);

  const total = models.reduce((s, m) => s + m.cost, 0);
  const cx = size / 2, cy = size / 2, r = 80, inner = 52;
  const GAP = 0.025; // radians gap between slices
  const MIN_ARC = 0.06; // minimum arc so tiny slices are visible

  // Use distinct per-model colors
  const colors = models.map(m => modelColor(m.model));

  if (!total || models.length === 0) {
    // Empty state: draw grey ring
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.arc(cx, cy, inner, Math.PI * 2, 0, true);
    ctx.closePath();
    ctx.fillStyle = '#E2E8F0';
    ctx.fill();
  } else {
    // Compute raw arcs with minimum size, then scale to fill 2π
    const rawArcs = models.map(m => Math.max(MIN_ARC, (m.cost / total) * Math.PI * 2));
    const rawSum = rawArcs.reduce((s, a) => s + a, 0);
    const scale = (Math.PI * 2 - GAP * models.length) / rawSum;
    const arcs = rawArcs.map(a => a * scale);

    let startAngle = -Math.PI / 2;
    models.forEach((m, i) => {
      const arc = arcs[i];
      const start = startAngle + GAP / 2;
      const end = start + arc;

      ctx.beginPath();
      ctx.arc(cx, cy, r, start, end);
      ctx.arc(cx, cy, inner, end, start, true);
      ctx.closePath();
      ctx.fillStyle = colors[i];
      ctx.fill();

      startAngle += arc + GAP;
    });
  }

  // Center text
  const displayTotal = total || 0;
  ctx.fillStyle = '#0F172A';
  ctx.font = '700 18px Inter, system-ui, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(fmtCost(displayTotal), cx, cy - 7);
  ctx.fillStyle = '#94A3B8';
  ctx.font = '500 10px Inter, system-ui, sans-serif';
  ctx.fillText('TOTAL COST', cx, cy + 10);
  ctx.textBaseline = 'alphabetic';

  const legendEl = document.getElementById('donutLegend');
  if (legendEl) {
    legendEl.innerHTML = models.map((m, i) => {
      const pct = total ? ((m.cost / total) * 100).toFixed(1) : '0.0';
      return '<div class="legend-item">' +
        '<span class="legend-dot" style="background:' + colors[i] + ';border-radius:50%;flex-shrink:0;"></span>' +
        '<span>' + shortModel(m.model) + ' — ' + fmtCost(m.cost) + ' <span style="color:#94A3B8">(' + pct + '%)</span></span>' +
        '</div>';
    }).join('');
  }
}

function buildPricingTable() {
  const tbody = document.getElementById('pricingTable');
  if (!tbody) return;
  tbody.innerHTML = PRICING_DATA.map(p => {
    const type = modelType(p.model);
    return '<tr>' +
      '<td>' + modelBadge(p.model) + '</td>' +
      '<td class="num">$' + p.input.toFixed(2) + '</td>' +
      '<td class="num">$' + p.output.toFixed(2) + '</td>' +
      '<td class="num">$' + p.cache_read.toFixed(3) + '</td>' +
      '<td class="num">$' + p.cache_write.toFixed(3) + '</td>' +
      '</tr>';
  }).join('');
}

async function refresh(showSpinner) {
  try {
    if (showSpinner) {
      document.getElementById('refreshBtn').classList.add('spinning');
    }
    const resp = await fetch('/api/data');
    const d = await resp.json();
    _lastData = d;
    _monthlyCost = d.monthlyCost || {};

    document.getElementById('loadingState').style.display = 'none';
    document.getElementById('appRoot').style.display = 'block';

    // Month picker
    const sel = document.getElementById('monthSelect');
    const months = Object.keys(_monthlyCost).sort().reverse();
    const currentMonth = d.today.slice(0, 7);
    const prevSel = sel.value || currentMonth;
    sel.innerHTML = months.map(m =>
      '<option value="' + m + '"' + (m === prevSel ? ' selected' : '') + '>' + m + '</option>'
    ).join('');
    updateMonthCost();

    document.getElementById('todayCost').textContent = fmtCost(d.todayCost);
    document.getElementById('todayDate').textContent = d.today;
    document.getElementById('totalSessions').textContent = d.totalSessions.toLocaleString();
    document.getElementById('totalMessages').textContent = d.totalMessages.toLocaleString();
    document.getElementById('dateRange').textContent = d.firstDate + ' → ' + d.today;
    document.getElementById('lastUpdate').textContent = new Date().toLocaleTimeString();

    // Model table
    const tbody = document.getElementById('modelTable');
    let rows = '';
    let totals = {input: 0, output: 0, cache_read: 0, cache_write: 0, cost: 0};
    for (const m of d.models) {
      totals.input += m.input;
      totals.output += m.output;
      totals.cache_read += m.cache_read;
      totals.cache_write += m.cache_write;
      totals.cost += m.cost;
      const total_t = m.input + m.output + m.cache_read + m.cache_write || 1;
      rows += '<tr>' +
        '<td>' + modelBadge(m.model) + '</td>' +
        '<td class="num">' + fmt(m.input) + '</td>' +
        '<td class="num">' + fmt(m.output) + '</td>' +
        '<td class="num">' + fmt(m.cache_read) + '</td>' +
        '<td class="num">' + fmt(m.cache_write) + '</td>' +
        '<td class="num" style="font-weight:700;">' + fmtCost(m.cost) + '</td>' +
        '<td><div class="token-bar-wrap">' +
          '<div class="token-bar-in" style="width:' + (m.input / total_t * 100) + '%"></div>' +
          '<div class="token-bar-cache" style="width:' + ((m.cache_read + m.cache_write) / total_t * 100) + '%"></div>' +
          '<div class="token-bar-out" style="width:' + (m.output / total_t * 100) + '%"></div>' +
        '</div></td>' +
        '</tr>';
    }
    const total_all = totals.input + totals.output + totals.cache_read + totals.cache_write || 1;
    rows += '<tr class="total-row">' +
      '<td>TOTAL</td>' +
      '<td class="num">' + fmt(totals.input) + '</td>' +
      '<td class="num">' + fmt(totals.output) + '</td>' +
      '<td class="num">' + fmt(totals.cache_read) + '</td>' +
      '<td class="num">' + fmt(totals.cache_write) + '</td>' +
      '<td class="num" style="color:var(--indigo);">' + fmtCost(totals.cost) + '</td>' +
      '<td><div class="token-bar-wrap">' +
        '<div class="token-bar-in" style="width:' + (totals.input / total_all * 100) + '%"></div>' +
        '<div class="token-bar-cache" style="width:' + ((totals.cache_read + totals.cache_write) / total_all * 100) + '%"></div>' +
        '<div class="token-bar-out" style="width:' + (totals.output / total_all * 100) + '%"></div>' +
      '</div></td>' +
      '</tr>';
    tbody.innerHTML = rows;

    // --- 5-hour rolling window ---
    const fh = d.fiveHour;
    document.getElementById('fiveHCost').textContent = fmtCost(fh.cost);
    document.getElementById('fiveHWindow').textContent = fh.window;
    document.getElementById('fiveHConvs').textContent = fh.conversations.toLocaleString();
    document.getElementById('fiveHMsgs').textContent = fh.messages.toLocaleString();
    document.getElementById('fiveHTok').textContent = fmt(fh.tokens);
    const resetEl = document.getElementById('fiveHReset');
    if (fh.resetAt) {
      resetEl.innerHTML = '<div class="reset-pill">↺ Resets at ' + fh.resetAt + ' (London)</div>';
    } else {
      resetEl.innerHTML = '<div class="reset-pill" style="background:#F0FDF4;color:#16A34A;">✓ Window clear</div>';
    }
    const fiveHModelsEl = document.getElementById('fiveHModels');
    if (fh.byModel && fh.byModel.length) {
      fiveHModelsEl.innerHTML = fh.byModel.map(m =>
        '<div class="today-model-row">' +
          '<span>' + modelBadge(m.model) + '</span>' +
          '<span class="today-model-cost">' + fmtCost(m.cost) + '</span>' +
        '</div>'
      ).join('');
    } else {
      fiveHModelsEl.innerHTML = '<div style="font-size:12px;color:var(--text-tertiary);padding:8px 0;">No usage in this window</div>';
    }

    // --- Today detail ---
    const td = d.todayDetail;
    document.getElementById('todayLabel').textContent = d.today;
    document.getElementById('todayDetailCost').textContent = fmtCost(td.cost);
    document.getElementById('todayDetailMeta').textContent = fmtCost(td.cost) + ' spent today';
    document.getElementById('todayConvs').textContent = td.conversations.toLocaleString();
    document.getElementById('todayMsgs').textContent = td.messages.toLocaleString();
    document.getElementById('todayTok').textContent = fmt(td.tokens);
    const todayModelsEl = document.getElementById('todayModels');
    if (td.byModel.length) {
      todayModelsEl.innerHTML = td.byModel.map(m =>
        '<div class="today-model-row">' +
          '<span>' + modelBadge(m.model) + '</span>' +
          '<span class="today-model-cost">' + fmtCost(m.cost) + '</span>' +
        '</div>'
      ).join('');
    } else {
      todayModelsEl.innerHTML = '<div style="font-size:12px;color:var(--text-tertiary);padding:8px 0;">No usage yet today</div>';
    }

    // --- This week ---
    const wk = d.week;
    document.getElementById('weekLabel').textContent = 'Week of ' + wk.weekStart;
    document.getElementById('weekCost').textContent = fmtCost(wk.cost);
    let weekMetaText = wk.messages.toLocaleString() + ' messages · ' + fmt(wk.tokens) + ' tokens';
    if (wk.prevWeekCost > 0) {
      const delta = wk.cost - wk.prevWeekCost;
      const pct = ((delta / wk.prevWeekCost) * 100).toFixed(0);
      const cls = delta > 0 ? 'up' : delta < 0 ? 'down' : 'same';
      const arrow = delta > 0 ? '↑' : delta < 0 ? '↓' : '→';
      weekMetaText += '<span class="period-delta ' + cls + '">' + arrow + ' ' + Math.abs(pct) + '% vs last week</span>';
    }
    document.getElementById('weekMeta').innerHTML = weekMetaText;

    // Week mini bar chart
    const weekBarsEl = document.getElementById('weekBars');
    const maxDay = Math.max(...wk.daily.map(x => x.cost), 1);
    const dayColors = {Mon:'#6366F1',Tue:'#8B5CF6',Wed:'#A855F7',Thu:'#14B8A6',Fri:'#06B6D4',Sat:'#94A3B8',Sun:'#94A3B8'};
    const barMaxH = 120; // px, matches .week-bar-area height
    weekBarsEl.innerHTML = wk.daily.map(x => {
      const h = Math.max(2, Math.round((x.cost / maxDay) * barMaxH));
      return '<div class="week-bar-col">' +
        '<div class="week-bar-cost">' + (x.cost > 0 ? '$' + x.cost.toFixed(0) : '') + '</div>' +
        '<div class="week-bar-area">' +
          '<div class="week-bar" style="height:' + h + 'px;background:' + (dayColors[x.day] || '#94A3B8') + ';"></div>' +
        '</div>' +
        '<div class="week-bar-label">' + x.day + '</div>' +
        '</div>';
    }).join('');

    // Charts (only draw if tab is visible)
    const activeTab = document.querySelector('.tab-panel.active');
    if (activeTab && activeTab.id === 'tab-overview') {
      drawBarChart('dailyChart', d.dailyCost, false);
      drawDonut('modelDonut', d.models);
    }

    buildPricingTable();

  } catch (e) {
    console.error('Refresh failed:', e);
  } finally {
    document.getElementById('refreshBtn').classList.remove('spinning');
  }
}

// ── SESSIONS ──
let _allSessions = null;

function fmtDuration(s) {
  if (s < 60) return s + 's';
  if (s < 3600) return Math.floor(s / 60) + 'm ' + (s % 60) + 's';
  return Math.floor(s / 3600) + 'h ' + Math.floor((s % 3600) / 60) + 'm';
}

function timeAgo(isoStr) {
  const diff = (Date.now() - new Date(isoStr)) / 1000;
  if (diff < 60) return 'just now';
  if (diff < 3600) return Math.floor(diff / 60) + 'm ago';
  if (diff < 86400) return Math.floor(diff / 3600) + 'h ago';
  if (diff < 604800) return Math.floor(diff / 86400) + 'd ago';
  return new Date(isoStr).toLocaleDateString('en-US', {month:'short', day:'numeric', year:'numeric'});
}

function renderSessions(sessions) {
  const el = document.getElementById('sessionsList');
  if (!sessions.length) {
    el.innerHTML = '<div style="color:var(--text-tertiary);font-size:13px;padding:40px 0;text-align:center;">No sessions found</div>';
    return;
  }
  el.innerHTML = sessions.map(s => {
    const modelBadges = s.models.map(m => modelBadge(m)).join('');
    const branch = s.gitBranch && s.gitBranch !== 'HEAD' && s.gitBranch !== ''
      ? '<span class="session-branch"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="6" y1="3" x2="6" y2="15"/><circle cx="18" cy="6" r="3"/><circle cx="6" cy="18" r="3"/><path d="M18 9a9 9 0 0 1-9 9"/></svg>' + s.gitBranch.slice(0,30) + '</span>'
      : '';
    const compaction = s.hasCompaction ? '<span class="compaction-badge">compacted</span>' : '';
    const fmtLondon = iso => new Date(iso).toLocaleString('en-GB', {month:'short',day:'numeric',hour:'2-digit',minute:'2-digit',timeZone:'Europe/London'});
    const endTime = s.endTime || s.startTime;
    const startLabel = fmtLondon(s.startTime);
    const endLabel = fmtLondon(endTime);
    const drawerMeta = startLabel + ' → ' + endLabel + ' · ' + fmtDuration(s.duration) + ' · ' + (s.userMessages + s.assistantMessages) + ' messages · ' + fmtCost(s.cost);
    return '<div class="session-card" onclick="openDrawer(\'' + s.id + '\',\'Session ' + s.id.slice(0,8) + '\',\'' + drawerMeta.replace(/'/g, "\\'") + '\')">' +
      '<div class="session-main">' +
        '<div class="session-top">' +
          '<span class="session-time">' + timeAgo(endTime) + '</span>' +
          '<span class="session-id">' + s.id.slice(0, 8) + '</span>' +
          branch + compaction +
        '</div>' +
        '<div class="session-models" style="margin-bottom:8px;">' + modelBadges + '</div>' +
        '<div class="session-meta">' +
          '<div class="session-meta-item"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>' + fmtDuration(s.duration) + '</div>' +
          '<div class="session-meta-item"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>' + (s.userMessages + s.assistantMessages) + ' msgs</div>' +
          '<div class="session-meta-item"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>' + s.toolCalls + ' tools</div>' +
          '<div class="session-meta-item"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/></svg>' + fmt(s.tokens) + ' tokens</div>' +
          '<div class="session-meta-item" style="color:var(--text-tertiary);font-size:11px;">' + startLabel + ' → ' + endLabel + '</div>' +
        '</div>' +
      '</div>' +
      '<div>' +
        '<div class="session-cost">' + fmtCost(s.cost) + '</div>' +
        '<div class="session-cost-sub">' + fmt(s.output) + ' out</div>' +
      '</div>' +
    '</div>';
  }).join('');
}

function filterSessions() {
  if (!_allSessions) return;
  const q = document.getElementById('sessionSearch').value.trim().toLowerCase();
  if (!q) { renderSessions(_allSessions); return; }
  renderSessions(_allSessions.filter(s =>
    s.id.toLowerCase().includes(q) ||
    s.models.some(m => m.toLowerCase().includes(q)) ||
    (s.gitBranch && s.gitBranch.toLowerCase().includes(q)) ||
    s.startTime.includes(q)
  ));
}

async function loadSessions() {
  try {
    const resp = await fetch('/api/sessions');
    _allSessions = await resp.json();
    filterSessions();
  } catch (e) {
    document.getElementById('sessionsList').innerHTML =
      '<div style="color:var(--rose);font-size:13px;padding:20px 0;">Failed to load sessions</div>';
  }
}

function doRefresh() { refresh(true); }

refresh(false);
setInterval(() => refresh(false), 30000);
window.addEventListener('resize', () => {
  if (_lastData) {
    const activeTab = document.querySelector('.tab-panel.active');
    if (!activeTab) return;
    if (activeTab.id === 'tab-overview') { drawBarChart('dailyChart', _lastData.dailyCost, false); drawDonut('modelDonut', _lastData.models); }
  }
});

// ── SESSION VIEWER DRAWER ──
function openDrawer(sessionId, title, meta) {
  document.getElementById('drawerTitle').textContent = title;
  document.getElementById('drawerMeta').textContent = meta;
  document.getElementById('drawerBody').innerHTML = '<div class="drawer-loading">Loading messages…</div>';
  document.getElementById('drawerOverlay').classList.add('open');
  document.getElementById('drawer').classList.add('open');
  document.body.style.overflow = 'hidden';
  fetch('/api/session/' + encodeURIComponent(sessionId))
    .then(r => r.json())
    .then(msgs => renderDrawerMessages(msgs))
    .catch(() => {
      document.getElementById('drawerBody').innerHTML = '<div class="drawer-loading" style="color:var(--rose)">Failed to load messages</div>';
    });
}

function closeDrawer() {
  document.getElementById('drawerOverlay').classList.remove('open');
  document.getElementById('drawer').classList.remove('open');
  document.body.style.overflow = '';
}

function renderDrawerMessages(msgs) {
  const el = document.getElementById('drawerBody');
  if (!msgs.length) {
    el.innerHTML = '<div class="drawer-loading">No messages found</div>';
    return;
  }
  el.innerHTML = msgs.map(m => {
    // Split tool lines from regular text
    const lines = m.text.split('\n');
    const html = lines.map(l => {
      if (l.startsWith('[Tool: ') && l.endsWith(']')) {
        return '<span class="msg-tool">' + escHtml(l) + '</span>';
      }
      return escHtml(l);
    }).join('\n');
    return '<div class="msg-bubble ' + m.role + '">' +
      '<div class="msg-role">' + (m.role === 'user' ? 'You' : 'Claude') + '</div>' +
      '<div class="msg-text">' + html + '</div>' +
      '</div>';
  }).join('');
  el.scrollTop = 0;
}

function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
</script>

<!-- Session viewer drawer -->
<div class="drawer-overlay" id="drawerOverlay" onclick="closeDrawer()"></div>
<div class="drawer" id="drawer">
  <div class="drawer-header">
    <div>
      <div class="drawer-title" id="drawerTitle">Session</div>
      <div class="drawer-meta" id="drawerMeta"></div>
    </div>
    <button class="drawer-close" onclick="closeDrawer()" title="Close">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
    </button>
  </div>
  <div class="drawer-body" id="drawerBody"></div>
</div>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/api/session/"):
            session_id = self.path[len("/api/session/"):]
            msgs = load_session_messages(session_id)
            body = json.dumps(msgs).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/sessions":
            data = scan_sessions()
            body = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/data":
            data = get_data()
            body = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/" or self.path == "/index.html":
            body = HTML_PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        # Quieter logging — only errors
        if args and "404" in str(args[0]):
            super().log_message(format, *args)


def main():
    parser = argparse.ArgumentParser(description="Claude Code Spend Tracker")
    parser.add_argument("--port", type=int, default=3456, help="Port to serve on (default: 3456)")
    parser.add_argument("--no-open", action="store_true", help="Don't auto-open browser")
    args = parser.parse_args()

    # Quick data check
    print("Loading data...")
    data = get_data()
    print(f"  Models: {len(data['models'])} | Messages: {data['totalMessages']:,}")
    print(f"  Total est. cost: ${data['totalCost']:,.2f}")
    print(f"  Today's est. cost: ${data['todayCost']:,.2f}")
    print()

    server = HTTPServer(("0.0.0.0", args.port), Handler)
    print(f"Serving on http://0.0.0.0:{args.port}")
    print(f"  → Port forward: ssh -L {args.port}:localhost:{args.port} <host>")
    print(f"  → Then open: http://localhost:{args.port}")
    print("Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


if __name__ == "__main__":
    main()
