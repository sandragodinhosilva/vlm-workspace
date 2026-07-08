"""
Shared prev/next/random/jump-to-index navigation for Gradio browse apps.

Extracted from browser-app/shared/components.py + image_row_viewer.py's
filter-scoped nav (2026-07-08 homogenization pass — see GRADIO_APPS_REPORT.md
and the /apps skill's "common features" section). Two usage shapes:

1. Precomputed filtered-index tuple (browser-app style): resolve a `pos` in
   [0, len(filtered)) to an absolute dataset index via `resolve_index()`.
2. Recompute-on-demand filter predicate (image-viewer style): pass a
   `matches(i) -> bool` callable to `filtered_indices()` each call, for apps
   where the filter set is cheap to recompute and there's no separate
   "apply filter" step.

Pure functions + component factories only — no gr.State wiring, no IO.
Caller owns state and wires .click()/.change() events.
"""

import random
from typing import Callable, Optional, Sequence, Tuple

import gradio as gr


# ---------------------------------------------------------------------------
# Component factories
# ---------------------------------------------------------------------------

def make_nav_row() -> Tuple[gr.Button, gr.Button, gr.Button, gr.Button, gr.Markdown]:
    """Returns (prev_btn, next_btn, random_btn, refresh_btn, counter_md)."""
    with gr.Row():
        prev_btn = gr.Button("◄ Prev", size="sm")
        with gr.Column(scale=3, min_width=200):
            counter_md = gr.Markdown("No samples loaded")
        next_btn = gr.Button("Next ►", size="sm")
        random_btn = gr.Button("Random", size="sm", variant="secondary")
        refresh_btn = gr.Button("⟳ Refresh", size="sm", variant="secondary")
    return prev_btn, next_btn, random_btn, refresh_btn, counter_md


def make_jump_row(label: str = "Jump to #") -> Tuple[gr.Number, gr.Button]:
    """Jump-to-index control (1-based input)."""
    with gr.Row():
        jump_input = gr.Number(label=label, minimum=1, value=1, precision=0)
        jump_btn = gr.Button("Go", size="sm")
    return jump_input, jump_btn


# ---------------------------------------------------------------------------
# Navigation logic — precomputed filtered-index tuple
# ---------------------------------------------------------------------------

def navigate(
    total: int,
    filtered: Optional[Sequence[int]],
    pos: int,
    delta: Optional[int],
) -> int:
    """
    Compute new position after navigation.
    total: total samples in dataset (used when filtered is None)
    filtered: sequence of matching indices, or None (all samples)
    pos: current position
    delta: +1 next, -1 prev, None=random
    Returns new pos (clamped, or random if delta is None).
    """
    n = max(0, len(filtered) if filtered is not None else total)
    if n == 0:
        return 0
    if delta is None:
        return random.randrange(n)
    return max(0, min(pos + delta, n - 1))


def resolve_index(filtered: Optional[Sequence[int]], pos: int) -> int:
    """Map filtered position to absolute dataset index."""
    if filtered is not None and pos < len(filtered):
        return filtered[pos]
    return pos


def get_total(filtered: Optional[Sequence[int]], dataset_total: int) -> int:
    """Effective total count (filtered or full)."""
    return len(filtered) if filtered is not None else dataset_total


# ---------------------------------------------------------------------------
# Navigation logic — recompute-on-demand filter predicate
# ---------------------------------------------------------------------------

def filtered_indices(total: int, matches: Callable[[int], bool]) -> list:
    """Every absolute index in [0, total) for which matches(i) is True."""
    return [j for j in range(total) if matches(j)]


def step_filtered(current: int, delta: int, sel: Sequence[int]) -> int:
    """
    Move +1/-1 within a filtered index subset `sel`, from the subset member
    nearest to `current` (handles `current` not itself being in `sel`).
    Falls back to `current` unchanged if `sel` is empty (an empty filter
    result should hold position, not jump — unlike random_filtered, which
    has no meaningful "hold" and instead falls back to the full range).

    NOTE: "nearest, then step" can skip a match when `current` sits between
    two entries of `sel` (e.g. sel=[5,10,15], current=12, delta=-1 → nearest
    is 10, one step back lands on 5, skipping 10). If the app's semantics
    need "the first match strictly after/before current" instead (e.g. a
    Prev/Next-Match button where the current index is often NOT itself a
    filter match), use next_or_prev_match() below.
    """
    if not sel:
        return current
    nearest = min(range(len(sel)), key=lambda k: abs(sel[k] - current))
    new_pos = max(0, min(len(sel) - 1, nearest + delta))
    return sel[new_pos]


def next_or_prev_match(current: int, delta: int, sel: Sequence[int]) -> int:
    """
    Find the first entry in sorted `sel` strictly after (delta>0) or before
    (delta<0) `current`, clamping to the nearest end if none exists. Unlike
    step_filtered, this never skips an entry that lies between `current` and
    the direction of travel — correct for "Prev/Next Match" style buttons
    where `current` is frequently not itself a member of `sel` (e.g. it was
    set by a free-typed row index or a different filter's last position).
    Returns `current` unchanged if `sel` is empty.
    """
    if not sel:
        return current
    if delta > 0:
        for m in sel:
            if m > current:
                return m
        return sel[-1]
    for m in reversed(sel):
        if m < current:
            return m
    return sel[0]


def random_filtered(sel: Sequence[int], fallback_total: int) -> int:
    """Random index from `sel`, or from range(fallback_total) if sel is empty."""
    pool = sel or list(range(fallback_total))
    return random.choice(pool)


# ---------------------------------------------------------------------------
# Counter formatting — image-viewer's richer format is the house convention
# ---------------------------------------------------------------------------

def format_counter(pos: int, total: int, scope: Optional[str] = None) -> str:
    """
    Plain "Sample i / N" counter (browser-app style). Use format_scoped_counter
    instead when the app has a filter dropdown — the scope name is real signal.
    """
    if total == 0:
        return "No samples loaded"
    return f"Sample **{pos + 1:,}** / **{total:,}**"


def format_scoped_counter(
    row_idx: int, total_rows: int, filtered_pos: int, filtered_total: int,
    scope: Optional[str] = None,
) -> str:
    """
    House convention: "**pos / N** (scope) · row index **i** of total".
    filtered_pos is 1-based position within the filtered subset;
    row_idx/total_rows are the absolute dataset index/count.
    """
    scope_label = "all" if not scope or scope in ("(all)", "None") else f"`{scope}`"
    return (
        f"**{filtered_pos} / {filtered_total}** ({scope_label}) · "
        f"row index **{row_idx}** of {total_rows}"
    )
