"""Colour registry for Sandra's non-3WC figures (reasoning-transfer project, VLM).

Colours are keyed by MEANING, never by position. Every plotting script imports from here;
no script defines its own hex list. Register a new model / category here first, then plot.
Model hues follow Vasco's rteval dashboard (viz/build_gpqa_dashboard.py): red = hurts,
blue = helps, and the same display names.

    python plot_style.py   # prints the index
"""

from __future__ import annotations

# --- models: display name + colour, keyed by rteval model key ---------------------------
MODEL_LABEL = {
    "qwen9": "Qwen3.5-9B",
    "qwen27": "Qwen3.6-27B",
    "gemma12": "Gemma-4-12B",
    "nemotron4": "Nemotron-4B",
    "glm52": "GLM-5.2",
    "glm53": "GLM-5.3",
}
MODEL_COLOR = {
    "qwen27": "#1c5cab",     # Qwen family = blues; the larger model is the darker one
    "qwen9": "#5aa0e6",
    "gemma12": "#eb6834",
    "nemotron4": "#2e9e5b",
    "glm52": "#17a2b8",
    "glm53": "#17a2b8",
}

# --- sign semantics for transfer deltas (Vasco's validated diverging ramp) ----------------
NEG = "#b52c2b"   # negative delta = the trace HURT the receiver
POS = "#1c5cab"   # positive delta = the trace HELPED
DIVERGING_CMAP = "RdBu"   # matplotlib name; centre at zero, red negative, blue positive

# --- W1 arms -----------------------------------------------------------------------------
ARM_COLOR = {
    "after": "#7d1a1a",    # prefix INCLUDES the verified error node
    "before": "#86b6ef",   # prefix stops just short of it
    "unused": "#8a8a8a",   # error node not on the path to the conclusion
}

# --- W2 behaviours (node, edge, node triplets) --------------------------------------------
BEHAVIOUR_LABEL = {
    "abandon_and_replan": "Repair: abandon line, re-plan",
    "abandon_plan": "Repair: abandon plan",
    "flag_only": "Flag only (note a problem, move on)",
    "self_challenge": "Self-challenge (attack own step)",
    "retract_conclusion": "Retract a conclusion",
}
BEHAVIOUR_COLOR = {
    "abandon_and_replan": "#2e7d32",
    "abandon_plan": "#7cb342",
    "flag_only": "#c62828",
    "self_challenge": "#f9a825",
    "retract_conclusion": "#757575",
}

# --- source-trace correctness ------------------------------------------------------------
SOURCE_CORRECT_COLOR = {"source_right": "#1c5cab", "source_wrong": "#b52c2b"}
SOURCE_CORRECT_LABEL = {"source_right": "Source trace answered right",
                        "source_wrong": "Source trace answered wrong"}

# --- stage-4 node verdicts ----------------------------------------------------------------
VERDICT_COLOR = {"correct": "#86b6ef", "error": "#b52c2b", "unverifiable": "#bdbdbd", "other": "#e0e0e0"}

FOOT = {"fontsize": 7.5, "color": "#6b6b6b", "ha": "left", "va": "bottom"}
NOISE_BAND = "#d9d9d9"

if __name__ == "__main__":
    for k, v in MODEL_COLOR.items():
        print(f"{k:12s} {MODEL_LABEL[k]:14s} {v}")
    for k, v in BEHAVIOUR_COLOR.items():
        print(f"{k:22s} {BEHAVIOUR_LABEL[k]:40s} {v}")
