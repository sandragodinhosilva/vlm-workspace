#!/usr/bin/env python3
"""
VLM Pose Estimation Pipeline Monitoring App
============================================

Comprehensive Gradio web application for monitoring:
- Dataset creation (SFT datasets)
- Training progress (checkpoints, metrics)
- Evaluation results (per-sample analysis, comparisons)

Features:
- Per-image lineage tracking
- Checkpoint performance comparison
- Interactive visualizations
- Pre-generated report display

Usage:
    python monitoring_app.py --port 7861 --share

Author: Generated with Claude Code
Date: 2026-02-02
"""

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from functools import lru_cache
from datetime import datetime
import logging
import yaml

# Third-party imports
import gradio as gr
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import plotly.graph_objects as go

# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================

REPO_ROOT = Path(__file__).resolve().parent
HOME_ROOT = REPO_ROOT.parent
VLM_EVAL_ROOT = HOME_ROOT / "vlm-evaluation"
SFT_DATA_ROOT = HOME_ROOT / "sft-data-vlm"

# Central config — override any path via environment variables
CONFIG = {
    # Paths (all overridable via env vars)
    "datasets_base_path": Path(os.environ.get("MONITOR_DATASETS_PATH", "/mnt/data/shared/vlm/data/image_aux_datasets")),
    "models_base_path": Path(os.environ.get("MONITOR_MODELS_PATH", "/mnt/data/sgsilva/models")),
    "results_base_path": Path(os.environ.get("MONITOR_RESULTS_PATH", str(VLM_EVAL_ROOT / "results" / "final"))),
    "evaluations_path": Path(os.environ.get("MONITOR_EVALUATIONS_PATH", str(VLM_EVAL_ROOT / "results" / "evaluations"))),
    "experiments_csv_path": Path(os.environ.get("MONITOR_EXPERIMENTS_CSV", str(VLM_EVAL_ROOT / "experiments-final.csv"))),
    "visualizations_path": Path(os.environ.get("MONITOR_VISUALIZATIONS_PATH", str(VLM_EVAL_ROOT / "results" / "visualizations"))),
    "reasoning_data_path": Path(os.environ.get("MONITOR_REASONING_DATA", "/mnt/data/shared/vlm/data/reasoning_data")),
    "reasoning_tests_path": Path(os.environ.get("MONITOR_REASONING_TESTS", "/mnt/data/sgsilva/reasoning_traces_tests")),
    "experiments_path": Path(os.environ.get("MONITOR_EXPERIMENTS_PATH", str(SFT_DATA_ROOT / "experiments"))),

    # Feature flags — set True to show hidden tabs
    "show_evaluation_dashboard": False,
    "show_benchmarks_eval": False,
    "show_mixed_vs_single": False,

    # Tuning
    "gallery_page_size": 50,
    "server_port": 7861,
}

# Backward-compat aliases (existing code references these directly)
DATASETS_BASE_PATH = CONFIG["datasets_base_path"]
MODELS_BASE_PATH = CONFIG["models_base_path"]
RESULTS_BASE_PATH = CONFIG["results_base_path"]
EVALUATIONS_PATH = CONFIG["evaluations_path"]
EXPERIMENTS_CSV_PATH = CONFIG["experiments_csv_path"]
VISUALIZATIONS_PATH = CONFIG["visualizations_path"]
REASONING_DATA_PATH = CONFIG["reasoning_data_path"]
REASONING_TESTS_PATH = CONFIG["reasoning_tests_path"]
EXPERIMENTS_PATH = CONFIG["experiments_path"]
REASONING_TASKS = ["task1", "task2", "task3a", "task3b", "task3c", "task3d", "task4_v5.3", "task4_v6.2"]

# Fallback: v2 test files in reasoning_traces_tests/ (used when full dataset not available)
_REASONING_TEST_FILE_MAP = {
    "task1": "task1_v2_test",
    "task2": "task2_v2_test",
    "task3a": "task3a_v2_test",
    "task3b": "task3b_v2_test",
    "task3c": "task3c_v2_test",
    "task3d": "task3d_v2_test",
    "task4_v5.3": "task4_v53_v2_test",
    "task4_v6.2": "task4_v62_v2_test",
}

# Global mapping from model name to (task, variant) - populated from experiments CSV
MODEL_TO_TASK_VARIANT = {}

# --- Experiment Archive ---
ARCHIVE_PATH = Path(__file__).parent / "archived_experiments.json"
_ARCHIVED_EXPERIMENTS: set[tuple[str, str]] = set()  # {(task, variant), ...}

def _load_archive():
    """Load archived experiments from JSON file."""
    global _ARCHIVED_EXPERIMENTS
    try:
        if ARCHIVE_PATH.exists():
            data = json.loads(ARCHIVE_PATH.read_text())
            _ARCHIVED_EXPERIMENTS = {(e["task"], e["variant"]) for e in data.get("archived", [])}
    except Exception as e:
        logging.warning(f"Failed to load archive: {e}")
        _ARCHIVED_EXPERIMENTS = set()

def _save_archive():
    """Persist archived experiments to JSON file."""
    data = {"archived": [{"task": t, "variant": v} for t, v in sorted(_ARCHIVED_EXPERIMENTS)]}
    ARCHIVE_PATH.write_text(json.dumps(data, indent=2) + "\n")

MIXED_DATASETS_PATH = Path(__file__).parent / "mixed_datasets.yaml"

def _load_mixed_datasets_md() -> str:
    """Load mixed dataset descriptions from YAML and render as compact markdown.
    Supports top-level 'active'/'archived' sections. Only 'active' is shown."""
    try:
        if MIXED_DATASETS_PATH.exists():
            data = yaml.safe_load(MIXED_DATASETS_PATH.read_text())
            if not data:
                return "*No mixed datasets configured.*"
            entries = data.get("active", data) if isinstance(data, dict) else {}
            if not isinstance(entries, dict):
                return "*Invalid mixed_datasets.yaml format.*"
            blocks = []
            for name, info in entries.items():
                if not isinstance(info, dict):
                    continue
                tasks = info.get("tasks", "")
                if isinstance(tasks, list):
                    tasks = ", ".join(str(t) for t in tasks)
                tasks_display = tasks.replace(", ", " · ")
                notes = info.get("notes", "")
                block = f"**{name}**"
                if notes:
                    block += f" — *{notes}*"
                block += f"\n{tasks_display}"
                blocks.append(block)
            return "\n\n".join(blocks) if blocks else "*No active mixed datasets.*"
    except Exception as e:
        logging.warning(f"Failed to load mixed_datasets.yaml: {e}")
    return "*Edit mixed_datasets.yaml to add descriptions.*"

def is_archived(task: str, variant: str) -> bool:
    if (task, variant) in _ARCHIVED_EXPERIMENTS:
        return True
    # Mixed variants may be archived under 'mixed' but appear under individual tasks (task1, task2, etc.)
    if variant.startswith('mixed_') and ('mixed', variant) in _ARCHIVED_EXPERIMENTS:
        return True
    # Global archive: ("*", variant) hides a variant across ALL tasks (e.g., qwen3-vl-4b-baseline)
    if ('*', variant) in _ARCHIVED_EXPERIMENTS:
        return True
    return False

def get_active_variants(task: str, all_variants: list[str]) -> list[str]:
    """Filter out archived variants for a given task."""
    return [v for v in all_variants if not is_archived(task, v)]

def get_all_known_variants(task: str) -> list[str]:
    """Get all variants for a task from BOTH filesystem (DATASET_INDEX) and CSV (EXPERIMENT_INDEX).
    Used by Archive Manager so CSV-only variants (e.g. v4.4_qwen_baseline) can be archived too.
    Pass task='*' to get ALL variants across all tasks (for global archival)."""
    if task == '*':
        # Global: collect all variants from all tasks
        variants: set[str] = set()
        for t_variants in DATASET_INDEX.values():
            variants.update(t_variants.keys())
        if EXPERIMENT_INDEX is not None:
            variants.update(EXPERIMENT_INDEX['dataset_variant'].unique())
        return sorted(variants)

    variants = set(DATASET_INDEX.get(task, {}).keys())
    if EXPERIMENT_INDEX is not None:
        # CSV uses original task names (task1c, task4, etc.)
        # For merged tasks (task1 includes task1b/task1c), scan subtasks too
        subtask_prefixes = [task]
        for child, (parent, _) in _SUBTASK_MERGE.items():
            if parent == task:
                subtask_prefixes.append(child)
        for prefix in subtask_prefixes:
            mask = EXPERIMENT_INDEX['task'] == prefix
            csv_variants = set(EXPERIMENT_INDEX[mask]['dataset_variant'].unique())
            variants.update(csv_variants)
    return sorted(variants)

_load_archive()

# Visualization constants
BASE_KEYPOINT_RADIUS = 4
BASE_SKELETON_THICKNESS = 2
COLOR_GROUND_TRUTH = (0, 255, 0)  # Green in BGR
COLOR_PREDICTED = (0, 0, 255)    # Red in BGR
COLOR_BASELINE = (128, 128, 128)  # Gray in BGR
COLOR_BEST = (255, 128, 0)       # Blue in BGR

# Keypoint subsets and skeletons
# KEYPOINT_SUBSETS defined later (after skeleton constants, ~line 691)

# Task display names — known tasks get friendly names; new tasks auto-discovered at startup
_TASK_DISPLAY_OVERRIDES = {
    'task1': 'Task 1: Keypoint Detection',
    'task2': 'Task 2: Keypoint Labeling',
    'task3a': 'Task 3a: Label Error Detection',
    'task3b': 'Task 3b: Missing Keypoint Detection',
    'task3c': 'Task 3c: Displaced Keypoint Correction',
    'task3d': 'Task 3d: Combined Error Correction',
    'task4': 'Task 4: Exercise Description MCQA',
    'mixed': 'Mixed Tasks: Multi-Task Evaluation',
}
TASK_NAMES = dict(_TASK_DISPLAY_OVERRIDES)

# Sub-tasks that should be merged into their parent as variants (not shown as separate tasks).
# Value: (parent_key, prefix_with_child_key)
#   prefix=True:  task1b/cropped_v1 → task1/task1b_cropped_v1 (avoids name collision)
#   prefix=False: mixed_tasks/mixed_v3 → mixed/mixed_v3 (names are already unique)
_SUBTASK_MERGE = {
    'task1b': ('task1', True),
    'task1c': ('task1', True),
    'task4a': ('task4', True),
    'task4b': ('task4', True),
    'mixed_tasks': ('mixed', False),
}

# Variant name normalization for mixed task datasets.
# Directory names → CSV-compatible names (prefix "mixed_" if missing)
def _normalize_mixed_variant(variant: str) -> str:
    """Ensure mixed variant names match CSV convention (mixed_ prefix)."""
    if not variant.startswith('mixed_') and not variant.startswith('balanced_'):
        return variant
    if variant.startswith('balanced_'):
        return f"mixed_{variant}"
    return variant

def _sync_task_names():
    """Add any newly discovered tasks from DATASET_INDEX to TASK_NAMES.

    Sub-tasks listed in _SUBTASK_MERGE are folded into their parent task
    with prefixed variant names (e.g. task4a/v2 → task4/task4a_v2).
    """
    merge_pending = []
    for task_key in list(DATASET_INDEX.keys()):
        entry = _SUBTASK_MERGE.get(task_key)
        if entry:
            merge_pending.append((task_key, entry[0], entry[1]))
            continue
        if task_key not in TASK_NAMES:
            TASK_NAMES[task_key] = task_key.replace('_', ' ').title()
            logging.info(f"Auto-discovered new task: {task_key}")

    # Merge sub-tasks into parent
    for child_key, parent_key, use_prefix in merge_pending:
        if parent_key not in DATASET_INDEX:
            DATASET_INDEX[parent_key] = {}
        for variant, data in DATASET_INDEX[child_key].items():
            if use_prefix:
                merged_name = f"{child_key}_{variant}"
            elif parent_key == 'mixed':
                # Normalize mixed variant names to match CSV convention
                merged_name = _normalize_mixed_variant(variant)
            else:
                merged_name = variant
            # Avoid overwriting existing variant with same name
            if merged_name in DATASET_INDEX[parent_key]:
                merged_name = f"{child_key}_{variant}"
            DATASET_INDEX[parent_key][merged_name] = data
            logging.info(f"Merged {child_key}/{variant} → {parent_key}/{merged_name}")
        del DATASET_INDEX[child_key]

# Mixed variant configurations: task columns, metrics, and steps per epoch
MIXED_VARIANT_CONFIGS = {
    'mixed_balanced_v1': {
        'steps_per_epoch': 315,
        'tasks': {
            'task1': ('oks_score', 'T1 OKS'),
            'task2': ('per_keypoint_accuracy', 'T2 Acc'),
            'task3': ('oks_score', 'T3 OKS'),
            'task4': ('accuracy', 'T4 Acc'),
        },
    },
    'mixed_v2_phase1': {
        'steps_per_epoch': 843,
        'tasks': {
            'task1': ('oks_score', 'T1 OKS'),
            'task1b': ('oks_score', 'T1b OKS'),
            'task1c': ('oks_score', 'T1c OKS'),
            'task3b': ('oks_score', 'T3b OKS'),
            'task3c': ('oks_score', 'T3c OKS'),
        },
    },
    'mixed_v2_phase2': {
        'steps_per_epoch': 928,
        'tasks': {
            'task1': ('oks_score', 'T1 OKS'),
            'task1b': ('oks_score', 'T1b OKS'),
            'task1c': ('oks_score', 'T1c OKS'),
            'task2': ('per_keypoint_accuracy', 'T2 Acc'),
            'task3b': ('oks_score', 'T3b OKS'),
            'task3c': ('oks_score', 'T3c OKS'),
            'task3d': ('oks_score', 'T3d OKS'),
            'task4': ('accuracy', 'T4 Acc'),
        },
    },
    'mixed_v3': {
        'steps_per_epoch': 1125,
        'tasks': {
            'task1c': ('oks_score', 'T1c OKS'),
            'task2': ('per_keypoint_accuracy', 'T2 Acc'),
            'task3b': ('oks_score', 'T3b OKS'),
            'task3c': ('oks_score', 'T3c OKS'),
            'task3d': ('oks_score', 'T3d OKS'),
            'task4': ('accuracy', 'T4 Acc'),
        },
    },
    'mixed_final_a': {
        'steps_per_epoch': 1125,
        'tasks': {
            'task1': ('oks_score', 'T1 OKS'),
            'task2': ('per_keypoint_accuracy', 'T2 Acc'),
            'task3a': ('f1_score', 'T3a F1'),
            'task3b': ('oks_score', 'T3b OKS'),
            'task3c': ('oks_score', 'T3c OKS'),
            'task3d': ('oks_score', 'T3d OKS'),
            'task4': ('accuracy', 'T4 Acc'),
        },
    },
    'mixed_final_b': {
        'steps_per_epoch': 562,
        'tasks': {
            'task1': ('oks_score', 'T1 OKS'),
            'task2': ('per_keypoint_accuracy', 'T2 Acc'),
            'task3a': ('f1_score', 'T3a F1'),
            'task3b': ('oks_score', 'T3b OKS'),
            'task3c': ('oks_score', 'T3c OKS'),
            'task3d': ('oks_score', 'T3d OKS'),
            'task4': ('accuracy', 'T4 Acc'),
        },
    },
    'mixed_v4_weighted': {
        'steps_per_epoch': 1125,
        'tasks': {
            'task1c': ('oks_score', 'T1c OKS'),
            'task2': ('per_keypoint_accuracy', 'T2 Acc'),
            'task3b': ('oks_score', 'T3b OKS'),
            'task3c': ('oks_score', 'T3c OKS'),
            'task3d': ('oks_score', 'T3d OKS'),
            'task4': ('accuracy', 'T4 Acc'),
        },
    },
    'mixed_v5': {
        'steps_per_epoch': 1352,
        'tasks': {
            'task1c': ('oks_score', 'T1c OKS'),
            'task2': ('per_keypoint_accuracy', 'T2 Acc'),
            'task3d': ('oks_score', 'T3d OKS'),
            'task4': ('accuracy', 'T4 Acc'),
        },
    },
    'mixed_v6': {
        'steps_per_epoch': 987,
        'tasks': {
            'task1c': ('oks_score', 'T1c OKS'),
            'task2': ('per_keypoint_accuracy', 'T2 Acc'),
            'task3d': ('oks_score', 'T3d OKS'),
        },
    },
    'mixed_v7': {
        'steps_per_epoch': 987,
        'tasks': {
            'task1c': ('oks_score', 'T1c OKS'),
            'task2': ('per_keypoint_accuracy', 'T2 Acc'),
            'task3d': ('oks_score', 'T3d OKS'),
        },
    },
    'mixed_balanced_v2': {
        'steps_per_epoch': 244,
        'tasks': {
            'task1': ('oks_score', 'T1 OKS'),
            'task1b': ('oks_score', 'T1b OKS'),
            'task1c': ('oks_score', 'T1c OKS'),
            'task2': ('per_keypoint_accuracy', 'T2 Acc'),
            'task3b': ('f1_score', 'T3b F1'),
            'task3c': ('f1_score', 'T3c F1'),
            'task3d': ('f1_score', 'T3d F1'),
            'task4': ('accuracy', 'T4 Acc'),
        },
    },
}

def get_mixed_config(variant: str) -> dict:
    """Get mixed variant config, with auto-detection fallback."""
    if variant in MIXED_VARIANT_CONFIGS:
        return MIXED_VARIANT_CONFIGS[variant]
    # Fallback: detect tasks from CSV data
    if EXPERIMENT_INDEX is not None:
        rows = EXPERIMENT_INDEX[EXPERIMENT_INDEX['dataset_variant'] == variant]
        if not rows.empty:
            tasks = {}
            for t in sorted(rows['task'].unique()):
                if t.startswith('task1'):
                    tasks[t] = ('oks_score', f'{t.replace("task", "T").upper()} OKS')
                elif t == 'task2':
                    tasks[t] = ('per_keypoint_accuracy', 'T2 Acc')
                elif t.startswith('task3'):
                    tasks[t] = ('f1_score', f'{t.replace("task", "T").upper()} F1')
                elif t.startswith('task4'):
                    tasks[t] = ('accuracy', 'T4 Acc')
            return {'steps_per_epoch': 323, 'tasks': tasks}
    return MIXED_VARIANT_CONFIGS.get('mixed_balanced_v1', {'steps_per_epoch': 323, 'tasks': {}})

# Global caches (Tier 1: Startup Cache)
DATASET_INDEX = {}
VALIDATOR_INDEX = {}  # {image_id: {validator_name: {status, issues, corrected_description, ...}}}
CONFUSION_FLAGS = {}  # {(image_id, template): {max_risk, flags}} — 2D/3D projection confusion flags
MODEL_INDEX = {}
EXPERIMENT_INDEX = None
BENCHMARKS_INDEX = None  # IFEval and SIBench benchmark results


def normalize_variant_aliases(variant: str) -> List[str]:
    """Return list of variant name aliases for CSV matching.

    Filesystem names may differ from experiments-final.csv names, e.g.:
    - mcqa_v6.1_qwen (filesystem) → mcqa_v6.1 (CSV)
    - mcqa_v4.4_kimi (filesystem) → v4.4_kimi_baseline or mcqa_v4.4 (CSV)
    """
    if not variant:
        return []
    aliases = [variant]
    # Strip _qwen/_kimi suffixes
    base = re.sub(r'_(qwen|kimi)$', '', variant)
    if base != variant:
        aliases.append(base)
    return aliases

# Generation prompt YAML files per MCQA variant (prefix match)
PROMPT_CONFIGS_DIR = SFT_DATA_ROOT / "prompts" / "task4"
VARIANT_PROMPT_MAP = {
    'mcqa_v1': 'v1_mcqa.yaml',
    'mcqa_v2': 'v2_mcqa.yaml',
    'mcqa_v3': 'v3_mcqa_generation.yaml',
    'mcqa_v4.1': 'v4.1_mcqa_generation.yaml',
    'mcqa_v4.2': 'v4.2_mcqa_generation.yaml',
    'mcqa_v4.3': 'v4.2_mcqa_generation.yaml',  # Same prompt, different filter
    'mcqa_v4.4': 'v4.4_mcqa.yaml',
    'mcqa_v6.1': 'v6.1_mcqa.yaml',
    # V6.2 is deterministic (no LLM prompt)
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

TRAIN_GLOBAL_BATCH_SIZE = 16  # From training configs

def extract_step(model_name):
    """Extract step number from model name."""
    step_match = re.search(r'step[_-]?(\d+)', model_name.lower())
    return int(step_match.group(1)) if step_match else 0


def steps_per_epoch(variant: str = None) -> int:
    """Compute steps per epoch for a variant from dataset sample count.
    Falls back to 323 if variant info unavailable."""
    if variant and DATASET_INDEX:
        for task_data in DATASET_INDEX.values():
            if variant in task_data:
                train_samples = task_data[variant].get('train_samples', 0)
                if train_samples > 0:
                    return max(1, train_samples // TRAIN_GLOBAL_BATCH_SIZE)
    return 323  # Fallback (V1: 5756 / 16 ≈ 360, but 323 was the original default)

# =============================================================================
# THEME & STYLING
# =============================================================================

# Custom theme — warm, Claude-inspired palette
custom_theme = gr.themes.Soft(
    primary_hue="amber",
    secondary_hue="stone",
    neutral_hue="stone",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
    text_size="md",
    spacing_size="md",
    radius_size="lg"
).set(
    body_background_fill="#FAF9F6",
    input_background_fill="white",
    block_background_fill="white",
    body_text_color="#292524",
    panel_background_fill="white",
    block_border_width="1px",
    block_border_color="#E7E5E4",
    block_shadow="0 1px 2px 0 rgba(0,0,0,0.04)",
    button_primary_background_fill="#D97706",
    button_primary_background_fill_hover="#B45309",
    button_primary_text_color="white",
    button_secondary_background_fill="#F5F5F4",
    button_secondary_background_fill_hover="#E7E5E4",
    button_secondary_text_color="#44403C",
)

# Custom CSS — warm, Claude-inspired design
custom_css = """
/* Typography */
.gradio-container {
    font-family: 'Inter', ui-sans-serif, system-ui, -apple-system, sans-serif !important;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

h1, h2, h3, h4 {
    font-weight: 700 !important;
    color: #1C1917 !important;
    letter-spacing: -0.02em;
}

h1 { font-size: 1.75rem !important; }
h3 { font-size: 1.1rem !important; }

/* Tabs */
.tab-nav button {
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    padding: 0.6rem 1.2rem !important;
    border-radius: 10px 10px 0 0 !important;
    transition: all 0.15s ease;
}

.tab-nav button.selected {
    font-weight: 600 !important;
    color: #B45309 !important;
    border-bottom: 2px solid #D97706 !important;
}

/* Sidebar */
.gr-column:first-child {
    background: #FAF9F6;
    border-right: 1px solid #E7E5E4;
}

/* Panels & cards */
.gr-panel, .gr-box {
    background: white !important;
    border-radius: 12px !important;
    border: 1px solid #E7E5E4 !important;
    box-shadow: 0 1px 2px 0 rgba(0,0,0,0.04) !important;
}

/* Tables */
table {
    font-size: 0.85rem !important;
    border-collapse: separate !important;
    border-spacing: 0 !important;
}

table th {
    background: #F5F5F4 !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    font-size: 0.75rem !important;
    letter-spacing: 0.05em;
    color: #57534E !important;
    padding: 0.6rem 0.8rem !important;
}

table td {
    padding: 0.5rem 0.8rem !important;
    border-bottom: 1px solid #F5F5F4 !important;
}

table tr:hover td {
    background: #FAFAF9 !important;
}

/* Highlight best checkpoint */
.best-checkpoint {
    background: #FFFBEB !important;
    font-weight: 600;
}

/* Stats cards */
.stat-card {
    background: white;
    padding: 1rem;
    border-radius: 12px;
    margin: 0.5rem 0;
    border: 1px solid #E7E5E4;
}

/* Accordions */
.gr-accordion {
    border-radius: 12px !important;
    border: 1px solid #E7E5E4 !important;
    overflow: hidden;
}

/* Markdown code blocks */
pre, code {
    font-family: 'IBM Plex Mono', ui-monospace, monospace !important;
    font-size: 0.85rem !important;
    background: #FAFAF9 !important;
    border-radius: 8px !important;
}

/* Dropdowns & inputs */
input, textarea, select, .gr-input {
    border-radius: 8px !important;
    border: 1px solid #D6D3D1 !important;
    font-size: 0.875rem !important;
    transition: border-color 0.15s ease;
}

input:focus, textarea:focus, select:focus {
    border-color: #D97706 !important;
    box-shadow: 0 0 0 2px rgba(217,119,6,0.15) !important;
}

/* Gallery images */
.gr-gallery img {
    border-radius: 8px !important;
    transition: transform 0.15s ease;
}

.gr-gallery img:hover {
    transform: scale(1.02);
}

/* Scrollbar styling */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #D6D3D1; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #A8A29E; }

/* JSON display */
.json-container {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.8rem !important;
}
"""

# =============================================================================
# CACHE MANAGEMENT
# =============================================================================

_ALL_CACHES = []

def cacheable(maxsize=128):
    """Decorator that registers the LRU cache for bulk invalidation."""
    def decorator(fn):
        cached_fn = lru_cache(maxsize=maxsize)(fn)
        _ALL_CACHES.append(cached_fn)
        return cached_fn
    return decorator

def clear_all_caches():
    """Clear every registered LRU cache and in-memory caches."""
    global _TRAINING_NOTES_CACHE
    for fn in _ALL_CACHES:
        fn.cache_clear()
    _TRAINING_NOTES_CACHE = None
    logging.info(f"Cleared {len(_ALL_CACHES)} caches + training notes")

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def empty_figure(message: str = "No data available", height: int = 350) -> go.Figure:
    """Create a blank Plotly figure with a centered message."""
    fig = go.Figure()
    fig.add_annotation(text=message, xref="paper", yref="paper",
                       x=0.5, y=0.5, showarrow=False, font=dict(size=14, color="gray"))
    fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False),
                      plot_bgcolor='white', paper_bgcolor='white', height=height)
    return fig

def safe_load_json(file_path: Path) -> Optional[Dict]:
    """Safely load JSON file, return None on error."""
    try:
        if not file_path.exists():
            return None
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.warning(f"Failed to load JSON from {file_path}: {e}")
        return None


def safe_load_text(file_path: Path) -> Optional[str]:
    """Safely load text file, return None on error."""
    try:
        if not file_path.exists():
            return None
        with open(file_path, 'r') as f:
            return f.read()
    except Exception as e:
        logging.warning(f"Failed to load text from {file_path}: {e}")
        return None


def parse_checkpoint_name(name: str) -> Tuple[str, str, int]:
    """
    Extract task, variant, step from checkpoint name.

    Example:
        "4b_4epochs_task1b_cropped-step320" → ("task1b", "cropped_v1", 320)

    Args:
        name: Checkpoint directory name

    Returns:
        Tuple of (task, variant, step)
    """
    # Extract task (task1, task1b, task2, etc.)
    task_match = re.search(r'task\d+[a-d]?', name)
    task = task_match.group() if task_match else "unknown"

    # Extract variant (cropped, original, etc.)
    variant = "unknown"

    # Try to infer variant from evaluation results if not in name
    if variant == "unknown":
        # Check if evaluation results exist that can tell us the variant
        task_for_lookup = task if task != "unknown" else None
        step_for_lookup = None
        step_match = re.search(r'step(\d+)', name)
        if step_match:
            step_for_lookup = int(step_match.group(1))

        if task_for_lookup and step_for_lookup:
            # Look for evaluation results that match this task/step
            results_pattern = f"{task_for_lookup}_*_test_step{step_for_lookup}_*.json"
            matches = list(RESULTS_BASE_PATH.glob(results_pattern))
            if matches:
                # Extract variant from result filename
                # Format: task1_cropped_v1_test_step646_timestamp.json
                result_filename = matches[0].name
                parts = result_filename.split('_')
                if len(parts) >= 3:
                    # variant is between task and "test"
                    variant_parts = []
                    for i, part in enumerate(parts[1:], 1):
                        if part == 'test':
                            break
                        variant_parts.append(part)
                    if variant_parts:
                        variant = '_'.join(variant_parts)

    if 'cropped' in name.lower():
        # Handle task2 visualized_cropped variants
        if 'visualized_cropped' in name.lower():
            # Extract version number
            version_match = re.search(r'v(\d+)', name)
            version = version_match.group(1) if version_match else "1"
            variant = f'visualized_cropped_v{version}'
        # Handle HuggingFace variants (cropped_v1_hf, cropped_v2_hf, etc.)
        elif '_hf' in name.lower() or 'hf' in name.lower():
            # Extract version number if present
            version_match = re.search(r'v(\d+)', name)
            version = version_match.group(1) if version_match else "1"
            variant = f'cropped_v{version}_hf'
        else:
            # Extract version number if present (e.g., "cropped-v2" → "cropped_v2")
            version_match = re.search(r'v(\d+)', name)
            version = version_match.group(1) if version_match else "1"
            variant = f'cropped_v{version}'
    elif 'original' in name.lower():
        # Handle HuggingFace original variants
        if '_hf' in name.lower() or 'hf' in name.lower():
            version_match = re.search(r'v(\d+)', name)
            version = version_match.group(1) if version_match else "1"
            variant = f'original_v{version}_hf'
        else:
            variant = 'original_v1'
    elif 'high_error' in name.lower():
        variant = 'v1_high_error'
    elif 'low_missing' in name.lower():
        variant = 'v1_low_missing'
    elif 'background_displacement' in name.lower():
        variant = 'v1_background_displacement'
    elif 'small_displacement' in name.lower():
        variant = 'v1_small_displacement'
    elif 'mcqa' in name.lower() or 'kpqa' in name.lower():
        # Detect mcqa version — check specific versions first (longest match first)
        nl = name.lower()
        if 'v6.2' in nl or 'v6_2' in nl:
            variant = 'mcqa_v6.2'
        elif 'v6.1.2' in nl or 'v6_1_2' in nl:
            variant = 'mcqa_v6.1.2'
        elif 'v6.1' in nl or 'v6_1' in nl:
            variant = 'mcqa_v6.1'
        elif 'v6' in nl:
            variant = 'mcqa_v6.1'  # default V6 → V6.1
        elif 'v5.3' in nl or 'v5_3' in nl:
            variant = 'mcqa_v5.3'
        elif 'v5.2' in nl or 'v5_2' in nl:
            variant = 'mcqa_v5.2'
        elif 'v5.1' in nl or 'v5_1' in nl:
            variant = 'mcqa_v5.1'
        elif 'mcqa_v4.4' in nl or 'mcqa_v4_4' in nl:
            variant = 'mcqa_v4.4_qwen' if 'qwen' in nl else 'mcqa_v4.4_kimi' if 'kimi' in nl else 'mcqa_v4.4_qwen'
        elif 'mcqa_v4.3' in nl or 'mcqa_v4_3' in nl:
            variant = 'mcqa_v4.3'
        elif 'mcqa_v4.2' in nl or 'mcqa_v4_2' in nl:
            variant = 'mcqa_v4.2'
        elif 'mcqa_v3' in nl or 'v3' in nl:
            variant = 'mcqa_v3'
        elif 'kpqa-v5' in nl or 'kpqa_v5' in nl or 'mcqa_v5' in nl:
            variant = 'mcqa_v5'
        else:
            variant = 'mcqa_v1'

    # Extract step
    step_match = re.search(r'step(\d+)', name)
    step = int(step_match.group(1)) if step_match else 0

    return task, variant, step


# =============================================================================
# KEYPOINT VISUALIZATION CONSTANTS
# =============================================================================

# COCO-25 keypoint skeleton connections (indices 0-24)
COCO25_SKELETON = [
    # Face connections
    [1, 0], [0, 2], [1, 3], [2, 4],
    # Torso
    [5, 6],
    # Left arm
    [5, 7], [7, 9], [9, 11], [9, 13],
    # Right arm
    [6, 8], [8, 10], [10, 12], [10, 14],
    # Shoulders to hips
    [5, 15], [6, 16],
    # Hip connection
    [15, 16],
    # Left leg
    [15, 17], [17, 19], [19, 21], [21, 23],
    # Right leg
    [16, 18], [18, 20], [20, 22], [22, 24]
]

# COCO-17 keypoint skeleton
COCO17_SKELETON = [
    # Face connections
    [1, 0], [0, 2], [1, 3], [2, 4],
    # Torso
    [5, 6],
    # Left arm
    [5, 7], [7, 9],
    # Right arm
    [6, 8], [8, 10],
    # Shoulders to hips
    [5, 11], [6, 12],
    # Hip connection
    [11, 12],
    # Left leg
    [11, 13], [13, 15],
    # Right leg
    [12, 14], [14, 16]
]

# Body-12 keypoint skeleton
BODY12_SKELETON = [
    # Torso
    [0, 1],
    # Left arm
    [0, 2], [2, 4],
    # Right arm
    [1, 3], [3, 5],
    # Shoulders to hips
    [0, 6], [1, 7],
    # Hip connection
    [6, 7],
    # Left leg
    [6, 8], [8, 10],
    # Right leg
    [7, 9], [9, 11]
]

# Keypoint subset configurations
KEYPOINT_SUBSETS = {
    'coco25': {
        'num_keypoints': 25,
        'skeleton': COCO25_SKELETON,
        'names': [
            'Nose', 'Left Eye', 'Right Eye', 'Left Ear', 'Right Ear',
            'Left Shoulder', 'Right Shoulder', 'Left Elbow', 'Right Elbow',
            'Left Wrist', 'Right Wrist', 'Left Pinky', 'Right Pinky',
            'Left Index', 'Right Index', 'Left Hip', 'Right Hip',
            'Left Knee', 'Right Knee', 'Left Ankle', 'Right Ankle',
            'Left Heel', 'Right Heel', 'Left Foot Index', 'Right Foot Index',
        ],
    },
    'coco17': {
        'num_keypoints': 17,
        'skeleton': COCO17_SKELETON,
        'names': [
            'Nose', 'Left Eye', 'Right Eye', 'Left Ear', 'Right Ear',
            'Left Shoulder', 'Right Shoulder', 'Left Elbow', 'Right Elbow',
            'Left Wrist', 'Right Wrist', 'Left Hip', 'Right Hip',
            'Left Knee', 'Right Knee', 'Left Ankle', 'Right Ankle',
        ],
    },
    'body12': {
        'num_keypoints': 12,
        'skeleton': BODY12_SKELETON,
        'names': [
            'Left Shoulder', 'Right Shoulder', 'Left Elbow', 'Right Elbow',
            'Left Wrist', 'Right Wrist', 'Left Hip', 'Right Hip',
            'Left Knee', 'Right Knee', 'Left Ankle', 'Right Ankle',
        ],
    },
}

# Visualization colors (BGR format for OpenCV)
COLOR_GREEN = (0, 255, 0)      # Ground truth
COLOR_RED = (0, 0, 255)        # Model 1 predictions
COLOR_BLUE = (255, 0, 0)       # Model 2 predictions
COLOR_YELLOW = (0, 255, 255)   # Model 3 predictions
COLOR_PURPLE = (255, 0, 255)   # Model 4 predictions

# Color labels for legend
COLOR_LABELS = {
    str(COLOR_RED): "🔴 Red",
    str(COLOR_BLUE): "🔵 Blue",
    str(COLOR_YELLOW): "🟡 Yellow",
    str(COLOR_PURPLE): "🟣 Purple"
}


def get_color_label(color_bgr: tuple) -> str:
    """Get human-readable color label from BGR tuple."""
    return COLOR_LABELS.get(str(color_bgr), "Unknown")


def parse_keypoints(text: str) -> List[Dict]:
    """
    Parse keypoint predictions/ground truth from text format.

    Handles formats:
    - Without confidence: 0. Nose: <point>(500, 200)</point>
    - With confidence: 0. Nose: <point>(500, 200)</point> <confidence>0.85</confidence>

    Args:
        text: Keypoint text in VLM format

    Returns:
        List of keypoint dicts with idx, name, x, y
    """
    keypoints = []
    pattern = r'(\d+)\.\s*([^:<>\n]+):\s*<point>\((\d+),\s*(\d+)\)</point>'

    for match in re.finditer(pattern, text):
        idx, name, x, y = match.groups()
        keypoints.append({
            'idx': int(idx),
            'name': name.strip(),
            'x': int(x),
            'y': int(y)
        })

    return keypoints


def scale_keypoint_to_image(kp: Dict, img_width: int, img_height: int) -> tuple:
    """
    Scale normalized keypoint [0-1000] to actual image coordinates.

    Args:
        kp: Keypoint dict with x, y in [0-1000] range
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        Tuple of (scaled_x, scaled_y)
    """
    scaled_x = int((kp['x'] / 1000.0) * img_width)
    scaled_y = int((kp['y'] / 1000.0) * img_height)
    return (scaled_x, scaled_y)


def visualize_keypoints_on_image(
    image: np.ndarray,
    keypoints_text: str,
    keypoint_subset: str = 'coco25',
    color: Tuple[int, int, int] = COLOR_GREEN
) -> np.ndarray:
    """
    Draw keypoints and skeleton on image.

    Args:
        image: Input image (numpy array, BGR format)
        keypoints_text: Keypoint text in VLM format
        keypoint_subset: 'coco25', 'coco17', or 'body12'
        color: BGR color tuple for visualization

    Returns:
        Image with keypoints drawn (numpy array)
    """
    img_height, img_width = image.shape[:2]
    output = image.copy()

    # Calculate scale factor for drawing parameters
    scale_factor = img_width / 640.0
    keypoint_radius = max(2, int(4 * scale_factor))
    skeleton_thickness = max(1, int(2 * scale_factor))

    # Parse keypoints
    kps = parse_keypoints(keypoints_text)
    if not kps:
        return output

    # Get subset configuration
    subset_config = KEYPOINT_SUBSETS.get(keypoint_subset, KEYPOINT_SUBSETS['coco25'])
    skeleton = subset_config['skeleton']
    subset_names = subset_config['names']
    name_to_idx = {name: idx for idx, name in enumerate(subset_names)}

    # Create coordinate mapping (subset idx -> (x, y))
    kp_map = {}
    for kp in kps:
        x, y = scale_keypoint_to_image(kp, img_width, img_height)
        subset_idx = name_to_idx.get(kp['name'])
        if subset_idx is not None:
            kp_map[subset_idx] = (x, y)

    # Draw skeleton (lines)
    for idx1, idx2 in skeleton:
        if idx1 in kp_map and idx2 in kp_map:
            cv2.line(output, kp_map[idx1], kp_map[idx2], color, skeleton_thickness)

    # Draw keypoints (circles)
    for pt in kp_map.values():
        cv2.circle(output, pt, keypoint_radius, color, -1)

    return output


def format_metric(value: float, format_type: str) -> str:
    """Format metric value for display."""
    if format_type == 'percent':
        return f"{value * 100:.1f}%"
    elif format_type == 'px':
        return f"{value:.1f}px"
    elif format_type == 'score':
        return f"{value:.3f}"
    else:
        return f"{value:.2f}"


# =============================================================================
# DATA LOADING FUNCTIONS (Tier 1: Startup Cache)
# =============================================================================

def build_dataset_index(base_path: Path = DATASETS_BASE_PATH) -> Dict:
    """
    Scan all datasets and build index with metadata.

    Returns:
        Dict mapping {task: {variant: {path, train_stats, test_stats, ...}}}
    """
    logging.info("Building dataset index...")
    index = {}

    if not base_path.exists():
        logging.warning(f"Dataset base path not found: {base_path}")
        return index

    # Directories to skip (not actual task datasets)
    _SKIP_DIRS = {'archive', '_metadata', '__pycache__', '.git', 'scripts'}

    def _count(stats, jsonl_path):
        """Extract sample count from stats dict, falling back to JSONL line count."""
        if stats:
            for key in ('total_processed', 'samples_written', 'total_samples', 'total'):
                val = stats.get(key)
                if val and val > 0:
                    return val
        if jsonl_path.exists():
            try:
                with open(jsonl_path) as f:
                    return sum(1 for _ in f)
            except OSError:
                pass
        return 0

    def _strip_date_suffix(name: str) -> str:
        """Strip _0503 date suffix from promoted dir names."""
        return re.sub(r'_\d{4}$', '', name)

    for task_dir in base_path.iterdir():
        if not task_dir.is_dir():
            continue

        task_name = task_dir.name  # "task1_0503", "task1", etc.
        if task_name in _SKIP_DIRS or task_name.startswith('.'):
            logging.debug(f"Skipping non-task directory: {task_name}")
            continue

        # Flat promoted dir: train.jsonl directly in task_dir (e.g. task1_0503/)
        if (task_dir / "train.jsonl").exists() or (task_dir / "test.jsonl").exists():
            clean_name = _strip_date_suffix(task_name)
            train_stats = safe_load_json(task_dir / "train_stats.json") or safe_load_json(task_dir / "train.stats.json")
            test_stats = safe_load_json(task_dir / "test_stats.json") or safe_load_json(task_dir / "test.stats.json")
            entry = {
                'path': str(task_dir),
                'train_stats': train_stats or {},
                'test_stats': test_stats or {},
                'train_samples': _count(train_stats, task_dir / "train.jsonl"),
                'test_samples': _count(test_stats, task_dir / "test.jsonl"),
            }
            # Group mixed_* promoted dirs under 'mixed' task
            if clean_name.startswith('mixed_'):
                if 'mixed' not in index:
                    index['mixed'] = {}
                index['mixed'][clean_name] = entry
            else:
                if clean_name not in index:
                    index[clean_name] = {}
                index[clean_name]["gold"] = entry
            continue

        # Nested layout: task_dir/{variant}/train.jsonl
        index[task_name] = {}

        for variant_dir in task_dir.iterdir():
            if not variant_dir.is_dir():
                continue

            variant_name = variant_dir.name  # "cropped_v1", etc.

            # Skip HuggingFace format datasets (Arrow format, not JSONL)
            if variant_name.endswith('_hf'):
                logging.debug(f"Skipping HuggingFace dataset: {task_name}/{variant_name}")
                continue

            # Skip backup, old, filtered (intermediate), and non-dataset directories
            if '.backup' in variant_name or '.OLD' in variant_name or '_filtered' in variant_name:
                logging.debug(f"Skipping backup/old/filtered dataset: {task_name}/{variant_name}")
                continue
            if variant_name == 'blazepose_3d':
                logging.debug(f"Skipping non-dataset directory: {task_name}/{variant_name}")
                continue

            train_stats = safe_load_json(variant_dir / "train_stats.json") or safe_load_json(variant_dir / "train.stats.json")
            test_stats = safe_load_json(variant_dir / "test_stats.json") or safe_load_json(variant_dir / "test.stats.json")

            index[task_name][variant_name] = {
                'path': str(variant_dir),
                'train_stats': train_stats or {},
                'test_stats': test_stats or {},
                'train_samples': _count(train_stats, variant_dir / "train.jsonl"),
                'test_samples': _count(test_stats, variant_dir / "test.jsonl"),
            }

    # Scan _archive/ for all historical variants (old sft_datasets_v4 structure)
    archive_path = base_path / "_archive"
    if archive_path.exists() and archive_path.is_dir():
        for task_dir in archive_path.iterdir():
            if not task_dir.is_dir() or task_dir.name.startswith('.'):
                continue
            task_name = task_dir.name
            if task_name not in index:
                index[task_name] = {}
            for variant_dir in task_dir.iterdir():
                if not variant_dir.is_dir():
                    continue
                variant_name = variant_dir.name
                if variant_name.endswith('_hf'):
                    continue
                if '.backup' in variant_name or '.OLD' in variant_name or '_filtered' in variant_name:
                    continue
                if variant_name == 'blazepose_3d':
                    continue
                # Skip if this variant already exists (gold takes precedence)
                if variant_name in index[task_name]:
                    continue
                train_stats = safe_load_json(variant_dir / "train_stats.json") or safe_load_json(variant_dir / "train.stats.json")
                test_stats = safe_load_json(variant_dir / "test_stats.json") or safe_load_json(variant_dir / "test.stats.json")
                index[task_name][variant_name] = {
                    'path': str(variant_dir),
                    'train_stats': train_stats or {},
                    'test_stats': test_stats or {},
                    'train_samples': _count(train_stats, variant_dir / "train.jsonl"),
                    'test_samples': _count(test_stats, variant_dir / "test.jsonl"),
                }

    # Special handling for mixed_tasks in _archive/
    mixed_tasks_path = archive_path / "mixed_tasks" if archive_path.exists() else base_path / "mixed_tasks"
    if mixed_tasks_path.exists() and mixed_tasks_path.is_dir():
        if 'mixed' not in index:
            index['mixed'] = {}
        for variant_dir in mixed_tasks_path.iterdir():
            if not variant_dir.is_dir():
                continue

            # Add 'mixed_' prefix only if dir name doesn't already start with 'mixed_'
            variant_name = variant_dir.name if variant_dir.name.startswith('mixed_') else f"mixed_{variant_dir.name}"

            # Skip HuggingFace format datasets
            if variant_dir.name.endswith('_hf'):
                logging.debug(f"Skipping HuggingFace dataset: mixed/{variant_name}")
                continue

            # Skip if already exists (gold takes precedence)
            if variant_name in index['mixed']:
                continue

            train_stats = safe_load_json(variant_dir / "train_stats.json") or safe_load_json(variant_dir / "train.stats.json")
            test_stats = safe_load_json(variant_dir / "test_stats.json") or safe_load_json(variant_dir / "test.stats.json")

            index['mixed'][variant_name] = {
                'path': str(variant_dir),
                'train_stats': train_stats or {},
                'test_stats': test_stats or {},
                'train_samples': _count(train_stats, variant_dir / "train.jsonl"),
                'test_samples': _count(test_stats, variant_dir / "test.jsonl"),
            }

    logging.info(f"Dataset index built: {len(index)} tasks, "
                f"{sum(len(v) for v in index.values())} variants")
    return index


# =============================================================================
# MCQA TAB HELPERS
# =============================================================================

def build_validator_index() -> Dict:
    """
    Build index of validator results keyed by image_id.
    Scans archive/, mcqa_v4.2/, and mcqa_v4.4_* dirs for validation JSONs.

    Validator keys:
        - 'qwen', 'gemini' for V4.2 (shared by V4.3)
        - 'qwen_v4.4' for V4.4 Qwen-validated (different descriptions, different outcomes)

    Returns:
        {image_id: {validator_key: {status, issues, corrected_description, image_path}}}
    """
    index = {}
    # (directory, key_suffix) — suffix appended to validator name to avoid collisions
    _archive = DATASETS_BASE_PATH / "_archive"
    validator_dirs = [
        (_archive / "archive", ""),
        (_archive / "task4" / "mcqa_v4.2", ""),
        (_archive / "task4" / "mcqa_v4.4_qwen", "_v4.4"),
        (_archive / "task4" / "mcqa_v4.4_kimi", "_v4.4_kimi"),
        (_archive / "task4" / "mcqa_v6.1_qwen", "_v6.1"),
        (_archive / "task4" / "mcqa_v6.1_kimi", "_v6.1_kimi"),
    ]

    for scan_dir, suffix in validator_dirs:
        if not scan_dir.exists():
            continue
        for json_file in scan_dir.glob("*validation*.json"):
            try:
                data = json.loads(json_file.read_text())
                if not isinstance(data, dict) or 'results' not in data:
                    continue
                # Determine validator name from filename
                fname = json_file.stem.lower()
                if 'qwen' in fname:
                    validator = 'qwen' + suffix
                elif 'gemini' in fname:
                    validator = 'gemini' + suffix
                else:
                    validator = json_file.stem + suffix

                for result in data['results']:
                    image_id = result.get('image_id', '')
                    if not image_id:
                        continue
                    if image_id not in index:
                        index[image_id] = {}
                    index[image_id][validator] = {
                        'status': result.get('validation_status', 'UNKNOWN'),
                        'issues': result.get('issues', ''),
                        'corrected_description': result.get('corrected_description', ''),
                        'image_path': result.get('image_path', ''),
                    }
            except Exception as e:
                logging.debug(f"Failed to load validator JSON {json_file}: {e}")

    logging.info(f"Validator index built: {len(index)} image_ids")
    return index


def build_confusion_flags_index() -> Dict:
    """
    Load 2D/3D projection confusion flags from *_2d3d_flags.jsonl files.
    Auto-discovers flag files in dataset directories.

    Returns:
        {(image_id, template): {max_risk, flags}}
    """
    index = {}
    task4_dir = DATASETS_BASE_PATH / "_archive" / "task4"
    if not task4_dir.exists():
        return index
    for flag_file in task4_dir.glob("*/*_2d3d_flags.jsonl"):
        try:
            with open(flag_file) as f:
                for line in f:
                    entry = json.loads(line)
                    key = (entry['image_id'], entry['template'])
                    index[key] = {
                        'max_risk': entry['max_risk'],
                        'flags': entry['flags'],
                        'answer': entry.get('answer', ''),
                    }
        except Exception as e:
            logging.debug(f"Failed to load confusion flags from {flag_file}: {e}")
    logging.info(f"Confusion flags index built: {len(index)} flagged samples")
    return index


@cacheable(maxsize=50)
def resolve_jsonl_path(dataset_path: str, split: str) -> Optional[Path]:
    """Find the JSONL file for a given dataset path and split."""
    dp = Path(dataset_path)

    # Check DATASET_INDEX first
    for task_data in DATASET_INDEX.values():
        for variant_data in task_data.values():
            if variant_data.get('path') == dataset_path:
                standard = dp / f"{split}.jsonl"
                if standard.exists():
                    return standard

    # Standard name
    standard = dp / f"{split}.jsonl"
    if standard.exists():
        return standard

    # Glob fallback for non-standard names
    pattern = f"*{split}*.jsonl"
    matches = sorted(dp.glob(pattern))
    if matches:
        return matches[0]

    # Any JSONL file
    all_jsonl = sorted(dp.glob("*.jsonl"))
    if all_jsonl:
        return all_jsonl[0]

    return None


@cacheable(maxsize=20)
def _load_jsonl_lines(jsonl_path: str) -> Tuple[str, ...]:
    """Cache raw JSONL lines for O(1) random access. First call loads file; subsequent calls are instant."""
    try:
        with open(jsonl_path) as f:
            return tuple(f)
    except Exception as e:
        logging.error(f"Failed to load JSONL lines from {jsonl_path}: {e}")
        return ()


# ---------------------------------------------------------------------------
# Reasoning Trace Data
# ---------------------------------------------------------------------------

REASONING_INDEX: Dict[str, Dict[str, Dict]] = {}      # {task: {split: {"path": str, "count": int}}}
REASONING_HALLUCINATIONS: Dict[str, Dict[str, set]] = {}  # {task: {split: set(image_ids)}}
REASONING_ORIENTATION_MISMATCHES: Dict[str, Dict[str, Dict[str, Dict]]] = {}  # {task: {split: {image_id: {gt_orientation, confidence, signals, ...}}}}
REASONING_BODY_POS_MISMATCHES: Dict[str, Dict[str, Dict[str, Dict]]] = {}  # {task: {split: {image_id: {claimed_position, gt_position, ...}}}}
REASONING_HIGH_ANGLE: Dict[str, Dict[str, Dict[str, Dict]]] = {}  # {task: {split: {image_id: {metadata_position, kp_position, torso_angle_deg}}}}
REASONING_PROMPT_CONFIGS: Dict[str, Dict[str, str]] = {}  # {task: {"system_message": str, "user_prompt": str}}
REASONING_AUDIT: Dict[str, Dict[str, Dict[str, Dict]]] = {}  # {task: {split: {image_id: {reason, timestamp, line_idx}}}}
REASONING_APPROVED: Dict[str, Dict[str, Dict[str, Dict]]] = {}  # {task: {split: {image_id: {timestamp, line_idx}}}}

AUDIT_REASONS = [
    "wrong_orientation",
    "wrong_body_position",
    "hallucinated_reasoning",
    "wrong_left_right",
    "incoherent_reasoning",
    "other",
]

_REASONING_YAML_PATHS = {
    "task1": "prompts/task1/v1_reas_traces.yaml",
    "task2": "prompts/task2/v1_reas_traces_task2.yaml",
    "task3a": "prompts/task3a/v1_reason_traces_task3a.yaml",
    "task3b": "prompts/task3b/v1_reason_traces_task3b.yaml",
    "task3c": "prompts/task3c/v1_reason_traces_task3c.yaml",
    "task3d": "prompts/task3d/v1_reason_traces_task3d.yaml",
    "task4_v5.3": "prompts/task4/v1_reason_traces_task4_v5.3.yaml",
    "task4_v6.2": "prompts/task4/v1_reas_traces_task4_v6.2.yaml",
}


def build_reasoning_index():
    """Scan reasoning_data directories and build index, hallucination sets, and prompt configs."""
    global REASONING_INDEX, REASONING_HALLUCINATIONS, REASONING_PROMPT_CONFIGS, REASONING_ORIENTATION_MISMATCHES, REASONING_BODY_POS_MISMATCHES, REASONING_HIGH_ANGLE
    for task in REASONING_TASKS:
        task_dir = REASONING_DATA_PATH / task
        if not task_dir.is_dir():
            continue
        REASONING_INDEX[task] = {}
        REASONING_HALLUCINATIONS[task] = {}
        for split in ("train", "test", "qwen3_regen"):
            jsonl_path = task_dir / f"{split}.jsonl"
            if jsonl_path.exists():
                count = sum(1 for _ in open(jsonl_path))
                REASONING_INDEX[task][split] = {"path": str(jsonl_path), "count": count}
            hall_path = task_dir / f"{split}_hallucination.jsonl"
            hall_ids = set()
            if hall_path.exists():
                for line in open(hall_path):
                    try:
                        d = json.loads(line)
                        hall_ids.add(d.get("metadata", {}).get("image_id", ""))
                    except Exception:
                        pass
            REASONING_HALLUCINATIONS[task][split] = hall_ids
            # Load orientation mismatch data (v2 format: full mismatch dicts)
            orient_path = task_dir / f"{split}_orientation_mismatches.json"
            if orient_path.exists():
                try:
                    orient_data = json.loads(orient_path.read_text())
                    REASONING_ORIENTATION_MISMATCHES.setdefault(task, {})[split] = {
                        m["image_id"]: m
                        for m in orient_data.get("mismatches", [])
                    }
                except Exception:
                    pass
            # Load body position mismatch data
            bp_path = task_dir / f"{split}_body_position_mismatches.json"
            if bp_path.exists():
                try:
                    bp_data = json.loads(bp_path.read_text())
                    REASONING_BODY_POS_MISMATCHES.setdefault(task, {})[split] = {
                        m["image_id"]: m
                        for m in bp_data.get("mismatches", [])
                    }
                except Exception:
                    pass
            # Load high-angle disagreement data
            ha_path = task_dir / f"{split}_high_angle_disagreements.json"
            if ha_path.exists():
                try:
                    ha_data = json.loads(ha_path.read_text())
                    REASONING_HIGH_ANGLE.setdefault(task, {})[split] = {
                        m["image_id"]: m
                        for m in ha_data.get("mismatches", [])
                    }
                except Exception:
                    pass

    # Fallback: load v2 test files from reasoning_traces_tests/ for tasks missing data
    if REASONING_TESTS_PATH.is_dir():
        for task, file_prefix in _REASONING_TEST_FILE_MAP.items():
            if task not in REASONING_INDEX:
                REASONING_INDEX[task] = {}
                REASONING_HALLUCINATIONS[task] = {}
            if "test" not in REASONING_INDEX[task]:
                jsonl_path = REASONING_TESTS_PATH / f"{file_prefix}.jsonl"
                if jsonl_path.exists():
                    count = sum(1 for _ in open(jsonl_path))
                    REASONING_INDEX[task]["test"] = {"path": str(jsonl_path), "count": count, "source": "tests"}
                    hall_path = REASONING_TESTS_PATH / f"{file_prefix}_hallucination.jsonl"
                    hall_ids = set()
                    if hall_path.exists():
                        for line in open(hall_path):
                            try:
                                d = json.loads(line)
                                hall_ids.add(d.get("metadata", {}).get("image_id", ""))
                            except Exception:
                                pass
                    REASONING_HALLUCINATIONS[task]["test"] = hall_ids

    # Load teacher prompt YAML templates for reconstruction
    sft_base = SFT_DATA_ROOT
    for task, rel_path in _REASONING_YAML_PATHS.items():
        yaml_path = sft_base / rel_path
        if yaml_path.exists():
            with open(yaml_path) as f:
                cfg = yaml.safe_load(f)
            REASONING_PROMPT_CONFIGS[task] = {
                "system_message": cfg.get("system_message", ""),
                "user_prompt": cfg.get("user_prompt", ""),
            }

    # Load audit exclusions and approvals
    _load_all_audits()
    _load_all_approvals()

    logging.info(
        f"Reasoning index: {sum(len(v) for v in REASONING_INDEX.values())} splits "
        f"across {len(REASONING_INDEX)} tasks, {len(REASONING_PROMPT_CONFIGS)} prompt configs loaded"
    )


# ========== PROMPT COMPARISON EXPERIMENT LOADING ==========

def _discover_comparison_experiments() -> list:
    """Auto-discover experiment directories from EXPERIMENTS_PATH."""
    if not EXPERIMENTS_PATH.is_dir():
        return []
    return sorted(d.name for d in EXPERIMENTS_PATH.iterdir()
                  if d.is_dir() and not d.name.startswith("."))


def _load_comparison_experiment(experiment_name: str) -> dict:
    """Load an experiment directory into memory.

    Returns {tasks, versions, data, hallucinated, counts} where:
    - data[task][ver] = tuple of raw JSONL line strings
    - hallucinated[task][ver] = set of hallucinated image_ids
    """
    exp_dir = EXPERIMENTS_PATH / experiment_name
    empty = {"tasks": [], "versions": {}, "data": {}, "hallucinated": {}, "counts": {}}
    if not exp_dir.is_dir():
        return empty

    # Discover {task}_{version}.jsonl files
    task_versions: dict = {}
    for f in exp_dir.glob("*.jsonl"):
        if "_hallucination" in f.stem or ".pretty" in f.name:
            continue
        parts = f.stem.rsplit("_", 1)
        if len(parts) != 2:
            continue
        task, version = parts
        task_versions.setdefault(task, []).append(version)

    if not task_versions:
        return empty

    tasks = sorted(task_versions.keys())
    versions = {t: sorted(vs) for t, vs in task_versions.items()}
    data: dict = {}
    hallucinated: dict = {}
    counts: dict = {}

    for task in tasks:
        data[task] = {}
        hallucinated[task] = {}
        for ver in versions[task]:
            # Load main data lines
            jsonl_path = exp_dir / f"{task}_{ver}.jsonl"
            with open(jsonl_path) as fh:
                data[task][ver] = tuple(fh.readlines())

            # Load hallucination image_ids
            hall_path = exp_dir / f"{task}_{ver}_hallucination.jsonl"
            hall_ids: set = set()
            if hall_path.exists() and hall_path.stat().st_size > 0:
                for line in open(hall_path):
                    try:
                        rec = json.loads(line)
                        iid = rec.get("metadata", {}).get("image_id", "")
                        if iid:
                            hall_ids.add(iid)
                    except Exception:
                        pass
            hallucinated[task][ver] = hall_ids

        counts[task] = len(data[task][versions[task][0]])

    return {"tasks": tasks, "versions": versions, "data": data,
            "hallucinated": hallucinated, "counts": counts}


def _parse_comparison_sample(raw_line: str) -> dict:
    """Parse a single JSONL line into display-ready components."""
    d = json.loads(raw_line)
    msgs = json.loads(d["messages"]) if isinstance(d["messages"], str) else d["messages"]
    meta = d.get("metadata", {})

    image_path = ""
    assistant_content = ""
    for m in msgs:
        if m["role"] == "user" and isinstance(m["content"], list):
            for part in m["content"]:
                if isinstance(part, dict) and "image" in part:
                    image_path = part["image"]
        elif m["role"] == "assistant":
            assistant_content = m["content"] if isinstance(m["content"], str) else ""

    think_text = ""
    think_m = re.search(r"<think>(.*?)</think>", assistant_content, re.DOTALL)
    if think_m:
        think_text = think_m.group(1).strip()

    answer_text = ""
    ans_m = re.search(r"<answer>(.*?)</answer>", assistant_content, re.DOTALL)
    if ans_m:
        answer_text = ans_m.group(1).strip()

    return {
        "image_path": image_path,
        "image_id": meta.get("image_id", ""),
        "think_text": think_text,
        "answer_text": answer_text,
        "word_count": meta.get("reasoning_word_count", len(think_text.split())),
        "metadata": meta,
    }


def _audit_path(task: str, split: str) -> Path:
    """Path to audit exclusions JSON for a task/split."""
    return REASONING_DATA_PATH / task / f"{split}_audit_exclusions.json"


def _load_all_audits():
    """Load audit exclusion files for all tasks/splits."""
    global REASONING_AUDIT
    for task in REASONING_TASKS:
        for split in ("train", "test", "qwen3_regen"):
            p = _audit_path(task, split)
            if p.exists():
                try:
                    REASONING_AUDIT.setdefault(task, {})[split] = json.loads(p.read_text())
                except Exception:
                    pass


def _save_audit(task: str, split: str):
    """Persist audit exclusions to disk."""
    data = REASONING_AUDIT.get(task, {}).get(split, {})
    p = _audit_path(task, split)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2))


def _approved_path(task: str, split: str) -> Path:
    """Path to approved samples JSON for a task/split."""
    return REASONING_DATA_PATH / task / f"{split}_approved.json"


def _load_all_approvals():
    """Load approval files for all tasks/splits."""
    global REASONING_APPROVED
    for task in REASONING_TASKS:
        for split in ("train", "test", "qwen3_regen"):
            p = _approved_path(task, split)
            if p.exists():
                try:
                    REASONING_APPROVED.setdefault(task, {})[split] = json.loads(p.read_text())
                except Exception:
                    pass


def _save_approved(task: str, split: str):
    """Persist approved samples to disk."""
    data = REASONING_APPROVED.get(task, {}).get(split, {})
    p = _approved_path(task, split)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2))


def _resolve_reasoning_key(task: str, variant: str) -> str:
    """Map sidebar task + variant to the REASONING_INDEX key.

    Most tasks map 1:1 (task1 → task1).  For task4, the variant encodes
    the reasoning sub-variant: mcqa_v5.3 → task4_v5.3, mcqa_v6.2 → task4_v6.2.
    Falls back to bare task name if no variant-specific match exists.
    """
    if task == 'task4' and variant:
        # mcqa_v5.3 → v5.3, mcqa_v6.2 → v6.2
        ver = variant.replace('mcqa_', '')
        candidate = f"task4_{ver}"
        if candidate in REASONING_INDEX:
            return candidate
    # Direct match (task1, task2, task3a, ...)
    if task in REASONING_INDEX:
        return task
    return task


def _reconstruct_teacher_prompt(task: str, answer_text: str, user_text: str, meta: Dict) -> Tuple[str, str]:
    """Reconstruct teacher system message and user prompt from YAML template + sample data.

    Returns (teacher_sys, teacher_prompt). Falls back gracefully if data is missing.
    """
    cfg = REASONING_PROMPT_CONFIGS.get(task)
    if not cfg:
        return "", "(prompt config not found for this task)"

    teacher_sys = cfg["system_message"]
    tpl = cfg["user_prompt"]

    if task in ("task1", "task2"):
        # Convert <point>(x, y)</point> → [x, y] for bracket format (task1)
        # For task2, the answer is just "N. Body Part" lines — no conversion needed
        gt_bracket = re.sub(r"<point>\((\d+),\s*(\d+)\)</point>", r"[\1, \2]", answer_text)
        teacher_prompt = tpl.replace("{{gt_keypoints_list}}", gt_bracket)

    elif task.startswith("task3"):
        # Extract keypoint mapping from student user text
        kp_match = re.search(
            r"Keypoint labels \(number: body part\):\n(.+?)(?:\n\n|\nVerify|\nIdentify|\nCheck|\nAnalyze)",
            user_text, re.DOTALL
        )
        keypoint_mapping = kp_match.group(1).strip() if kp_match else ""
        if not keypoint_mapping:
            # Fallback: lines matching "N: BodyPart"
            keypoint_mapping = "\n".join(
                ln.strip() for ln in user_text.split("\n") if re.match(r"^\d+:\s+\w", ln.strip())
            )
        # Format GT corrections from answer (JSON)
        gt_corrections = answer_text
        try:
            gt = json.loads(answer_text)
            if not gt.get("has_errors", False):
                gt_corrections = "No errors detected — all keypoints are correctly labeled and positioned."
            else:
                corrections = gt.get("corrections", [])
                if isinstance(corrections, list) and corrections and isinstance(corrections[0], dict):
                    lines = []
                    for c in corrections:
                        pos = c.get("correct_position", [])
                        pos_str = f" → corrected position: [{pos[0]}, {pos[1]}]" if len(pos) == 2 else ""
                        lines.append(f"- Keypoint {c.get('keypoint_number', '?')} ({c.get('label', '?')}){pos_str}")
                    gt_corrections = "\n".join(lines)
        except (json.JSONDecodeError, TypeError):
            pass
        teacher_prompt = tpl.replace("{{keypoint_mapping}}", keypoint_mapping).replace("{{gt_corrections}}", gt_corrections)

    elif task.startswith("task4"):
        # Reconstruct geometric context from verification attributes
        verification = meta.get("verification", {})
        attributes = verification.get("attributes", [])
        if attributes:
            geo_lines = []
            for attr in attributes:
                attr_type = attr.get("type", "unknown")
                key = attr.get("key", "unknown")
                category = attr.get("category", "unknown")
                if "__" in key:
                    parts = key.split("__")
                    fk = " \u2194 ".join(p.replace("_", " ").title() for p in parts)
                else:
                    fk = key.replace("_", " ").title()
                type_labels = {"angle": "angle", "quadrant": "position", "height": "height comparison",
                               "distance": "distance", "alignment": "alignment"}
                geo_lines.append(f"- {fk} {type_labels.get(attr_type, attr_type)}: {category}")
            geo_ctx = "\n".join(geo_lines)
        else:
            geo_ctx = "(geometric context not available)"
        # Extract question and choices from student user text
        q_lines, c_lines, in_choices = [], [], False
        for line in user_text.strip().split("\n"):
            stripped = line.strip()
            if re.match(r"^[A-D]\)", stripped):
                in_choices = True
            if stripped.startswith("Select the letter"):
                continue
            (c_lines if in_choices else q_lines).append(line)
        question_text = "\n".join(q_lines).strip()
        choices_text = "\n".join(c_lines).strip()
        correct_answer = meta.get("correct_answer", answer_text.strip())
        teacher_prompt = (
            tpl.replace("{{geometric_context}}", geo_ctx)
               .replace("{{question_text}}", question_text)
               .replace("{{choices_text}}", choices_text)
               .replace("{{correct_answer}}", correct_answer)
        )
    else:
        teacher_prompt = "(reconstruction not implemented for this task)"

    return teacher_sys, teacher_prompt


def load_reasoning_sample(task: str, split: str, idx: int) -> Optional[Dict]:
    """Load one reasoning trace sample by index."""
    info = REASONING_INDEX.get(task, {}).get(split)
    if not info:
        return None
    lines = _load_jsonl_lines(info["path"])
    if idx < 0 or idx >= len(lines):
        return None
    return json.loads(lines[idx])


def parse_reasoning_sample(raw: Dict, task: str, split: str) -> Dict:
    """Parse a raw JSONL dict into display-ready components."""
    msgs = json.loads(raw["messages"]) if isinstance(raw["messages"], str) else raw["messages"]
    meta = raw.get("metadata", {})

    # Extract from messages
    sys_text, image_path, user_text, assistant_content = "", "", "", ""
    for m in msgs:
        if m["role"] == "system":
            sys_text = m["content"]
        elif m["role"] == "user":
            if isinstance(m["content"], list):
                for part in m["content"]:
                    if "image" in part:
                        image_path = part["image"]
                    elif "text" in part:
                        user_text = part["text"]
            else:
                user_text = m["content"]
        elif m["role"] == "assistant":
            assistant_content = m["content"]

    # Extract think/answer
    think_text, answer_text = "", ""
    think_m = re.search(r"<think>(.*?)</think>", assistant_content, re.DOTALL)
    if think_m:
        think_text = think_m.group(1).strip()
    ans_m = re.search(r"<answer>(.*?)</answer>", assistant_content, re.DOTALL)
    if ans_m:
        answer_text = ans_m.group(1).strip()

    # Teacher prompt — use stored if available, otherwise reconstruct from YAML template
    teacher_prompt = meta.get("teacher_prompt", "")
    teacher_sys = meta.get("teacher_system_message", "")
    if not teacher_prompt:
        teacher_sys, teacher_prompt = _reconstruct_teacher_prompt(task, answer_text, user_text, meta)

    # Hallucination check
    hall_ids = REASONING_HALLUCINATIONS.get(task, {}).get(split, set())
    is_hallucinated = meta.get("image_id", "") in hall_ids

    # Orientation mismatch check (v2: full mismatch dict with confidence + signals)
    orient_map = REASONING_ORIENTATION_MISMATCHES.get(task, {}).get(split, {})
    image_id = meta.get("image_id", "")
    orient_info = orient_map.get(image_id)
    orientation_mismatch = orient_info is not None
    # Backward compat: orient_info may be str (v1) or dict (v2)
    if isinstance(orient_info, str):
        gt_orientation = orient_info
        orient_confidence = "unknown"
        orient_signals = {}
    elif isinstance(orient_info, dict):
        gt_orientation = orient_info.get("gt_orientation", "")
        orient_confidence = orient_info.get("confidence", "unknown")
        orient_signals = orient_info.get("signals", {})
    else:
        gt_orientation = ""
        orient_confidence = ""
        orient_signals = {}

    # Body position mismatch check
    bp_map = REASONING_BODY_POS_MISMATCHES.get(task, {}).get(split, {})
    bp_info = bp_map.get(image_id)
    body_pos_mismatch = bp_info is not None
    if isinstance(bp_info, dict):
        claimed_body_pos = bp_info.get("claimed_position", "")
        gt_body_pos = bp_info.get("gt_position", "")
        body_pos_detail = bp_info.get("body_position", {})
    else:
        claimed_body_pos = ""
        gt_body_pos = ""
        body_pos_detail = {}

    # High-angle disagreement check
    ha_map = REASONING_HIGH_ANGLE.get(task, {}).get(split, {})
    ha_info = ha_map.get(image_id)

    return {
        "image_path": image_path,
        "image_id": image_id,
        "teacher_sys": teacher_sys,
        "teacher_prompt": teacher_prompt,
        "think_text": think_text,
        "answer_text": answer_text,
        "train_sys": sys_text,
        "train_user": user_text,
        "train_assistant": assistant_content,
        "metadata": meta,
        "reasoning_word_count": meta.get("reasoning_word_count", 0),
        "is_hallucinated": is_hallucinated,
        "orientation_mismatch": orientation_mismatch,
        "gt_orientation": gt_orientation,
        "orient_confidence": orient_confidence,
        "orient_signals": orient_signals,
        "body_pos_mismatch": body_pos_mismatch,
        "claimed_body_pos": claimed_body_pos,
        "gt_body_pos": gt_body_pos,
        "body_pos_detail": body_pos_detail,
        "high_angle_disagreement": ha_info,
    }


def search_reasoning_image_id(image_id: str) -> List[Dict]:
    """Find all (task, split, idx) containing this image_id. Returns list of dicts."""
    results = []
    query = image_id.strip()
    for task in REASONING_TASKS:
        for split in ("train", "test", "qwen3_regen"):
            info = REASONING_INDEX.get(task, {}).get(split)
            if not info:
                continue
            lines = _load_jsonl_lines(info["path"])
            for idx, line in enumerate(lines):
                try:
                    d = json.loads(line)
                    iid = d.get("metadata", {}).get("image_id", "")
                    if query in iid:
                        results.append({"task": task, "split": split, "idx": idx, "image_id": iid})
                except Exception:
                    pass
    return results


def count_mcqa_samples(dataset_path: str, split: str) -> int:
    """Count total JSONL lines in a dataset split (O(1) via cached lines)."""
    jsonl_path = resolve_jsonl_path(dataset_path, split)
    if not jsonl_path or not jsonl_path.exists():
        return 0
    return len(_load_jsonl_lines(str(jsonl_path)))


def get_generation_prompt_md(variant: str) -> str:
    """Load generation prompt YAML for a variant and format as markdown."""
    if not variant:
        return ""
    # Match variant to prompt file (longest prefix match)
    yaml_file = None
    for prefix in sorted(VARIANT_PROMPT_MAP.keys(), key=len, reverse=True):
        if variant.startswith(prefix):
            yaml_file = PROMPT_CONFIGS_DIR / VARIANT_PROMPT_MAP[prefix]
            break
    if not yaml_file or not yaml_file.exists():
        if any(variant.startswith(p) for p in ('mcqa_v5', 'kpqa')):
            return "*V5.x variants use algorithmic keypoint QA generation (no LLM prompt)*"
        return ""
    try:
        import yaml
        with open(yaml_file) as f:
            cfg = yaml.safe_load(f)
        parts = [f"**Prompt:** `{yaml_file.name}` (v{cfg.get('version', '?')})"]
        sys_msg = cfg.get('system_message', '')
        if sys_msg:
            parts.append(f"**System:**\n```\n{sys_msg.strip()}\n```")
        user_prompt = cfg.get('user_prompt', '')
        if user_prompt:
            parts.append(f"**User template:**\n```\n{user_prompt.strip()}\n```")
        return "\n\n".join(parts)
    except Exception as e:
        logging.error(f"Error loading prompt YAML {yaml_file}: {e}")
        return f"*Error loading prompt: {e}*"


def load_mcqa_sample(dataset_path: str, split: str, idx: int) -> Optional[Dict]:
    """Load a single MCQA sample by line index (O(1) via cached lines)."""
    jsonl_path = resolve_jsonl_path(dataset_path, split)
    if not jsonl_path or not jsonl_path.exists():
        return None
    try:
        lines = _load_jsonl_lines(str(jsonl_path))
        if 0 <= idx < len(lines):
            return json.loads(lines[idx])
    except Exception as e:
        logging.error(f"Error loading MCQA sample {idx}: {e}")
        gr.Warning(f"Failed to load MCQA sample {idx}: {e}")
    return None


def parse_mcqa_messages(sample: Dict) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Parse messages from MCQA sample (handles both string and list formats).

    Returns:
        (image_path, question_text, correct_answer)
    """
    messages = sample.get('messages', [])
    if isinstance(messages, str):
        messages = json.loads(messages)

    image_path = None
    question_text = None
    correct_answer = None

    for msg in messages:
        role = msg.get('role', '')
        content = msg.get('content', '')

        if role == 'user':
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        if 'image' in item:
                            image_path = item['image']
                        elif 'text' in item:
                            question_text = item['text']
            elif isinstance(content, str) and not image_path:
                question_text = content

        elif role == 'assistant':
            if isinstance(content, str):
                correct_answer = content.strip()

    return image_path, question_text, correct_answer


def _is_qwen_only_variant(variant: str) -> bool:
    """V4.3/V4.4/V6.1 were filtered with Qwen only — Gemini results don't apply."""
    return variant and ('v4.3' in variant or 'v4.4' in variant or 'v6.1' in variant)


def _get_variant_validator_keys(variant: str) -> Optional[list]:
    """Return the validator keys relevant for a given MCQA variant.

    V4.2 uses 'qwen' and 'gemini' (consensus).
    V4.3 uses 'qwen' only (same descriptions as V4.2, just Qwen-filtered).
    V4.4 uses 'qwen_v4.4' (different descriptions, different outcomes).
    Returns None to show all validators (e.g. for V3 or unknown variants).
    """
    if not variant:
        return None
    if 'v6.1' in variant:
        return ['qwen_v6.1', 'qwen_v6.1_kimi']
    if 'v4.4' in variant:
        if 'kimi' in variant:
            return ['qwen_v4.4_kimi']
        return ['qwen_v4.4']
    if 'v4.3' in variant:
        return ['qwen']
    if 'v4.2' in variant:
        return ['qwen', 'gemini']
    return None


@cacheable(maxsize=200)
def get_filtered_dataset_indices(dataset_path: str, split: str, status_filter: str, variant: str = "", exercise_prefix: str = "All", question_template: str = "All", confusion_filter: str = "All", error_label: str = "All") -> Optional[Tuple]:
    """
    Get indices of samples matching filters (validator status + exercise prefix + question template + confusion risk + error label).
    Returns tuple of indices (hashable for lru_cache), or None if no filters active.
    For Qwen-only variants (v4.3), only considers Qwen validator status.
    """
    has_status = status_filter != "All"
    has_exercise = exercise_prefix != "All"
    has_template = question_template != "All"
    has_confusion = confusion_filter != "All"
    has_error_label = error_label != "All"
    if not has_status and not has_exercise and not has_template and not has_confusion and not has_error_label:
        return None

    jsonl_path = resolve_jsonl_path(dataset_path, split)
    if not jsonl_path:
        return tuple()

    allowed_keys = _get_variant_validator_keys(variant)
    lines = _load_jsonl_lines(str(jsonl_path))
    indices = []
    for i, line in enumerate(lines):
        try:
            sample = json.loads(line)
            metadata = sample.get('metadata', {})
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            image_id = metadata.get('image_id', '')

            # Exercise prefix filter
            if has_exercise:
                prefix = image_id.split('_')[0] if '_' in image_id else ''
                if prefix != exercise_prefix:
                    continue

            # Question template filter
            if has_template:
                tmpl = metadata.get('question_template', '')
                if tmpl != question_template:
                    continue

            # Validator status filter
            if has_status:
                if image_id not in VALIDATOR_INDEX:
                    continue
                validators = VALIDATOR_INDEX[image_id]
                if allowed_keys:
                    statuses = [v.get('status', '') for k, v in validators.items() if k in allowed_keys]
                else:
                    statuses = [v.get('status', '') for v in validators.values()]
                if status_filter == "CORRECT" and not (statuses and all(s == "CORRECT" for s in statuses)):
                    continue
                elif status_filter == "INCORRECT" and not any(s == "INCORRECT" for s in statuses):
                    continue

            # 2D/3D confusion risk filter
            if has_confusion:
                tmpl = metadata.get('question_template', '')
                flag_key = (image_id, tmpl)
                flag_data = CONFUSION_FLAGS.get(flag_key)
                if confusion_filter == "Not flagged":
                    if flag_data is not None:
                        continue
                elif confusion_filter == "Any flagged":
                    if flag_data is None:
                        continue
                elif confusion_filter == "HIGH":
                    if not flag_data or flag_data['max_risk'] != 'HIGH':
                        continue
                elif confusion_filter == "MEDIUM+":
                    if not flag_data or flag_data['max_risk'] not in ('HIGH', 'MEDIUM'):
                        continue

            # Error label filter (normalized error_category from verification)
            if has_error_label:
                verification = metadata.get('verification', {})
                raw_cat = verification.get('error_category', '')
                if not raw_cat:
                    continue  # No error_category → skip when filtering by error label
                if normalize_error_label(raw_cat) != error_label:
                    continue

            indices.append(i)
        except Exception:
            continue

    return tuple(indices)


def get_filtered_validation_ids(status_filter: str, variant: str = "") -> List[str]:
    """Get list of image_ids matching a validator status filter.
    Uses variant-specific validator keys (e.g. V4.4 checks 'qwen_v4.4', V4.3 checks 'qwen')."""
    if not VALIDATOR_INDEX:
        return []
    if status_filter == "All":
        return sorted(VALIDATOR_INDEX.keys())

    allowed_keys = _get_variant_validator_keys(variant)
    result = []
    for image_id, validators in VALIDATOR_INDEX.items():
        if allowed_keys:
            statuses = [v.get('status', '') for k, v in validators.items() if k in allowed_keys]
        else:
            statuses = [v.get('status', '') for v in validators.values()]
        if status_filter == "CORRECT" and statuses and all(s == "CORRECT" for s in statuses):
            result.append(image_id)
        elif status_filter == "INCORRECT" and any(s == "INCORRECT" for s in statuses):
            result.append(image_id)
    return sorted(result)


@cacheable(maxsize=20)
def get_exercise_prefixes(dataset_path: str, split: str) -> Tuple[str, ...]:
    """Get sorted unique exercise prefixes from dataset image_ids.
    Exercise prefix = first segment of image_id before '_'."""
    jsonl_path = resolve_jsonl_path(dataset_path, split)
    if not jsonl_path:
        return ()
    lines = _load_jsonl_lines(str(jsonl_path))
    prefixes = set()
    for line in lines:
        try:
            meta = json.loads(line).get('metadata', {})
            if isinstance(meta, str):
                meta = json.loads(meta)
            iid = meta.get('image_id', '')
            prefix = iid.split('_')[0] if '_' in iid else ''
            if prefix:
                prefixes.add(prefix)
        except Exception:
            continue
    return tuple(sorted(prefixes))


@cacheable(maxsize=200)
def get_filtered_explorer_indices(dataset_path: str, split: str, exercise_prefix: str) -> Tuple[int, ...]:
    """Get sample indices matching exercise prefix filter.

    Returns tuple of 0-based indices where the sample's exercise_prefix matches.
    """
    jsonl_path = resolve_jsonl_path(dataset_path, split)
    if not jsonl_path:
        return ()
    lines = _load_jsonl_lines(str(jsonl_path))
    indices = []
    for i, line in enumerate(lines):
        try:
            meta = json.loads(line).get('metadata', {})
            if isinstance(meta, str):
                meta = json.loads(meta)
            iid = meta.get('image_id', '')
            prefix = iid.split('_')[0] if '_' in iid else ''
            if prefix == exercise_prefix:
                indices.append(i)
        except Exception:
            continue
    return tuple(indices)


@cacheable(maxsize=20)
def get_question_templates(dataset_path: str, split: str) -> Tuple[str, ...]:
    """Get sorted unique question_template values from dataset metadata."""
    jsonl_path = resolve_jsonl_path(dataset_path, split)
    if not jsonl_path:
        return ()
    lines = _load_jsonl_lines(str(jsonl_path))
    templates = set()
    for line in lines:
        try:
            meta = json.loads(line).get('metadata', {})
            if isinstance(meta, str):
                meta = json.loads(meta)
            tmpl = meta.get('question_template', '')
            if tmpl:
                templates.add(tmpl)
        except Exception:
            continue
    return tuple(sorted(templates))


# Mapping from raw CSV error_category strings to normalized groups
_ERROR_LABEL_PATTERNS = [
    (re.compile(r"insufficient|incomplete|range of motion", re.I), {
        re.compile(r"shoulder", re.I): "Insufficient ROM (shoulder)",
        re.compile(r"hip", re.I): "Insufficient ROM (hip)",
        re.compile(r"knee", re.I): "Insufficient ROM (knee)",
        re.compile(r"trunk|spinal|side\s*bend|rotation|extension|flexion", re.I): "Insufficient ROM (trunk)",
        re.compile(r"elbow", re.I): "Insufficient ROM (elbow)",
        re.compile(r"wrist|finger|thumb|hand", re.I): "Insufficient ROM (wrist/hand)",
        re.compile(r"scapul", re.I): "Insufficient ROM (scapular)",
        re.compile(r"plantar|ankle|dorsi", re.I): "Insufficient ROM (ankle)",
        re.compile(r"thoracic", re.I): "Insufficient ROM (thoracic)",
    }),
    (re.compile(r"momentum|swinging|bouncing|speed", re.I), None),
    (re.compile(r"lack of|poor control|jerky|rapid|uncontrolled|abrupt|shaky|choppy", re.I), None),
    (re.compile(r"asymmetr|uneven|one side|lateral\s*shift", re.I), None),
    (re.compile(r"pelvic|pelvis", re.I), None),
    (re.compile(r"knee|valgus|varus", re.I), None),
    (re.compile(r"fatigue|loss of precision|tiring", re.I), None),
]

_ERROR_LABEL_NAMES = {
    0: "Insufficient ROM",  # base name when no sub-match
    1: "Momentum/swinging",
    2: "Poor control/jerky",
    3: "Asymmetry",
    4: "Pelvic drop/rotation",
    5: "Knee misalignment",
    6: "Fatigue",
}


def normalize_error_label(raw_category: str) -> str:
    """Normalize a raw CSV error_category string into a clean group label."""
    if not raw_category:
        return "Other"
    cat = re.sub(r"^\d+\.\s*", "", raw_category)  # Strip leading numbering
    for i, (pattern, sub_patterns) in enumerate(_ERROR_LABEL_PATTERNS):
        if pattern.search(cat):
            if sub_patterns:
                for sub_pat, sub_label in sub_patterns.items():
                    if sub_pat.search(cat):
                        return sub_label
                return _ERROR_LABEL_NAMES.get(i, "Other")
            return _ERROR_LABEL_NAMES.get(i, "Other")
    return "Other"


@cacheable(maxsize=20)
def get_error_labels(dataset_path: str, split: str) -> Tuple[str, ...]:
    """Get sorted unique normalized error labels from dataset metadata."""
    jsonl_path = resolve_jsonl_path(dataset_path, split)
    if not jsonl_path:
        return ()
    lines = _load_jsonl_lines(str(jsonl_path))
    labels = set()
    for line in lines:
        try:
            meta = json.loads(line).get('metadata', {})
            if isinstance(meta, str):
                meta = json.loads(meta)
            verification = meta.get('verification', {})
            raw_cat = verification.get('error_category', '')
            if raw_cat:
                labels.add(normalize_error_label(raw_cat))
        except Exception:
            continue
    return tuple(sorted(labels))


def build_checkpoint_index(models_path: Path = MODELS_BASE_PATH) -> Dict:
    """
    Scan all checkpoints and build index with metadata.

    Returns:
        Dict mapping {task: {variant: [checkpoint_info_dicts]}}
    """
    logging.info("Building checkpoint index...")
    index = {}

    if not models_path.exists():
        logging.warning(f"Models path not found: {models_path}")
        return index

    for cp_dir in models_path.iterdir():
        if not cp_dir.is_dir():
            continue

        # Try to load training_info.json (optional)
        training_info = safe_load_json(cp_dir / "training_info.json")

        # Check if this looks like a valid checkpoint (has model files)
        has_model_files = (cp_dir / "model.safetensors.index.json").exists() or \
                         (cp_dir / "config.json").exists() or \
                         list(cp_dir.glob("model-*.safetensors"))

        if not has_model_files:
            logging.debug(f"Skipping {cp_dir.name} - no model files found")
            continue

        # Get task/variant from experiments CSV mapping (authoritative source)
        # Fall back to parsing name if not in mapping
        if cp_dir.name in MODEL_TO_TASK_VARIANT:
            task, variant = MODEL_TO_TASK_VARIANT[cp_dir.name]
            # Extract step from name
            step_match = re.search(r'step(\d+)', cp_dir.name)
            step = int(step_match.group(1)) if step_match else 0
            logging.debug(f"Using CSV mapping for {cp_dir.name}: {task}/{variant}/step{step}")
        else:
            # Fallback to parsing checkpoint name
            task, variant, step = parse_checkpoint_name(cp_dir.name)
            logging.debug(f"Using name parsing for {cp_dir.name}: {task}/{variant}/step{step}")

        if task not in index:
            index[task] = {}
        if variant not in index[task]:
            index[task][variant] = []

        epoch = step // steps_per_epoch(variant) if step > 0 else 0

        index[task][variant].append({
            'name': cp_dir.name,
            'path': str(cp_dir),
            'step': step,
            'epoch': epoch,
            'training_info': training_info,
            'date': datetime.fromtimestamp(cp_dir.stat().st_mtime).strftime('%Y-%m-%d')
        })

    # Sort by step
    for task in index:
        for variant in index[task]:
            index[task][variant].sort(key=lambda x: x['step'])

    logging.info(f"Checkpoint index built: {len(index)} tasks, "
                f"{sum(len(v) for v in index.values())} variants, "
                f"{sum(len(cps) for task_variants in index.values() for cps in task_variants.values())} checkpoints")
    return index


def load_experiments_csv(csv_path: Path = EXPERIMENTS_CSV_PATH) -> Optional[pd.DataFrame]:
    """Load experiments tracking CSV and build model-to-task-variant mapping."""
    global MODEL_TO_TASK_VARIANT
    try:
        if csv_path.exists():
            df = pd.read_csv(csv_path)

            # Build mapping from model name to (task, variant)
            MODEL_TO_TASK_VARIANT = {}
            for _, row in df.iterrows():
                model_name = row['model']
                task = row['task']
                variant = row['dataset_variant']
                MODEL_TO_TASK_VARIANT[model_name] = (task, variant)

            logging.info(f"Built model mapping for {len(MODEL_TO_TASK_VARIANT)} models")
            return df
        else:
            logging.warning(f"Experiments CSV not found: {csv_path}")
            return None
    except Exception as e:
        logging.error(f"Failed to load experiments CSV: {e}")
        return None


def load_benchmarks_index() -> Optional[Dict]:
    """
    Load IFEval and SIBench benchmark results.

    Returns:
        Dict with structure:
        {
            'ifeval': {
                'baseline': {'prompt_strict': 52.87, 'instr_strict': 64.03},
                'models': {model_name: {metrics...}}
            },
            'sibench': {
                'baseline': {'overall': 38.86},
                'models': {model_name: {metrics...}}
            },
            'summary': {str} # Key findings from BENCHMARKS_TESTED.md
        }
    """
    index = {
        'ifeval': {'baseline': {}, 'models': {}},
        'sibench': {'baseline': {}, 'models': {}},
        'summary': ''
    }

    try:
        # Read BENCHMARKS_TESTED.md for summary (extract only key findings)
        benchmarks_file = VLM_EVAL_ROOT / "results" / "BENCHMARKS_TESTED.md"
        if benchmarks_file.exists():
            with open(benchmarks_file, 'r') as f:
                content = f.read()
                # Extract only Key Findings section for UI display
                key_findings_match = re.search(r'## Key Findings Summary(.*?)(?=##|\Z)', content, re.DOTALL)
                if key_findings_match:
                    index['summary'] = key_findings_match.group(1).strip()
                else:
                    index['summary'] = "No summary available"

        # Load IFEval reports from vlm-evaluation/results/reports/
        reports_dir = VLM_EVAL_ROOT / "results" / "reports"
        if reports_dir.exists():
            # First pass: identify baseline model
            baseline_model_name = None
            baseline_metrics = None

            for report_file in reports_dir.glob("*_report.md"):
                model_name = report_file.stem.replace('_report', '')

                # Detect baseline model (exact match or Qwen patterns)
                is_baseline = (
                    model_name == 'Qwen__Qwen3-VL-4B-Instruct' or
                    model_name == 'qwen3-vl-4b-instruct' or
                    'Qwen3-VL-4B-Instruct' in model_name
                )

                try:
                    with open(report_file, 'r') as f:
                        report_content = f.read()

                        # Extract metrics
                        prompt_match = re.search(r'Prompt-Level\s+Strict.*?(\d+\.\d+)%', report_content)
                        instr_match = re.search(r'Instruction-Level\s+Strict.*?(\d+\.\d+)%', report_content)

                        if prompt_match and instr_match:
                            prompt_strict = float(prompt_match.group(1))
                            instr_strict = float(instr_match.group(1))

                            # Set baseline if this is baseline model
                            if is_baseline and baseline_metrics is None:
                                baseline_model_name = model_name
                                baseline_metrics = {
                                    'prompt_strict': prompt_strict,
                                    'instr_strict': instr_strict
                                }
                                index['ifeval']['baseline'] = baseline_metrics

                except Exception as e:
                    logging.warning(f"Failed to parse IFEval report {report_file}: {e}")

            # Second pass: calculate deltas for all models
            if not index['ifeval']['baseline']:
                # Fallback to hardcoded values if baseline not found
                logging.warning("IFEval baseline model not found in reports, using hardcoded values")
                index['ifeval']['baseline'] = {
                    'prompt_strict': 52.87,
                    'instr_strict': 64.03
                }

            baseline = index['ifeval']['baseline']

            for report_file in reports_dir.glob("*_report.md"):
                model_name = report_file.stem.replace('_report', '')
                try:
                    with open(report_file, 'r') as f:
                        report_content = f.read()

                        # Extract metrics
                        prompt_match = re.search(r'Prompt-Level\s+Strict.*?(\d+\.\d+)%', report_content)
                        instr_match = re.search(r'Instruction-Level\s+Strict.*?(\d+\.\d+)%', report_content)

                        if prompt_match and instr_match:
                            prompt_strict = float(prompt_match.group(1))
                            instr_strict = float(instr_match.group(1))

                            index['ifeval']['models'][model_name] = {
                                'prompt_strict': prompt_strict,
                                'instr_strict': instr_strict,
                                'delta_prompt': prompt_strict - baseline['prompt_strict'],
                                'delta_instr': instr_strict - baseline['instr_strict'],
                                'report_path': str(report_file)
                            }
                except Exception as e:
                    logging.warning(f"Failed to parse IFEval report {report_file}: {e}")

        # Load SIBench results from outputs/sibench/
        sibench_dir = Path("/mnt/data/sgsilva/outputs/sibench")
        if sibench_dir.exists():
            # Look for markdown report files
            report_files = sorted(sibench_dir.glob("report_*.md"), reverse=True)
            if report_files:
                report_path = report_files[0]
                try:
                    with open(report_path, 'r') as f:
                        content = f.read()

                    # Parse each model section
                    # Pattern: ### model_name\n**Run:** ...\n| Task | Correct | Total | Accuracy |
                    model_sections = re.split(r'###\s+(\S+)', content)[1:]  # Skip first empty split

                    for i in range(0, len(model_sections), 2):
                        if i + 1 >= len(model_sections):
                            break

                        model_name = model_sections[i].strip()
                        section_content = model_sections[i + 1]

                        # Extract overall accuracy
                        overall_match = re.search(r'\|\s+\*\*OVERALL\*\*\s+\|[^|]+\|[^|]+\|\s+\*\*(\d+\.?\d*)%\*\*', section_content)
                        if overall_match:
                            overall = float(overall_match.group(1))

                            # Extract per-task accuracies
                            per_task = {}
                            task_matches = re.finditer(r'\|\s+([^|]+?)\s+\|\s+\d+\s+\|\s+\d+\s+\|\s+(\d+\.?\d*)%', section_content)
                            for match in task_matches:
                                task_name = match.group(1).strip()
                                accuracy = float(match.group(2))
                                if task_name != '**OVERALL**':
                                    per_task[task_name] = accuracy

                            # Find result directory
                            model_dir = sibench_dir / model_name
                            result_path = ""
                            if model_dir.exists():
                                timestamp_dirs = sorted([d for d in model_dir.iterdir() if d.is_dir()], reverse=True)
                                if timestamp_dirs:
                                    result_path = str(timestamp_dirs[0])

                            index['sibench']['models'][model_name] = {
                                'overall': overall,
                                'per_task': per_task,
                                'result_path': result_path
                            }

                            # Set baseline if this is the baseline model (exact match)
                            is_baseline = (
                                model_name == 'qwen3-vl-4b-baseline' or
                                model_name.endswith('-baseline') or
                                model_name.startswith('baseline-')
                            )
                            if is_baseline and not index['sibench']['baseline'].get('overall'):
                                index['sibench']['baseline']['overall'] = overall
                                logging.info(f"Set SIBench baseline from model: {model_name} ({overall:.2f}%)")

                except Exception as e:
                    logging.warning(f"Failed to parse SIBench report {report_path}: {e}")

        logging.info(f"Loaded benchmarks: IFEval={len(index['ifeval']['models'])} models, SIBench={len(index['sibench']['models'])} models")
        return index

    except Exception as e:
        logging.error(f"Failed to load benchmarks index: {e}")
        return None


# =============================================================================
# DATA ACCESS FUNCTIONS (Tier 2: Session Cache)
# =============================================================================

def extract_image_path_from_sample(sample: Dict) -> Optional[str]:
    """
    Extract image path from sample, handling multiple JSONL formats.

    Formats supported:
    1. Direct keys: cropped_image, original_image, image
    2. Messages format: image path in messages[user][content][image]

    Args:
        sample: Sample dict from JSONL

    Returns:
        Image path string or None
    """
    # Try direct image keys first
    img_path = sample.get('cropped_image') or sample.get('original_image') or sample.get('image')
    if img_path:
        return img_path

    # Try extracting from messages field (JSON string or dict)
    messages = sample.get('messages')
    if messages:
        try:
            # Parse if it's a JSON string
            if isinstance(messages, str):
                messages = json.loads(messages)

            # Look for image in user message content
            if isinstance(messages, list):
                for msg in messages:
                    if msg.get('role') == 'user':
                        content = msg.get('content', [])
                        if isinstance(content, list):
                            for item in content:
                                if isinstance(item, dict) and 'image' in item:
                                    return item['image']
        except Exception as e:
            logging.debug(f"Failed to parse messages: {e}")

    return None


def extract_annotation_from_sample(sample: Dict) -> Optional[str]:
    """
    Extract ground truth annotation from sample, handling multiple JSONL formats.

    Formats supported:
    1. Direct keys: cropped_annotation, original_annotation, annotation
    2. Messages format: annotation in messages[assistant][content]

    Args:
        sample: Sample dict from JSONL

    Returns:
        Annotation text or None
    """
    # Try direct annotation keys first
    annotation = sample.get('cropped_annotation') or sample.get('original_annotation') or sample.get('annotation')
    if annotation:
        return annotation

    # Try extracting from messages field (JSON string or dict)
    messages = sample.get('messages')
    if messages:
        try:
            # Parse if it's a JSON string
            if isinstance(messages, str):
                messages = json.loads(messages)

            # Look for annotation in assistant message content
            if isinstance(messages, list):
                for msg in messages:
                    if msg.get('role') == 'assistant':
                        content = msg.get('content', '')
                        if isinstance(content, str) and content:
                            return content
        except Exception as e:
            logging.debug(f"Failed to parse messages for annotation: {e}")

    return None


def get_paginated_samples(dataset_path: str, split: str, page: int, page_size: int = 50) -> List[Dict]:
    """Load paginated samples from JSONL dataset (O(1) via cached lines)."""
    jsonl_file = Path(dataset_path) / f"{split}.jsonl"
    if not jsonl_file.exists():
        logging.warning(f"JSONL file not found: {jsonl_file}")
        return []

    try:
        lines = _load_jsonl_lines(str(jsonl_file))
        start_idx = page * page_size
        end_idx = min(start_idx + page_size, len(lines))
        return [json.loads(lines[i]) for i in range(start_idx, end_idx)]
    except Exception as e:
        logging.error(f"Failed to load samples: {e}")
        return []


def load_sample_with_image(dataset_path: str, split: str, sample_idx: int, image_field: str = 'cropped_image') -> Optional[Tuple[Dict, np.ndarray]]:
    """
    Load a specific sample with its image from dataset.

    Args:
        dataset_path: Path to dataset directory
        split: 'train' or 'test'
        sample_idx: Sample index (0-based)
        image_field: Which image field to load ('cropped_image', 'original_image', etc.)

    Returns:
        Tuple of (sample_dict, image_array) or None if not found
    """
    try:
        jsonl_file = Path(dataset_path) / f"{split}.jsonl"
        if not jsonl_file.exists():
            return None

        # Load the specific sample via cached lines (O(1) access)
        lines = _load_jsonl_lines(str(jsonl_file))
        if sample_idx < 0 or sample_idx >= len(lines):
            return None

        sample = json.loads(lines[sample_idx])

        # Get the image path using helper (handles multiple formats)
        image_path = extract_image_path_from_sample(sample)
        if not image_path or not Path(image_path).exists():
            return sample, None

        # Load image
        image = cv2.imread(image_path)
        return sample, image
    except Exception as e:
        logging.error(f"Failed to load sample {sample_idx}: {e}")
        gr.Warning(f"Failed to load sample {sample_idx}: {e}")
        return None


def create_image_gallery(dataset_path: str, split: str, page: int = 0, page_size: int = 50) -> List[str]:
    """
    Create list of image paths for gallery display.

    Args:
        dataset_path: Path to dataset directory
        split: 'train' or 'test'
        page: Page number (0-indexed)
        page_size: Number of images per page

    Returns:
        List of image file paths
    """
    try:
        samples = get_paginated_samples(dataset_path, split, page, page_size)
        if not samples:
            logging.warning(f"No samples found for {dataset_path}/{split} page {page}")
            return []

        image_paths = []
        for sample in samples:
            # Extract image path from sample (handles multiple formats)
            img_path = extract_image_path_from_sample(sample)
            if img_path and Path(img_path).exists():
                image_paths.append(img_path)
            else:
                logging.debug(f"Image not found: {img_path}")

        return image_paths
    except Exception as e:
        logging.error(f"Error creating image gallery: {e}")
        return []


def create_filtered_image_gallery(
    dataset_path: str, split: str, indices: Tuple[int, ...],
    page: int = 0, page_size: int = 50
) -> List[str]:
    """Create gallery from specific sample indices (for filtered views)."""
    try:
        start = page * page_size
        end = min(start + page_size, len(indices))
        if start >= len(indices):
            return []

        jsonl_path = resolve_jsonl_path(dataset_path, split)
        if not jsonl_path:
            return []
        lines = _load_jsonl_lines(str(jsonl_path))

        image_paths = []
        for idx in indices[start:end]:
            if idx < len(lines):
                sample = json.loads(lines[idx])
                img_path = extract_image_path_from_sample(sample)
                if img_path and Path(img_path).exists():
                    image_paths.append(img_path)
        return image_paths
    except Exception as e:
        logging.error(f"Error creating filtered image gallery: {e}")
        return []


def get_sample_visualization(
    dataset_path: str,
    split: str,
    sample_idx: int,
    show_annotation: bool = True,
    keypoint_subset: str = 'coco25'
) -> Optional[np.ndarray]:
    """
    Get annotated image for a sample.

    Args:
        dataset_path: Path to dataset directory
        split: 'train' or 'test'
        sample_idx: Sample index
        show_annotation: Whether to draw keypoints
        keypoint_subset: Keypoint subset to use

    Returns:
        Annotated image as numpy array or None
    """
    try:
        result = load_sample_with_image(dataset_path, split, sample_idx, 'cropped_image')
        if not result:
            logging.warning(f"Could not load sample {sample_idx} from {dataset_path}/{split}")
            return None

        sample, image = result
        if image is None:
            logging.warning(f"Image data is None for sample {sample_idx}")
            return None

        if show_annotation:
            # Get ground truth annotation using helper (handles multiple formats)
            annotation_text = extract_annotation_from_sample(sample)
            if annotation_text:
                try:
                    image = visualize_keypoints_on_image(
                        image,
                        annotation_text,
                        keypoint_subset=keypoint_subset,
                        color=COLOR_GREEN
                    )
                except Exception as viz_error:
                    logging.error(f"Error visualizing keypoints: {viz_error}")
                    # Return image without annotations rather than failing completely
                    pass

        return image
    except Exception as e:
        logging.error(f"Error in get_sample_visualization: {e}")
        gr.Warning(f"Failed to load visualization: {e}")
        return None


# Placeholder for future implementation
# This file is created in Phase 1, additional functions will be added in subsequent phases

# Configure logging to write to logs directory
LOGS_DIR = Path(__file__).parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)
log_file = LOGS_DIR / f"app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  # Also log to console
    ]
)
logging.info(f"Monitoring app module loaded successfully. Logging to: {log_file}")

@cacheable(maxsize=50)
def load_evaluation_results(result_file: str) -> Optional[Dict]:
    """
    Load and cache evaluation results JSON.

    Args:
        result_file: Path to results JSON file

    Returns:
        Results dict or None
    """
    return safe_load_json(Path(result_file))


def find_result_file(checkpoint_name: str, results_dir: Path = RESULTS_BASE_PATH, task: str = None, variant: str = None, split: str = "test") -> Optional[str]:
    """
    Find evaluation results JSON for a checkpoint (fine-tuned or baseline).

    Args:
        checkpoint_name: Name of checkpoint directory or baseline model
        results_dir: Directory containing results
        task: Task name (optional, used for baseline models)
        variant: Variant name (optional, used for baseline models)
        split: 'test' or 'train'

    Returns:
        Path to results file or None
    """
    if not results_dir.exists():
        return None

    # Special handling for mixed tasks
    if task == 'mixed':
        # Try v1 format: mixed_{model_name}_test*.json
        pattern = f"mixed_*{checkpoint_name}*.json"
        matches = list(results_dir.glob(pattern))
        if matches:
            return str(sorted(matches)[-1])
        # Try v2 format: task1_{variant}_test_{checkpoint}*.json (return first task file found)
        for prefix in ['task1_', 'task1b_', 'task1c_', 'task3b_', 'task3c_', 'task2_', 'task3d_', 'task4_']:
            pattern = f"{prefix}*{checkpoint_name}*.json"
            matches = [m for m in results_dir.glob(pattern) if not str(m).endswith('.checkpoint.json')]
            if matches:
                return str(sorted(matches)[-1])
        return None

    # Check if this is a baseline model or fine-tuned checkpoint
    parsed_task, parsed_variant, step = parse_checkpoint_name(checkpoint_name)

    # Use provided task/variant if available, otherwise use parsed values
    task = task or parsed_task
    variant = variant or parsed_variant

    if step > 0:
        # Fine-tuned model with step: task1_cropped_v1_test_step646_*.json
        pattern = f"{task}_{variant}_{split}_step{step}_*.json"
        matches = list(results_dir.glob(pattern))
    else:
        # Baseline model without step: task1_cropped_v1_test_4b_*.json
        # Extract model shortname (4b, 235b, 8b, gemini, etc.)
        model_shortname = None
        if '4b' in checkpoint_name.lower():
            model_shortname = '4b'
        elif '8b' in checkpoint_name.lower():
            model_shortname = '8b'
        elif '235b' in checkpoint_name.lower():
            model_shortname = '235b'
        elif 'gemini' in checkpoint_name.lower():
            model_shortname = 'gemini3'

        if model_shortname and task != 'unknown' and variant != 'unknown':
            # Try baseline-specific pattern first, then broader fallback
            pattern = f"{task}_{variant}_{split}_baseline_{model_shortname}_*.json"
            matches = list(results_dir.glob(pattern))
            if not matches:
                # Fallback: broader pattern but exclude step results
                pattern = f"{task}_{variant}_{split}*_{model_shortname}_*.json"
                matches = [m for m in results_dir.glob(pattern) if '_step' not in m.name]
        else:
            # Fallback: try any file matching task and variant (exclude step results)
            pattern = f"{task}_{variant}_{split}_*.json"
            matches = [m for m in results_dir.glob(pattern) if '_step' not in m.name]

    if not matches and 'mixed' in checkpoint_name.lower() and step > 0:
        # Mixed model fallback: results stored under mixed_vN variant, not single-task variant.
        # E.g. checkpoint "qwen3-vl-4b-4epochs-mixed-v3-step4500" with task=task1, variant=cropped_v1
        # → actual file: task1c_mixed_v3_test_step4500_*.json
        # First try specific mixed variant pattern, then broad fallback
        mixed_match = re.search(r'mixed[-_](.+?)[-_]step\d+', checkpoint_name.lower())
        if mixed_match:
            mixed_variant = 'mixed_' + mixed_match.group(1).replace('-', '_')
            specific_pattern = f"{task}*_{mixed_variant}_{split}_step{step}_*.json"
            matches = list(results_dir.glob(specific_pattern))
        if not matches:
            # Broad fallback: any mixed result for this task/step
            mixed_pattern = f"{task}*_mixed_*_{split}_step{step}_*.json"
            matches = list(results_dir.glob(mixed_pattern))

    if not matches:
        return None
    # Return most recent if multiple matches
    return str(sorted(matches)[-1])


def find_comparison_report(task: str, variant: str, evaluations_dir: Path = EVALUATIONS_PATH) -> Optional[str]:
    """
    Find pre-generated checkpoint comparison report.

    Args:
        task: Task name (e.g., "task1")
        variant: Variant name (e.g., "cropped_v1")
        evaluations_dir: Directory containing comparison reports

    Returns:
        Path to report file or None
    """
    # Pattern: checkpoint_comparison_4b_task{N}_{variant}.txt
    pattern = f"checkpoint_comparison_4b_{task}_{variant}.txt"

    report_file = evaluations_dir / pattern
    if report_file.exists():
        return str(report_file)
    return None


@cacheable(maxsize=20)
def load_comparison_report(report_file: str) -> str:
    """Load and cache comparison report text."""
    text = safe_load_text(Path(report_file))
    return text if text else "Report not available"


# =============================================================================
# UI HELPER FUNCTIONS
# =============================================================================

def create_stats_card(task: str, variant: str, split: str) -> str:
    """Create dataset statistics markdown card."""
    if task not in DATASET_INDEX or variant not in DATASET_INDEX[task]:
        return "### Dataset Statistics\n\nNo dataset found for this selection."

    dataset_info = DATASET_INDEX[task][variant]
    stats = dataset_info.get(f'{split}_stats', {})
    num_samples = dataset_info.get(f'{split}_samples', 0)
    # Fallback: count JSONL lines if stats file was missing
    if num_samples == 0:
        num_samples = count_mcqa_samples(dataset_info['path'], split)

    return f"""### Dataset Statistics

**Task**: {TASK_NAMES.get(task, task)}
**Variant**: {variant}
**Split**: {split}
**Samples**: {num_samples:,}
**Path**: `{dataset_info['path']}`
"""


def create_variant_summary_table(task: str, variant: str) -> str:
    """Create a markdown summary table of all variants for a task with key metrics."""
    if task not in DATASET_INDEX or EXPERIMENT_INDEX is None or EXPERIMENT_INDEX.empty:
        return ""

    task_experiments = EXPERIMENT_INDEX[EXPERIMENT_INDEX['task'] == task]

    # Pick primary metric per task
    if task in ('task1', 'task1b', 'task1c'):
        metric_col = "oks_score"
    elif task == 'task2':
        metric_col = "per_keypoint_accuracy"
    elif task.startswith('task3'):
        metric_col = "f1_score"
    elif task == 'task4':
        metric_col = "accuracy"
    else:
        metric_col = "oks_score"

    rows = ""
    for v, v_info in DATASET_INDEX.get(task, {}).items():
        train_n = v_info.get('train_samples', 0)
        test_n = v_info.get('test_samples', 0)

        # Best SFT
        v_exp = task_experiments[
            (task_experiments['dataset_variant'] == v) &
            (task_experiments['is_best'] == True) &
            (task_experiments['is_sft'] == True)
        ]
        # Baseline
        v_base = task_experiments[
            (task_experiments['dataset_variant'] == v) &
            (task_experiments['is_sft'] == False) &
            (task_experiments['is_best'] == True)
        ]

        base_val = ""
        if not v_base.empty and metric_col in v_base.columns:
            val = v_base.iloc[0].get(metric_col)
            if pd.notna(val):
                base_val = f"{val:.3f}" if val < 1 else f"{val:.1f}%"

        best_val, best_step = "", ""
        if not v_exp.empty and metric_col in v_exp.columns:
            row = v_exp.iloc[0]
            val = row.get(metric_col)
            if pd.notna(val):
                best_val = f"{val:.3f}" if val < 1 else f"{val:.1f}%"
            model = row.get('model', '')
            step_match = re.search(r'step(\d+)', model)
            if step_match:
                best_step = f"step{step_match.group(1)}"

        marker = " **←**" if v == variant else ""
        rows += f"| {v}{marker} | {train_n:,} | {test_n:,} | {base_val} | {best_val} | {best_step} |\n"

    if not rows:
        return ""

    return f"""### {TASK_NAMES.get(task, task)} — All Variants

| Variant | Train | Test | Baseline | Best SFT | Step |
|---------|------:|-----:|---------:|---------:|------|
{rows}"""


def get_task_prompt(task: str, variant: str, split: str = 'test') -> str:
    """
    Extract the prompt/instruction for a specific task and variant.
    Loads one sample from the dataset to get the prompt.
    """
    # Special handling for mixed tasks - show variant-specific description
    if task == 'mixed':
        if variant and 'v2_phase1' in variant:
            return """**Mixed v2 Phase 1 — Coordinate Foundation**

Two-phase curriculum training. Phase 1 focuses on coordinate regression:
- **Task 1 (25kp)**: Full skeleton keypoint detection (~20%)
- **Task 1b (17kp)**: COCO-17 keypoint detection (~20%)
- **Task 1c (12kp)**: Body-only keypoint detection (~20%)
- **Task 3b**: Missing keypoint correction (~20%)
- **Task 3c**: Displaced keypoint correction (~20%)

15,000 train samples (3K each), 3 epochs, LR 1e-5."""
        elif variant and 'v2_phase2' in variant:
            return """**Mixed v2 Phase 2 — Full Multi-Task**

Initialized from Phase 1 best checkpoint, adds classification tasks:
- **Task 1/1b/1c**: Coordinate regression (2K each, 36%)
- **Task 2**: Keypoint labeling (2K, 12%)
- **Task 3b/3c/3d**: Error correction (1.5K each, 27%)
- **Task 4 V5.3/V6.2**: Pose understanding (1.5K + 2.5K, 24%)

16,500 train samples, 3 epochs, LR 5e-6 (lower to protect Phase 1 learning)."""
        return """**Multi-Task Dataset (balanced_v1)**

This dataset contains samples from 4 different task types:
- **Task 1: Keypoint Detection** (~25% of samples)
- **Task 2: Keypoint Labeling** (~25% of samples)
- **Task 3: Error Detection** (~25% of samples)
- **Task 4: Exercise MCQA** (~25% of samples)

Each sample has its own task-specific instruction."""

    if task not in DATASET_INDEX or variant not in DATASET_INDEX[task]:
        return "No dataset found for this selection."

    dataset_path = DATASET_INDEX[task][variant]['path']
    jsonl_file = Path(dataset_path) / f"{split}.jsonl"

    if not jsonl_file.exists():
        return f"Dataset file not found: {jsonl_file}"

    try:
        # Read first sample to extract prompt
        with open(jsonl_file, 'r') as f:
            first_line = f.readline()
            if not first_line:
                return "Empty dataset file"

            sample = json.loads(first_line)

            # Extract prompt from messages (new format)
            messages = sample.get('messages')
            if messages:
                if isinstance(messages, str):
                    messages = json.loads(messages)

                # Find user message with prompt
                for msg in messages:
                    if msg.get('role') == 'user':
                        content = msg.get('content', [])
                        # Extract text from content (skip images)
                        if isinstance(content, str):
                            return content
                        prompt_parts = []
                        for item in content:
                            if isinstance(item, dict):
                                if 'text' in item:
                                    prompt_parts.append(item['text'])
                            elif isinstance(item, str):
                                prompt_parts.append(item)

                        if prompt_parts:
                            return '\n'.join(prompt_parts)

            # Fallback: check if there's a direct prompt field
            if 'prompt' in sample:
                return sample['prompt']

            return "Prompt not found in dataset format"

    except Exception as e:
        logging.error(f"Error extracting prompt: {e}")
        return f"Error loading prompt: {str(e)}"


def create_checkpoint_list(task: str, variant: str) -> List[str]:
    """
    Get list of EVALUATED checkpoints for task/variant.
    Only returns checkpoints that have evaluation results in EXPERIMENT_INDEX.
    Includes both fine-tuned checkpoints and baseline models.
    """
    checkpoint_names = []

    if EXPERIMENT_INDEX is None:
        return checkpoint_names

    # Special handling for mixed tasks
    if task == 'mixed':
        # Filter by specific variant if provided, else show all mixed variants
        mixed_variant = variant if variant and variant.startswith('mixed_') else f"mixed_{variant}" if variant else None
        if mixed_variant:
            evaluated_models = EXPERIMENT_INDEX[
                EXPERIMENT_INDEX['dataset_variant'] == mixed_variant
            ].copy()
        else:
            evaluated_models = EXPERIMENT_INDEX[
                EXPERIMENT_INDEX['dataset_variant'].str.startswith('mixed_', na=False)
            ].copy()

        # Filter out archived variants
        evaluated_models = evaluated_models[
            ~evaluated_models['dataset_variant'].apply(lambda v: is_archived('mixed', v))
        ]

        # Group by model name (each model has 4 rows - one per task)
        unique_models = evaluated_models.groupby('model').first().reset_index()

        for _, row in unique_models.iterrows():
            model_name = row['model']
            is_sft = row['is_sft']

            if is_sft:
                if model_name not in checkpoint_names:
                    checkpoint_names.append(model_name)
            else:
                baseline_entry = f"[Baseline] {model_name}"
                if baseline_entry not in checkpoint_names:
                    checkpoint_names.append(baseline_entry)
    else:
        # Standard single-task handling - include subtasks
        task_filter = EXPERIMENT_INDEX['task'].str.startswith(task)

        variant_filter = EXPERIMENT_INDEX['dataset_variant'].isin(normalize_variant_aliases(variant))
        evaluated_models = EXPERIMENT_INDEX[
            task_filter & variant_filter
        ]

        for _, row in evaluated_models.iterrows():
            model_name = row['model']
            is_sft = row['is_sft']

            if is_sft:
                # Fine-tuned model - use name as-is
                if model_name not in checkpoint_names:
                    checkpoint_names.append(model_name)
            else:
                # Baseline model - add prefix
                baseline_entry = f"[Baseline] {model_name}"
                if baseline_entry not in checkpoint_names:
                    checkpoint_names.append(baseline_entry)

        # Also include mixed model checkpoints AND their baselines for this task
        mixed_models = EXPERIMENT_INDEX[
            task_filter &
            (EXPERIMENT_INDEX['dataset_variant'].str.contains('mixed', na=False))
        ]
        for _, row in mixed_models.iterrows():
            model_name = row['model']
            row_variant = row['dataset_variant']
            is_sft = row['is_sft']
            # Skip archived variants (check both task-specific and 'mixed' archives)
            if is_archived(task, row_variant) or is_archived('mixed', row_variant):
                continue
            if is_sft:
                if model_name not in checkpoint_names:
                    checkpoint_names.append(model_name)
            else:
                baseline_entry = f"[Baseline] {model_name}"
                if baseline_entry not in checkpoint_names:
                    checkpoint_names.append(baseline_entry)

    return checkpoint_names


def create_checkpoint_table(task: str, variant: str = None, all_variants: bool = False) -> pd.DataFrame:
    """
    Create checkpoint table for Training Monitor tab.
    Uses experiments-final.csv as source to show all evaluated checkpoints,
    even if model files have been deleted.

    Args:
        task: Task name
        variant: Variant name (optional if all_variants=True)
        all_variants: If True, show all checkpoints for task across all variants

    Returns:
        DataFrame with checkpoint information
    """
    if EXPERIMENT_INDEX is None:
        return pd.DataFrame(columns=["Checkpoint", "Variant", "Epoch", "Step", "Consumed Samples", "Date", "Status", "OKS/Acc"])

    # Special handling for mixed tasks
    if task == 'mixed':
        # Filter by specific variant if provided, else show all mixed variants
        if variant and not variant.startswith('mixed_'):
            variant = f"mixed_{variant}"
        if variant:
            # Also include MCQA-prefixed rows (e.g. mcqa_v5.3_mixed_final_a for task4)
            experiments = EXPERIMENT_INDEX[
                (EXPERIMENT_INDEX['dataset_variant'] == variant) |
                (EXPERIMENT_INDEX['dataset_variant'].str.endswith(f'_{variant}', na=False))
            ].copy()
        else:
            experiments = EXPERIMENT_INDEX[
                EXPERIMENT_INDEX['dataset_variant'].str.startswith('mixed_', na=False) |
                EXPERIMENT_INDEX['dataset_variant'].str.contains('_mixed_', na=False)
            ].copy()

        # Determine the variant for config lookup
        mixed_variant = variant or (experiments['dataset_variant'].iloc[0] if not experiments.empty else 'mixed_balanced_v1')
        config = get_mixed_config(mixed_variant)
        task_defs = config['tasks']
        mixed_spe = config['steps_per_epoch']

        # Build dynamic column names
        metric_cols = [col_name for _, (_, col_name) in task_defs.items()]
        empty_cols = ["Checkpoint", "Variant", "Step", "Date", "Status"] + metric_cols

        if experiments.empty:
            return pd.DataFrame(columns=empty_cols)

        # Group by model (each model has N rows - one per task)
        experiments['step'] = experiments['model'].apply(extract_step)
        experiments['epoch'] = experiments['step'] // max(1, mixed_spe)

        # Group by model name and aggregate metrics
        grouped = experiments.groupby('model').first().reset_index()

        # Build table with dynamic task columns
        table_data = []
        for _, model_row in grouped.iterrows():
            model_name = model_row['model']
            task_rows = experiments[experiments['model'] == model_name]

            row_data = {
                "Checkpoint": model_name,
                "Variant": mixed_variant,
                "Step": model_row['step'],
                "Date": model_row.get('date', model_row.get('timestamp', '')[:10] if pd.notna(model_row.get('timestamp')) else ''),
                "Status": "Evaluated" if not model_row.get('is_sft', False) else "Fine-tuned",
            }

            # Add dynamic task metric columns
            for task_key, (metric_col, col_name) in task_defs.items():
                t_row = task_rows[task_rows['task'] == task_key]
                if not t_row.empty and metric_col in t_row.columns and pd.notna(t_row.iloc[0].get(metric_col)):
                    row_data[col_name] = f"{float(t_row.iloc[0][metric_col]):.3f}"
                else:
                    row_data[col_name] = "-"

            table_data.append(row_data)

        df = pd.DataFrame(table_data)
        df = df.sort_values('Step', ascending=True)
        return df

    # Filter experiments by task (non-mixed tasks)
    if all_variants:
        # Show all variants for this task - include subtasks
        # Filter orphan baselines (only show baselines for variants with SFT checkpoints)
        task_filter = EXPERIMENT_INDEX['task'].str.startswith(task)
        sft_only = EXPERIMENT_INDEX[
            task_filter &
            (EXPERIMENT_INDEX['is_sft'] == True)
        ]
        sft_variants = set(sft_only['dataset_variant'].unique())
        # Filter out archived variants
        sft_variants = {v for v in sft_variants if not is_archived(task, v)}
        experiments = EXPERIMENT_INDEX[
            task_filter &
            (EXPERIMENT_INDEX['dataset_variant'].isin(sft_variants))
        ].copy()
    else:
        # Show specific variant - include subtasks
        if variant is None:
            return pd.DataFrame(columns=["Checkpoint", "Variant", "Epoch", "Step", "Consumed Samples", "Date", "Status", "OKS/Acc"])
        task_filter = EXPERIMENT_INDEX['task'].str.startswith(task)
        experiments = EXPERIMENT_INDEX[
            task_filter &
            EXPERIMENT_INDEX['dataset_variant'].isin(normalize_variant_aliases(variant))
        ].copy()

    if experiments.empty:
        return pd.DataFrame(columns=["Checkpoint", "Variant", "Epoch", "Step", "Consumed Samples", "Date", "Status", "OKS/Acc"])

    # Parse step numbers from model names (using global extract_step function)
    experiments['step'] = experiments['model'].apply(extract_step)
    experiments['epoch'] = experiments.apply(
        lambda row: row['step'] // steps_per_epoch(row.get('dataset_variant')) if row['step'] > 0 else 0,
        axis=1
    )

    # Check if model files still exist
    def check_model_exists(model_name):
        model_path = MODELS_BASE_PATH / model_name
        return model_path.exists()

    # Find best checkpoint
    best_idx = None
    if task in ['task1', 'task1b', 'task1c', 'task3a', 'task3b', 'task3c', 'task3d']:
        # Use OKS for keypoint detection/correction tasks
        metric_col = 'oks_score'
        best_idx = experiments[metric_col].idxmax() if metric_col in experiments.columns else None
    elif task == 'task2':
        # Use per_keypoint_accuracy for labeling tasks
        if 'per_keypoint_accuracy' in experiments.columns:
            best_idx = experiments['per_keypoint_accuracy'].idxmax()
    elif task == 'task4':
        # Use accuracy for MCQA tasks
        if 'accuracy' in experiments.columns:
            best_idx = experiments['accuracy'].idxmax()

    # Build table data
    table_data = []
    for idx, row in experiments.iterrows():
        model_name = row['model']
        is_sft = row.get('is_sft', False)

        # Determine status
        if idx == best_idx:
            status = "✓ Best"
        elif is_sft:
            status = "Evaluated (Fine-tuned)"
        else:
            status = "Evaluated (Baseline)"

        # Check if model files exist
        model_exists = check_model_exists(model_name.replace('[Baseline] ', ''))
        if not model_exists and is_sft:
            status += " [Deleted]"

        # Build row with all available metrics
        row_data = {
            "Checkpoint": model_name,
            "Variant": row['dataset_variant'],
            "Epoch": row['epoch'],
            "Step": row['step'],
            "Samples": row.get('num_samples', row['step']),
            "Date": row.get('date', row.get('timestamp', '')[:10] if pd.notna(row.get('timestamp')) else ''),
            "Status": status,
        }

        # Add task-specific metrics
        if task in ['task1', 'task1b', 'task1c', 'task3a', 'task3b', 'task3c', 'task3d']:
            # Keypoint detection/correction tasks
            row_data["OKS"] = f"{row.get('oks_score', 0):.3f}" if pd.notna(row.get('oks_score')) else "-"
            row_data["OKS (CW)"] = f"{row.get('oks_confidence_weighted', 0):.3f}" if pd.notna(row.get('oks_confidence_weighted')) else "-"
            row_data["F1"] = f"{row.get('f1_score', 0):.3f}" if pd.notna(row.get('f1_score')) else "-"
            row_data["Precision"] = f"{row.get('precision', 0):.3f}" if pd.notna(row.get('precision')) else "-"
            row_data["Recall"] = f"{row.get('recall', 0):.3f}" if pd.notna(row.get('recall')) else "-"
            row_data["MAE"] = f"{row.get('mae_total', 0):.1f}" if pd.notna(row.get('mae_total')) else "-"
            row_data["PCK@50"] = f"{row.get('pck_50', 0):.3f}" if pd.notna(row.get('pck_50')) else "-"
            row_data["L/R Confusion"] = f"{row.get('left_right_confusion', 0):.3f}" if pd.notna(row.get('left_right_confusion')) else "-"
        elif task == 'task2':
            # Keypoint labeling task
            row_data["Accuracy"] = f"{row.get('per_keypoint_accuracy', 0):.3f}" if pd.notna(row.get('per_keypoint_accuracy')) else "-"
            row_data["L/R Confusion"] = f"{row.get('left_right_confusion', 0):.3f}" if pd.notna(row.get('left_right_confusion')) else "-"
            row_data["Exact Match"] = f"{row.get('exact_match', 0):.3f}" if pd.notna(row.get('exact_match')) else "-"
        elif task == 'task4':
            # MCQA task — only accuracy and parse_rate matter
            row_data["Accuracy"] = f"{row.get('accuracy', 0):.3f}" if pd.notna(row.get('accuracy')) else "-"
            row_data["Parse Rate"] = f"{row.get('parse_rate', 0):.3f}" if pd.notna(row.get('parse_rate')) else "-"

        table_data.append(row_data)

    df = pd.DataFrame(table_data)
    # Sort by step (ascending)
    df = df.sort_values('Step', ascending=True)
    return df


def create_mixed_vs_single_comparison() -> tuple:
    """
    Generate dynamic mixed vs single-task comparison from experiments-final.csv.
    Returns (summary_markdown, plotly_figure).
    """
    if EXPERIMENT_INDEX is None or EXPERIMENT_INDEX.empty:
        return "No experiment data available.", empty_figure("No experiment data")

    # Task-specific primary metrics
    task_metrics = {
        'task1': ('oks_score', 'OKS'),
        'task2': ('per_keypoint_accuracy', 'Keypoint Accuracy'),
        'task3': ('f1_score', 'F1 Score'),
        'task4': ('accuracy', 'Accuracy'),
    }

    # Get mixed-task data
    mixed = EXPERIMENT_INDEX[
        EXPERIMENT_INDEX['dataset_variant'].str.startswith('mixed_', na=False)
    ].copy()

    if mixed.empty:
        return "No mixed-task evaluations found in experiments-final.csv.", empty_figure("No mixed-task evaluations")

    mixed['step'] = mixed['model'].apply(extract_step)

    # Get single-task SFT data (exclude mixed)
    single_sft = EXPERIMENT_INDEX[
        (EXPERIMENT_INDEX['is_sft'] == True) &
        (~EXPERIMENT_INDEX['dataset_variant'].str.startswith('mixed_', na=False))
    ].copy()

    # Get baselines
    baselines = EXPERIMENT_INDEX[
        (EXPERIMENT_INDEX['is_sft'] == False)
    ].copy()

    # Build summary per task
    summary_parts = ["# Mixed-Task vs Single-Task Comparison\n",
                     "*Live data from experiments-final.csv*\n"]
    comparison_rows = []

    for task_key, (metric_col, metric_label) in task_metrics.items():
        mixed_task = mixed[mixed['task'] == task_key]
        single_task = single_sft[single_sft['task'] == task_key]
        baseline_task = baselines[baselines['task'] == task_key]

        if mixed_task.empty:
            continue

        # Best mixed
        if metric_col in mixed_task.columns and mixed_task[metric_col].notna().any():
            best_mixed_idx = mixed_task[metric_col].idxmax()
            best_mixed_val = mixed_task.loc[best_mixed_idx, metric_col]
            best_mixed_model = mixed_task.loc[best_mixed_idx, 'model']
            best_mixed_step = extract_step(best_mixed_model)
        else:
            best_mixed_val = None
            best_mixed_model = "N/A"
            best_mixed_step = 0

        # Best single
        if not single_task.empty and metric_col in single_task.columns and single_task[metric_col].notna().any():
            best_single_idx = single_task[metric_col].idxmax()
            best_single_val = single_task.loc[best_single_idx, metric_col]
            best_single_model = single_task.loc[best_single_idx, 'model']
            best_single_variant = single_task.loc[best_single_idx, 'dataset_variant']
        else:
            best_single_val = None
            best_single_model = "N/A"
            best_single_variant = ""

        # Baseline
        if not baseline_task.empty and metric_col in baseline_task.columns and baseline_task[metric_col].notna().any():
            best_baseline_val = baseline_task[metric_col].max()
        else:
            best_baseline_val = None

        comparison_rows.append({
            'task': task_key,
            'metric_label': metric_label,
            'mixed_val': best_mixed_val,
            'mixed_model': best_mixed_model,
            'mixed_step': best_mixed_step,
            'single_val': best_single_val,
            'single_model': best_single_model,
            'single_variant': best_single_variant,
            'baseline_val': best_baseline_val,
        })

    # Build summary table
    summary_parts.append("| Task | Metric | Mixed Best | Single Best | Baseline | Winner |")
    summary_parts.append("|------|--------|-----------|-------------|----------|--------|")

    for row in comparison_rows:
        m_val = f"{row['mixed_val']:.4f}" if row['mixed_val'] is not None else "N/A"
        s_val = f"{row['single_val']:.4f}" if row['single_val'] is not None else "N/A"
        b_val = f"{row['baseline_val']:.4f}" if row['baseline_val'] is not None else "N/A"

        # Determine winner
        if row['mixed_val'] is not None and row['single_val'] is not None:
            if row['mixed_val'] > row['single_val']:
                winner = "Mixed"
            elif row['single_val'] > row['mixed_val']:
                winner = "Single"
            else:
                winner = "Tie"
        else:
            winner = "—"

        task_name = TASK_NAMES.get(row['task'], row['task'])
        summary_parts.append(
            f"| {task_name} | {row['metric_label']} | "
            f"**{m_val}** (step{row['mixed_step']}) | "
            f"**{s_val}** ({row['single_variant']}) | "
            f"{b_val} | {winner} |"
        )

    # Mixed checkpoint count
    mixed_steps = sorted(mixed['step'].unique())
    single_count = len(single_sft['model'].unique())
    summary_parts.append(f"\n**Mixed checkpoints:** {len(mixed_steps)} steps ({', '.join(f'step{s}' for s in mixed_steps)})")
    summary_parts.append(f"**Single-task checkpoints:** {single_count} models")

    summary_md = "\n".join(summary_parts)

    # Build comparison plot: grouped bar chart per task
    fig = go.Figure()

    tasks_with_data = [r for r in comparison_rows if r['mixed_val'] is not None or r['single_val'] is not None]
    task_labels = [TASK_NAMES.get(r['task'], r['task']).replace('Task ', 'T') for r in tasks_with_data]

    mixed_vals = [r['mixed_val'] if r['mixed_val'] is not None else 0 for r in tasks_with_data]
    single_vals = [r['single_val'] if r['single_val'] is not None else 0 for r in tasks_with_data]
    baseline_vals = [r['baseline_val'] if r['baseline_val'] is not None else 0 for r in tasks_with_data]

    fig.add_trace(go.Bar(
        name='Mixed-Task Best',
        x=task_labels, y=mixed_vals,
        marker_color='#3b82f6',
        text=[f"{v:.3f}" for v in mixed_vals],
        textposition='outside'
    ))
    fig.add_trace(go.Bar(
        name='Single-Task Best',
        x=task_labels, y=single_vals,
        marker_color='#ef4444',
        text=[f"{v:.3f}" for v in single_vals],
        textposition='outside'
    ))
    fig.add_trace(go.Bar(
        name='Baseline',
        x=task_labels, y=baseline_vals,
        marker_color='#9ca3af',
        text=[f"{v:.3f}" for v in baseline_vals],
        textposition='outside'
    ))

    fig.update_layout(
        title="Mixed vs Single-Task: Best Checkpoint per Task",
        barmode='group',
        yaxis_title="Metric Value",
        paper_bgcolor='#FAF9F6',
        plot_bgcolor='white',
        font=dict(family="Inter", color='#292524'),
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )

    # Also create a progression plot for task1 OKS (most important comparison)
    return summary_md, fig


def create_mixed_vs_single_progression() -> go.Figure:
    """Create OKS progression plot comparing mixed vs best single-task for Task 1."""
    if EXPERIMENT_INDEX is None:
        return empty_figure("No experiment data")

    fig = go.Figure()

    # Mixed task1 data
    mixed_t1 = EXPERIMENT_INDEX[
        (EXPERIMENT_INDEX['dataset_variant'].str.startswith('mixed_', na=False)) &
        (EXPERIMENT_INDEX['task'] == 'task1') &
        (EXPERIMENT_INDEX['oks_score'].notna())
    ].copy()

    if not mixed_t1.empty:
        mixed_t1['step'] = mixed_t1['model'].apply(extract_step)
        mixed_t1 = mixed_t1.sort_values('step')
        fig.add_trace(go.Scatter(
            x=mixed_t1['step'], y=mixed_t1['oks_score'],
            mode='lines+markers', name='Mixed-Task',
            line=dict(color='#3b82f6', width=3),
            marker=dict(size=10)
        ))

    # Best single-task variant for task1 (by max OKS)
    single_t1 = EXPERIMENT_INDEX[
        (EXPERIMENT_INDEX['task'] == 'task1') &
        (EXPERIMENT_INDEX['is_sft'] == True) &
        (~EXPERIMENT_INDEX['dataset_variant'].str.startswith('mixed_', na=False)) &
        (EXPERIMENT_INDEX['oks_score'].notna())
    ].copy()

    if not single_t1.empty:
        # Find best variant
        best_variant = single_t1.groupby('dataset_variant')['oks_score'].max().idxmax()
        best_single = single_t1[single_t1['dataset_variant'] == best_variant].copy()
        best_single['step'] = best_single['model'].apply(extract_step)
        best_single = best_single.sort_values('step')
        fig.add_trace(go.Scatter(
            x=best_single['step'], y=best_single['oks_score'],
            mode='lines+markers', name=f'Single-Task ({best_variant})',
            line=dict(color='#ef4444', width=3),
            marker=dict(size=10)
        ))

    # Baseline
    baseline_t1 = EXPERIMENT_INDEX[
        (EXPERIMENT_INDEX['task'] == 'task1') &
        (EXPERIMENT_INDEX['is_sft'] == False) &
        (EXPERIMENT_INDEX['oks_score'].notna())
    ]
    if not baseline_t1.empty:
        for _, row in baseline_t1.iterrows():
            fig.add_hline(
                y=row['oks_score'],
                line_dash="dash", line_color="#9ca3af",
                annotation_text=f"Baseline: {row['model'][:30]} ({row['oks_score']:.3f})",
                annotation_position="bottom right"
            )

    fig.update_layout(
        title="Task 1 OKS Progression: Mixed vs Single-Task",
        xaxis_title="Training Step",
        yaxis_title="OKS Score",
        paper_bgcolor='#FAF9F6',
        plot_bgcolor='white',
        font=dict(family="Inter", color='#292524'),
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    return fig


def create_metrics_plot(task: str, variant: str = None, all_variants: bool = False, metric: str = "OKS") -> go.Figure:
    """
    Create metrics progression plot for Training Monitor tab.
    Uses experiments-final.csv to show metrics for all evaluated checkpoints.

    Args:
        task: Task name
        variant: Variant name (optional if all_variants=True)
        all_variants: If True, show all checkpoints for task across all variants
        metric: Metric to plot (OKS, F1 Score, Precision, Recall, MAE, PCK@50)

    Returns:
        Plotly figure with metrics over training steps
    """
    if EXPERIMENT_INDEX is None:
        return empty_figure("No evaluation data available")

    # Special handling for mixed tasks - show task traces dynamically per variant
    if task == 'mixed':
        if variant and not variant.startswith('mixed_'):
            variant = f"mixed_{variant}"
        if variant:
            # Also include MCQA-prefixed rows (e.g. mcqa_v5.3_mixed_final_a for task4)
            experiments = EXPERIMENT_INDEX[
                (EXPERIMENT_INDEX['dataset_variant'] == variant) |
                (EXPERIMENT_INDEX['dataset_variant'].str.endswith(f'_{variant}', na=False))
            ].copy()
        else:
            experiments = EXPERIMENT_INDEX[
                EXPERIMENT_INDEX['dataset_variant'].str.startswith('mixed_', na=False) |
                EXPERIMENT_INDEX['dataset_variant'].str.contains('_mixed_', na=False)
            ].copy()

        if experiments.empty:
            return empty_figure("No mixed task evaluations found")

        mixed_variant = variant or experiments['dataset_variant'].iloc[0]
        config = get_mixed_config(mixed_variant)
        experiments['step'] = experiments['model'].apply(extract_step)

        # Create figure with one trace per task
        fig = go.Figure()
        color_palette = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#84cc16', '#f97316']

        for i, (task_key, (metric_col, col_name)) in enumerate(config['tasks'].items()):
            task_data = experiments[experiments['task'] == task_key].copy()
            task_data = task_data.sort_values('step')

            if not task_data.empty and metric_col in task_data.columns:
                metric_label = col_name.split(' ')[-1]  # e.g. "OKS" from "T1 OKS"
                fig.add_trace(go.Scatter(
                    x=task_data['step'],
                    y=task_data[metric_col],
                    mode='lines+markers',
                    name=f"{col_name}",
                    line=dict(color=color_palette[i % len(color_palette)], width=2),
                    marker=dict(size=8)
                ))

        fig.update_layout(
            title=f"Mixed Tasks: {mixed_variant} Performance Progression",
            xaxis_title="Training Step",
            yaxis_title="Metric Value",
            hovermode='x unified',
            showlegend=True,
            paper_bgcolor='#ffffff',
            plot_bgcolor='white',
            font=dict(family="Inter, system-ui, sans-serif", color='#292524'),
            height=500,
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
        )
        return fig

    # Filter experiments by task (get both baseline and fine-tuned)
    # Include related tasks (e.g., task1, task1b, task1c all show under task1)
    if all_variants:
        # Get fine-tuned models - include subtasks (task1 includes task1b, task1c, etc.)
        task_filter = EXPERIMENT_INDEX['task'].str.startswith(task)
        experiments_sft = EXPERIMENT_INDEX[
            task_filter &
            (EXPERIMENT_INDEX['is_sft'] == True)
        ].copy()
        # Get baseline models for variants that have SFT checkpoints
        sft_variants = set(experiments_sft['dataset_variant'].unique())
        # Filter out archived variants
        sft_variants = {v for v in sft_variants if not is_archived(task, v)}
        experiments_sft = experiments_sft[experiments_sft['dataset_variant'].isin(sft_variants)].copy()
        experiments_baseline = EXPERIMENT_INDEX[
            task_filter &
            (EXPERIMENT_INDEX['is_sft'] == False) &
            (EXPERIMENT_INDEX['dataset_variant'].isin(sft_variants))
        ].copy()

        # Also include baseline-only variants (no SFT yet) so they appear as dots at step 0
        # But exclude archived variants
        baseline_only = EXPERIMENT_INDEX[
            task_filter &
            (EXPERIMENT_INDEX['is_sft'] == False) &
            (~EXPERIMENT_INDEX['dataset_variant'].isin(sft_variants))
        ].copy()
        if not baseline_only.empty:
            baseline_only = baseline_only[~baseline_only['dataset_variant'].apply(lambda v: is_archived(task, v))]
        if not baseline_only.empty:
            experiments_baseline = pd.concat([experiments_baseline, baseline_only], ignore_index=True)

        # For SFT variants without their own baseline (e.g. mixed models),
        # create a synthetic baseline using the matching single-task variant's baseline.
        # E.g. mcqa_v5.3_mixed_v3 → use mcqa_v5.3 baseline, mixed_v3 task1c → use cropped_v1 baseline.
        baseline_variants = set(experiments_baseline['dataset_variant'].unique())
        missing_baseline = sft_variants - baseline_variants
        if missing_baseline and not experiments_baseline.empty:
            primary_col = 'accuracy' if task == 'task4' else 'oks_score' if task.startswith('task1') or task.startswith('task3') else 'per_keypoint_accuracy'
            candidates = experiments_baseline.dropna(subset=[primary_col]) if primary_col in experiments_baseline.columns else experiments_baseline
            if candidates.empty:
                candidates = experiments_baseline
            for missing_var in missing_baseline:
                matched = pd.DataFrame()
                # 1. Try explicit baseline_* pattern: mixed_final_a → baseline_mixed_final
                #    Strip trailing _a/_b/etc. suffix and check for baseline_ prefix
                base_prefix = re.sub(r'_[a-z]$', '', missing_var)  # mixed_final_a → mixed_final
                baseline_candidate = f"baseline_{base_prefix}"
                matched = candidates[candidates['dataset_variant'] == baseline_candidate]
                if matched.empty:
                    # 2. Try stripping the mixed suffix: mcqa_v6.2_mixed_final_a → mcqa_v6.2
                    base_variant = re.sub(r'_mixed_\w+$', '', missing_var)
                    matched = candidates[candidates['dataset_variant'] == base_variant]
                if matched.empty:
                    # 3. Fallback: prefer v1 baselines, then alphabetically first
                    matched = candidates.sort_values('dataset_variant', key=lambda s: s.apply(
                        lambda x: ('0_' + x) if x.endswith('_v1') or x == 'mcqa_v1' else ('1_' + x)
                    )).iloc[0:1]
                else:
                    matched = matched.iloc[0:1]
                row = matched.copy()
                donor_variant = matched.iloc[0]['dataset_variant'] if not matched.empty else None
                row['dataset_variant'] = missing_var
                experiments_baseline = pd.concat([experiments_baseline, row], ignore_index=True)
                # Remove donor baseline_* rows — they've been consumed and would show as disconnected dots
                if donor_variant and donor_variant.startswith('baseline_'):
                    experiments_baseline = experiments_baseline[
                        experiments_baseline['dataset_variant'] != donor_variant
                    ].copy()
    else:
        if variant is None:
            return empty_figure("No variant specified")
        # Get fine-tuned models - include subtasks
        task_filter = EXPERIMENT_INDEX['task'].str.startswith(task)
        variant_filter = EXPERIMENT_INDEX['dataset_variant'].isin(normalize_variant_aliases(variant))
        experiments_sft = EXPERIMENT_INDEX[
            task_filter & variant_filter &
            (EXPERIMENT_INDEX['is_sft'] == True)
        ].copy()
        # Get baseline models
        experiments_baseline = EXPERIMENT_INDEX[
            task_filter & variant_filter &
            (EXPERIMENT_INDEX['is_sft'] == False)
        ].copy()

    # Combine: baseline models get step=0, fine-tuned models keep their steps
    experiments = experiments_sft.copy() if not experiments_sft.empty else pd.DataFrame()

    if experiments.empty and experiments_baseline.empty:
        return empty_figure("No fine-tuned checkpoints found for this task")

    # Assign steps to fine-tuned models (using global extract_step function)
    if not experiments.empty:
        experiments['step'] = experiments['model'].apply(extract_step)
        experiments = experiments[experiments['step'] > 0]  # Filter out step=0

    # Add baseline models as step 0 (average across all baseline models per variant)
    if not experiments_baseline.empty:
        experiments_baseline = experiments_baseline.copy()
        experiments_baseline['step'] = 0

        # If showing all variants, average baseline performance per variant
        if all_variants:
            # Build aggregation dict only for columns that exist
            agg_dict = {'step': 'first'}  # Always 0

            # Add numeric columns that exist
            possible_metrics = ['oks_score', 'f1_score', 'precision', 'recall', 'mae_total', 'pck_50',
                               'per_keypoint_accuracy', 'left_right_confusion', 'exact_match', 'accuracy']
            for col in possible_metrics:
                if col in experiments_baseline.columns:
                    agg_dict[col] = 'mean'

            # Group by BOTH variant and task to avoid merging task1/task1b/task1c baselines
            # (e.g., cropped_v1 exists for task1, task1b, and task1c with different metrics)
            group_cols = ['dataset_variant']
            if 'task' in experiments_baseline.columns:
                group_cols.append('task')

            baseline_aggregated = experiments_baseline.groupby(group_cols).agg(agg_dict).reset_index()
            experiments_baseline = baseline_aggregated
        else:
            # For single variant, average all baseline models
            baseline_mean = experiments_baseline.mean(numeric_only=True)
            experiments_baseline = pd.DataFrame([baseline_mean])
            experiments_baseline['step'] = 0
            # Preserve the variant column
            experiments_baseline['dataset_variant'] = variant

        # Combine baseline and fine-tuned
        experiments = pd.concat([experiments_baseline, experiments], ignore_index=True)

    experiments = experiments.sort_values('step')

    if experiments.empty:
        return empty_figure("No evaluation results found")

    # Map metric names to dataframe columns
    metric_map = {
        "OKS": "oks_score",
        "OKS (Conf-Weighted)": "oks_confidence_weighted",
        "F1 Score": "f1_score",
        "Precision": "precision",
        "Recall": "recall",
        "MAE": "mae_total",
        "PCK@50": "pck_50",
        "Accuracy": "accuracy" if task == "task4" else "per_keypoint_accuracy",
        "Parse Rate": "parse_rate",
        "L/R Confusion (%)": "left_right_confusion",
    }

    metric_col = metric_map.get(metric, "oks_score")

    # Check if metric exists in data
    if metric_col not in experiments.columns:
        return empty_figure(f"Metric '{metric}' not available for this task")

    # Create plot
    fig = go.Figure()

    # Initialize variants list for legend configuration
    variants = []

    # If showing all variants, group by variant and use different styles
    if all_variants:
        # Merge orphan baselines with their SFT traces when they differ only by subtask.
        # E.g. mixed_v3 baseline under task1 should connect to mixed_v3 SFT under task1c.
        # Only reassign baselines that have NO SFT data under their own task (orphans).
        if 'task' in experiments.columns and not experiments.empty:
            sft_rows = experiments[experiments['step'] > 0]
            baseline_rows = experiments[experiments['step'] == 0]
            if not sft_rows.empty and not baseline_rows.empty:
                # Build set of (variant, task) combos that have SFT data
                sft_pairs = set(zip(sft_rows['dataset_variant'], sft_rows['task']))
                # For orphan baselines, find which subtask has SFT data for their variant
                sft_variant_tasks = sft_rows.groupby('dataset_variant')['task'].first().to_dict()
                for idx, row in baseline_rows.iterrows():
                    var, btask = row['dataset_variant'], row['task']
                    # Only reassign if this baseline has no SFT under its own task
                    if (var, btask) not in sft_pairs and var in sft_variant_tasks:
                        experiments.at[idx, 'task'] = sft_variant_tasks[var]

        # CRITICAL: Group by BOTH task and variant to avoid mixing task1/cropped_v1 with task1c/cropped_v1
        # Create combined identifier for each unique task+variant combination
        experiments['display_variant'] = experiments.apply(
            lambda row: f"{row['dataset_variant']} ({TASK_NAMES.get(row['task'], row['task'])})"
            if row['task'] != task else row['dataset_variant'],
            axis=1
        )
        variants = sorted(experiments['display_variant'].unique())

        # Highly distinct color palette with better visual separation
        color_palette = [
            '#e11d48',  # Bright red
            '#0ea5e9',  # Sky blue
            '#22c55e',  # Green
            '#f97316',  # Orange
            '#8b5cf6',  # Purple
            '#ec4899',  # Pink
            '#06b6d4',  # Cyan
            '#eab308',  # Yellow
            '#6366f1',  # Indigo
            '#14b8a6',  # Teal
            '#f43f5e',  # Rose
            '#84cc16',  # Lime
        ]

        # Different line styles for better distinction
        line_styles = ['solid', 'dash', 'dot', 'dashdot']

        # Assign colors and styles to variants
        colors_map = {}
        styles_map = {}
        for i, var in enumerate(variants):
            colors_map[var] = color_palette[i % len(color_palette)]
            styles_map[var] = line_styles[i % len(line_styles)]

        for var in variants:
            var_data = experiments[experiments['display_variant'] == var].sort_values('step')
            # Drop rows where the metric is NaN
            var_data = var_data.dropna(subset=[metric_col])
            if var_data.empty:
                continue
            steps = var_data['step'].tolist()
            values = var_data[metric_col].tolist()
            # Convert ratio to percentage for L/R confusion
            if metric == "L/R Confusion (%)":
                values = [v * 100 for v in values]

            # Determine mode: markers only if single point (baseline only), lines+markers if multiple
            plot_mode = 'markers' if len(steps) == 1 else 'lines+markers'

            fig.add_trace(go.Scatter(
                x=steps, y=values,
                mode=plot_mode,
                name=f'{var}',
                line=dict(
                    color=colors_map[var],
                    width=3,  # Thicker lines for better visibility
                    dash=styles_map[var]
                ),
                marker=dict(size=10, line=dict(width=1, color='white')),  # Larger markers with white border
                opacity=0.85  # Slight transparency to see overlaps
            ))
    else:
        # Single variant - plot selected metric
        steps = experiments['step'].tolist()
        values = experiments[metric_col].tolist()
        # Convert ratio to percentage for L/R confusion
        if metric == "L/R Confusion (%)":
            values = [v * 100 for v in values]

        # Determine mode: markers only if single point (baseline only), lines+markers if multiple
        plot_mode = 'markers' if len(steps) == 1 else 'lines+markers'

        fig.add_trace(go.Scatter(
            x=steps, y=values,
            mode=plot_mode,
            name=metric,
            line=dict(color='#10b981', width=2),
            marker=dict(size=8)
        ))

    # Update layout
    title_text = f"{metric} Progression - {TASK_NAMES.get(task, task)}"
    if all_variants:
        title_text += " (All Variants)"
    elif variant:
        title_text += f" ({variant})"

    # Set y-axis range based on metric
    if metric == "MAE":
        y_range = None  # MAE can be > 1
    elif metric == "L/R Confusion (%)":
        y_range = [0, 105]  # Percentage scale
    else:
        y_range = [0, 1.05]

    # Configure legend based on number of variants
    if all_variants and len(variants) > 4:
        # Vertical legend for many variants
        legend_config = dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.02,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='#e5e7eb',
            borderwidth=1
        )
        # Add more space for legend
        margin_config = dict(l=80, r=200, t=100, b=80)
    else:
        # Horizontal legend for few variants
        legend_config = dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(255, 255, 255, 0.9)'
        )
        margin_config = dict(l=80, r=80, t=100, b=80)

    fig.update_layout(
        title=dict(
            text=title_text,
            font=dict(size=18, color='#292524', family='Inter'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title=dict(text="Training Step", font=dict(size=14)),
            showgrid=True,
            gridcolor='#E7E5E4',
            gridwidth=1,
            zeroline=False,
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            title=dict(text=metric, font=dict(size=14)),
            showgrid=True,
            gridcolor='#E7E5E4',
            gridwidth=1,
            zeroline=False,
            range=y_range,
            tickfont=dict(size=12)
        ),
        plot_bgcolor='white',
        paper_bgcolor='#ffffff',
        font=dict(family="Inter, system-ui, sans-serif", color='#292524'),
        hovermode='x unified',
        legend=legend_config,
        margin=margin_config,
        height=600  # Taller plot for better visibility
    )

    return fig


def create_mcqa_analysis_plots(checkpoint_name: str, task: str, variant: str, split: str = "test"):
    """
    Create confusion matrix heatmap and answer distribution chart for Task 4 MCQA.

    Returns: (confusion_fig, distribution_fig, tier_fig, summary_markdown)
    """
    clean_name = checkpoint_name.replace('[Baseline] ', '')
    result_file = find_result_file(clean_name, task=task, variant=variant, split=split)
    if not result_file:
        return empty_figure("No results found"), empty_figure("No results found"), empty_figure("No results found"), "*No evaluation results found for this checkpoint.*"

    results = load_evaluation_results(result_file)
    if not results:
        return empty_figure("Failed to load"), empty_figure("Failed to load"), empty_figure("Failed to load"), "*Failed to load results.*"

    summary = results.get('summary', {})
    cm_data = summary.get('confusion_matrix', {})
    pred_dist = summary.get('answer_distribution', {})

    if not cm_data:
        return empty_figure("No confusion matrix"), empty_figure("No confusion matrix"), empty_figure("No data"), "*No confusion matrix in results.*"

    labels = ['A', 'B', 'C', 'D']
    matrix = np.zeros((4, 4), dtype=int)
    for i, gt in enumerate(labels):
        for j, pred in enumerate(labels):
            matrix[i][j] = cm_data.get(gt, {}).get(pred, 0)

    row_sums = matrix.sum(axis=1, keepdims=True)
    pct = np.where(row_sums > 0, matrix / row_sums * 100, 0)
    text = [[f"{matrix[i][j]}<br>({pct[i][j]:.0f}%)" for j in range(4)] for i in range(4)]

    # Confusion matrix heatmap
    cm_fig = go.Figure(data=go.Heatmap(
        z=matrix.tolist(), x=labels, y=labels,
        colorscale='Blues', text=text, texttemplate="%{text}",
        hovertemplate="GT: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>"
    ))
    cm_fig.update_layout(
        title=f"Confusion Matrix: {checkpoint_name}",
        xaxis_title="Predicted Answer", yaxis_title="Ground Truth Answer",
        yaxis=dict(autorange='reversed'),
        height=420, width=480,
        paper_bgcolor='#ffffff', plot_bgcolor='white',
        font=dict(family="Inter, system-ui, sans-serif", color='#292524')
    )

    # Distribution bar chart
    gt_counts = [int(row_sums[i][0]) for i in range(4)]
    pred_counts = [pred_dist.get(l, 0) for l in labels]

    dist_fig = go.Figure()
    dist_fig.add_trace(go.Bar(name='Ground Truth', x=labels, y=gt_counts, marker_color='#3b82f6'))
    dist_fig.add_trace(go.Bar(name='Predicted', x=labels, y=pred_counts, marker_color='#ef4444'))
    dist_fig.update_layout(
        title=f"Answer Distribution: {checkpoint_name}",
        xaxis_title="Answer Letter", yaxis_title="Count",
        barmode='group', height=350,
        paper_bgcolor='#ffffff', plot_bgcolor='white',
        font=dict(family="Inter, system-ui, sans-serif", color='#292524'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # Summary markdown
    total = int(matrix.sum())
    correct = int(np.trace(matrix))
    accuracy = correct / total * 100 if total > 0 else 0
    lines = [f"**Total**: {total} samples | **Accuracy**: {accuracy:.1f}%\n"]
    lines.append("| Letter | GT Count | Predicted | Per-class Acc |")
    lines.append("|--------|----------|-----------|---------------|")
    for i, l in enumerate(labels):
        gt_n = int(row_sums[i][0])
        pred_n = pred_counts[i]
        cls_acc = matrix[i][i] / gt_n * 100 if gt_n > 0 else 0
        lines.append(f"| {l} | {gt_n} | {pred_n} | {cls_acc:.1f}% |")

    # Prediction bias
    total_pred = sum(pred_counts) or 1
    max_pred_pct = max(pred_counts) / total_pred * 100
    max_pred_letter = labels[pred_counts.index(max(pred_counts))]
    lines.append(f"\n**Prediction bias**: {max_pred_letter} = {max_pred_pct:.1f}% (expected ~25%)")

    # Most confused pair
    np.fill_diagonal(matrix, 0)
    if matrix.max() > 0:
        idx = np.unravel_index(matrix.argmax(), matrix.shape)
        lines.append(f"**Most confused**: GT={labels[idx[0]]} predicted as {labels[idx[1]]} ({matrix[idx[0]][idx[1]]} times)")

    # Per-tier accuracy: cross-reference eval results with source JSONL metadata
    tier_fig = _build_tier_accuracy_fig(results, variant, split, checkpoint_name, empty_figure)

    return cm_fig, dist_fig, tier_fig, "\n".join(lines)


def _build_tier_accuracy_fig(results: Dict, variant: str, split: str, checkpoint_name: str, _empty_fig):
    """Build per-tier accuracy line plot across ALL checkpoints for the variant.

    Each line = one difficulty tier, each dot = one checkpoint (baseline + SFT steps).
    """
    tier_list = _build_tier_list_for_variant(variant, split)
    if not any(tier_list):
        return _empty_fig("No difficulty_tier metadata in JSONL")

    # Collect all checkpoints for this variant
    all_checkpoints = create_checkpoint_list('task4', variant)
    if not all_checkpoints:
        return _empty_fig("No evaluated checkpoints found")

    # Compute per-tier accuracy for each checkpoint (matching by position)
    def _tier_acc_for_results(res):
        tier_correct = {}
        for i, r in enumerate(res.get('detailed_results', [])):
            tier = tier_list[i] if i < len(tier_list) else None
            if not tier:
                continue
            if tier not in tier_correct:
                tier_correct[tier] = [0, 0]
            tier_correct[tier][1] += 1
            if r.get('metrics', {}).get('correct', 0):
                tier_correct[tier][0] += 1
        return {t: c[0] / c[1] * 100 if c[1] > 0 else 0 for t, c in tier_correct.items()}

    # checkpoint_data: [(label, step_num, {tier: accuracy})]
    checkpoint_data = []
    for cp in all_checkpoints:
        clean = cp.replace('[Baseline] ', '')
        rf = find_result_file(clean, task='task4', variant=variant, split=split)
        if not rf:
            continue
        res = load_evaluation_results(rf)
        if not res:
            continue
        step = extract_step(cp)
        label = "Baseline" if step == 0 else f"step{step}"
        tier_accs = _tier_acc_for_results(res)
        if tier_accs:
            checkpoint_data.append((label, step, tier_accs))

    if not checkpoint_data:
        return _empty_fig("No tier data for any checkpoint")

    # Sort by step number
    checkpoint_data.sort(key=lambda x: x[1])

    # Discover all tiers across all checkpoints
    all_tiers = sorted(set(t for _, _, accs in checkpoint_data for t in accs))

    # Shorten tier labels
    def _short_tier(t):
        parts = t.split('_', 1)
        num = parts[0].replace('TIER', 'T') if parts[0].startswith('TIER') else parts[0]
        desc = parts[1].replace('_', ' ').title() if len(parts) > 1 else ''
        return f"{num}: {desc}" if desc else num

    # Count samples per tier (from first checkpoint that has data)
    # Count total questions per tier (not unique images — V6.2 has 2 questions/image)
    tier_counts = {}
    for t in all_tiers:
        tier_counts[t] = sum(1 for tier in tier_list if tier == t)

    colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#84cc16']
    x_labels = [d[0] for d in checkpoint_data]

    tier_fig = go.Figure()
    for i, tier in enumerate(all_tiers):
        y_vals = [d[2].get(tier, None) for d in checkpoint_data]
        short = _short_tier(tier)
        n = tier_counts.get(tier, 0)
        tier_fig.add_trace(go.Scatter(
            x=x_labels, y=y_vals,
            mode='lines+markers+text',
            name=f"{short} (n={n})",
            line=dict(color=colors[i % len(colors)], width=2),
            marker=dict(size=8),
            text=[f"{v:.1f}" if v is not None else "" for v in y_vals],
            textposition='top center',
            textfont=dict(size=9),
        ))

    tier_fig.update_layout(
        title=f"Per-Tier Accuracy Progression: {variant}",
        xaxis_title="Checkpoint", yaxis_title="Accuracy (%)",
        yaxis=dict(range=[0, 105]),
        height=450,
        paper_bgcolor='#ffffff', plot_bgcolor='white',
        font=dict(family="Inter, system-ui, sans-serif", color='#292524'),
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        hovermode='x unified',
    )
    return tier_fig


V5_TIER_NAMES = {
    1: "TIER1_SINGLE_KEYPOINT", 2: "TIER2_BINARY_RELATION",
    3: "TIER3_COMPARATIVE", 4: "TIER4_MULTI_HOP", 5: "TIER5_BIOMECHANICS",
}

def _build_tier_list_for_variant(variant: str, split: str) -> List[Optional[str]]:
    """Build ordered tier list from source JSONL. Returns list aligned with JSONL/eval order."""
    dataset_info = DATASET_INDEX.get('task4', {}).get(variant, {})
    dataset_path = dataset_info.get('path')
    if not dataset_path:
        return []
    jsonl_path = resolve_jsonl_path(dataset_path, split)
    if not jsonl_path or not jsonl_path.exists():
        return []
    try:
        jsonl_lines = _load_jsonl_lines(str(jsonl_path))
        tier_list = []
        for line in jsonl_lines:
            sample = json.loads(line)
            meta = sample.get('metadata', {})
            if isinstance(meta, str):
                meta = json.loads(meta)
            tier = meta.get('difficulty_tier', '')
            if tier.startswith('TIER_'):
                template = meta.get('question_template', '')
                tier_nums = re.findall(r'tier(\d)', template)
                if tier_nums:
                    max_tier = max(int(t) for t in tier_nums)
                    tier = V5_TIER_NAMES.get(max_tier, f"TIER{max_tier}")
            tier_list.append(tier if tier else None)
        return tier_list
    except Exception:
        return []


def _filter_eval_indices(checkpoint: str, task: str, variant: str, split: str,
                         filter_mode: str = "All", tier_filter: str = "All") -> tuple:
    """Filter evaluation results, returning (result_file, filtered_indices, tier_list).

    Only stores lightweight indices in Gradio State — full results loaded on demand.
    """
    if not checkpoint or task != 'task4':
        return None, [], []
    clean = checkpoint.replace('[Baseline] ', '')
    rf = find_result_file(clean, task=task, variant=variant, split=split)
    if not rf:
        return None, [], []
    results = load_evaluation_results(rf)
    if not results or 'detailed_results' not in results:
        return None, [], []

    tier_list = _build_tier_list_for_variant(variant, split)
    detailed = results['detailed_results']

    filtered_indices = []
    for i, r in enumerate(detailed):
        correct = r.get('metrics', {}).get('correct', 0)
        if filter_mode == "Incorrect Only" and correct:
            continue
        if filter_mode == "Correct Only" and not correct:
            continue
        tier = tier_list[i] if i < len(tier_list) else None
        if tier_filter != "All" and tier:
            tier_num = tier_filter.split(":")[0].replace("T", "TIER")
            if not tier.startswith(tier_num):
                continue
        filtered_indices.append(i)
    return rf, filtered_indices, tier_list


def _format_eval_at_index(result_file: str, index: int, tier_list: list) -> tuple:
    """Load and format a single eval result by index. Returns (image_path, verdict_md, detail_md)."""
    if not result_file:
        return None, "", ""
    results = load_evaluation_results(result_file)
    if not results or index >= len(results.get('detailed_results', [])):
        return None, "", ""
    r = results['detailed_results'][index]

    metrics = r.get('metrics', {})
    correct = metrics.get('correct', 0)
    predicted = metrics.get('predicted_answer', '?')
    gt = r.get('ground_truth', r.get('correct_answer', '?'))
    response = r.get('response', '')
    prompt = r.get('prompt', '')
    image_path = r.get('image_path', '')
    tier = tier_list[index] if index < len(tier_list) else ''

    if correct:
        verdict = f"### &#x2705; CORRECT &mdash; Predicted: **{predicted}** | GT: **{gt}**"
    else:
        verdict = f"### &#x274C; INCORRECT &mdash; Predicted: **{predicted}** | GT: **{gt}**"
    if tier:
        short_tier = tier.split('_', 1)
        num = short_tier[0].replace('TIER', 'T')
        desc = short_tier[1].replace('_', ' ').title() if len(short_tier) > 1 else ''
        verdict += f" &nbsp;|&nbsp; {num}: {desc}"

    detail_parts = [
        "**Question:**",
        f"\n{prompt}\n",
        "---",
        f"**Model Response:**\n\n> {response}",
    ]
    return image_path, verdict, "\n".join(detail_parts)


def _is_result_correct(r: Dict, task: str) -> bool:
    """Determine if a single eval result is 'correct' based on task-specific metrics."""
    m = r.get('metrics', {})
    # Task 4 MCQA: binary correct field
    if 'correct' in m:
        return bool(m['correct'])
    # Task 2: keypoint labeling — use per_keypoint_accuracy > 0.5
    if 'per_keypoint_accuracy' in m:
        return m['per_keypoint_accuracy'] > 0.5
    # Task 3: error correction — use error_detection_correct AND f1_score > 0.5
    if 'error_detection_correct' in m:
        return bool(m['error_detection_correct']) and m.get('f1_score', 0) > 0.5
    # Task 1: keypoint detection — use oks_score threshold
    if 'oks_score' in m:
        return m['oks_score'] > 0.5
    return False


def _filter_eval_indices_generic(checkpoint: str, task: str, variant: str, split: str,
                                  filter_mode: str = "All") -> tuple:
    """Filter eval results for any task. Returns (result_file, filtered_indices)."""
    if not checkpoint:
        return None, []
    clean = checkpoint.replace('[Baseline] ', '')
    rf = find_result_file(clean, task=task, variant=variant, split=split)
    if not rf:
        return None, []
    results = load_evaluation_results(rf)
    if not results or 'detailed_results' not in results:
        return None, []

    filtered = []
    for i, r in enumerate(results['detailed_results']):
        correct = _is_result_correct(r, task)
        if filter_mode == "Incorrect Only" and correct:
            continue
        if filter_mode == "Correct Only" and not correct:
            continue
        filtered.append(i)
    return rf, filtered


def _format_eval_generic(result_file: str, index: int, task: str) -> tuple:
    """Format a single eval result for any task. Returns (image_path, verdict_md, detail_md)."""
    if not result_file:
        return None, "", ""
    results = load_evaluation_results(result_file)
    if not results or index >= len(results.get('detailed_results', [])):
        return None, "", ""
    r = results['detailed_results'][index]

    m = r.get('metrics', {})
    correct = _is_result_correct(r, task)
    prediction = r.get('prediction', r.get('response', ''))
    gt = r.get('ground_truth', '')
    image_path = r.get('image_path', '')

    # Verdict line with key metrics
    icon = "&#x2705;" if correct else "&#x274C;"
    status = "CORRECT" if correct else "INCORRECT"

    if task.startswith('task2'):
        acc = m.get('per_keypoint_accuracy', 0)
        n_correct = m.get('num_keypoints_correct', 0)
        n_total = m.get('num_keypoints_total', 0)
        lr_conf = m.get('left_right_confusions', 0)
        verdict = f"### {icon} {status} &mdash; Accuracy: **{acc:.0%}** ({n_correct}/{n_total} keypoints)"
        if lr_conf:
            verdict += f" &nbsp;|&nbsp; L/R confusions: **{lr_conf}**"
    elif task.startswith('task3'):
        f1 = m.get('f1_score', 0)
        tp = m.get('true_positives', 0)
        fp = m.get('false_positives', 0)
        fn = m.get('false_negatives', 0)
        oks = m.get('oks_score', 0)
        verdict = f"### {icon} {status} &mdash; F1: **{f1:.2f}** (TP={tp} FP={fp} FN={fn}) | OKS: **{oks:.3f}**"
    else:
        verdict = f"### {icon} {status}"

    # Prediction and GT as separate outputs for side-by-side display
    pred_display = prediction[:2000] + '...' if len(prediction) > 2000 else prediction
    gt_display = gt[:2000] + '...' if len(gt) > 2000 else gt

    pred_md = f"**Model Prediction:**\n\n```\n{pred_display}\n```"
    gt_md = f"**Ground Truth:**\n\n```\n{gt_display}\n```"
    return image_path, verdict, pred_md, gt_md


_TRAINING_NOTES_CACHE: dict[str, str] | None = None

def _load_training_notes() -> dict[str, str]:
    """Load task descriptions from external markdown file, parsing ## headers as task keys.
    Prefers task_descriptions.md, falls back to training_notes.md."""
    global _TRAINING_NOTES_CACHE
    if _TRAINING_NOTES_CACHE is not None:
        return _TRAINING_NOTES_CACHE

    app_dir = Path(__file__).parent
    notes_path = app_dir / "task_descriptions.md"
    if not notes_path.exists():
        notes_path = app_dir / "training_notes.md"
    result: dict[str, str] = {}
    try:
        if notes_path.exists():
            current_task = None
            lines: list[str] = []
            for line in notes_path.read_text().splitlines():
                if line.startswith("## "):
                    if current_task and lines:
                        result[current_task] = "\n".join(lines).strip()
                    current_task = line[3:].strip()
                    lines = []
                elif current_task is not None:
                    if line.strip() == "---":
                        continue
                    lines.append(line)
            if current_task and lines:
                result[current_task] = "\n".join(lines).strip()
    except Exception as e:
        logging.warning(f"Failed to load task descriptions: {e}")

    _TRAINING_NOTES_CACHE = result
    return result

def get_training_notes(task: str) -> str:
    """Return per-task variant notes, filtering out archived variants.
    For merged tasks (e.g. task1), also includes subtask descriptions (task1b, task1c)."""
    notes = _load_training_notes()

    # Collect sections: main task + any merged subtasks
    task_keys = [task]
    for child, (parent, _) in _SUBTASK_MERGE.items():
        if parent == task and child in notes:
            task_keys.append(child)

    description_lines = []
    variant_lines = []
    for tkey in task_keys:
        raw = notes.get(tkey, "")
        if not raw:
            continue
        for para in raw.split('\n\n'):
            stripped = para.strip()
            if not stripped:
                continue
            m = re.match(r'\*\*(\S+?)\*\*', stripped)
            if m:
                name = m.group(1)
                archive_key = name
                if task == 'task4' and not name.lower().startswith('mcqa_'):
                    archive_key = f"mcqa_{name.lower()}"
                if is_archived(task, archive_key):
                    continue
                variant_lines.append(stripped)
            else:
                description_lines.append(stripped)

    if not description_lines and not variant_lines:
        return "*All variants for this task have been archived.*"

    parts = []
    if description_lines:
        parts.append(" ".join(description_lines))
    if variant_lines:
        parts.append("\n".join(f"- {v}" for v in variant_lines))
    return "\n\n".join(parts)


# =============================================================================
# BENCHMARK VISUALIZATION FUNCTIONS
# =============================================================================

def create_ifeval_table(filtered_models=None) -> pd.DataFrame:
    """Create IFEval results table with delta from baseline.

    Args:
        filtered_models: Optional dict of models to display. If None, shows all models.
    """
    if not BENCHMARKS_INDEX or 'ifeval' not in BENCHMARKS_INDEX:
        return pd.DataFrame(columns=["Model", "Prompt Strict (%)", "Δ Prompt", "Instr Strict (%)", "Δ Instr", "Status"])

    baseline = BENCHMARKS_INDEX['ifeval']['baseline']
    models = filtered_models if filtered_models is not None else BENCHMARKS_INDEX['ifeval']['models']

    rows = []

    # Add baseline row
    rows.append({
        'Model': 'Baseline (Qwen3-VL-4B-Instruct)',
        'Prompt Strict (%)': f"{baseline['prompt_strict']:.2f}",
        'Δ Prompt': '-',
        'Instr Strict (%)': f"{baseline['instr_strict']:.2f}",
        'Δ Instr': '-',
        'Status': '✅ Baseline'
    })

    # Add model rows sorted by delta (worst degradation first)
    sorted_models = sorted(models.items(), key=lambda x: x[1]['delta_prompt'])

    for model_name, metrics in sorted_models:
        delta_prompt = metrics['delta_prompt']
        delta_instr = metrics['delta_instr']

        # Determine status
        if delta_prompt >= 0:
            status = '✅ Improved'
        elif delta_prompt >= -5:
            status = '⚠️ Minor degradation'
        else:
            status = '🔴 Severe degradation'

        rows.append({
            'Model': model_name,
            'Prompt Strict (%)': f"{metrics['prompt_strict']:.2f}",
            'Δ Prompt': f"{delta_prompt:+.2f}",
            'Instr Strict (%)': f"{metrics['instr_strict']:.2f}",
            'Δ Instr': f"{delta_instr:+.2f}",
            'Status': status
        })

    return pd.DataFrame(rows)


def create_ifeval_chart(filtered_models=None) -> go.Figure:
    """Create IFEval degradation bar chart.

    Args:
        filtered_models: Optional dict of models to display. If None, shows all models.
    """
    if not BENCHMARKS_INDEX or 'ifeval' not in BENCHMARKS_INDEX:
        return empty_figure("No IFEval data available")

    models = filtered_models if filtered_models is not None else BENCHMARKS_INDEX['ifeval']['models']

    # Sort by delta
    sorted_items = sorted(models.items(), key=lambda x: x[1]['delta_prompt'])

    model_names = [item[0] for item in sorted_items]
    deltas = [item[1]['delta_prompt'] for item in sorted_items]

    # Color based on delta
    colors = []
    for delta in deltas:
        if delta >= 0:
            colors.append('#10b981')  # Green - improved
        elif delta >= -5:
            colors.append('#f59e0b')  # Orange - minor degradation
        else:
            colors.append('#ef4444')  # Red - severe degradation

    fig = go.Figure(data=[
        go.Bar(
            x=model_names,
            y=deltas,
            marker_color=colors,
            text=[f"{d:+.1f}%" for d in deltas],
            textposition='outside'
        )
    ])

    fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Baseline")
    fig.add_hline(y=-5, line_dash="dot", line_color="orange", annotation_text="-5% threshold")

    fig.update_layout(
        title="IFEval: Delta from Baseline (Prompt-Level Strict)",
        xaxis_title="Model",
        yaxis_title="Δ from Baseline (%)",
        showlegend=False,
        paper_bgcolor='#ffffff',
        plot_bgcolor='white',
        font=dict(family="Inter, system-ui, sans-serif", color='#292524'),
        xaxis=dict(tickangle=-45),
        height=500
    )

    return fig


def create_sibench_table(filtered_models=None) -> pd.DataFrame:
    """Create SIBench results table.

    Args:
        filtered_models: Optional dict of models to display. If None, shows all models.
    """
    if not BENCHMARKS_INDEX or 'sibench' not in BENCHMARKS_INDEX:
        return pd.DataFrame(columns=["Model", "Overall Accuracy (%)", "Δ from Baseline", "Status"])

    baseline = BENCHMARKS_INDEX['sibench'].get('baseline', {})
    models = filtered_models if filtered_models is not None else BENCHMARKS_INDEX['sibench'].get('models', {})

    # Validate baseline exists
    if not baseline or 'overall' not in baseline:
        logging.warning("SIBench baseline not found")
        return pd.DataFrame(columns=["Model", "Overall Accuracy (%)", "Δ from Baseline", "Status"])

    if not models:
        logging.warning("No SIBench models found")
        return pd.DataFrame(columns=["Model", "Overall Accuracy (%)", "Δ from Baseline", "Status"])

    rows = []
    baseline_overall = baseline['overall']

    # Add baseline row
    rows.append({
        'Model': 'Baseline (qwen3-vl-4b)',
        'Overall Accuracy (%)': f"{baseline_overall:.2f}",
        'Δ from Baseline': '-',
        'Status': '✅ Baseline'
    })

    # Add model rows (sort by delta, worst degradation first for consistency with IFEval)
    model_items = []
    for model_name, metrics in models.items():
        if 'overall' in metrics:
            delta = metrics['overall'] - baseline_overall
            model_items.append((model_name, metrics, delta))

    # Sort by delta (worst degradation first)
    for model_name, metrics, delta in sorted(model_items, key=lambda x: x[2]):

        # Determine status
        if delta >= 0:
            status = '✅ Improved'
        elif delta >= -5:
            status = '⚠️ Minor degradation'
        else:
            status = '🔴 Severe degradation'

        rows.append({
            'Model': model_name,
            'Overall Accuracy (%)': f"{metrics['overall']:.2f}",
            'Δ from Baseline': f"{delta:+.2f}",
            'Status': status
        })

    return pd.DataFrame(rows)


def create_sibench_summary() -> str:
    """Create SIBench summary markdown."""
    if not BENCHMARKS_INDEX or 'sibench' not in BENCHMARKS_INDEX:
        return "*No SIBench data available*"

    models = BENCHMARKS_INDEX['sibench'].get('models', {})
    baseline = BENCHMARKS_INDEX['sibench'].get('baseline', {})

    baseline_overall = baseline.get('overall', 0)

    summary = f"""### SIBench Summary

**Models Evaluated**: {len(models)}
**Baseline Accuracy**: {baseline_overall:.2f}%

**Status**: ⚠️ Only {len(models)} models evaluated - needs expansion to all 20+ models

**Key Insight**: Vision training appears to degrade spatial reasoning capabilities
"""

    return summary


def create_sibench_chart(filtered_models=None) -> go.Figure:
    """Create SIBench per-task comparison chart.

    Args:
        filtered_models: Optional dict of models to display. If None, shows all models.
    """
    if not BENCHMARKS_INDEX or 'sibench' not in BENCHMARKS_INDEX:
        return empty_figure("No SIBench data available")

    models = filtered_models if filtered_models is not None else BENCHMARKS_INDEX['sibench'].get('models', {})

    if not models:
        return empty_figure("No SIBench models found")

    # Create grouped bar chart comparing models on each task
    fig = go.Figure()

    # Get all unique tasks from all models
    all_tasks = set()
    for model_data in models.values():
        per_task = model_data.get('per_task', {})
        all_tasks.update(per_task.keys())

    # Sort tasks alphabetically
    tasks = sorted(all_tasks)

    # Add a trace for each model
    colors = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6']
    for idx, (model_name, model_data) in enumerate(sorted(models.items())):
        per_task = model_data.get('per_task', {})
        accuracies = [per_task.get(task, 0) for task in tasks]

        fig.add_trace(go.Bar(
            name=model_name,
            x=tasks,
            y=accuracies,
            marker_color=colors[idx % len(colors)],
            text=[f"{acc:.1f}%" for acc in accuracies],
            textposition='outside'
        ))

    fig.update_layout(
        title="SIBench: Per-Task Performance Comparison",
        xaxis_title="Task",
        yaxis_title="Accuracy (%)",
        barmode='group',
        showlegend=True,
        paper_bgcolor='#ffffff',
        plot_bgcolor='white',
        font=dict(family="Inter, system-ui, sans-serif", color='#292524'),
        xaxis=dict(tickangle=-45),
        yaxis=dict(range=[0, 105]),
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig


def get_checkpoint_details(checkpoint_name: str) -> Tuple[str, Dict, bool]:
    """
    Load checkpoint configuration and training info.

    Args:
        checkpoint_name: Name of checkpoint

    Returns:
        Tuple of (config_yaml_text, training_info_dict, files_exist)
    """
    if not checkpoint_name:
        return "No checkpoint selected", {}, False

    # Remove [Baseline] prefix if present
    clean_name = checkpoint_name.replace('[Baseline] ', '')

    # Check if checkpoint directory exists
    checkpoint_path = MODELS_BASE_PATH / clean_name

    if not checkpoint_path.exists():
        return "Checkpoint directory not found", {}, False

    # Check if training files exist
    config_file = checkpoint_path / "config.yaml"
    training_info_file = checkpoint_path / "training_info.json"

    has_config = config_file.exists()
    has_training_info = training_info_file.exists()

    if not has_config and not has_training_info:
        # Training files were never saved for this checkpoint
        msg = "Training configuration files not available for this checkpoint.\n\n"
        msg += "These files (config.yaml, training_info.json) were not saved during training."
        return msg, {}, False

    # Load files if they exist
    config_text = safe_load_text(config_file) if has_config else "config.yaml not available"
    training_info = safe_load_json(training_info_file) if has_training_info else {}

    return config_text, training_info, True


def generate_custom_comparison(checkpoint_names: List[str], task: str, variant: str) -> Tuple[pd.DataFrame, go.Figure, str]:
    """
    Generate custom checkpoint comparison with table, radar chart, and summary.

    Args:
        checkpoint_names: List of checkpoint names to compare
        task: Current task (e.g., 'task1', 'task2')
        variant: Current variant (e.g., 'cropped_v1', 'visualized_cropped_v4')

    Returns:
        Tuple of (comparison_table_df, radar_chart_fig, summary_markdown)
    """
    try:
        if not checkpoint_names or len(checkpoint_names) == 0:
            empty_df = pd.DataFrame(columns=["Checkpoint", "Step", "OKS", "OKS (CW)", "F1", "Precision", "Recall"])
            return empty_df, empty_figure("Select checkpoints to compare"), "*Select checkpoints and click 'Generate Comparison'*"

        # Limit to 5 checkpoints
        checkpoint_names = checkpoint_names[:5]
        logging.info(f"📊 Generating comparison for {len(checkpoint_names)} checkpoints:")
        logging.info(f"   Task: {task}, Variant: {variant}")
        for i, cp in enumerate(checkpoint_names, 1):
            logging.info(f"   {i}. {cp}")

        # Special handling for mixed tasks
        if task == 'mixed':
            mixed_variant = variant if variant and variant.startswith('mixed_') else f"mixed_{variant}" if variant else 'mixed_balanced_v1'
            config = get_mixed_config(mixed_variant)
            task_defs = config['tasks']

            comparison_data = []
            metrics_for_radar = {}

            for cp_name in checkpoint_names:
                clean_name = cp_name.replace('[Baseline] ', '')

                matching_rows = EXPERIMENT_INDEX[
                    (EXPERIMENT_INDEX['model'] == clean_name) &
                    (EXPERIMENT_INDEX['dataset_variant'] == mixed_variant)
                ]

                if matching_rows.empty:
                    # Fallback: try startswith for backward compat
                    matching_rows = EXPERIMENT_INDEX[
                        (EXPERIMENT_INDEX['model'] == clean_name) &
                        (EXPERIMENT_INDEX['dataset_variant'].str.startswith('mixed_', na=False))
                    ]

                if matching_rows.empty:
                    logging.warning(f"Mixed task checkpoint not found: {clean_name}")
                    continue

                step = extract_step(clean_name) if matching_rows.iloc[0]['is_sft'] else 0

                row_data = {'Checkpoint': cp_name, 'Step': step}
                radar_data = {}

                for task_key, (metric_col, col_name) in task_defs.items():
                    t_row = matching_rows[matching_rows['task'] == task_key]
                    if not t_row.empty and metric_col in t_row.columns and pd.notna(t_row.iloc[0].get(metric_col)):
                        val = float(t_row.iloc[0][metric_col])
                        row_data[col_name] = f"{val:.3f}"
                        radar_data[col_name] = val
                    else:
                        row_data[col_name] = 'N/A'
                        radar_data[col_name] = 0

                comparison_data.append(row_data)
                metrics_for_radar[cp_name] = radar_data

            df = pd.DataFrame(comparison_data)

            # Create radar chart
            fig = go.Figure()
            if metrics_for_radar:
                categories = list(next(iter(metrics_for_radar.values())).keys())
                for cp_name, radar_vals in metrics_for_radar.items():
                    values = list(radar_vals.values())
                    values.append(values[0])  # Close the radar
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=categories + [categories[0]],
                        fill='toself',
                        name=cp_name
                    ))

            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                title=f"Mixed Tasks: {mixed_variant} Performance Comparison",
                paper_bgcolor='#ffffff',
                font=dict(family="Inter, system-ui, sans-serif", color='#292524'),
                height=500
            )

            task_list = "\n".join(f"- **{col_name}**" for _, (_, col_name) in task_defs.items())
            summary = f"### Mixed Tasks Comparison ({mixed_variant})\n\n"
            summary += f"Comparing {len(checkpoint_names)} checkpoints across {len(task_defs)} task types:\n\n"
            summary += task_list + "\n"

            return df, fig, summary

        # Collect metrics from all checkpoints using EXPERIMENT_INDEX (non-mixed tasks)
        comparison_data = []
        metrics_for_radar = {}

        for cp_name in checkpoint_names:
            # Clean checkpoint name (remove [Baseline] prefix if present)
            clean_name = cp_name.replace('[Baseline] ', '')

            # Find checkpoint in EXPERIMENT_INDEX (filter by model, task, and variant)
            # Include subtasks when filtering by task
            task_filter = EXPERIMENT_INDEX['task'].str.startswith(task)
            matching_rows = EXPERIMENT_INDEX[
                (EXPERIMENT_INDEX['model'] == clean_name) &
                task_filter &
                EXPERIMENT_INDEX['dataset_variant'].isin(normalize_variant_aliases(variant))
            ]

            if matching_rows.empty:
                logging.warning(f"Checkpoint not found in experiments: {clean_name} (task={task}, variant={variant})")
                continue

            # Get first matching row (should be unique per model/task/variant combination)
            row = matching_rows.iloc[0]

            # Extract step (0 for baseline, actual step for fine-tuned)
            step = 0 if not row['is_sft'] else extract_step(row['model'])

            # Build row data based on task type
            row_data = {
                'Checkpoint': cp_name,
                'Step': step
            }

            # Extract task-specific metrics
            if task in ['task1', 'task1b', 'task1c', 'task3a', 'task3b', 'task3c', 'task3d']:
                # Keypoint detection/correction tasks
                oks = row.get('oks_score', 0)
                f1 = row.get('f1_score', 0)
                precision = row.get('precision', 0)
                recall = row.get('recall', 0)

                # VALIDATION: Check metrics are in valid ranges
                if oks < 0 or oks > 1:
                    logging.error(f"❌ Invalid OKS {oks} for {cp_name} in comparison")
                if f1 < 0 or f1 > 1:
                    logging.error(f"❌ Invalid F1 {f1} for {cp_name} in comparison")
                if precision < 0 or precision > 1:
                    logging.error(f"❌ Invalid Precision {precision} for {cp_name} in comparison")
                if recall < 0 or recall > 1:
                    logging.error(f"❌ Invalid Recall {recall} for {cp_name} in comparison")

                lr_confusion = row.get('left_right_confusion', 0)
                oks_cw = row.get('oks_confidence_weighted', 0)

                row_data['OKS'] = f"{oks:.3f}" if oks > 0 else 'N/A'
                row_data['OKS (CW)'] = f"{oks_cw:.3f}" if pd.notna(oks_cw) and oks_cw > 0 else 'N/A'
                row_data['F1'] = f"{f1:.3f}" if f1 > 0 else 'N/A'
                row_data['Precision'] = f"{precision:.3f}" if precision > 0 else 'N/A'
                row_data['Recall'] = f"{recall:.3f}" if recall > 0 else 'N/A'
                row_data['L/R Confusion'] = f"{lr_confusion:.3f}" if lr_confusion > 0 else 'N/A'

                metrics_for_radar[cp_name] = {
                    'OKS': oks if oks > 0 else 0,
                    'F1': f1 if f1 > 0 else 0,
                    'Precision': precision if precision > 0 else 0,
                    'Recall': recall if recall > 0 else 0,
                    'L/R Confusion': lr_confusion if lr_confusion > 0 else 0,
                }
            elif task == 'task2':
                # Keypoint labeling task
                accuracy = row.get('per_keypoint_accuracy', 0)
                lr_confusion = row.get('left_right_confusion', 0)
                exact_match = row.get('exact_match', 0)

                row_data['Accuracy'] = f"{accuracy:.3f}" if accuracy > 0 else 'N/A'
                row_data['L/R Confusion'] = f"{lr_confusion:.3f}" if lr_confusion > 0 else 'N/A'
                row_data['Exact Match'] = f"{exact_match:.3f}" if exact_match > 0 else 'N/A'

                metrics_for_radar[cp_name] = {
                    'Accuracy': accuracy if accuracy > 0 else 0,
                    'L/R Confusion': lr_confusion if lr_confusion > 0 else 0,
                    'Exact Match': exact_match if exact_match > 0 else 0
                }
            elif task == 'task4':
                # MCQA task
                accuracy = row.get('accuracy', 0)
                parse_rate = row.get('parse_rate', 0)

                row_data['Accuracy'] = f"{accuracy:.3f}" if accuracy > 0 else 'N/A'
                row_data['Parse Rate'] = f"{parse_rate:.3f}" if parse_rate > 0 else 'N/A'

                metrics_for_radar[cp_name] = {
                    'Accuracy': accuracy if accuracy > 0 else 0,
                    'Parse Rate': parse_rate if parse_rate > 0 else 0
                }

            comparison_data.append(row_data)

        # Create comparison table
        df = pd.DataFrame(comparison_data)

        # Create radar chart
        radar_fig = go.Figure()

        # Determine metric names based on task type
        colors = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6']

        if task in ['task1', 'task1b', 'task1c', 'task3a', 'task3b', 'task3c', 'task3d']:
            # Keypoint detection/correction metrics
            metric_names = ['OKS', 'OKS (CW)', 'F1', 'Precision', 'Recall', 'L/R Confusion']
        elif task == 'task2':
            # Keypoint labeling metrics
            metric_names = ['Accuracy', 'Exact Match', 'L/R Confusion']
        elif task == 'task4':
            # MCQA metrics
            metric_names = ['Accuracy', 'Parse Rate']
        else:
            metric_names = []

        # Add trace for each checkpoint
        if metric_names and len(comparison_data) > 0:
            for idx, (cp_name, metrics) in enumerate(metrics_for_radar.items()):
                values = [metrics.get(m, 0) for m in metric_names]
                # Close the radar by appending first value
                values_closed = values + [values[0]]
                metric_names_closed = metric_names + [metric_names[0]]

                # Use checkpoint name for legend instead of just step
                display_name = cp_name.replace('[Baseline] ', '')

                radar_fig.add_trace(go.Scatterpolar(
                    r=values_closed,
                    theta=metric_names_closed,
                    fill='toself',
                    name=display_name,
                    line=dict(color=colors[idx % len(colors)], width=2),
                    opacity=0.6
                ))

        radar_fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    showgrid=True,
                    gridcolor='#E7E5E4'
                ),
                angularaxis=dict(
                    showgrid=True,
                    gridcolor='#E7E5E4'
                )
            ),
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            ),
            paper_bgcolor='#ffffff',
            plot_bgcolor='white',
            font=dict(family="Inter, system-ui, sans-serif", color='#292524'),
            title=dict(
                text="Multi-Metric Comparison",
                font=dict(size=16, color='#292524')
            )
        )

        # Generate summary
        if len(comparison_data) == 0:
            summary = "*No evaluation results found for selected checkpoints*"
        else:
            best_checkpoint = comparison_data[0]['Checkpoint']
            best_metric = 0

            # Find best checkpoint based on primary metric
            for row in comparison_data:
                metric_val = 0
                # Use task-appropriate primary metric
                if 'OKS' in row and row['OKS'] != 'N/A':
                    metric_val = float(row['OKS'])
                elif 'Accuracy' in row and row['Accuracy'] != 'N/A':
                    metric_val = float(row['Accuracy'])

                if metric_val > best_metric:
                    best_metric = metric_val
                    best_checkpoint = row['Checkpoint']

            # Calculate insights
            steps = [int(r['Step']) for r in comparison_data]
            min_step = min(steps)
            max_step = max(steps)

            # Check if the best model is the most trained one
            best_step = next(int(r['Step']) for r in comparison_data if r['Checkpoint'] == best_checkpoint)
            trend = 'Improving with training' if best_step == max_step else 'Mixed - best is not most trained'

            summary = f"""### Comparison Summary

**Total Checkpoints Compared**: {len(comparison_data)}

**Best Checkpoint**: `{best_checkpoint}` with score: **{best_metric:.3f}**

**Key Insights**:
- Training steps range: {min_step} to {max_step}
- Performance trend: {trend}
"""

        return df, radar_fig, summary

    except Exception as e:
        error_msg = f"Error generating comparison: {str(e)}"
        logging.error(error_msg)

        empty_df = pd.DataFrame(columns=["Checkpoint", "Step", "OKS", "OKS (CW)", "F1", "Precision", "Recall"])
        return empty_df, empty_figure(f"Error: {str(e)}"), f"**Error**: {error_msg}"


def find_prediction_for_sample(checkpoint_name: str, image_id: str, task: str = None, variant: str = None, split: str = "test") -> Optional[Dict]:
    """
    Find prediction for a specific sample from evaluation results.

    Args:
        checkpoint_name: Name of checkpoint (may include "[Baseline]" prefix)
        image_id: Image ID to find
        task: Task name (optional, used for baseline models)
        variant: Variant name (optional, used for baseline models)
        split: 'test' or 'train'

    Returns:
        Dict with prediction, ground_truth, and metrics, or None if not found
    """
    try:
        # Strip "[Baseline]" prefix if present
        clean_checkpoint_name = checkpoint_name.replace('[Baseline] ', '')

        # Find evaluation results file
        result_file = find_result_file(clean_checkpoint_name, task=task, variant=variant, split=split)
        if not result_file:
            logging.warning(f"⚠️  No evaluation results file found for checkpoint: {checkpoint_name}, task: {task}, variant: {variant}")
            return None

        # Load results
        results = load_evaluation_results(result_file)
        if not results or 'detailed_results' not in results:
            logging.warning(f"⚠️  No detailed results in evaluation file: {result_file}")
            return None

        # Validate results metadata matches expected checkpoint/task/variant
        results_metadata = results.get('metadata', {})
        results_checkpoint = results_metadata.get('checkpoint_name', '')
        results_task = results_metadata.get('task', '')
        results_variant = results_metadata.get('variant', '')

        # Log metadata validation (helpful for debugging)
        logging.debug(f"Results file: {Path(result_file).name}")
        logging.debug(f"  Expected checkpoint: {clean_checkpoint_name}, Found: {results_checkpoint}")
        logging.debug(f"  Expected task: {task}, Found: {results_task}")
        logging.debug(f"  Expected variant: {variant}, Found: {results_variant}")

        # Search for sample by image_id (EXACT MATCH ONLY)
        for sample in results['detailed_results']:
            # Use image_id field (not sample_id which is just numeric index)
            sample_image_id = str(sample.get('image_id', ''))

            # Use exact matching only to ensure predictions match the correct image
            if sample_image_id == str(image_id):
                metrics = sample.get('metrics', {})

                # Validate metrics are in reasonable ranges
                oks = metrics.get('oks_score')
                if oks is not None and (oks < 0 or oks > 1):
                    logging.error(f"❌ Invalid OKS score {oks} for image {image_id} in checkpoint {checkpoint_name}")

                oks_str = f"{oks:.3f}" if oks is not None else "N/A"
                logging.info(f"✓ Found prediction for image {image_id[:50]}... in checkpoint {checkpoint_name} (OKS: {oks_str})")

                return {
                    'prediction': sample.get('prediction', ''),
                    'ground_truth': sample.get('ground_truth', ''),
                    'metrics': metrics,
                    'sample_id': sample_image_id,
                    'image_path': sample.get('image_path', ''),
                    'checkpoint_verified': clean_checkpoint_name,
                    'result_file': result_file
                }

        logging.warning(f"⚠️  Image {image_id[:50]}... NOT FOUND in {Path(result_file).name} (has {len(results['detailed_results'])} samples)")
        return None

    except Exception as e:
        logging.error(f"❌ Error finding prediction for sample {image_id}: {e}")
        return None


# =============================================================================
# MAIN UI LAYOUT
# =============================================================================

def build_ui():
    """Build the main Gradio interface."""

    with gr.Blocks(title="Image SFT Dataset Monitor") as app:
        gr.Markdown("""
        # Image SFT Dataset Monitor

        Dataset exploration, training progress, and evaluation results.
        """)

        with gr.Row():
            # ========== LEFT SIDEBAR (1 unit) ==========
            with gr.Column(scale=1):
                gr.Markdown("### Dataset Selection")

                task_dropdown = gr.Dropdown(
                    choices=list(TASK_NAMES.keys()),
                    value='task1' if 'task1' in TASK_NAMES else None,
                    label="Task Type",
                    info="Select the task type to monitor"
                )

                # Initialize variant dropdown with default task's variants
                default_task = 'task1' if 'task1' in TASK_NAMES else list(TASK_NAMES.keys())[0] if TASK_NAMES else None
                initial_variants = get_active_variants(default_task, list(DATASET_INDEX.get(default_task, {}).keys())) if default_task else []
                default_checkpoints = create_checkpoint_list(default_task, initial_variants[0] if initial_variants else '')

                variant_dropdown = gr.Dropdown(
                    choices=initial_variants,
                    value=initial_variants[0] if initial_variants else None,
                    label="Dataset Variant",
                    info="Select the dataset variant",
                    allow_custom_value=True,
                )

                split_radio = gr.Radio(
                    choices=["train", "test", "qwen3_regen"],
                    value="test",
                    label="Dataset Split",
                    info="Train, test, or Qwen3 regenerated traces"
                )

                with gr.Accordion("Dataset Statistics", open=True):
                    stats_display = gr.Markdown("Select a dataset to view statistics")

                with gr.Accordion("Task Prompt", open=False):
                    prompt_display = gr.Textbox(
                        label="Instruction sent to model",
                        value="Select a task to view the prompt",
                        lines=12,
                        max_lines=20
                    )

                with gr.Accordion("Training Checkpoints", open=False):
                    gr.Markdown("**Available Checkpoints**")
                    gr.Markdown("*Checkpoints for this task/variant (for information only)*", elem_classes="text-sm text-gray-600")
                    checkpoint_list = gr.Textbox(
                        value="",
                        label="",
                        lines=8,
                        max_lines=15,
                        interactive=False,
                        show_label=False
                    )

                with gr.Accordion("Archive Manager", open=False):
                    gr.Markdown("*Hide experiments you no longer need.*")
                    # --- Archive a new experiment ---
                    _arch_tasks = sorted(DATASET_INDEX.keys()) + ["*  (all tasks)"]
                    _arch_default_task = _arch_tasks[0] if _arch_tasks else None
                    _arch_variants = [v for v in get_all_known_variants(_arch_default_task) if not is_archived(_arch_default_task, v)] if _arch_default_task else []
                    arch_task_dd = gr.Dropdown(choices=_arch_tasks, value=_arch_default_task, label="Task")
                    arch_variant_dd = gr.Dropdown(choices=_arch_variants, value=_arch_variants[0] if _arch_variants else None, label="Variant")
                    archive_btn = gr.Button("Archive this experiment", size="sm")
                    # --- Currently archived (restore) ---
                    _archived_labels = [f"{t} / {v}" for t, v in sorted(_ARCHIVED_EXPERIMENTS)]
                    arch_restore_dd = gr.Dropdown(
                        choices=_archived_labels, value=None,
                        label="Archived experiments (select to restore)",
                        allow_custom_value=True,
                    )
                    restore_btn = gr.Button("Restore selected", size="sm")
                    archive_status = gr.Markdown("")

                with gr.Accordion("Mixed Dataset Composition", open=False):
                    gr.Markdown(_load_mixed_datasets_md(), elem_classes=["mixed-datasets-info"])

            # ========== RIGHT MAIN CONTENT (2 units) ==========
            with gr.Column(scale=2):
                with gr.Tabs() as main_tabs:
                    # === Tab 1: Dataset Explorer ===
                    with gr.Tab("Dataset Explorer"):
                        gr.Markdown("### Explore Dataset Samples")

                        with gr.Row():
                            prev_btn = gr.Button("← Previous Page", size="sm")
                            page_info = gr.Markdown("Page 1 of 1 (0 samples)")
                            next_btn = gr.Button("Next Page →", size="sm")

                        with gr.Row():
                            explorer_search_input = gr.Textbox(
                                label="Search Image ID",
                                placeholder="Enter image_id or partial match...",
                                scale=3,
                                lines=1,
                                max_lines=1,
                            )
                            explorer_search_btn = gr.Button("Go", size="sm", scale=1)
                            explorer_random_btn = gr.Button("Random", size="sm", variant="secondary", scale=1)
                            explorer_refresh_btn = gr.Button("Refresh", size="sm", variant="secondary", scale=1)

                        with gr.Row():
                            explorer_exercise_filter = gr.Dropdown(
                                choices=["All"],
                                value="All",
                                label="Filter by Exercise",
                                scale=1,
                                allow_custom_value=False,
                            )

                        image_gallery = gr.Gallery(
                            label="Dataset Samples",
                            columns=4,
                            rows=3,
                            height=600,
                            object_fit="contain"
                        )

                        # Page state tracking
                        current_page = gr.State(value=0)

                        # Prediction overlay controls (test split only)
                        with gr.Row(visible=True) as prediction_controls:
                            show_predictions = gr.Checkbox(
                                label="Show Model Predictions",
                                value=False,
                                info="Compare up to 4 models against ground truth"
                            )
                            prediction_checkpoints = gr.CheckboxGroup(
                                choices=default_checkpoints,
                                value=[],
                                label="Select Checkpoints to Compare (max 4)",
                                info="Choose up to 4 models - Colors: Red, Blue, Yellow, Purple"
                            )

                        with gr.Accordion("Selected Image Details", open=False):
                            with gr.Row():
                                with gr.Column(scale=4):
                                    image_id_display = gr.Markdown("*Select an image to view details*", elem_id="image-id-header")
                                with gr.Column(scale=1):
                                    refresh_btn = gr.Button("Refresh", size="sm", variant="secondary")

                            # Hidden state to track current sample
                            current_sample_idx = gr.State(value=None)

                            gr.Markdown("### Multi-Model Comparison")

                            # For keypoint detection (task1): show images
                            with gr.Row(visible=True) as comparison_images:
                                with gr.Column(scale=1):
                                    gr.Markdown("**Ground Truth (Green)**")
                                    gt_image = gr.Image(label="Ground Truth")
                                with gr.Column(scale=1):
                                    gr.Markdown("**Model Predictions (No GT)**")
                                    combined_pred_image = gr.Image(label="Predictions Only")

                            # For other tasks (task2, task3, task4): show text
                            with gr.Row(visible=False) as comparison_texts:
                                with gr.Column(scale=1):
                                    gr.Markdown("**Ground Truth**")
                                    gt_text_display = gr.Textbox(label="", lines=20, max_lines=30, show_label=False, interactive=False)
                                with gr.Column(scale=1):
                                    gr.Markdown("**Model Predictions**")
                                    pred_text_display = gr.Textbox(label="", lines=20, max_lines=30, show_label=False, interactive=False)

                            gr.Markdown("### Individual Model Predictions")
                            with gr.Row():
                                pred_image1 = gr.Image(label="Model 1 (Red)", visible=False)
                                pred_image2 = gr.Image(label="Model 2 (Blue)", visible=False)
                                pred_image3 = gr.Image(label="Model 3 (Yellow)", visible=False)
                                pred_image4 = gr.Image(label="Model 4 (Purple)", visible=False)

                            with gr.Tabs():
                                # Test-only tab
                                with gr.Tab("Metrics Comparison", visible=True) as metrics_tab:
                                    metrics_comparison_table = gr.Dataframe(
                                        headers=["Color", "Model", "OKS", "OKS (CW)", "PCK@0.5", "MAE (px)"],
                                        label="Performance Comparison"
                                    )

                                with gr.Tab("Ground Truth"):
                                    gr.Markdown("**Ground truth keypoint annotations**")
                                    gt_text = gr.Textbox(label="Ground Truth Data", lines=15, max_lines=30)

                                # Test-only tab
                                with gr.Tab("Model Predictions", visible=True) as predictions_tab:
                                    gr.Markdown("**Raw prediction text from model output**")
                                    prediction_text1 = gr.Textbox(label="Model 1 Predictions", value="", lines=15, max_lines=30)
                                    prediction_text2 = gr.Textbox(label="Model 2 Predictions", value="", lines=15, max_lines=30)
                                    prediction_text3 = gr.Textbox(label="Model 3 Predictions", value="", lines=15, max_lines=30)
                                    prediction_text4 = gr.Textbox(label="Model 4 Predictions", value="", lines=15, max_lines=30)

                                # Test-only tab
                                with gr.Tab("Full Evaluation JSON", visible=True) as eval_json_tab:
                                    gr.Markdown("**Combined evaluation results for all selected models**\n\nShows predictions, ground truth, and metrics for the current image across all selected checkpoints.")
                                    full_eval_json = gr.JSON(label="Multi-Model Comparison Data")

                                with gr.Tab("Metadata"):
                                    gr.Markdown("**Image metadata and lineage information**")
                                    metadata_json = gr.JSON(label="Image Metadata")
                                    lineage_md = gr.Markdown("*Lineage tracking: Track image transformations and augmentations (coming soon)*")

                        # Evaluation Results Browser (generic, for task2/task3)
                        with gr.Accordion("Evaluation Results Browser", open=False, visible=False) as generic_eval_browser_tab:
                            with gr.Row():
                                generic_eval_cp = gr.Dropdown(
                                    choices=[], value=None, label="Checkpoint",
                                    info="Select checkpoint to browse model responses", scale=2,
                                    allow_custom_value=True)
                                generic_eval_filter = gr.Radio(
                                    choices=["All", "Incorrect Only", "Correct Only"],
                                    value="Incorrect Only", label="Filter", scale=1)
                            with gr.Row():
                                generic_eval_prev = gr.Button("◄ Prev", size="sm", scale=1)
                                generic_eval_counter = gr.Markdown("*Select a checkpoint*")
                                generic_eval_next = gr.Button("Next ►", size="sm", scale=1)
                            with gr.Row():
                                generic_eval_image = gr.Image(label="Image", height=350, interactive=False, scale=1)
                                with gr.Column(scale=1):
                                    generic_eval_verdict = gr.Markdown("")
                                    with gr.Row():
                                        generic_eval_pred = gr.Markdown("")
                                        generic_eval_gt = gr.Markdown("")
                            generic_eval_rf = gr.State(value=None)
                            generic_eval_indices = gr.State(value=[])
                            generic_eval_idx = gr.State(value=0)

                        # Variant summary table — last item in Dataset Explorer
                        with gr.Accordion("All Variants Summary", open=False):
                            variant_summary_md = gr.Markdown("")

                    # === Tab 2: Training Monitor ===
                    with gr.Tab("Training Monitor"):
                        gr.Markdown("### Checkpoint Training Progress")

                        checkpoint_table = gr.Dataframe(
                            label="All Evaluated Checkpoints (from experiments-final.csv) - Click column headers to sort",
                            interactive=False,
                            wrap=True
                        )

                        # Metric selector for plot
                        with gr.Row():
                            metric_selector = gr.Dropdown(
                                choices=["OKS", "OKS (Conf-Weighted)", "F1 Score", "Precision", "Recall", "MAE", "PCK@50", "Accuracy", "Parse Rate", "L/R Confusion (%)", "Multi-Task"],
                                value="OKS",
                                label="Select Metric to Plot",
                                scale=4
                            )
                            refresh_training_btn = gr.Button("Refresh", scale=1, variant="secondary")

                        metrics_plot = gr.Plot(label="Metrics Progression Over Training Steps")

                        training_notes = gr.Markdown(value="", visible=True)

                        with gr.Accordion("⚙️ Checkpoint Details", open=False, visible=False) as checkpoint_details_accordion:
                            # Initialize with default task/variant checkpoints
                            default_checkpoints = create_checkpoint_list(default_task, initial_variants[0] if initial_variants else '')

                            selected_checkpoint = gr.Dropdown(
                                choices=default_checkpoints,
                                value=None,
                                label="Select Checkpoint",
                                info="Choose a checkpoint to view details",
                                allow_custom_value=True,
                            )

                            with gr.Row():
                                config_display = gr.Code(language="yaml", label="Training Config")
                                training_info_display = gr.JSON(label="Training Metadata")

                    # === Tab 3: Evaluation Dashboard ===
                    with gr.Tab("Evaluation Dashboard", visible=CONFIG["show_evaluation_dashboard"]):
                        gr.Markdown("### Checkpoint Performance Analysis")

                        # Only Summary Comparison is implemented
                        with gr.Group(visible=True) as summary_view:
                            gr.Markdown("#### Summary Comparison")

                            comparison_mode = gr.Radio(
                                choices=["Quick View (Pre-generated Report)", "Custom Comparison"],
                                value="Quick View (Pre-generated Report)",
                                label="Comparison Mode"
                            )

                            # Quick view - pre-generated report
                            with gr.Group(visible=True) as quick_view:
                                comparison_report = gr.Textbox(
                                    label="Checkpoint Comparison Report",
                                    lines=30,
                                    max_lines=50
                                )

                            # Custom comparison
                            with gr.Group(visible=False) as custom_view:
                                selected_checkpoints = gr.CheckboxGroup(
                                    choices=[],
                                    label="Select Checkpoints to Compare (max 5)",
                                    info="Choose checkpoints to compare against baseline"
                                )

                                compare_btn = gr.Button("Generate Comparison", variant="primary")

                                comparison_table = gr.Dataframe(
                                    label="Performance Comparison"
                                )

                                radar_chart = gr.Plot(label="Multi-Metric Radar Chart")

                                improvement_summary = gr.Markdown("*Click 'Generate Comparison' to see analysis*")

                    # === Tab 4: Benchmarks eval ===
                    with gr.Tab("Benchmarks", visible=CONFIG["show_benchmarks_eval"]):
                        gr.Markdown("### Benchmark Evaluation Results")
                        gr.Markdown("Track IFEval and SIBench performance across all evaluated models")

                        # Show warning if no data, but still create components
                        has_benchmark_data = BENCHMARKS_INDEX and BENCHMARKS_INDEX.get('ifeval', {}).get('models')

                        if not has_benchmark_data:
                            gr.Markdown("⚠️ No benchmark results available. Make sure evaluation results exist in:")
                            gr.Markdown(f"- IFEval: `{VLM_EVAL_ROOT / 'results' / 'reports'}/`")
                            gr.Markdown("- SIBench: `/mnt/data/sgsilva/outputs/sibench/`")

                        # Always create components (needed for event handler wiring)
                        # Benchmark selector
                        benchmark_selector = gr.Radio(
                            choices=["IFEval", "SIBench"],
                            value="IFEval",
                            label="Select Benchmark",
                            info="Choose which benchmark results to view",
                            visible=has_benchmark_data
                        )

                        # Filter controls
                        with gr.Row(visible=has_benchmark_data):
                            # Extract unique tasks and variants from evaluated models
                            evaluated_tasks = set()
                            evaluated_variants = set()
                            if BENCHMARKS_INDEX:
                                for model_name in BENCHMARKS_INDEX.get('ifeval', {}).get('models', {}).keys():
                                    # Parse task from model name (e.g., "task1", "task2")
                                    task_match = re.search(r'task(\d+[a-z]?)', model_name)
                                    if task_match:
                                        evaluated_tasks.add(task_match.group(0))
                                    # Parse variant if present
                                    if 'cropped' in model_name:
                                        evaluated_variants.add('cropped')
                                    if 'original' in model_name:
                                        evaluated_variants.add('original')
                                    if 'visualized' in model_name:
                                        evaluated_variants.add('visualized')

                            benchmark_task_filter = gr.Dropdown(
                                choices=["All Tasks"] + sorted(list(evaluated_tasks)),
                                value="All Tasks",
                                label="Filter by Task",
                                info="Show only models from specific task",
                                scale=1
                            )

                            benchmark_variant_filter = gr.Dropdown(
                                choices=["All Variants"] + sorted(list(evaluated_variants)),
                                value="All Variants",
                                label="Filter by Variant",
                                info="Show only models with specific variant",
                                scale=1
                            )

                        # Summary card
                        with gr.Row():
                            benchmark_summary = gr.Markdown(
                                value=BENCHMARKS_INDEX.get('summary', '') if BENCHMARKS_INDEX else "No summary available",
                                label="Key Findings",
                                visible=has_benchmark_data
                            )

                        # Results table
                        benchmark_table = gr.Dataframe(
                            label="Benchmark Results",
                            interactive=False,
                            wrap=True,
                            visible=has_benchmark_data
                        )

                        # Visualization
                        benchmark_chart = gr.Plot(
                            label="Performance Comparison",
                            visible=has_benchmark_data
                        )

                        # Model detail view
                        with gr.Accordion("Model Details", open=False, visible=has_benchmark_data):
                            model_selector = gr.Dropdown(
                                choices=[],
                                value=None,
                                label="Select Model for Details",
                                info="View detailed metrics and full report",
                                allow_custom_value=True  # Prevent errors during choice updates
                            )

                            model_details = gr.Markdown("*Select a model to view detailed metrics*")

                            report_link = gr.Markdown("")


                    with gr.Tab("MCQA") as mcqa_tab:
                        gr.Markdown("### Task 4: Multiple Choice Question Answering")
                        gr.Markdown("Browse MCQA samples with questions, images, and correct answers.")

                        # MCQA Analysis (confusion matrix, distributions, per-tier accuracy)
                        with gr.Accordion("MCQA Analysis (Confusion Matrix & Per-Tier Accuracy)", open=False) as mcqa_analysis_tab:
                            gr.Markdown("**Confusion matrix, answer distribution, and per-tier accuracy (full dataset)**")
                            mcqa_analysis_dropdown = gr.Dropdown(choices=[], value=None, label="Checkpoint",
                                                                 info="Select checkpoint to view aggregate MCQA analysis",
                                                                 allow_custom_value=True)
                            with gr.Row():
                                mcqa_confusion_plot = gr.Plot(label="Confusion Matrix")
                                mcqa_distribution_plot = gr.Plot(label="Prediction Distribution")
                            mcqa_tier_plot = gr.Plot(label="Per-Tier Accuracy")
                            mcqa_analysis_md = gr.Markdown("*Select a checkpoint above to view MCQA analysis*")

                        # Evaluation Results Browser
                        with gr.Accordion("Evaluation Results Browser (Model Responses)", open=False) as eval_browser_tab:
                            with gr.Row():
                                eval_browser_checkpoint = gr.Dropdown(
                                    choices=[], value=None, label="Checkpoint",
                                    info="Select checkpoint to browse model responses", scale=2,
                                    allow_custom_value=True)
                                eval_browser_filter = gr.Radio(
                                    choices=["All", "Incorrect Only", "Correct Only"],
                                    value="Incorrect Only", label="Filter", scale=1)
                                eval_browser_tier = gr.Dropdown(
                                    choices=["All", "T1: Single Keypoint", "T2: Binary Relation",
                                             "T3: Comparative", "T4: Multi Hop", "T5: Biomechanics"],
                                    value="All", label="Tier", scale=1)
                            with gr.Row():
                                eval_browser_prev = gr.Button("◄ Prev", size="sm", scale=1)
                                eval_browser_counter = gr.Markdown("*Select a checkpoint*", elem_classes=["sample-counter"])
                                eval_browser_next = gr.Button("Next ►", size="sm", scale=1)
                            with gr.Row():
                                eval_browser_image = gr.Image(label="Exercise Image", height=350, interactive=False, scale=1)
                                with gr.Column(scale=1):
                                    eval_browser_verdict = gr.Markdown("")
                                    eval_browser_detail = gr.Markdown("")
                            eval_browser_rf = gr.State(value=None)      # result file path (lightweight)
                            eval_browser_indices = gr.State(value=[])   # filtered indices (ints only)
                            eval_browser_tiers = gr.State(value=[])     # tier list (for display)
                            eval_browser_idx = gr.State(value=0)        # current position in filtered list

                        # Browse mode info + controls
                        gr.Markdown("""**Dataset Samples**: Browse MCQA questions, choices, and images from the JSONL dataset.
            **Validation Results**: Browse Qwen/Gemini validator judgments (CORRECT/INCORRECT) with comments. Use status filter to find disagreements.""")
                        with gr.Row():
                            mcqa_browse_mode = gr.Radio(
                                choices=["Dataset Samples", "Validation Results"],
                                value="Dataset Samples",
                                label="Browse Mode",
                                scale=2
                            )
                            mcqa_status_filter = gr.Radio(
                                choices=["All", "INCORRECT", "CORRECT"],
                                value="All",
                                label="Filter by Validator Status",
                                scale=2
                            )
                            mcqa_exercise_filter = gr.Dropdown(
                                choices=["All"],
                                value="All",
                                label="Filter by Exercise",
                                scale=1,
                                allow_custom_value=False,
                            )
                        with gr.Row():
                            mcqa_template_filter = gr.Dropdown(
                                choices=["All"],
                                value="All",
                                label="Filter by Question Template",
                                scale=2,
                                allow_custom_value=False,
                            )
                            mcqa_confusion_filter = gr.Dropdown(
                                choices=["All", "HIGH", "MEDIUM+", "Any flagged", "Not flagged"],
                                value="All",
                                label="2D/3D Confusion Risk",
                                scale=1,
                                allow_custom_value=False,
                            )
                            mcqa_error_label_filter = gr.Dropdown(
                                choices=["All"],
                                value="All",
                                label="Filter by Error Type",
                                scale=1,
                                allow_custom_value=False,
                            )

                        # Navigation controls
                        with gr.Row():
                            mcqa_prev_btn = gr.Button("◄ Previous", size="sm", scale=1)
                            with gr.Column(scale=2):
                                mcqa_sample_counter = gr.Markdown("Sample 0 of 0")
                            mcqa_next_btn = gr.Button("Next ►", size="sm", scale=1)

                        with gr.Row():
                            mcqa_jump_input = gr.Number(label="Jump to Sample (1-based)", value=1, minimum=1, step=1, precision=0, scale=3)
                            mcqa_jump_btn = gr.Button("Go", size="sm", scale=1)
                            mcqa_random_btn = gr.Button("Random", size="sm", variant="secondary", scale=1)
                            mcqa_refresh_btn = gr.Button("Refresh", size="sm", variant="secondary", scale=1)

                        # Image display
                        mcqa_image_display = gr.Image(label="Exercise Image", height=400, interactive=False)
                        mcqa_image_path_display = gr.Markdown("", elem_classes=["image-path-label"])

                        # Question text
                        mcqa_question_display = gr.Markdown("**Question:** Select a Task 4 MCQA variant to begin")

                        # Multiple choice options
                        mcqa_choices_display = gr.Radio(choices=[], value=None, label="Answer Choices", interactive=False)

                        # Metadata accordion
                        with gr.Accordion("Sample Metadata", open=False):
                            mcqa_metadata_display = gr.JSON(label="Full Metadata")

                        # Validator comments
                        mcqa_validator_display = gr.Markdown("*No validator comments available*")

                        # Generation prompt (per variant, not per sample)
                        with gr.Accordion("Generation Prompt", open=False):
                            mcqa_prompt_display = gr.Markdown("*Select a variant to see its generation prompt*")

                        # Reference docs — last accordions
                        with gr.Accordion("Dataset Version Guide", open=False):
                            gr.Markdown("""
            | Version | Method | Train | Test | Baseline | Best SFT | Status |
            |---------|--------|------:|-----:|---------:|---------:|--------|
            | **V1** | LLM descriptions + random distractors | 5,756 | 4,399 | 84.2% | 95.2% | Done (too easy) |
            | **V3** | Single-call VLM generation | 4,739 | 1,812 | 97.4% | — | Too easy |
            | **V4.2** | Description + keypoint hints, consensus filter | 1,515 | 1,013 | 94.0% | — | Too easy |
            | **V4.3** | Description + keypoint hints, Qwen-only filter | 2,480 | 1,931 | 91.3% | — | Too easy |
            | **V4.4** | Calibrated difficulty, tiered distractors (Qwen/Kimi) | — | 4,399 | 71.4% | — | Generation done |
            | **V5** | Geometric QA, Tier 1-2 only | 1,926 | 484 | 40.3% | 77.7% | Done |
            | **V5.1** | Balanced all-tier QA, Tier 1-5 | 6,361 | 3,936 | 32.7% | — | Training done |
            | **V5.2** | All-tier geometric QA (initial thresholds) | 6,732 | 3,916 | 36.5% | — | Superseded by V5.3 |
            | **V5.3** | V5.2 + foreshortening + tuned thresholds | 6,732 | 3,916 | 32.5% | 37.2% | Done (marginal) |
            | **V6.1.2** | LLM desc + geometric context (filtered) | 5,697 | 4,399 | 65.6% | 98.8% | Done (inflated) |
            | **V6.2** | Deterministic template descriptions (no LLM) | 6,500 | 7,872 | 49.4% | 75.4% | Done (best gain) |

            **V1–V4.x**: LLM-generated descriptions. All baselines too high (84-97%).

            **V5.x**: Pure geometric QA. V5.3 final: marginal gain (32.5→37.2%), model struggles with raw geometric reasoning.

            **V6.1.2**: 98.8% accuracy is **inflated** — distractors differ by 1-2 words. Not suitable for benchmarking.

            **V6.2**: Deterministic templates. Best honest benchmark: 49.4→75.4%. No LLM cost.
            """)

                        with gr.Accordion("V5.x Question Reference (Tiers, Angles, Distances)", open=False):
                            gr.Markdown("""
#### Question Tiers

| Tier | Name | Templates | Example |
|------|------|-----------|---------|
| 1 | Single Keypoint | Quadrant location (5 answers incl. "Middle") | "Where is the left wrist?" |
| 2 | Binary Relation | Joint angles (8) + distances (8) | "What is the angle at the left elbow?" |
| 3 | Comparative | Limb bend + height comparison (upright only) | "Which arm is more bent?" |
| 4 | Multi-Hop | Alignment + torso position | "Are the shoulders level?" |
| 5 | Geometric | Pose symmetry + most-bent joint | "Which joint is most bent?" |

---

#### Angles (Tier 2) — 8 templates

Measured at a **vertex keypoint** with two adjacent rays. Foreshortened segments rejected via per-segment anatomical ratios.

**Categories**: Acute (<75°), Right angle (75-115°), Obtuse (115-165°), Straight (>165°)
**Dead zones**: 65-75°, 105-115°, 165-175° — rejected to avoid ambiguity.

---

#### Distances (Tier 2) — 8 templates, normalized by **torso diameter**

Torso diameter = max pairwise distance among 4 torso keypoints (robust to foreshortening).

**Categories**: Very close (<25%), Close (25-110%), Far (110-200%), Very far (>200%)

---

#### Coordinate System

Quadrant labels use **anatomical left/right** (person's perspective), NOT camera left/right.
Camera mirror correction applied for front-facing poses.
            """)

                        # Hidden state
                        mcqa_current_idx = gr.State(value=0)

                    # ===== MCQA EVENT HANDLER FUNCTIONS =====

                    def is_mcqa_variant(variant):
                        return variant and ('mcqa' in variant.lower() or 'kpqa' in variant.lower())

                    def render_mcqa_sample(variant, split, sample_idx):
                        """Render a single MCQA dataset sample. Returns (image, image_path_md, question_md, choices_update, metadata, validator_md)."""
                        empty_validator = "*No validator comments available*"
                        no_path = ""
                        dataset_path = DATASET_INDEX.get('task4', {}).get(variant, {}).get('path')
                        if not dataset_path:
                            return None, no_path, "**Question:** Dataset not found", gr.update(choices=[], value=None), {}, empty_validator

                        sample = load_mcqa_sample(dataset_path, split, sample_idx)
                        if not sample:
                            return None, no_path, "**Question:** Sample not found", gr.update(choices=[], value=None), {}, empty_validator

                        image_path, question_text, correct_answer_raw = parse_mcqa_messages(sample)

                        # Load image
                        image = str(image_path) if image_path and Path(image_path).exists() else None
                        image_path_md = f"`{image_path}`" if image_path else ""

                        # Parse metadata
                        metadata = sample.get('metadata', {})
                        if isinstance(metadata, str):
                            metadata = json.loads(metadata)

                        correct_answer = metadata.get('correct_answer', correct_answer_raw or '?')
                        choices = metadata.get('choices', [])

                        # Build question markdown
                        if question_text:
                            # Extract just the question part (before A) )
                            parts = question_text.split('\nA)', 1)
                            if len(parts) == 1:
                                parts = question_text.split('A)', 1)
                            question_only = parts[0].strip()
                        else:
                            question_only = "Question not found"
                        question_md = f"**Question:** {question_only}"

                        # Format choices with labels
                        if choices:
                            labels = ['A', 'B', 'C', 'D']
                            choice_list = [f"{labels[i]}) {c}" for i, c in enumerate(choices) if i < len(labels)]
                            correct_idx = metadata.get('correct_index', -1)
                            correct_choice = choice_list[correct_idx] if 0 <= correct_idx < len(choice_list) else None
                            choices_update = gr.update(
                                choices=choice_list,
                                value=correct_choice,
                                label=f"Answer Choices (Correct: {correct_answer})"
                            )
                        else:
                            # Fallback: extract from question text
                            choice_list = []
                            if question_text:
                                for letter in ['A)', 'B)', 'C)', 'D)']:
                                    if letter in question_text:
                                        start = question_text.index(letter)
                                        # Find next letter or end
                                        end = len(question_text)
                                        for next_letter in ['B)', 'C)', 'D)']:
                                            if next_letter in question_text and question_text.index(next_letter) > start:
                                                end = min(end, question_text.index(next_letter))
                                                break
                                        choice_text = question_text[start:end].strip()
                                        # Strip trailing instruction text that may follow the last choice
                                        for suffix in ['Select the letter', 'Choose the letter', 'Pick the letter']:
                                            if suffix in choice_text:
                                                choice_text = choice_text[:choice_text.index(suffix)].rstrip(' .\n')
                                        choice_list.append(choice_text)
                            choices_update = gr.update(
                                choices=choice_list if choice_list else [],
                                value=None,
                                label=f"Answer Choices (Correct: {correct_answer})"
                            )

                        # Validator comments — only for description-based variants (v3, v4.x)
                        # V5 is keypoint QA and V1 uses random distractors; validators checked descriptions only
                        image_id = metadata.get('image_id', '')
                        validator_md = empty_validator
                        has_validators = variant and any(tag in variant for tag in ['v3', 'v4'])
                        if has_validators and image_id and image_id in VALIDATOR_INDEX:
                            parts = []
                            allowed_keys = _get_variant_validator_keys(variant)
                            for vname, vdata in sorted(VALIDATOR_INDEX[image_id].items()):
                                if allowed_keys and vname not in allowed_keys:
                                    continue
                                status = vdata.get('status', 'UNKNOWN')
                                icon = "✅" if status == "CORRECT" else "❌" if status == "INCORRECT" else "❓"
                                # Clean display name: 'qwen_v4.4' → 'Qwen (V4.4)'
                                display_name = vname.split('_')[0].title()
                                if '_v' in vname:
                                    display_name += f" ({vname.split('_', 1)[1].upper()})"
                                parts.append(f"**{display_name}**: {icon} {status}")
                                issues = vdata.get('issues', '')
                                if issues and issues != 'None':
                                    parts.append(f"  Issues: {issues}")
                                corrected = vdata.get('corrected_description', '')
                                if corrected and corrected != 'None':
                                    parts.append(f"  Description: {corrected}")
                            if parts:
                                validator_md = "\n\n".join(parts)

                        # 2D/3D confusion flags
                        tmpl = metadata.get('question_template', '')
                        flag_key = (image_id, tmpl)
                        flag_data = CONFUSION_FLAGS.get(flag_key)
                        if flag_data:
                            risk = flag_data['max_risk']
                            risk_icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(risk, "⚪")
                            flag_parts = [f"**2D/3D Confusion Risk:** {risk_icon} **{risk}**"]
                            for fl in flag_data['flags']:
                                reason = fl.get('reason', '')
                                detail_items = []
                                if 'min_ratio' in fl:
                                    detail_items.append(f"segment ratio: {fl['min_ratio']:.3f}")
                                if 'segment1' in fl:
                                    detail_items.append(f"{fl['segment1']}={fl.get('segment1_length', '?')}px")
                                if 'segment2' in fl:
                                    detail_items.append(f"{fl['segment2']}={fl.get('segment2_length', '?')}px")
                                if 'torso_diameter' in fl:
                                    detail_items.append(f"torso_d={fl['torso_diameter']}px")
                                if 'flags' in fl:
                                    detail_items.extend(fl['flags'])
                                if 'real_angle' in fl and fl['real_angle'] is not None:
                                    detail_items.append(f"3D angle: {fl['real_angle']}\u00b0")
                                if 'category' in fl:
                                    detail_items.append(f"2D category: {fl['category']}")
                                if 'ratio' in fl:
                                    detail_items.append(f"shoulder/torso ratio: {fl['ratio']:.3f}")
                                detail_str = ", ".join(detail_items) if detail_items else str(fl)
                                flag_parts.append(f"- **{reason}**: {detail_str}")
                            confusion_md = "\n".join(flag_parts)
                            if validator_md == empty_validator:
                                validator_md = confusion_md
                            else:
                                validator_md = validator_md + "\n\n---\n\n" + confusion_md

                        # Reasoning traces (e.g. Kimi K2.5 chain-of-thought)
                        reasoning = metadata.get('reasoning_content', '')
                        if reasoning:
                            reasoning_escaped = reasoning.replace('<', '&lt;').replace('>', '&gt;')
                            reasoning_md = (
                                f"\n\n---\n\n"
                                f"<details><summary><b>Generation Reasoning Trace</b> ({len(reasoning):,} chars)</summary>\n\n"
                                f"<pre style=\"white-space: pre-wrap; word-wrap: break-word; font-size: 0.85em;\">"
                                f"{reasoning_escaped}</pre>\n\n</details>"
                            )
                            if validator_md == empty_validator:
                                validator_md = reasoning_md.lstrip('\n-')
                            else:
                                validator_md += reasoning_md

                        return image, image_path_md, question_md, choices_update, metadata, validator_md

                    def render_validation_result(image_id, variant=""):
                        """Render a validation result by image_id. Returns (image, image_path_md, question_md, choices_update, metadata, validator_md)."""
                        if image_id not in VALIDATOR_INDEX:
                            return None, "", "**No result**", gr.update(choices=[]), {}, ""

                        validators = VALIDATOR_INDEX[image_id]
                        allowed_keys = _get_variant_validator_keys(variant)
                        # Pick the first relevant validator for image/description display
                        if allowed_keys:
                            first_v = next((validators[k] for k in allowed_keys if k in validators), next(iter(validators.values())))
                        else:
                            first_v = next(iter(validators.values()))
                        image_path = first_v.get('image_path', '')
                        image = str(image_path) if image_path and Path(image_path).exists() else None
                        image_path_md = f"`{image_path}`" if image_path else ""

                        # Build validator summary
                        parts = [f"**Image ID:** `{image_id}`\n"]
                        for vname, vdata in sorted(validators.items()):
                            if allowed_keys and vname not in allowed_keys:
                                continue
                            status = vdata.get('status', 'UNKNOWN')
                            icon = "✅" if status == "CORRECT" else "❌" if status == "INCORRECT" else "❓"
                            display_name = vname.split('_')[0].title()
                            if '_v' in vname:
                                display_name += f" ({vname.split('_', 1)[1].upper()})"
                            parts.append(f"### {display_name} Validator: {icon} {status}")
                            issues = vdata.get('issues', '')
                            if issues and issues != 'None':
                                parts.append(f"**Issues:** {issues}")
                            corrected = vdata.get('corrected_description', '')
                            if corrected and corrected != 'None':
                                parts.append(f"**Corrected:** {corrected}")

                        desc = first_v.get('corrected_description', '') or ''
                        question_md = f"**Description:** {desc[:200]}..." if len(desc) > 200 else f"**Description:** {desc}" if desc else "**Description:** N/A"

                        return image, image_path_md, question_md, gr.update(choices=[], value=None), {'image_id': image_id}, "\n\n".join(parts)

                    # ===== MCQA NAVIGATION FUNCTIONS =====

                    mcqa_nav_outputs = [mcqa_current_idx, mcqa_sample_counter, mcqa_image_display,
                                       mcqa_image_path_display, mcqa_question_display, mcqa_choices_display,
                                       mcqa_metadata_display, mcqa_validator_display]

                    def load_mcqa_initial(task, variant, split, status_filter="All", exercise_prefix="All", question_template="All", confusion_filter="All", error_label="All"):
                        """Load first MCQA sample when variant changes."""
                        if task != 'task4' or not is_mcqa_variant(variant):
                            return 0, "Sample 0 of 0", None, "", "**Question:** Select a Task 4 MCQA variant", gr.update(choices=[]), {}, ""
                        dataset_path = DATASET_INDEX.get('task4', {}).get(variant, {}).get('path')
                        if not dataset_path:
                            return 0, "Sample 0 of 0", None, "", "**Question:** Dataset not found", gr.update(choices=[]), {}, ""

                        filtered = get_filtered_dataset_indices(dataset_path, split, status_filter, variant, exercise_prefix, question_template, confusion_filter, error_label)
                        if filtered is not None:
                            total = len(filtered)
                            if total == 0:
                                return 0, "0 matching samples", None, "", "**No samples match filter**", gr.update(choices=[]), {}, ""
                            idx = filtered[0]
                            counter = f"Sample 1 of {total:,} (filtered)"
                        else:
                            total = count_mcqa_samples(dataset_path, split)
                            if total == 0:
                                return 0, "Sample 0 of 0", None, "", "**Question:** No samples", gr.update(choices=[]), {}, ""
                            idx = 0
                            counter = f"Sample 1 of {total:,}"

                        image, ip, q, c, m, v = render_mcqa_sample(variant, split, idx)
                        return 0, counter, image, ip, q, c, m, v

                    def _nav_dataset(pos, variant, split, status_filter, exercise_prefix="All", question_template="All", confusion_filter="All", error_label="All"):
                        """Navigate dataset samples by filtered position."""
                        dataset_path = DATASET_INDEX.get('task4', {}).get(variant, {}).get('path')
                        if not dataset_path:
                            return 0, "Sample 0 of 0", None, "", "**Question:** No dataset", gr.update(choices=[]), {}, ""

                        filtered = get_filtered_dataset_indices(dataset_path, split, status_filter, variant, exercise_prefix, question_template, confusion_filter, error_label)
                        if filtered is not None:
                            total = len(filtered)
                            if total == 0:
                                return 0, "0 matching samples", None, "", "**No samples match filter**", gr.update(choices=[]), {}, ""
                            pos = max(0, min(total - 1, pos))
                            sample_idx = filtered[pos]
                            counter = f"Sample {pos + 1} of {total:,} (filtered)"
                        else:
                            total = count_mcqa_samples(dataset_path, split)
                            if total == 0:
                                return 0, "Sample 0 of 0", None, "", "**Question:** No samples", gr.update(choices=[]), {}, ""
                            pos = max(0, min(total - 1, pos))
                            sample_idx = pos
                            counter = f"Sample {pos + 1} of {total:,}"

                        image, ip, q, c, m, v = render_mcqa_sample(variant, split, sample_idx)
                        return pos, counter, image, ip, q, c, m, v

                    def _variant_has_validators(variant):
                        """Validators only exist for description-based variants (v3, v4.x, v6.1)."""
                        return variant and any(tag in variant for tag in ['v3', 'v4', 'v6.1'])

                    _no_validators_msg = (0, "Result 0 of 0", None, "",
                        "**Validation results are only available for description-based variants (V3, V4.x, V6.1).** "
                        "The selected variant uses a different question format (keypoint QA or deterministic descriptions).",
                        gr.update(choices=[]), {}, "*No validator data for this variant*")

                    def _nav_validation(idx, status_filter, variant=""):
                        """Navigate validation results by index."""
                        ids = get_filtered_validation_ids(status_filter, variant)
                        if not ids:
                            return 0, "Result 0 of 0", None, "", "**No results**", gr.update(choices=[]), {}, ""
                        idx = max(0, min(len(ids) - 1, idx))
                        image, ip, q, c, m, v = render_validation_result(ids[idx], variant)
                        return idx, f"Result {idx + 1} of {len(ids):,}", image, ip, q, c, m, v

                    def _on_mcqa_filter_change(browse_mode, status_filter, task, variant, split, exercise_prefix, question_template, confusion_filter, error_label):
                        """Unified handler for all MCQA filter changes."""
                        if browse_mode == "Validation Results":
                            if not _variant_has_validators(variant):
                                return _no_validators_msg
                            return _nav_validation(0, status_filter, variant)
                        if not _variant_has_validators(variant) and status_filter != "All":
                            status_filter = "All"
                        return load_mcqa_initial(task, variant, split, status_filter, exercise_prefix, question_template, confusion_filter, error_label)

                    def on_mcqa_prev(task, variant, split, current_idx, browse_mode, status_filter, exercise_prefix, question_template, confusion_filter, error_label):
                        if browse_mode == "Validation Results":
                            if not _variant_has_validators(variant):
                                return _no_validators_msg
                            return _nav_validation(current_idx - 1, status_filter, variant)
                        if task != 'task4' or not is_mcqa_variant(variant):
                            return current_idx, "Sample 0 of 0", None, "", "**Question:** No dataset", gr.update(choices=[]), {}, ""
                        return _nav_dataset(current_idx - 1, variant, split, status_filter, exercise_prefix, question_template, confusion_filter, error_label)

                    def on_mcqa_next(task, variant, split, current_idx, browse_mode, status_filter, exercise_prefix, question_template, confusion_filter, error_label):
                        if browse_mode == "Validation Results":
                            if not _variant_has_validators(variant):
                                return _no_validators_msg
                            return _nav_validation(current_idx + 1, status_filter, variant)
                        if task != 'task4' or not is_mcqa_variant(variant):
                            return current_idx, "Sample 0 of 0", None, "", "**Question:** No dataset", gr.update(choices=[]), {}, ""
                        return _nav_dataset(current_idx + 1, variant, split, status_filter, exercise_prefix, question_template, confusion_filter, error_label)

                    def on_mcqa_jump(task, variant, split, jump_value, browse_mode, status_filter, exercise_prefix, question_template, confusion_filter, error_label):
                        try:
                            target = max(0, int(jump_value or 1) - 1)  # Convert 1-based to 0-based
                        except (TypeError, ValueError):
                            target = 0
                        if browse_mode == "Validation Results":
                            return _nav_validation(target, status_filter, variant)
                        if task != 'task4' or not is_mcqa_variant(variant):
                            return 0, "Sample 0 of 0", None, "", "**Question:** No dataset", gr.update(choices=[]), {}, ""
                        return _nav_dataset(target, variant, split, status_filter, exercise_prefix, question_template, confusion_filter, error_label)

                    def on_mcqa_random(task, variant, split, browse_mode, status_filter, exercise_prefix, question_template, confusion_filter, error_label):
                        if browse_mode == "Validation Results":
                            ids = get_filtered_validation_ids(status_filter, variant)
                            target = random.randint(0, max(0, len(ids) - 1)) if ids else 0
                            return _nav_validation(target, status_filter, variant)
                        if task != 'task4' or not is_mcqa_variant(variant):
                            return 0, "Sample 0 of 0", None, "", "**Question:** No dataset", gr.update(choices=[]), {}, ""
                        dataset_path = DATASET_INDEX.get('task4', {}).get(variant, {}).get('path')
                        if not dataset_path:
                            return 0, "Sample 0 of 0", None, "", "**Question:** No dataset", gr.update(choices=[]), {}, ""
                        filtered = get_filtered_dataset_indices(dataset_path, split, status_filter, variant, exercise_prefix, question_template, confusion_filter, error_label)
                        if filtered is not None:
                            total = len(filtered)
                        else:
                            total = count_mcqa_samples(dataset_path, split)
                        target = random.randint(0, max(0, total - 1)) if total > 0 else 0
                        return _nav_dataset(target, variant, split, status_filter, exercise_prefix, question_template, confusion_filter, error_label)

                    def on_mcqa_refresh(task, variant, split, browse_mode, status_filter, exercise_prefix, question_template, confusion_filter, error_label):
                        """Refresh validator index and dataset index, then reload current view."""
                        global VALIDATOR_INDEX, DATASET_INDEX, CONFUSION_FLAGS
                        VALIDATOR_INDEX = build_validator_index()
                        DATASET_INDEX = build_dataset_index()
                        CONFUSION_FLAGS = build_confusion_flags_index()
                        clear_all_caches()
                        logging.info(f"Refreshed: {len(VALIDATOR_INDEX)} validated IDs, {len(DATASET_INDEX.get('task4', {}))} task4 variants")
                        # Reload current view
                        nav = load_mcqa_initial(task, variant, split, status_filter, exercise_prefix, question_template, confusion_filter, error_label)
                        dataset_path = DATASET_INDEX.get('task4', {}).get(variant, {}).get('path')
                        if dataset_path:
                            prefixes = get_exercise_prefixes(dataset_path, split)
                            exercise_choices = ["All"] + list(prefixes)
                            templates = get_question_templates(dataset_path, split)
                            template_choices = ["All"] + list(templates)
                            elabels = get_error_labels(dataset_path, split)
                            error_label_choices = ["All"] + list(elabels)
                        else:
                            exercise_choices = ["All"]
                            template_choices = ["All"]
                            error_label_choices = ["All"]
                        prompt_md = get_generation_prompt_md(variant)
                        return (*nav, gr.update(choices=exercise_choices, value="All"), gr.update(choices=template_choices, value="All"), gr.update(value="All"), gr.update(choices=error_label_choices, value="All"), prompt_md)

                    # ===== MCQA EVENT WIRING =====

                    _mcqa_filter_inputs = [mcqa_browse_mode, mcqa_status_filter, task_dropdown, variant_dropdown, split_radio, mcqa_exercise_filter, mcqa_template_filter, mcqa_confusion_filter, mcqa_error_label_filter]
                    for component in [mcqa_browse_mode, mcqa_status_filter, mcqa_exercise_filter, mcqa_template_filter, mcqa_confusion_filter, mcqa_error_label_filter]:
                        component.change(
                            fn=_on_mcqa_filter_change,
                            inputs=_mcqa_filter_inputs,
                            outputs=mcqa_nav_outputs
                        )
                    mcqa_prev_btn.click(
                        fn=on_mcqa_prev,
                        inputs=[task_dropdown, variant_dropdown, split_radio, mcqa_current_idx, mcqa_browse_mode, mcqa_status_filter, mcqa_exercise_filter, mcqa_template_filter, mcqa_confusion_filter, mcqa_error_label_filter],
                        outputs=mcqa_nav_outputs
                    )
                    mcqa_next_btn.click(
                        fn=on_mcqa_next,
                        inputs=[task_dropdown, variant_dropdown, split_radio, mcqa_current_idx, mcqa_browse_mode, mcqa_status_filter, mcqa_exercise_filter, mcqa_template_filter, mcqa_confusion_filter, mcqa_error_label_filter],
                        outputs=mcqa_nav_outputs
                    )
                    mcqa_jump_btn.click(
                        fn=on_mcqa_jump,
                        inputs=[task_dropdown, variant_dropdown, split_radio, mcqa_jump_input, mcqa_browse_mode, mcqa_status_filter, mcqa_exercise_filter, mcqa_template_filter, mcqa_confusion_filter, mcqa_error_label_filter],
                        outputs=mcqa_nav_outputs
                    )
                    mcqa_random_btn.click(
                        fn=on_mcqa_random,
                        inputs=[task_dropdown, variant_dropdown, split_radio, mcqa_browse_mode, mcqa_status_filter, mcqa_exercise_filter, mcqa_template_filter, mcqa_confusion_filter, mcqa_error_label_filter],
                        outputs=mcqa_nav_outputs
                    )

                    mcqa_refresh_btn.click(
                        fn=on_mcqa_refresh,
                        inputs=[task_dropdown, variant_dropdown, split_radio, mcqa_browse_mode, mcqa_status_filter, mcqa_exercise_filter, mcqa_template_filter, mcqa_confusion_filter, mcqa_error_label_filter],
                        outputs=mcqa_nav_outputs + [mcqa_exercise_filter, mcqa_template_filter, mcqa_confusion_filter, mcqa_error_label_filter, mcqa_prompt_display]
                    )

                    # Also reload MCQA and update exercise + template + error label filters when variant/split changes
                    mcqa_variant_outputs = mcqa_nav_outputs + [mcqa_exercise_filter, mcqa_template_filter, mcqa_confusion_filter, mcqa_error_label_filter, mcqa_prompt_display]

                    def on_mcqa_variant_change(task, variant, split):
                        nav = load_mcqa_initial(task, variant, split)
                        dataset_path = DATASET_INDEX.get('task4', {}).get(variant, {}).get('path')
                        if dataset_path:
                            prefixes = get_exercise_prefixes(dataset_path, split)
                            exercise_choices = ["All"] + list(prefixes)
                            templates = get_question_templates(dataset_path, split)
                            template_choices = ["All"] + list(templates)
                            elabels = get_error_labels(dataset_path, split)
                            error_label_choices = ["All"] + list(elabels)
                        else:
                            exercise_choices = ["All"]
                            template_choices = ["All"]
                            error_label_choices = ["All"]
                        prompt_md = get_generation_prompt_md(variant)
                        return (*nav, gr.update(choices=exercise_choices, value="All"), gr.update(choices=template_choices, value="All"), gr.update(value="All"), gr.update(choices=error_label_choices, value="All"), prompt_md)

                    variant_dropdown.change(
                        fn=on_mcqa_variant_change,
                        inputs=[task_dropdown, variant_dropdown, split_radio],
                        outputs=mcqa_variant_outputs
                    )
                    split_radio.change(
                        fn=on_mcqa_variant_change,
                        inputs=[task_dropdown, variant_dropdown, split_radio],
                        outputs=mcqa_variant_outputs
                    )


                    # === Tab 6: Mixed vs Single-Task Comparison (Dynamic) ===
                    with gr.Tab("Mixed vs Single", visible=CONFIG["show_mixed_vs_single"]):
                        # Generate dynamic comparison from experiments-final.csv
                        comparison_summary, comparison_bar_chart = create_mixed_vs_single_comparison()
                        progression_chart = create_mixed_vs_single_progression()

                        gr.Markdown(comparison_summary)
                        gr.Markdown("## Visualizations")
                        gr.Plot(comparison_bar_chart, label="Best Checkpoint Comparison")
                        gr.Plot(progression_chart, label="Task 1 OKS Progression")

                    with gr.Tab("Metrics"):
                        gr.Markdown("""
## Metric Definitions

### Shared Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **OKS** | `exp(-d² / (2 × 0.53 × area × σ²))` | Object Keypoint Similarity. Per-keypoint score averaged across all matched keypoints. `d` = Euclidean distance (px), `area` = image width × height, `σ` = per-keypoint tolerance (e.g. 0.026 for nose, 0.107 for hip). Range [0, 1], higher is better. |
| **F1 Score** | `2 × P × R / (P + R)` | Harmonic mean of precision and recall. |
| **Precision** | `matched / predicted` | Fraction of predictions that match a ground truth keypoint. |
| **Recall** | `matched / ground_truth` | Fraction of ground truth keypoints that were predicted. |
| **PCK@T** | `count(d ≤ T) / total` | Percentage of Correct Keypoints within T pixels. Reported at T = 50, 100, 150. |
| **MAE** | `mean(\|pred - gt\|)` | Mean Absolute Error in pixels. Reported per-axis (x, y) and total (sum of both axes). |
| **L/R Confusion** | _(see per-task definitions below)_ | Rate at which the model confuses left/right counterpart keypoints. Lower is better. |

---

### Task 1: Keypoint Coordinate Prediction

Given an image, predict `(x, y)` coordinates for all keypoints in [0–1000] range.

| Metric | Details |
|--------|---------|
| **OKS** | Primary metric. Uses COCO-style per-keypoint sigmas and heuristic factor 0.53. |
| **Precision / Recall / F1** | A keypoint is "matched" if predicted name exists in GT, with distance within 10% of torso diagonal (score threshold 0.35). |
| **MAE** | Average pixel error across matched keypoints. |
| **PCK@50/100/150** | Fraction of matched keypoints within 50/100/150 px. |
| **L/R Confusion** | For each anatomical L/R pair (e.g. left_shoulder / right_shoulder): compute `d_current = dist(pred_L, gt_L) + dist(pred_R, gt_R)` vs `d_swapped = dist(pred_L, gt_R) + dist(pred_R, gt_L)`. If swapped distance is smaller, the model confused left and right. Rate = swaps / evaluable pairs. |

---

### Task 2: Keypoint Labeling (Identification)

Given an image with numbered keypoints, predict the body part name for each number.

| Metric | Details |
|--------|---------|
| **Per-Keypoint Accuracy** | Primary metric. `correct_names / total_labels`. Matched by index — correct if `predicted_name == gt_name`. |
| **Exact Match** | 1 if ALL keypoint names in the sample are correct, 0 otherwise. |
| **L/R Confusion** | Fraction of predictions where the model predicted the left/right counterpart of the correct label (e.g. predicted "left_eye" when GT is "right_eye"). Rate = confusions / total labels. |

---

### Task 3a: Error Detection

Given an image with a pose overlay, identify which keypoint indices have errors (wrong position).

| Metric | Details |
|--------|---------|
| **F1 / Precision / Recall** | Primary metric. Based on set overlap between predicted error indices and GT error indices. TP = correctly flagged, FP = flagged but not an error, FN = missed error. |
| **Error Detection Accuracy** | Fraction of samples where the exact set of error indices was predicted correctly. |
| **L/R Confusion** | Always 0 for 3a — detection is index-based with no spatial component. |

---

### Task 3b/c/d: Error Correction

Given an image with a pose overlay, identify erroneous keypoints AND predict their corrected coordinates.

| Metric | Details |
|--------|---------|
| **F1 / Precision / Recall** | Same as 3a — based on which indices the model flagged. |
| **OKS** | Applied only to true positives (correctly flagged errors). Measures how accurately the model corrected the position. |
| **MAE** | Average pixel error of corrected positions vs GT positions (true positives only). |
| **PCK@50/100/150** | Fraction of corrected positions within threshold (true positives only). |
| **L/R Confusion** | When the model makes a false positive (flags index X, not an error) AND misses a false negative (index Y, actual error), checks if X and Y are L/R counterparts (e.g. left_knee vs right_knee). Rate = L/R confusions / total GT errors. |

**Subtask variants:** 3b = missing keypoints (low/high), 3c = displaced keypoints (small/background), 3d = mixed/challenging.

---

### Task 4: MCQA (Multiple Choice Question Answering)

Given an image and a 4-option question about keypoint geometry, select the correct answer (A/B/C/D).

| Metric | Details |
|--------|---------|
| **Accuracy** | Primary metric. `correct_answers / total_samples`. Exact match on letter (A/B/C/D). |
| **Parse Rate** | Fraction of model responses from which a valid answer letter could be extracted. |
| **Correct Count** | Absolute number of correct answers. |

**Variant families and comparability:**

| Family | Variants | Question Type | Baseline Range | Notes |
|--------|----------|---------------|----------------|-------|
| **V1–V4.x** | V1, V3, V4.2, V4.3, V4.4 | LLM-generated descriptions | 71–97% | Baselines too high; model already understands descriptions without SFT. |
| **V5.x** | V5, V5.1, V5.2, V5.3 | Pure geometric QA (angles, distances) | 32–40% | Hard for the model; marginal SFT gains (+5pp). Tests raw geometric reasoning. |
| **V6.x** | V6.1, V6.1.2, V6.2 | Descriptions from geometric analysis | 49–66% | V6.2 (templates, 49→75%) is the honest benchmark. |

**V6.1.2 caveat:** 98.8% accuracy is inflated. LLM-generated distractors differ from the correct answer by only 1–2 words (avg 90.9% similarity). The model learns word-level pattern matching, not pose comprehension. Use V6.2 for benchmarking.
""")

                    # ========== REASONING TRACES TAB ==========
                    with gr.Tab("Reasoning Traces") as reasoning_tab:
                        with gr.Row():
                            # --- Sidebar ---
                            with gr.Column(scale=1):
                                # Uses sidebar task_dropdown + variant_dropdown (no duplicate selectors)
                                with gr.Row():
                                    reas_prev_btn = gr.Button("◄ Prev", size="sm")
                                    reas_next_btn = gr.Button("Next ►", size="sm")
                                with gr.Row():
                                    reas_jump = gr.Number(value=1, minimum=1, label="Sample #", scale=2)
                                    reas_jump_btn = gr.Button("Go", size="sm", scale=1)
                                reas_random_btn = gr.Button("Random", size="sm")
                                reas_filter = gr.Dropdown(
                                    choices=["All",
                                             "Orientation Mismatch (All)",
                                             "Orientation Mismatch (High)",
                                             "Orientation Mismatch (Medium)",
                                             "Orientation Mismatch (Low)",
                                             "Body Position Mismatch",
                                             "High-Angle Disagreement",
                                             "Hallucinated",
                                             "Audit Excluded",
                                             "Approved"],
                                    value="All", label="Filter", interactive=True)
                                with gr.Accordion("Search", open=False):
                                    reas_search = gr.Textbox(label="Image ID", placeholder="paste image_id...")
                                    reas_search_btn = gr.Button("Search all tasks", size="sm")
                                reas_stats = gr.Markdown("*Select a task*")

                                # --- Audit controls (in sidebar, below stats) ---
                                with gr.Accordion("Audit", open=True):
                                    reas_audit_status = gr.Markdown("*No sample loaded*")
                                    with gr.Row():
                                        reas_audit_reason = gr.Dropdown(
                                            choices=AUDIT_REASONS,
                                            value="wrong_orientation",
                                            label="Reason", scale=2, interactive=True)
                                    with gr.Row():
                                        reas_approve_btn = gr.Button("Approve", variant="primary", size="sm", scale=1)
                                        reas_exclude_btn = gr.Button("Exclude", variant="stop", size="sm", scale=1)
                                        reas_undo_btn = gr.Button("Undo", size="sm", scale=1)

                            # --- Main content ---
                            with gr.Column(scale=4):
                                reas_header = gr.Markdown("*Select a task and navigate to a sample*")
                                reas_image = gr.Image(label="Image", height=400, interactive=False)

                                with gr.Accordion("1. Teacher Input", open=True):
                                    reas_teacher_sys = gr.Code(label="System Message", language=None, lines=6, interactive=False)
                                    reas_teacher_prompt = gr.Code(
                                        label="User Prompt (full rendering with GT)",
                                        language=None, lines=20, interactive=False)

                                with gr.Accordion("2. Teacher Reasoning Trace", open=True):
                                    reas_reasoning = gr.Markdown("*No sample loaded*")

                                with gr.Accordion("3. Training Sample (what student learns)", open=False):
                                    reas_train_sys = gr.Code(label="System", language=None, lines=4, interactive=False)
                                    reas_train_user = gr.Code(label="User Prompt (no GT)", language=None, lines=15, interactive=False)
                                    reas_train_assistant = gr.Code(
                                        label="Assistant (<think> + <answer>)",
                                        language=None, lines=20, interactive=False)

                                with gr.Accordion("4. Metadata", open=False):
                                    reas_metadata = gr.JSON(label="Full Metadata")

                                with gr.Accordion("5. Cross-Task View", open=False):
                                    reas_cross_task_btn = gr.Button("Find this image in all tasks")
                                    reas_cross_task = gr.Markdown("*Click button to search*")

                        # State
                        reas_current_idx = gr.State(value=0)
                        reas_current_image_id = gr.State(value="")
                        reas_filtered_indices = gr.State(value=None)  # None = no filter, list = filtered indices

                    # ========== PROMPT COMPARISON TAB ==========
                    with gr.Tab("Prompt Comparison") as prompt_cmp_tab:
                        with gr.Row():
                            # --- Sidebar ---
                            with gr.Column(scale=1):
                                gr.Markdown("### Prompt Comparison")
                                pcmp_experiment = gr.Dropdown(
                                    choices=_discover_comparison_experiments(),
                                    value=None, label="Experiment",
                                    info="Select comparison experiment")
                                pcmp_task = gr.Dropdown(
                                    choices=[], value=None,
                                    label="Task", interactive=True)
                                with gr.Row():
                                    pcmp_prev_btn = gr.Button("< Prev", size="sm")
                                    pcmp_next_btn = gr.Button("Next >", size="sm")
                                with gr.Row():
                                    pcmp_jump = gr.Number(value=1, minimum=1, label="Sample #", scale=2)
                                    pcmp_jump_btn = gr.Button("Go", size="sm", scale=1)
                                pcmp_random_btn = gr.Button("Random", size="sm")
                                pcmp_filter = gr.Dropdown(
                                    choices=["All", "Any Hallucinated", "Versions Disagree (word count)"],
                                    value="All", label="Filter", interactive=True)
                                pcmp_stats = gr.Markdown("*Select an experiment*")

                            # --- Main content ---
                            with gr.Column(scale=4):
                                pcmp_header = gr.Markdown("*Select an experiment and task*")
                                pcmp_image = gr.Image(label="Image", height=400, interactive=False)

                                with gr.Row():
                                    with gr.Column(visible=True) as pcmp_col1:
                                        pcmp_label1 = gr.Markdown("**v1**")
                                        pcmp_think1 = gr.Textbox(label="Reasoning", lines=15, interactive=False)
                                        pcmp_answer1 = gr.Code(label="Answer", language="json", lines=6, interactive=False)
                                        pcmp_badge1 = gr.Markdown("")
                                    with gr.Column(visible=True) as pcmp_col2:
                                        pcmp_label2 = gr.Markdown("**v2**")
                                        pcmp_think2 = gr.Textbox(label="Reasoning", lines=15, interactive=False)
                                        pcmp_answer2 = gr.Code(label="Answer", language="json", lines=6, interactive=False)
                                        pcmp_badge2 = gr.Markdown("")
                                    with gr.Column(visible=True) as pcmp_col3:
                                        pcmp_label3 = gr.Markdown("**v3**")
                                        pcmp_think3 = gr.Textbox(label="Reasoning", lines=15, interactive=False)
                                        pcmp_answer3 = gr.Code(label="Answer", language="json", lines=6, interactive=False)
                                        pcmp_badge3 = gr.Markdown("")
                                    with gr.Column(visible=True) as pcmp_col4:
                                        pcmp_label4 = gr.Markdown("**v4**")
                                        pcmp_think4 = gr.Textbox(label="Reasoning", lines=15, interactive=False)
                                        pcmp_answer4 = gr.Code(label="Answer", language="json", lines=6, interactive=False)
                                        pcmp_badge4 = gr.Markdown("")

                        # State
                        pcmp_current_idx = gr.State(value=0)
                        pcmp_exp_data = gr.State(value=None)
                        pcmp_filtered_indices = gr.State(value=None)


        # ========== EVENT HANDLERS ==========

        def get_metric_choices(task):
            """Get appropriate metric choices based on task type."""
            if task == 'mixed':
                return ["Multi-Task"]  # Mixed task plot shows all 4 tasks automatically
            elif task == 'task2':
                return ["Accuracy", "L/R Confusion (%)"]
            elif task in ['task1', 'task1b', 'task1c']:
                return ["OKS", "OKS (Conf-Weighted)", "F1 Score", "Precision", "Recall", "MAE", "PCK@50", "L/R Confusion (%)"]
            elif task in ['task3a', 'task3b', 'task3c', 'task3d']:
                return ["F1 Score", "Precision", "Recall", "OKS", "PCK@50", "L/R Confusion (%)"]
            elif task == 'task4':
                return ["Accuracy", "Parse Rate"]
            else:
                return ["OKS", "F1 Score", "Precision", "Recall", "MAE", "PCK@50"]

        def on_task_change(task):
            """Update variant dropdown and metric selector when task changes."""
            if task not in DATASET_INDEX:
                return gr.Dropdown(choices=[], value=None), gr.Dropdown(choices=[], value=None)

            variants = get_active_variants(task, list(DATASET_INDEX[task].keys()))
            metric_choices = get_metric_choices(task)

            return (
                gr.Dropdown(choices=variants, value=variants[0] if variants else None),
                gr.Dropdown(choices=metric_choices, value=metric_choices[0] if metric_choices else "OKS")
            )

        def on_variant_change(task, variant, split, metric):
            """Update stats, gallery, prompt, and checkpoint list when variant changes."""
            # Update stats card
            stats = create_stats_card(task, variant, split)

            # Get task prompt
            prompt = get_task_prompt(task, variant, split)

            # Get checkpoint list
            checkpoints = create_checkpoint_list(task, variant)

            # Update gallery with first page
            if task in DATASET_INDEX and variant in DATASET_INDEX[task]:
                dataset_path = DATASET_INDEX[task][variant]['path']
                gallery_images = create_image_gallery(dataset_path, split, page=0, page_size=50)
                num_samples = DATASET_INDEX[task][variant].get(f'{split}_samples', 0)
                num_pages = (num_samples + 49) // 50  # Ceiling division
                page_text = f"Page 1 of {num_pages} ({num_samples:,} total samples)"
            else:
                gallery_images = []
                page_text = "No dataset found"

            # Create checkpoint table and metrics plot
            # Training Monitor shows all checkpoints for task (all variants)
            cp_table = create_checkpoint_table(task, variant=None, all_variants=True)
            metrics_figure = create_metrics_plot(task, variant=None, all_variants=True, metric=metric)

            # Checkpoint dropdown choices (for details view)
            cp_dropdown_choices = checkpoints

            # Format checkpoint list as text (informational only)
            if checkpoints:
                checkpoint_text = "\n".join([f"• {cp}" for cp in checkpoints])
            else:
                checkpoint_text = "No checkpoints available for this task/variant"

            # Control visibility based on split — show predictions/metrics for both train and test
            # (train eval results are useful for diagnosing whether SFT learned the training data)
            is_test_split = True

            # Populate exercise filter
            if task in DATASET_INDEX and variant in DATASET_INDEX.get(task, {}):
                dataset_path_for_ex = DATASET_INDEX[task][variant]['path']
                prefixes = get_exercise_prefixes(dataset_path_for_ex, split)
                exercise_choices = ["All"] + list(prefixes)
            else:
                exercise_choices = ["All"]

            return (
                stats,  # stats_display
                prompt,  # prompt_display
                checkpoint_text,  # checkpoint_list (now a text display)
                gr.update(choices=checkpoints, value=[]),  # selected_checkpoints
                gallery_images,  # image_gallery
                page_text,  # page_info
                0,  # current_page (reset to first page)
                cp_table,  # checkpoint_table
                metrics_figure,  # metrics_plot
                get_training_notes(task),  # training_notes
                gr.update(choices=cp_dropdown_choices, value=None),  # selected_checkpoint
                gr.update(choices=cp_dropdown_choices, value=[]),  # prediction_checkpoints
                gr.update(visible=is_test_split),  # prediction_controls
                gr.update(visible=is_test_split),  # metrics_tab
                gr.update(visible=is_test_split),  # predictions_tab
                gr.update(visible=is_test_split),  # eval_json_tab
                gr.update(choices=exercise_choices, value="All"),  # explorer_exercise_filter
                create_variant_summary_table(task, variant),  # variant_summary_md
                gr.update(visible=(task == 'task4')),  # mcqa_analysis_tab
                gr.update(choices=cp_dropdown_choices, value=None),  # mcqa_analysis_dropdown
                gr.update(visible=(task == 'task4')),  # eval_browser_tab
                gr.update(choices=cp_dropdown_choices, value=None),  # eval_browser_checkpoint
                gr.update(visible=(task in ['task2', 'task3a', 'task3b', 'task3c', 'task3d'])),  # generic_eval_browser_tab
                gr.update(choices=cp_dropdown_choices, value=None),  # generic_eval_cp
                # Detail panel resets
                "*Select an image to view details*",  # image_id_display
                None,  # gt_image
                None,  # combined_pred_image
                "",  # gt_text
                "",  # gt_text_display
                "",  # pred_text_display
                None,  # current_sample_idx
                gr.update(visible=(task in ['task1', 'task1b', 'task1c'])),  # comparison_images
                gr.update(visible=(task not in ['task1', 'task1b', 'task1c'])),  # comparison_texts
            )

        def format_metrics_row(task: str, color_label: str, checkpoint_name: str, metrics: Dict) -> List:
            """Format metrics row based on task type."""
            if task in ['task3a', 'task3b', 'task3c', 'task3d']:
                # Error detection/correction tasks
                return [
                    color_label,
                    checkpoint_name,
                    f"{metrics.get('f1_score', 0):.3f}",
                    f"{metrics.get('precision', 0):.3f}",
                    f"{metrics.get('recall', 0):.3f}",
                    f"{metrics.get('error_detection_correct', 0):.3f}",
                ]
            elif task in ['task1', 'task1b', 'task1c']:
                # Keypoint coordinate prediction tasks
                oks_cw = metrics.get('oks_confidence_weighted', 0)
                oks_cw_str = f"{oks_cw:.3f}" if oks_cw else "-"
                return [
                    color_label,
                    checkpoint_name,
                    f"{metrics.get('oks_score', 0):.3f}",
                    oks_cw_str,
                    f"{metrics.get('pck_0.5', metrics.get('pck_50', 0)):.3f}",
                    f"{metrics.get('coordinate_mae_total', metrics.get('mae_total', 0)):.2f}",
                    f"{metrics.get('left_right_confusion_rate', metrics.get('left_right_confusion', 0)):.3f}"
                ]
            elif task == 'task2':
                # Keypoint labeling task
                return [
                    color_label,
                    checkpoint_name,
                    f"{metrics.get('per_keypoint_accuracy', 0):.3f}",
                    f"{metrics.get('left_right_confusion_rate', metrics.get('left_right_confusion', 0)):.3f}",
                    f"{metrics.get('exact_match', 0):.3f}"
                ]
            elif task == 'task4':
                # MCQA task
                return [
                    color_label,
                    checkpoint_name,
                    f"{metrics.get('accuracy', 0):.3f}",
                    f"{metrics.get('parse_rate', 0):.3f}"
                ]
            else:
                # Default to task1 format
                oks_cw = metrics.get('oks_confidence_weighted', 0)
                oks_cw_str = f"{oks_cw:.3f}" if oks_cw else "-"
                return [
                    color_label,
                    checkpoint_name,
                    f"{metrics.get('oks_score', 0):.3f}",
                    oks_cw_str,
                    f"{metrics.get('pck_0.5', 0):.3f}",
                    f"{metrics.get('mae_total', 0):.2f}",
                    f"{metrics.get('left_right_confusion_rate', metrics.get('left_right_confusion', 0)):.3f}"
                ]

        def get_metrics_headers(task: str) -> List[str]:
            """Get appropriate metrics headers based on task type."""
            if task in ['task3a', 'task3b', 'task3c', 'task3d']:
                return ["Color", "Model", "F1", "Precision", "Recall", "Error Det."]
            elif task in ['task1', 'task1b', 'task1c']:
                return ["Color", "Model", "OKS", "OKS (CW)", "PCK@0.5", "MAE (px)", "L/R Confusion"]
            elif task == 'task2':
                return ["Color", "Model", "Accuracy", "L/R Confusion", "Exact Match"]
            elif task == 'task4':
                return ["Color", "Model", "Accuracy", "Parse Rate"]
            else:
                return ["Color", "Model", "OKS", "OKS (CW)", "PCK@0.5", "MAE (px)", "L/R Confusion"]

        def on_gallery_select(evt: gr.SelectData, task, variant, split, show_preds, pred_checkpoints, page):
            """Handle gallery image selection - show details with optional predictions."""
            try:
                # evt.index is the gallery index (0-49 for current page)
                # Calculate absolute sample index: page * page_size + gallery_index
                gallery_idx = evt.index if isinstance(evt.index, int) else evt.index[0] if isinstance(evt.index, (list, tuple)) else 0
                sample_idx = page * 50 + gallery_idx
                logging.debug(f"Gallery select: evt.index={evt.index}, gallery_idx={gallery_idx}, page={page}, sample_idx={sample_idx}")

                if task not in DATASET_INDEX or variant not in DATASET_INDEX[task]:
                    show_images = task in ['task1', 'task1b', 'task1c']
                    return ("*Select an image*", None, None, None, None, None, None,
                           gr.update(value=[], headers=get_metrics_headers(task)),
                           {"error": "No dataset selected"}, "**Error**: No dataset selected",
                           "", "", "", "", "", {}, None,
                           "", "", gr.update(visible=show_images), gr.update(visible=not show_images))

                dataset_path = DATASET_INDEX[task][variant]['path']

                # Load sample data
                result = load_sample_with_image(dataset_path, split, sample_idx)
                if not result:
                    error_msg = f"Failed to load sample {sample_idx}"
                    logging.warning(error_msg)
                    show_images = task in ['task1', 'task1b', 'task1c']
                    return (None, None, None, None, None, None, None,
                           gr.update(value=[], headers=get_metrics_headers(task)),
                           {"error": error_msg}, f"**Error**: {error_msg}",
                           "", "", "", "", "", {}, None,
                           "", "", gr.update(visible=show_images), gr.update(visible=not show_images))

                sample, image = result

                # Create ground truth visualization
                gt_annotated = get_sample_visualization(dataset_path, split, sample_idx)
                if gt_annotated is None:
                    logging.warning(f"Failed to create GT visualization for sample {sample_idx}")

                # Prepare metadata
                metadata = {}
                if 'metadata' in sample:
                    try:
                        metadata = json.loads(sample['metadata']) if isinstance(sample['metadata'], str) else sample['metadata']
                    except Exception as meta_error:
                        logging.error(f"Error parsing metadata: {meta_error}")

                # Get image ID
                image_id = metadata.get('image_id') or sample.get('image_id', sample_idx)

                # Image ID header
                image_id_header = f"### 🖼️ Image ID: `{image_id}` | Sample Index: `{sample_idx}`"

                # Metadata display
                metadata_display = {
                    'image_id': image_id,
                    'exercise': metadata.get('exercise', 'N/A'),
                    'dimensions': f"{metadata.get('cropped_width', metadata.get('image_width', 'N/A'))} x {metadata.get('cropped_height', metadata.get('image_height', 'N/A'))}",
                    'visible_keypoints': f"{metadata.get('num_visible_keypoints', 'N/A')} / {metadata.get('num_keypoints', 'N/A')}",
                }

                # Lineage info
                lineage = f"""**Sample ID:** {image_id}
**Dataset**: {task} / {variant}
**Split**: {split}

*Lineage tracking: Track image transformations and augmentations (coming soon)*"""

                # Ground truth annotation using helper (handles multiple formats)
                gt_data = extract_annotation_from_sample(sample) or ''

                # Determine keypoint subset based on task
                if task == 'task1':
                    keypoint_subset = 'coco25'
                elif task == 'task1b':
                    keypoint_subset = 'coco17'
                elif task == 'task1c':
                    keypoint_subset = 'body12'
                else:
                    keypoint_subset = 'coco25'  # Default

                # Handle multi-model predictions if enabled
                pred_images = [None, None, None, None]  # Up to 4 models
                pred_texts = ["", "", "", ""]
                combined_image = None
                metrics_table = []
                full_eval_data = {
                    "image_id": image_id,
                    "sample_index": sample_idx,
                    "ground_truth": "",
                    "models": {}
                }

                if show_preds and pred_checkpoints:
                    # Limit to 4 checkpoints
                    selected_checkpoints = pred_checkpoints[:4]
                    colors = [COLOR_RED, COLOR_BLUE, COLOR_YELLOW, COLOR_PURPLE]

                    # Load base image
                    image_path = extract_image_path_from_sample(sample)
                    if image_path and Path(image_path).exists():
                        image_filename = Path(image_path).name

                        # Load base image for visualization
                        base_img = cv2.imread(str(image_path))

                        # If only one checkpoint selected, try to optimize with prediction lookup
                        if len(selected_checkpoints) == 1 and base_img is not None:
                            pred_data = find_prediction_for_sample(selected_checkpoints[0], image_id, task=task, variant=variant, split=split)
                            if pred_data:
                                # VALIDATION: Verify the prediction data matches what we requested
                                loaded_image_id = pred_data.get('sample_id', '')
                                if loaded_image_id != str(image_id):
                                    logging.error(f"❌ IMAGE ID MISMATCH! Requested: {image_id}, Got: {loaded_image_id}")
                                    logging.error(f"   Checkpoint: {selected_checkpoints[0]}, Task: {task}, Variant: {variant}")
                                    pred_data = None  # Invalidate the prediction data

                            if pred_data:
                                pred_texts[0] = pred_data.get('prediction', '')
                                metrics = pred_data.get('metrics', {})

                                # Validate metrics
                                oks = metrics.get('oks_score')
                                if oks is not None and (oks < 0 or oks > 1):
                                    logging.error(f"❌ Invalid OKS {oks} for {selected_checkpoints[0]}")

                                # Add to combined eval data
                                full_eval_data["ground_truth"] = pred_data.get('ground_truth', '')
                                full_eval_data["models"][selected_checkpoints[0]] = {
                                    "prediction": pred_data.get('prediction', ''),
                                    "metrics": metrics,
                                    "metadata": pred_data.get('metadata', {}),
                                    "image_path": pred_data.get('image_path', '')
                                }

                                metrics_table.append(format_metrics_row(
                                    task,
                                    get_color_label(colors[0]),
                                    selected_checkpoints[0],
                                    metrics
                                ))

                                # Create predictions-only image (right side)
                                # Only visualize keypoints for task1/task1b/task1c
                                if task in ['task1', 'task1b', 'task1c']:
                                    try:
                                        combined_image = visualize_keypoints_on_image(
                                            base_img.copy(), pred_texts[0],
                                            keypoint_subset=keypoint_subset, color=colors[0]
                                        )
                                    except Exception as e:
                                        logging.error(f"Error creating predictions image: {e}")
                                        combined_image = base_img.copy()
                                else:
                                    # For non-keypoint tasks, just show the base image
                                    combined_image = base_img.copy()

                                # Individual view (same as combined for single model)
                                pred_images[0] = combined_image.copy() if combined_image is not None else None

                        # If multiple checkpoints or no pre-gen available, generate on-the-fly
                        if combined_image is None:
                            base_img = cv2.imread(str(image_path))
                            if base_img is not None:
                                # Create predictions-only image (no GT on right side)
                                combined_image = base_img.copy()

                                # Process each checkpoint
                                for idx, checkpoint in enumerate(selected_checkpoints):
                                    logging.info(f"📊 Looking for predictions: checkpoint={checkpoint}, image_id={image_id[:50]}...")
                                    pred_data = find_prediction_for_sample(checkpoint, image_id, task=task, variant=variant, split=split)

                                    if pred_data:
                                        # VALIDATION: Verify the prediction data matches what we requested
                                        loaded_image_id = pred_data.get('sample_id', '')
                                        if loaded_image_id != str(image_id):
                                            logging.error(f"❌ IMAGE ID MISMATCH! Requested: {image_id}, Got: {loaded_image_id}")
                                            logging.error(f"   Checkpoint: {checkpoint}, Task: {task}, Variant: {variant}")
                                            continue  # Skip this prediction - it's wrong!

                                        pred_text = pred_data.get('prediction', '')
                                        metrics = pred_data.get('metrics', {})
                                        pred_texts[idx] = pred_text

                                        # Validate metrics
                                        oks = metrics.get('oks_score')
                                        if oks is not None:
                                            if oks < 0 or oks > 1:
                                                logging.error(f"❌ Invalid OKS {oks} for {checkpoint}")
                                            else:
                                                logging.info(f"✓ Validated: {checkpoint} → OKS={oks:.3f} for image {image_id[:30]}...")

                                        logging.info(f"✓ Loaded predictions from: {Path(pred_data.get('result_file', 'unknown')).name}")

                                        # Add to combined eval data (all models)
                                        if idx == 0:
                                            full_eval_data["ground_truth"] = pred_data.get('ground_truth', '')

                                        full_eval_data["models"][checkpoint] = {
                                            "color": get_color_label(colors[idx]),
                                            "prediction": pred_data.get('prediction', ''),
                                            "metrics": metrics,
                                            "metadata": pred_data.get('metadata', {}),
                                            "image_path": pred_data.get('image_path', '')
                                        }

                                        # Add to metrics table
                                        metrics_table.append(format_metrics_row(
                                            task,
                                            get_color_label(colors[idx]),
                                            checkpoint,
                                            metrics
                                        ))

                                        # Create individual prediction image
                                        # Only visualize keypoints for task1/task1b/task1c
                                        if task in ['task1', 'task1b', 'task1c']:
                                            try:
                                                pred_img = base_img.copy()
                                                pred_img = visualize_keypoints_on_image(
                                                    pred_img, pred_text, keypoint_subset=keypoint_subset, color=colors[idx]
                                                )
                                                pred_images[idx] = pred_img
                                            except Exception as e:
                                                logging.error(f"Error visualizing {checkpoint}: {e}")
                                                pred_images[idx] = base_img.copy()

                                            # Add predictions to combined image (NO GT)
                                            try:
                                                combined_image = visualize_keypoints_on_image(
                                                    combined_image, pred_text, keypoint_subset=keypoint_subset, color=colors[idx]
                                                )
                                            except Exception as e:
                                                logging.error(f"Error adding {checkpoint} to combined: {e}")
                                        else:
                                            # For non-keypoint tasks, just show the base image
                                            pred_images[idx] = base_img.copy()
                                    else:
                                        logging.warning(f"No predictions found for checkpoint={checkpoint}, image_id={image_id}")

                # Convert all images from BGR to RGB for Gradio display
                if gt_annotated is not None:
                    gt_annotated = cv2.cvtColor(gt_annotated, cv2.COLOR_BGR2RGB)
                if combined_image is not None:
                    combined_image = cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)
                for i in range(len(pred_images)):
                    if pred_images[i] is not None:
                        pred_images[i] = cv2.cvtColor(pred_images[i], cv2.COLOR_BGR2RGB)

                # Update metrics table with task-appropriate headers
                metrics_df_update = gr.update(
                    value=metrics_table,
                    headers=get_metrics_headers(task)
                )

                # Determine display mode based on task type and split
                # Only show images for keypoint tasks when test split is selected
                show_images = task in ['task1', 'task1b', 'task1c']
                show_text = not show_images

                # Format text displays for non-keypoint tasks
                gt_text_for_display = gt_data if show_text else ""
                # Combine all predictions into one text for the right panel
                pred_text_combined = ""
                if show_text:
                    # Use pred_texts array which was populated during prediction loading
                    color_names = ["🔴 Red", "🔵 Blue", "🟡 Yellow", "🟣 Purple"]
                    checkpoint_names = pred_checkpoints[:4] if pred_checkpoints else []

                    for i in range(4):
                        if i < len(checkpoint_names) and pred_texts[i]:
                            pred_text_combined += f"### {color_names[i]}: {checkpoint_names[i]}\n\n{pred_texts[i]}\n\n{'='*60}\n\n"

                    # If no predictions loaded, show message
                    if not pred_text_combined and show_preds:
                        pred_text_combined = "No predictions loaded. Make sure 'Show Model Predictions' is checked and checkpoints are selected."
                    elif not show_preds:
                        pred_text_combined = "Enable 'Show Model Predictions' and select checkpoints to view predictions."

                return (image_id_header, gt_annotated, combined_image,
                       pred_images[0], pred_images[1], pred_images[2], pred_images[3],
                       metrics_df_update, metadata_display, lineage, gt_data,
                       pred_texts[0], pred_texts[1], pred_texts[2], pred_texts[3], full_eval_data, sample_idx,
                       gt_text_for_display, pred_text_combined,
                       gr.update(visible=show_images), gr.update(visible=show_text))

            except Exception as e:
                error_msg = f"Error loading sample: {str(e)}"
                logging.error(error_msg)
                gr.Warning(f"Failed to load sample: {e}")
                show_images = task in ['task1', 'task1b', 'task1c']
                return (None, None, None, None, None, None, None,
                       gr.update(value=[], headers=get_metrics_headers(task)),
                       {"error": error_msg}, f"**Error**: {error_msg}",
                       "", "", "", "", "", {}, None,
                       "", "", gr.update(visible=show_images), gr.update(visible=not show_images))

        def on_refresh_image(task, variant, split, show_preds, pred_checkpoints, sample_idx):
            """Refresh/reload the current image visualizations."""
            if sample_idx is None:
                show_images = task in ['task1', 'task1b', 'task1c']
                return ("*Select an image first*", None, None, None, None, None, None,
                       gr.update(value=[], headers=get_metrics_headers(task)),
                       {}, "*Select an image*", "", "", "", "", "", {}, None,
                       "", "", gr.update(visible=show_images), gr.update(visible=not show_images))

            # Calculate page and gallery index from absolute sample_idx
            page_size = 50
            page = sample_idx // page_size
            gallery_idx = sample_idx % page_size

            # Create a mock SelectData event with the gallery-relative index
            class MockEvent:
                def __init__(self, idx):
                    self.index = idx

            mock_evt = MockEvent(gallery_idx)

            # Call on_gallery_select with the current sample
            return on_gallery_select(mock_evt, task, variant, split, show_preds, pred_checkpoints, page)

        # --- Explorer navigation helpers ---

        def _explorer_gallery_page(task, variant, split, page, exercise_filter):
            """Load gallery page respecting exercise filter. Returns (page, gallery, page_text)."""
            if task not in DATASET_INDEX or variant not in DATASET_INDEX.get(task, {}):
                return page, [], "No dataset found"
            dataset_path = DATASET_INDEX[task][variant]['path']

            if exercise_filter and exercise_filter != "All":
                indices = get_filtered_explorer_indices(dataset_path, split, exercise_filter)
                total = len(indices)
                num_pages = max(1, (total + 49) // 50)
                page = max(0, min(page, num_pages - 1))
                gallery = create_filtered_image_gallery(dataset_path, split, indices, page=page, page_size=50)
                page_text = f"Page {page + 1} of {num_pages} ({total:,} filtered samples)"
            else:
                num_samples = DATASET_INDEX[task][variant].get(f'{split}_samples', 0)
                num_pages = max(1, (num_samples + 49) // 50)
                page = max(0, min(page, num_pages - 1))
                gallery = create_image_gallery(dataset_path, split, page=page, page_size=50)
                page_text = f"Page {page + 1} of {num_pages} ({num_samples:,} total samples)"
            return page, gallery, page_text

        def on_explorer_random(task, variant, split, exercise_filter):
            """Navigate to a random page in the gallery."""
            if task not in DATASET_INDEX or variant not in DATASET_INDEX.get(task, {}):
                return 0, [], "No dataset found"
            dataset_path = DATASET_INDEX[task][variant]['path']

            if exercise_filter and exercise_filter != "All":
                indices = get_filtered_explorer_indices(dataset_path, split, exercise_filter)
                total = len(indices)
            else:
                total = DATASET_INDEX[task][variant].get(f'{split}_samples', 0)

            if total <= 0:
                return 0, [], "No samples found"
            num_pages = (total + 49) // 50
            random_page = random.randint(0, num_pages - 1)
            return _explorer_gallery_page(task, variant, split, random_page, exercise_filter)

        def on_explorer_search(task, variant, split, search_query):
            """Search for an image by image_id (substring match)."""
            if not search_query or not search_query.strip():
                return 0, [], "Enter an image ID to search"
            search_query = search_query.strip()

            if task not in DATASET_INDEX or variant not in DATASET_INDEX.get(task, {}):
                return 0, [], "No dataset found"
            dataset_path = DATASET_INDEX[task][variant]['path']
            jsonl_path = resolve_jsonl_path(dataset_path, split)
            if not jsonl_path:
                return 0, [], "No dataset file found"

            lines = _load_jsonl_lines(str(jsonl_path))
            for i, line in enumerate(lines):
                try:
                    meta = json.loads(line).get('metadata', {})
                    if isinstance(meta, str):
                        meta = json.loads(meta)
                    image_id = meta.get('image_id', '')
                    if search_query in image_id:
                        target_page = i // 50
                        gallery = create_image_gallery(dataset_path, split, page=target_page, page_size=50)
                        num_samples = len(lines)
                        num_pages = (num_samples + 49) // 50
                        page_text = f"Page {target_page + 1} of {num_pages} ({num_samples:,} total samples) — Found: {image_id}"
                        return target_page, gallery, page_text
                except Exception:
                    continue
            return 0, [], f"No match found for '{search_query}'"

        def on_explorer_refresh(task, variant, split, page, exercise_filter):
            """Refresh dataset caches and reload current gallery view."""
            global DATASET_INDEX
            clear_all_caches()
            DATASET_INDEX = build_dataset_index()
            logging.info("Explorer refresh: caches cleared, dataset index rebuilt")

            # Reload gallery
            _, gallery, page_text = _explorer_gallery_page(task, variant, split, page, exercise_filter)

            # Repopulate exercise filter
            if task in DATASET_INDEX and variant in DATASET_INDEX.get(task, {}):
                dataset_path = DATASET_INDEX[task][variant]['path']
                prefixes = get_exercise_prefixes(dataset_path, split)
                exercise_choices = ["All"] + list(prefixes)
            else:
                exercise_choices = ["All"]

            return gallery, page_text, gr.update(choices=exercise_choices, value="All")

        def on_explorer_exercise_change(task, variant, split, exercise_filter):
            """Filter gallery by exercise type."""
            return _explorer_gallery_page(task, variant, split, 0, exercise_filter)

        def on_page_prev(task, variant, split, page, exercise_filter):
            """Handle previous page button."""
            if page <= 0:
                return page, [], "Page 1 of 1"
            return _explorer_gallery_page(task, variant, split, page - 1, exercise_filter)

        def on_page_next(task, variant, split, page, exercise_filter):
            """Handle next page button."""
            if task not in DATASET_INDEX or variant not in DATASET_INDEX.get(task, {}):
                return page, [], "No dataset found"
            dataset_path = DATASET_INDEX[task][variant]['path']

            if exercise_filter and exercise_filter != "All":
                indices = get_filtered_explorer_indices(dataset_path, split, exercise_filter)
                total = len(indices)
            else:
                total = DATASET_INDEX[task][variant].get(f'{split}_samples', 0)
            num_pages = max(1, (total + 49) // 50)

            if page >= num_pages - 1:
                return page, [], f"Page {page + 1} of {num_pages}"
            return _explorer_gallery_page(task, variant, split, page + 1, exercise_filter)

        def on_comparison_mode_change(mode):
            """Toggle between quick view and custom comparison."""
            if mode == "Quick View (Pre-generated Report)":
                return gr.Group(visible=True), gr.Group(visible=False)
            else:
                return gr.Group(visible=False), gr.Group(visible=True)

        def load_quick_view_report(task, variant):
            """Load pre-generated comparison report."""
            report_file = find_comparison_report(task, variant)
            if report_file:
                return load_comparison_report(report_file)
            else:
                return f"No pre-generated comparison report found for {task}/{variant}"

        def on_checkpoint_select(checkpoint_name):
            """Load and display checkpoint configuration and training info."""
            config_text, training_info, files_exist = get_checkpoint_details(checkpoint_name)
            # Show accordion only if training files exist
            return config_text, training_info, gr.update(visible=files_exist)

        def on_benchmark_change(benchmark_type, task_filter="All Tasks", variant_filter="All Variants"):
            """Update benchmark results when benchmark type or filters change."""
            try:
                if not BENCHMARKS_INDEX:
                    return pd.DataFrame(), empty_figure("No benchmark data available"), gr.update(choices=[], value=None), "No benchmark data available"

                # Get all models for the selected benchmark
                all_models = BENCHMARKS_INDEX.get(benchmark_type.lower(), {}).get('models', {})

                # Apply filters
                filtered_models = {}
                for model_name, model_data in all_models.items():
                    # Task filter
                    if task_filter and task_filter != "All Tasks":
                        if task_filter not in model_name:
                            continue

                    # Variant filter
                    if variant_filter and variant_filter != "All Variants":
                        if variant_filter not in model_name:
                            continue

                    filtered_models[model_name] = model_data

                if benchmark_type == "IFEval":
                    table = create_ifeval_table(filtered_models)
                    chart = create_ifeval_chart(filtered_models)
                    model_choices = list(filtered_models.keys())
                    baseline_prompt = BENCHMARKS_INDEX.get('ifeval', {}).get('baseline', {}).get('prompt_strict', 0)

                    total_models = len(all_models)
                    filtered_count = len(filtered_models)
                    filter_text = f" (filtered: {filtered_count}/{total_models})" if filtered_count < total_models else ""

                    summary = f"""**IFEval Results Summary**

✅ Models Evaluated: {filtered_count}{filter_text}
📊 Baseline (Qwen3-VL-4B-Instruct): {baseline_prompt:.2f}% Prompt Strict

Select a model below to view detailed metrics and full report."""
                else:  # SIBench
                    table = create_sibench_table(filtered_models)
                    chart = create_sibench_chart(filtered_models)
                    summary_text = create_sibench_summary()
                    model_choices = list(filtered_models.keys())

                    total_models = len(all_models)
                    filtered_count = len(filtered_models)
                    filter_text = f" (filtered: {filtered_count}/{total_models})" if filtered_count < total_models else ""

                    summary = f"""**SIBench Results Summary**

{summary_text}

Models shown: {filtered_count}{filter_text}

Select a model below to view detailed per-task breakdown."""

                return table, chart, gr.update(choices=model_choices, value=None), summary

            except Exception as e:
                logging.error(f"Error in on_benchmark_change: {e}")
                gr.Warning(f"Failed to load benchmark data: {e}")
                return pd.DataFrame(), empty_figure(f"Error: {str(e)}"), gr.update(choices=[], value=None), f"Error: {str(e)}"

        def on_model_select(model_name, benchmark_type):
            """Show detailed metrics for selected model."""
            try:
                if not model_name or not BENCHMARKS_INDEX:
                    return "*Select a model to view detailed metrics*", ""

                if benchmark_type == "IFEval":
                    model_data = BENCHMARKS_INDEX.get('ifeval', {}).get('models', {}).get(model_name, {})
                    if not model_data:
                        return f"*Model '{model_name}' not found in IFEval results*", ""

                    baseline = BENCHMARKS_INDEX.get('ifeval', {}).get('baseline', {})
                    baseline_prompt = baseline.get('prompt_strict', 0)
                    baseline_instr = baseline.get('instr_strict', 0)

                    # Simplified status logic
                    delta_prompt = model_data.get('delta_prompt', 0)
                    if abs(delta_prompt) <= 5:
                        status = '✅ Maintained'
                    elif delta_prompt >= -15:
                        status = '⚠️ Degraded'
                    else:
                        status = '🔴 Severely Degraded'

                    details = f"""### {model_name}

**IFEval Metrics:**
- **Prompt-Level Strict Accuracy**: {model_data.get('prompt_strict', 0):.2f}% (Δ {delta_prompt:+.2f}%)
- **Instruction-Level Strict Accuracy**: {model_data.get('instr_strict', 0):.2f}% (�� {model_data.get('delta_instr', 0):+.2f}%)

**Baseline Comparison:**
- Baseline Prompt Strict: {baseline_prompt:.2f}%
- Baseline Instruction Strict: {baseline_instr:.2f}%

**Status**: {status}
"""

                    report_path = model_data.get('report_path', '')
                    if report_path:
                        report_link = f"\n\n📄 **Full Report**: [{Path(report_path).name}]({report_path})"
                    else:
                        report_link = ""

                else:  # SIBench
                    model_data = BENCHMARKS_INDEX.get('sibench', {}).get('models', {}).get(model_name, {})
                    if not model_data:
                        return f"*Model '{model_name}' not found in SIBench results*", ""

                    baseline = BENCHMARKS_INDEX.get('sibench', {}).get('baseline', {})
                    baseline_overall = baseline.get('overall', 0)
                    model_overall = model_data.get('overall', 0)
                    delta = model_overall - baseline_overall

                    # Simplified status logic
                    if abs(delta) <= 5:
                        status = '✅ Maintained'
                    elif delta >= -15:
                        status = '⚠️ Degraded'
                    else:
                        status = '🔴 Severely Degraded'

                    details = f"""### {model_name}

**SIBench Overall Accuracy**: {model_overall:.2f}%
**Baseline Overall**: {baseline_overall:.2f}%
**Delta**: {delta:+.2f}%
**Status**: {status}

**Per-Task Breakdown:**
"""

                    per_task = model_data.get('per_task', {})
                    if per_task:
                        for task, score in sorted(per_task.items()):
                            details += f"\n- **{task}**: {score:.1f}%"
                    else:
                        details += "\n*No per-task data available*"

                    result_path = model_data.get('result_path', '')
                    if result_path:
                        report_link = f"\n\n📂 **Results Directory**: [{Path(result_path).name}]({result_path})"
                    else:
                        report_link = ""

                return details, report_link

            except Exception as e:
                logging.error(f"Error in on_model_select: {e}")
                gr.Warning(f"Failed to load model details: {e}")
                return f"Error loading model details: {str(e)}", ""

        # === Tab 5: MCQA Browser ===
        # Wire up event handlers
        task_dropdown.change(
            fn=on_task_change,
            inputs=[task_dropdown],
            outputs=[variant_dropdown, metric_selector]
        )

        # Auto-select task4 when MCQA tab is opened.
        # Use mcqa_tab.select() on the specific Tab (not Tabs) to avoid
        # Gradio 6.x SelectData KeyError: 'value' on Tabs.select().
        def _select_task4_for_mcqa():
            variants = get_active_variants('task4', list(DATASET_INDEX.get('task4', {}).keys()))
            metric_choices = get_metric_choices('task4')
            return (
                gr.Dropdown(value='task4'),
                gr.Dropdown(choices=variants, value=variants[0] if variants else None),
                gr.Dropdown(choices=metric_choices, value=metric_choices[0] if metric_choices else "Accuracy"),
            )

        mcqa_tab.select(
            fn=_select_task4_for_mcqa,
            inputs=[],
            outputs=[task_dropdown, variant_dropdown, metric_selector]
        )

        _variant_change_outputs = [
            stats_display, prompt_display, checkpoint_list, selected_checkpoints, image_gallery, page_info, current_page,
            checkpoint_table, metrics_plot, training_notes, selected_checkpoint, prediction_checkpoints,
            prediction_controls, metrics_tab, predictions_tab, eval_json_tab, explorer_exercise_filter, variant_summary_md,
            mcqa_analysis_tab, mcqa_analysis_dropdown, eval_browser_tab, eval_browser_checkpoint,
            generic_eval_browser_tab, generic_eval_cp,
            # Detail panel resets (clear stale data when switching task/variant)
            image_id_display, gt_image, combined_pred_image,
            gt_text, gt_text_display, pred_text_display,
            current_sample_idx, comparison_images, comparison_texts]

        variant_dropdown.change(
            fn=on_variant_change,
            inputs=[task_dropdown, variant_dropdown, split_radio, metric_selector],
            outputs=_variant_change_outputs
        )

        split_radio.change(
            fn=on_variant_change,
            inputs=[task_dropdown, variant_dropdown, split_radio, metric_selector],
            outputs=_variant_change_outputs
        )

        # Metric selector change - update plot only
        def on_metric_change(task, metric):
            """Update metrics plot when metric selection changes."""
            return create_metrics_plot(task, variant=None, all_variants=True, metric=metric)

        metric_selector.change(
            fn=on_metric_change,
            inputs=[task_dropdown, metric_selector],
            outputs=[metrics_plot]
        )

        # Refresh Training Monitor — reloads CSV, clears caches, re-renders plot, table, and notes
        def on_refresh_training(task, metric):
            global EXPERIMENT_INDEX
            EXPERIMENT_INDEX = load_experiments_csv()
            clear_all_caches()
            cp_table = create_checkpoint_table(task, variant=None, all_variants=True)
            plot = create_metrics_plot(task, variant=None, all_variants=True, metric=metric)
            notes = get_training_notes(task)
            return cp_table, plot, notes

        refresh_training_btn.click(
            fn=on_refresh_training,
            inputs=[task_dropdown, metric_selector],
            outputs=[checkpoint_table, metrics_plot, training_notes]
        )

        # Gallery selection
        image_gallery.select(
            fn=on_gallery_select,
            inputs=[task_dropdown, variant_dropdown, split_radio, show_predictions, prediction_checkpoints, current_page],
            outputs=[image_id_display, gt_image, combined_pred_image, pred_image1, pred_image2, pred_image3, pred_image4,
                    metrics_comparison_table, metadata_json, lineage_md, gt_text,
                    prediction_text1, prediction_text2, prediction_text3, prediction_text4, full_eval_json, current_sample_idx,
                    gt_text_display, pred_text_display, comparison_images, comparison_texts]
        )

        # Refresh button
        refresh_btn.click(
            fn=on_refresh_image,
            inputs=[task_dropdown, variant_dropdown, split_radio, show_predictions, prediction_checkpoints, current_sample_idx],
            outputs=[image_id_display, gt_image, combined_pred_image, pred_image1, pred_image2, pred_image3, pred_image4,
                    metrics_comparison_table, metadata_json, lineage_md, gt_text,
                    prediction_text1, prediction_text2, prediction_text3, prediction_text4, full_eval_json, current_sample_idx,
                    gt_text_display, pred_text_display, comparison_images, comparison_texts]
        )

        # Pagination controls
        prev_btn.click(
            fn=on_page_prev,
            inputs=[task_dropdown, variant_dropdown, split_radio, current_page, explorer_exercise_filter],
            outputs=[current_page, image_gallery, page_info]
        )

        next_btn.click(
            fn=on_page_next,
            inputs=[task_dropdown, variant_dropdown, split_radio, current_page, explorer_exercise_filter],
            outputs=[current_page, image_gallery, page_info]
        )

        # Explorer: random, search, refresh, exercise filter
        explorer_random_btn.click(
            fn=on_explorer_random,
            inputs=[task_dropdown, variant_dropdown, split_radio, explorer_exercise_filter],
            outputs=[current_page, image_gallery, page_info]
        )

        explorer_search_btn.click(
            fn=on_explorer_search,
            inputs=[task_dropdown, variant_dropdown, split_radio, explorer_search_input],
            outputs=[current_page, image_gallery, page_info]
        )

        # Also trigger search on Enter key
        explorer_search_input.submit(
            fn=on_explorer_search,
            inputs=[task_dropdown, variant_dropdown, split_radio, explorer_search_input],
            outputs=[current_page, image_gallery, page_info]
        )

        explorer_refresh_btn.click(
            fn=on_explorer_refresh,
            inputs=[task_dropdown, variant_dropdown, split_radio, current_page, explorer_exercise_filter],
            outputs=[image_gallery, page_info, explorer_exercise_filter]
        )

        explorer_exercise_filter.change(
            fn=on_explorer_exercise_change,
            inputs=[task_dropdown, variant_dropdown, split_radio, explorer_exercise_filter],
            outputs=[current_page, image_gallery, page_info]
        )

        comparison_mode.change(
            fn=on_comparison_mode_change,
            inputs=[comparison_mode],
            outputs=[quick_view, custom_view]
        )

        # Load comparison report when task/variant changes
        variant_dropdown.change(
            fn=load_quick_view_report,
            inputs=[task_dropdown, variant_dropdown],
            outputs=[comparison_report]
        )

        # Checkpoint details selection
        selected_checkpoint.change(
            fn=on_checkpoint_select,
            inputs=[selected_checkpoint],
            outputs=[config_display, training_info_display, checkpoint_details_accordion]
        )

        # Custom comparison generation
        compare_btn.click(
            fn=generate_custom_comparison,
            inputs=[selected_checkpoints, task_dropdown, variant_dropdown],
            outputs=[comparison_table, radar_chart, improvement_summary]
        )

        # MCQA Analysis dropdown handler
        def on_mcqa_analysis_change(checkpoint, task, variant, split):
            if not checkpoint or task != 'task4':
                ef = empty_figure("Select a Task 4 checkpoint")
                return ef, ef, ef, "*Select a checkpoint above to view MCQA analysis*"
            return create_mcqa_analysis_plots(checkpoint, task, variant, split)

        mcqa_analysis_dropdown.change(
            fn=on_mcqa_analysis_change,
            inputs=[mcqa_analysis_dropdown, task_dropdown, variant_dropdown, split_radio],
            outputs=[mcqa_confusion_plot, mcqa_distribution_plot, mcqa_tier_plot, mcqa_analysis_md]
        )

        # --- Evaluation Results Browser handlers ---
        _eval_browser_display_outputs = [
            eval_browser_image, eval_browser_verdict, eval_browser_detail, eval_browser_counter]
        _eval_browser_state_outputs = [eval_browser_rf, eval_browser_indices, eval_browser_tiers, eval_browser_idx]

        def on_eval_browser_load(checkpoint, task, variant, split, filter_mode, tier_filter):
            """Load filtered indices and show first result."""
            rf, indices, tiers = _filter_eval_indices(checkpoint, task, variant, split, filter_mode, tier_filter)
            if not indices:
                return None, [], [], 0, None, "", "*No results found for this selection*", f"**0** results"
            img, verdict, detail = _format_eval_at_index(rf, indices[0], tiers)
            counter = f"**1** of **{len(indices)}**"
            return rf, indices, tiers, 0, img, verdict, detail, counter

        def on_eval_browser_nav(direction, idx, rf, indices, tiers):
            """Navigate prev/next through filtered results."""
            if not indices:
                return 0, None, "", "*No results*", "*Select a checkpoint*"
            new_idx = max(0, min(len(indices) - 1, idx + direction))
            img, verdict, detail = _format_eval_at_index(rf, indices[new_idx], tiers)
            counter = f"**{new_idx + 1}** of **{len(indices)}**"
            return new_idx, img, verdict, detail, counter

        _load_inputs = [eval_browser_checkpoint, task_dropdown, variant_dropdown, split_radio,
                        eval_browser_filter, eval_browser_tier]
        _load_outputs = _eval_browser_state_outputs + _eval_browser_display_outputs

        for trigger in [eval_browser_checkpoint, eval_browser_filter, eval_browser_tier]:
            trigger.change(fn=on_eval_browser_load, inputs=_load_inputs, outputs=_load_outputs)

        _nav_inputs = [eval_browser_idx, eval_browser_rf, eval_browser_indices, eval_browser_tiers]
        eval_browser_prev.click(
            fn=lambda idx, rf, indices, tiers: on_eval_browser_nav(-1, idx, rf, indices, tiers),
            inputs=_nav_inputs,
            outputs=[eval_browser_idx] + _eval_browser_display_outputs)
        eval_browser_next.click(
            fn=lambda idx, rf, indices, tiers: on_eval_browser_nav(1, idx, rf, indices, tiers),
            inputs=_nav_inputs,
            outputs=[eval_browser_idx] + _eval_browser_display_outputs)

        # --- Generic Evaluation Results Browser handlers (task2/task3) ---
        _generic_eval_display = [generic_eval_image, generic_eval_verdict, generic_eval_pred, generic_eval_gt, generic_eval_counter]
        _generic_eval_states = [generic_eval_rf, generic_eval_indices, generic_eval_idx]

        def on_generic_eval_load(checkpoint, task, variant, split, filter_mode):
            rf, indices = _filter_eval_indices_generic(checkpoint, task, variant, split, filter_mode)
            if not indices:
                return None, [], 0, None, "", "", "", f"**0** results"
            img, verdict, pred_md, gt_md = _format_eval_generic(rf, indices[0], task)
            return rf, indices, 0, img, verdict, pred_md, gt_md, f"**1** of **{len(indices)}**"

        def on_generic_eval_nav(direction, idx, rf, indices, task):
            if not indices:
                return 0, None, "", "", "", "*Select a checkpoint*"
            new_idx = max(0, min(len(indices) - 1, idx + direction))
            img, verdict, pred_md, gt_md = _format_eval_generic(rf, indices[new_idx], task)
            return new_idx, img, verdict, pred_md, gt_md, f"**{new_idx + 1}** of **{len(indices)}**"

        _gen_load_inputs = [generic_eval_cp, task_dropdown, variant_dropdown, split_radio, generic_eval_filter]
        _gen_load_outputs = _generic_eval_states + _generic_eval_display

        for trigger in [generic_eval_cp, generic_eval_filter]:
            trigger.change(fn=on_generic_eval_load, inputs=_gen_load_inputs, outputs=_gen_load_outputs)

        _gen_nav_inputs = [generic_eval_idx, generic_eval_rf, generic_eval_indices, task_dropdown]
        generic_eval_prev.click(
            fn=lambda idx, rf, indices, task: on_generic_eval_nav(-1, idx, rf, indices, task),
            inputs=_gen_nav_inputs,
            outputs=[generic_eval_idx] + _generic_eval_display)
        generic_eval_next.click(
            fn=lambda idx, rf, indices, task: on_generic_eval_nav(1, idx, rf, indices, task),
            inputs=_gen_nav_inputs,
            outputs=[generic_eval_idx] + _generic_eval_display)

        # Benchmarks tab event handlers (always wire, handlers deal with missing data)
        benchmark_selector.change(
            fn=on_benchmark_change,
            inputs=[benchmark_selector, benchmark_task_filter, benchmark_variant_filter],
            outputs=[benchmark_table, benchmark_chart, model_selector, benchmark_summary]
        )

        benchmark_task_filter.change(
            fn=on_benchmark_change,
            inputs=[benchmark_selector, benchmark_task_filter, benchmark_variant_filter],
            outputs=[benchmark_table, benchmark_chart, model_selector, benchmark_summary]
        )

        benchmark_variant_filter.change(
            fn=on_benchmark_change,
            inputs=[benchmark_selector, benchmark_task_filter, benchmark_variant_filter],
            outputs=[benchmark_table, benchmark_chart, model_selector, benchmark_summary]
        )

        model_selector.change(
            fn=on_model_select,
            inputs=[model_selector, benchmark_selector],
            outputs=[model_details, report_link]
        )

        # Load IFEval results on initial render (if data exists, will populate; if not, will show empty state)
        app.load(
            fn=lambda: on_benchmark_change("IFEval", "All Tasks", "All Variants"),
            inputs=[],
            outputs=[benchmark_table, benchmark_chart, model_selector, benchmark_summary]
        )

        # ========== REASONING TRACES EVENT HANDLERS ==========

        reas_outputs = [
            reas_header, reas_image, reas_teacher_sys, reas_teacher_prompt,
            reas_reasoning, reas_train_sys, reas_train_user, reas_train_assistant,
            reas_metadata, reas_current_idx, reas_current_image_id,
            reas_audit_status,
        ]

        def render_reasoning(task, split, idx):
            """Core renderer — returns all UI component values for one sample."""
            defaults = (
                "*No data available*", None, "", "", "*No sample loaded*",
                "", "", "", {}, 0, "", "*No sample loaded*",
            )
            if not task:
                return defaults
            info = REASONING_INDEX.get(task, {}).get(split)
            if not info:
                return defaults
            count = info["count"]
            idx = max(0, min(idx, count - 1))
            raw = load_reasoning_sample(task, split, idx)
            if not raw:
                return defaults

            s = parse_reasoning_sample(raw, task, split)
            badge = "⚠️ HALLUCINATED" if s["is_hallucinated"] else "✅ Clean"
            orient_badge = ""
            if s.get("orientation_mismatch"):
                conf = s.get("orient_confidence", "unknown")
                conf_tag = f", confidence: {conf}" if conf != "unknown" else ""
                orient_badge = f" | 🧭 ORIENTATION MISMATCH (GT: {s['gt_orientation']}{conf_tag})"
                # Add signal breakdown if v2 data available
                sigs = s.get("orient_signals", {})
                if sigs:
                    sig_parts = []
                    if "box_ratio" in sigs:
                        v = sigs["box_ratio"].get("vote", "?")
                        val = sigs["box_ratio"].get("value")
                        sig_parts.append(f"box={val:.2f}→{v[:3]}" if val is not None else f"box→{v[:3]}")
                    if "shoulder_dx" in sigs:
                        v = sigs["shoulder_dx"].get("vote", "?")
                        val = sigs["shoulder_dx"].get("value")
                        sig_parts.append(f"shldr={val*100:.1f}%→{v[:3]}" if val is not None else f"shldr→{v[:3]}")
                    if "ear_asymmetry" in sigs:
                        v = sigs["ear_asymmetry"].get("vote", "?")
                        sig_parts.append(f"ear→{v[:3]}")
                    if "symmetry" in sigs:
                        v = sigs["symmetry"].get("vote", "?")
                        vl = sigs["symmetry"].get("vis_left", "?")
                        vr = sigs["symmetry"].get("vis_right", "?")
                        sig_parts.append(f"sym({vl}L/{vr}R)→{v[:3]}")
                    if sig_parts:
                        orient_badge += f"\n   Signals: {', '.join(sig_parts)}"
            # Body position mismatch badge
            bp_badge = ""
            if s.get("body_pos_mismatch"):
                bp_badge = f" | 🛏️ BODY POSITION MISMATCH (claimed: {s['claimed_body_pos']}, GT: {s['gt_body_pos']})"
                bp_detail = s.get("body_pos_detail", {})
                if bp_detail:
                    angle = bp_detail.get("torso_angle_deg", "?")
                    face_vis = "visible" if bp_detail.get("face_visible") else "occluded"
                    bp_badge += f"\n   Torso angle: {angle}°, face: {face_vis}"
            # High-angle disagreement badge
            ha_badge = ""
            ha_info = s.get("high_angle_disagreement")
            if ha_info:
                ha_angle = ha_info.get("torso_angle_deg", "?")
                ha_meta = ha_info.get("metadata_position", "?")
                ha_badge = f" | **HIGH-ANGLE** (metadata: {ha_meta}, torso: {ha_angle}°)"
            # Audit exclusion / approval badge
            audit_entry = REASONING_AUDIT.get(task, {}).get(split, {}).get(s["image_id"])
            approved_entry = REASONING_APPROVED.get(task, {}).get(split, {}).get(s["image_id"])
            audit_badge = ""
            if audit_entry:
                audit_badge = f" | **EXCLUDED** ({audit_entry.get('reason', '?')})"
                audit_status_md = f"**EXCLUDED** — reason: `{audit_entry.get('reason', '?')}`\n\nTimestamp: {audit_entry.get('timestamp', '?')}"
            elif approved_entry:
                audit_badge = " | **APPROVED**"
                audit_status_md = f"**APPROVED**\n\nTimestamp: {approved_entry.get('timestamp', '?')}"
            else:
                audit_status_md = "Not reviewed"

            source_tag = " *(v2 test sample)*" if info.get("source") == "tests" else ""
            header = (
                f"### {task} / {split} — Sample {idx + 1} of {count}{source_tag}\n"
                f"**Image**: `{s['image_id']}`\n"
                f"**Words**: {s['reasoning_word_count']} | {badge}{orient_badge}{bp_badge}{ha_badge}{audit_badge}"
            )
            # Render reasoning as markdown (preserve formatting)
            reasoning_md = s["think_text"] if s["think_text"] else "*Empty reasoning*"

            img = s["image_path"] if Path(s["image_path"]).exists() else None

            return (
                header, img, s["teacher_sys"], s["teacher_prompt"],
                reasoning_md, s["train_sys"], s["train_user"], s["train_assistant"],
                s["metadata"], idx, s["image_id"], audit_status_md,
            )

        def _build_filtered_indices(task, split, filter_type):
            """Build list of JSONL line indices matching the filter. Returns None for 'All'."""
            if filter_type == "All":
                return None
            info = REASONING_INDEX.get(task, {}).get(split)
            if not info:
                return []

            # Orientation mismatch filters (with optional confidence level)
            if filter_type.startswith("Orientation Mismatch"):
                orient_map = REASONING_ORIENTATION_MISMATCHES.get(task, {}).get(split, {})
                if not orient_map:
                    return []
                # Extract confidence filter: "(All)", "(High)", "(Medium)", "(Low)"
                conf_filter = None
                if "(High)" in filter_type:
                    conf_filter = "high"
                elif "(Medium)" in filter_type:
                    conf_filter = "medium"
                elif "(Low)" in filter_type:
                    conf_filter = "low"
                # Filter orient_map by confidence if needed
                if conf_filter:
                    filtered_ids = set()
                    for iid, info_dict in orient_map.items():
                        conf = info_dict.get("confidence", "unknown") if isinstance(info_dict, dict) else "unknown"
                        if conf == conf_filter:
                            filtered_ids.add(iid)
                else:
                    filtered_ids = set(orient_map.keys())
                if not filtered_ids:
                    return []
                lines = _load_jsonl_lines(info["path"])
                indices = []
                for idx, line in enumerate(lines):
                    try:
                        d = json.loads(line)
                        iid = d.get("metadata", {}).get("image_id", "")
                        if iid in filtered_ids:
                            indices.append(idx)
                    except Exception:
                        pass
                return indices
            elif filter_type == "Body Position Mismatch":
                bp_map = REASONING_BODY_POS_MISMATCHES.get(task, {}).get(split, {})
                if not bp_map:
                    return []
                bp_ids = set(bp_map.keys())
                lines = _load_jsonl_lines(info["path"])
                indices = []
                for idx, line in enumerate(lines):
                    try:
                        d = json.loads(line)
                        iid = d.get("metadata", {}).get("image_id", "")
                        if iid in bp_ids:
                            indices.append(idx)
                    except Exception:
                        pass
                return indices
            elif filter_type == "High-Angle Disagreement":
                ha_map = REASONING_HIGH_ANGLE.get(task, {}).get(split, {})
                if not ha_map:
                    return []
                ha_ids = set(ha_map.keys())
                lines = _load_jsonl_lines(info["path"])
                indices = []
                for idx, line in enumerate(lines):
                    try:
                        d = json.loads(line)
                        iid = d.get("metadata", {}).get("image_id", "")
                        if iid in ha_ids:
                            indices.append(idx)
                    except Exception:
                        pass
                return indices
            elif filter_type == "Hallucinated":
                hall_ids = REASONING_HALLUCINATIONS.get(task, {}).get(split, set())
                if not hall_ids:
                    return []
                lines = _load_jsonl_lines(info["path"])
                indices = []
                for idx, line in enumerate(lines):
                    try:
                        d = json.loads(line)
                        iid = d.get("metadata", {}).get("image_id", "")
                        if iid in hall_ids:
                            indices.append(idx)
                    except Exception:
                        pass
                return indices
            elif filter_type == "Audit Excluded":
                audit_ids = set(REASONING_AUDIT.get(task, {}).get(split, {}).keys())
                if not audit_ids:
                    return []
                lines = _load_jsonl_lines(info["path"])
                indices = []
                for idx, line in enumerate(lines):
                    try:
                        d = json.loads(line)
                        iid = d.get("metadata", {}).get("image_id", "")
                        if iid in audit_ids:
                            indices.append(idx)
                    except Exception:
                        pass
                return indices
            elif filter_type == "Approved":
                approved_ids = set(REASONING_APPROVED.get(task, {}).get(split, {}).keys())
                if not approved_ids:
                    return []
                lines = _load_jsonl_lines(info["path"])
                indices = []
                for idx, line in enumerate(lines):
                    try:
                        d = json.loads(line)
                        iid = d.get("metadata", {}).get("image_id", "")
                        if iid in approved_ids:
                            indices.append(idx)
                    except Exception:
                        pass
                return indices
            return None

        def _make_stats(rkey, split, filtered_indices):
            """Build stats markdown for sidebar."""
            info = REASONING_INDEX.get(rkey, {}).get(split)
            if not info:
                return "*No data*"
            count = info["count"]
            hall_count = len(REASONING_HALLUCINATIONS.get(rkey, {}).get(split, set()))
            orient_map = REASONING_ORIENTATION_MISMATCHES.get(rkey, {}).get(split, {})
            orient_count = len(orient_map)
            stats = (
                f"**Reasoning key**: {rkey}\n"
                f"**Samples**: {count:,}\n"
                f"**Hallucinated**: {hall_count:,} ({hall_count / max(1, count) * 100:.1f}%)\n"
                f"**Orientation mismatches**: {orient_count:,} ({orient_count / max(1, count) * 100:.1f}%)"
            )
            # Show confidence breakdown if v2 data
            if orient_map:
                conf_counts = {"high": 0, "medium": 0, "low": 0, "unknown": 0}
                for info_dict in orient_map.values():
                    c = info_dict.get("confidence", "unknown") if isinstance(info_dict, dict) else "unknown"
                    conf_counts[c] = conf_counts.get(c, 0) + 1
                if conf_counts.get("high", 0) or conf_counts.get("medium", 0):
                    stats += (
                        f"\n  ↳ High: {conf_counts['high']}, "
                        f"Medium: {conf_counts['medium']}, "
                        f"Low: {conf_counts['low']}"
                    )
            # Body position mismatches
            bp_map = REASONING_BODY_POS_MISMATCHES.get(rkey, {}).get(split, {})
            bp_count = len(bp_map)
            if bp_count:
                stats += f"\n**Body pos mismatches**: {bp_count:,} ({bp_count / max(1, count) * 100:.1f}%)"
            ha_map = REASONING_HIGH_ANGLE.get(rkey, {}).get(split, {})
            ha_count = len(ha_map)
            if ha_count:
                stats += f"\n**High-angle disagreements**: {ha_count:,} (metadata vs KP >40°)"
            audit_count = len(REASONING_AUDIT.get(rkey, {}).get(split, {}))
            if audit_count:
                stats += f"\n**Audit excluded**: {audit_count:,} ({audit_count / max(1, count) * 100:.1f}%)"
            approved_count = len(REASONING_APPROVED.get(rkey, {}).get(split, {}))
            if approved_count:
                stats += f"\n**Approved**: {approved_count:,} ({approved_count / max(1, count) * 100:.1f}%)"
            if filtered_indices is not None:
                stats += f"\n\n**Showing**: {len(filtered_indices)} filtered samples"
            return stats

        def on_reas_task_or_split_change(task, variant, split, filter_type):
            """When sidebar task/variant/split changes, resolve reasoning key and load first sample."""
            rkey = _resolve_reasoning_key(task, variant)
            filtered = _build_filtered_indices(rkey, split, filter_type)
            info = REASONING_INDEX.get(rkey, {}).get(split)
            if not info:
                stats = f"*No reasoning traces for {task}" + (f" / {variant}" if variant else "") + f" ({split})*"
                defaults = render_reasoning(rkey, split, 0)
                return list(defaults) + [stats, filtered]
            first_idx = filtered[0] if filtered else 0
            result = render_reasoning(rkey, split, first_idx)
            stats = _make_stats(rkey, split, filtered)
            return list(result) + [stats, filtered]

        def on_reas_filter_change(task, variant, split, filter_type):
            """When filter changes, rebuild index and jump to first match."""
            rkey = _resolve_reasoning_key(task, variant)
            filtered = _build_filtered_indices(rkey, split, filter_type)
            if filtered is not None and len(filtered) == 0:
                stats = _make_stats(rkey, split, filtered)
                defaults = (
                    f"*No samples match filter '{filter_type}'*", None, "", "",
                    "*No matching samples*", "", "", "", {}, 0, "", "*No sample loaded*",
                )
                return list(defaults) + [stats, filtered]
            first_idx = filtered[0] if filtered else 0
            result = render_reasoning(rkey, split, first_idx)
            stats = _make_stats(rkey, split, filtered)
            return list(result) + [stats, filtered]

        def on_reas_next(task, variant, split, idx, filtered):
            rkey = _resolve_reasoning_key(task, variant)
            if filtered is not None:
                # Find next index in filtered list after current idx
                for fi in filtered:
                    if fi > idx:
                        return render_reasoning(rkey, split, fi)
                # Wrap around
                return render_reasoning(rkey, split, filtered[0]) if filtered else render_reasoning(rkey, split, idx)
            return render_reasoning(rkey, split, idx + 1)

        def on_reas_prev(task, variant, split, idx, filtered):
            rkey = _resolve_reasoning_key(task, variant)
            if filtered is not None:
                # Find previous index in filtered list before current idx
                for fi in reversed(filtered):
                    if fi < idx:
                        return render_reasoning(rkey, split, fi)
                # Wrap around
                return render_reasoning(rkey, split, filtered[-1]) if filtered else render_reasoning(rkey, split, idx)
            return render_reasoning(rkey, split, max(0, idx - 1))

        def on_reas_jump(task, variant, split, num):
            rkey = _resolve_reasoning_key(task, variant)
            return render_reasoning(rkey, split, int(num) - 1)

        def on_reas_random(task, variant, split, filtered):
            rkey = _resolve_reasoning_key(task, variant)
            if filtered is not None and filtered:
                return render_reasoning(rkey, split, random.choice(filtered))
            info = REASONING_INDEX.get(rkey, {}).get(split)
            count = info["count"] if info else 1
            return render_reasoning(rkey, split, random.randint(0, count - 1))

        def on_reas_search(image_id):
            """Search across all tasks for an image_id, return markdown summary."""
            if not image_id or not image_id.strip():
                return "*Enter an image ID to search*"
            results = search_reasoning_image_id(image_id.strip())
            if not results:
                return f"*No results found for `{image_id.strip()}`*"
            lines = [f"**Found {len(results)} matches:**\n"]
            for r in results:
                hall_ids = REASONING_HALLUCINATIONS.get(r["task"], {}).get(r["split"], set())
                orient_map = REASONING_ORIENTATION_MISMATCHES.get(r["task"], {}).get(r["split"], {})
                badge = "⚠️" if r["image_id"] in hall_ids else "✅"
                orient_info = orient_map.get(r["image_id"])
                orient = ""
                if orient_info:
                    conf = orient_info.get("confidence", "") if isinstance(orient_info, dict) else ""
                    orient = f" 🧭({conf})" if conf else " 🧭"
                bp_map = REASONING_BODY_POS_MISMATCHES.get(r["task"], {}).get(r["split"], {})
                bp_tag = " 🛏️" if r["image_id"] in bp_map else ""
                ha_map = REASONING_HIGH_ANGLE.get(r["task"], {}).get(r["split"], {})
                ha_tag = " 📐" if r["image_id"] in ha_map else ""
                lines.append(f"- {badge}{orient}{bp_tag}{ha_tag} **{r['task']}** / {r['split']} — sample #{r['idx'] + 1} — `{r['image_id']}`")
            return "\n".join(lines)

        def on_reas_cross_task(image_id):
            """Find current image in all tasks."""
            if not image_id:
                return "*No image loaded*"
            return on_reas_search(image_id)

        # Wire reasoning trace events — uses sidebar task_dropdown + variant_dropdown
        reas_nav_inputs = [task_dropdown, variant_dropdown, split_radio, reas_current_idx, reas_filtered_indices]
        reas_task_inputs = [task_dropdown, variant_dropdown, split_radio, reas_filter]
        reas_full_outputs = reas_outputs + [reas_stats, reas_filtered_indices]

        # Load reasoning data when tab is selected or sidebar changes while on tab
        reasoning_tab.select(
            fn=on_reas_task_or_split_change,
            inputs=reas_task_inputs,
            outputs=reas_full_outputs)

        reas_filter.change(
            fn=on_reas_filter_change,
            inputs=reas_task_inputs,
            outputs=reas_full_outputs)

        reas_next_btn.click(fn=on_reas_next, inputs=reas_nav_inputs, outputs=reas_outputs)
        reas_prev_btn.click(fn=on_reas_prev, inputs=reas_nav_inputs, outputs=reas_outputs)
        reas_jump_btn.click(
            fn=on_reas_jump,
            inputs=[task_dropdown, variant_dropdown, split_radio, reas_jump],
            outputs=reas_outputs)
        reas_random_btn.click(
            fn=on_reas_random,
            inputs=[task_dropdown, variant_dropdown, split_radio, reas_filtered_indices],
            outputs=reas_outputs)

        reas_search_btn.click(fn=on_reas_search, inputs=[reas_search], outputs=[reas_cross_task])
        reas_cross_task_btn.click(fn=on_reas_cross_task, inputs=[reas_current_image_id], outputs=[reas_cross_task])

        # --- Audit exclude/approve/undo ---
        def on_reas_exclude(task, variant, split, idx, image_id, reason):
            """Mark current sample as excluded."""
            rkey = _resolve_reasoning_key(task, variant)
            if not image_id:
                return render_reasoning(rkey, split, idx)
            # Remove from approved if it was there
            approved = REASONING_APPROVED.get(rkey, {}).get(split, {})
            if image_id in approved:
                del approved[image_id]
                _save_approved(rkey, split)
            REASONING_AUDIT.setdefault(rkey, {}).setdefault(split, {})[image_id] = {
                "reason": reason,
                "timestamp": datetime.now().isoformat(),
                "line_idx": idx,
            }
            _save_audit(rkey, split)
            gr.Info(f"Excluded: {image_id} ({reason})")
            return render_reasoning(rkey, split, idx)

        def on_reas_approve(task, variant, split, idx, image_id):
            """Mark current sample as approved (reviewed and OK)."""
            rkey = _resolve_reasoning_key(task, variant)
            if not image_id:
                return render_reasoning(rkey, split, idx)
            # Remove from excluded if it was there
            audit = REASONING_AUDIT.get(rkey, {}).get(split, {})
            if image_id in audit:
                del audit[image_id]
                _save_audit(rkey, split)
            REASONING_APPROVED.setdefault(rkey, {}).setdefault(split, {})[image_id] = {
                "timestamp": datetime.now().isoformat(),
                "line_idx": idx,
            }
            _save_approved(rkey, split)
            gr.Info(f"Approved: {image_id}")
            return render_reasoning(rkey, split, idx)

        def on_reas_undo(task, variant, split, idx, image_id):
            """Remove audit exclusion or approval for current sample."""
            rkey = _resolve_reasoning_key(task, variant)
            if not image_id:
                return render_reasoning(rkey, split, idx)
            audit = REASONING_AUDIT.get(rkey, {}).get(split, {})
            if image_id in audit:
                del audit[image_id]
                _save_audit(rkey, split)
                gr.Info(f"Restored: {image_id}")
            approved = REASONING_APPROVED.get(rkey, {}).get(split, {})
            if image_id in approved:
                del approved[image_id]
                _save_approved(rkey, split)
                gr.Info(f"Unapproved: {image_id}")
            return render_reasoning(rkey, split, idx)

        reas_approve_btn.click(
            fn=on_reas_approve,
            inputs=[task_dropdown, variant_dropdown, split_radio, reas_current_idx, reas_current_image_id],
            outputs=reas_outputs)
        reas_exclude_btn.click(
            fn=on_reas_exclude,
            inputs=[task_dropdown, variant_dropdown, split_radio, reas_current_idx, reas_current_image_id, reas_audit_reason],
            outputs=reas_outputs)
        reas_undo_btn.click(
            fn=on_reas_undo,
            inputs=[task_dropdown, variant_dropdown, split_radio, reas_current_idx, reas_current_image_id],
            outputs=reas_outputs)

        # ========== PROMPT COMPARISON EVENT HANDLERS ==========

        def pcmp_render(exp_data, task, idx):
            """Render one sample across all versions."""
            empty_col = ("", "", "", "")
            empty = ("*No data*", None) + empty_col * 4 + (0,)
            empty_vis = empty + (gr.update(visible=True),) * 4
            if not exp_data or not task or task not in exp_data.get("tasks", []):
                return empty_vis

            count = exp_data["counts"].get(task, 0)
            if count == 0:
                return empty_vis
            idx = max(0, min(idx, count - 1))

            versions = exp_data["versions"][task]
            samples = []
            for ver in versions:
                lines = exp_data["data"][task][ver]
                s = _parse_comparison_sample(lines[idx])
                s["is_hallucinated"] = s["image_id"] in exp_data["hallucinated"][task].get(ver, set())
                s["version"] = ver
                samples.append(s)

            # Image from first version (all share the same image)
            img_path = samples[0]["image_path"]
            img = img_path if img_path and Path(img_path).exists() else None

            header = (
                f"### {task} — Sample {idx + 1} of {count}\n"
                f"**Image ID**: `{samples[0]['image_id']}`"
            )

            # Build 4 column outputs
            col_outputs = []
            for i in range(4):
                if i < len(samples):
                    s = samples[i]
                    badge = f"Words: **{s['word_count']}**"
                    if s["is_hallucinated"]:
                        badge += " | :warning: **HALLUCINATED**"
                    else:
                        badge += " | Clean"
                    col_outputs.extend([
                        f"**{s['version']}**",
                        s["think_text"] or "*Empty reasoning*",
                        s["answer_text"],
                        badge,
                    ])
                else:
                    col_outputs.extend(["", "", "", ""])

            n = len(versions)
            vis = (
                gr.update(visible=n >= 1),
                gr.update(visible=n >= 2),
                gr.update(visible=n >= 3),
                gr.update(visible=n >= 4),
            )
            return (header, img) + tuple(col_outputs) + (idx,) + vis

        def on_pcmp_experiment_change(experiment_name):
            """Load experiment, populate task dropdown + stats."""
            if not experiment_name:
                return None, gr.update(choices=[], value=None), "*Select an experiment*"
            exp_data = _load_comparison_experiment(experiment_name)
            if not exp_data["tasks"]:
                return None, gr.update(choices=[], value=None), f"*No data in {experiment_name}*"

            lines = [f"**{experiment_name}**\n"]
            for t in exp_data["tasks"]:
                vers = ", ".join(exp_data["versions"][t])
                n = exp_data["counts"][t]
                hall_parts = []
                for v in exp_data["versions"][t]:
                    h = len(exp_data["hallucinated"][t].get(v, set()))
                    if h > 0:
                        hall_parts.append(f"{v}:{h}")
                hall_str = f" | hall: {', '.join(hall_parts)}" if hall_parts else ""
                lines.append(f"- **{t}**: {n} samples ({vers}){hall_str}")
            stats = "\n".join(lines)

            return (
                exp_data,
                gr.update(choices=exp_data["tasks"], value=exp_data["tasks"][0]),
                stats,
            )

        def on_pcmp_task_change(exp_data, task):
            """Task changed — render first sample, reset filter."""
            result = pcmp_render(exp_data, task, 0)
            return list(result) + [None]  # +filtered_indices reset

        def on_pcmp_next(exp_data, task, idx, filtered):
            if filtered is not None and filtered:
                for fi in filtered:
                    if fi > idx:
                        return pcmp_render(exp_data, task, fi)
                return pcmp_render(exp_data, task, filtered[0])
            return pcmp_render(exp_data, task, idx + 1)

        def on_pcmp_prev(exp_data, task, idx, filtered):
            if filtered is not None and filtered:
                for fi in reversed(filtered):
                    if fi < idx:
                        return pcmp_render(exp_data, task, fi)
                return pcmp_render(exp_data, task, filtered[-1])
            return pcmp_render(exp_data, task, max(0, idx - 1))

        def on_pcmp_jump(exp_data, task, num):
            return pcmp_render(exp_data, task, int(num) - 1)

        def on_pcmp_random(exp_data, task, filtered):
            if filtered is not None and filtered:
                return pcmp_render(exp_data, task, random.choice(filtered))
            if not exp_data or not task:
                return pcmp_render(exp_data, task, 0)
            count = exp_data["counts"].get(task, 1)
            return pcmp_render(exp_data, task, random.randint(0, count - 1))

        def on_pcmp_filter_change(exp_data, task, filter_type):
            """Build filtered indices and render first match."""
            filtered = None
            if filter_type != "All" and exp_data and task and task in exp_data.get("tasks", []):
                count = exp_data["counts"].get(task, 0)
                versions = exp_data["versions"].get(task, [])
                indices = []
                for i in range(count):
                    if filter_type == "Any Hallucinated":
                        for ver in versions:
                            s = _parse_comparison_sample(exp_data["data"][task][ver][i])
                            if s["image_id"] in exp_data["hallucinated"][task].get(ver, set()):
                                indices.append(i)
                                break
                    elif filter_type == "Versions Disagree (word count)":
                        wcs = []
                        for ver in versions:
                            s = _parse_comparison_sample(exp_data["data"][task][ver][i])
                            wcs.append(s["word_count"])
                        if wcs and max(wcs) > 1.5 * max(min(wcs), 1):
                            indices.append(i)
                filtered = indices if indices else []

            first_idx = filtered[0] if filtered else 0
            result = pcmp_render(exp_data, task, first_idx)
            return list(result) + [filtered]

        # --- Prompt Comparison wiring ---
        pcmp_render_outputs = [
            pcmp_header, pcmp_image,
            pcmp_label1, pcmp_think1, pcmp_answer1, pcmp_badge1,
            pcmp_label2, pcmp_think2, pcmp_answer2, pcmp_badge2,
            pcmp_label3, pcmp_think3, pcmp_answer3, pcmp_badge3,
            pcmp_label4, pcmp_think4, pcmp_answer4, pcmp_badge4,
            pcmp_current_idx,
            pcmp_col1, pcmp_col2, pcmp_col3, pcmp_col4,
        ]

        pcmp_experiment.change(
            fn=on_pcmp_experiment_change,
            inputs=[pcmp_experiment],
            outputs=[pcmp_exp_data, pcmp_task, pcmp_stats])

        pcmp_task.change(
            fn=on_pcmp_task_change,
            inputs=[pcmp_exp_data, pcmp_task],
            outputs=pcmp_render_outputs + [pcmp_filtered_indices])

        pcmp_nav_inputs = [pcmp_exp_data, pcmp_task, pcmp_current_idx, pcmp_filtered_indices]
        pcmp_next_btn.click(fn=on_pcmp_next, inputs=pcmp_nav_inputs, outputs=pcmp_render_outputs)
        pcmp_prev_btn.click(fn=on_pcmp_prev, inputs=pcmp_nav_inputs, outputs=pcmp_render_outputs)
        pcmp_jump_btn.click(
            fn=on_pcmp_jump,
            inputs=[pcmp_exp_data, pcmp_task, pcmp_jump],
            outputs=pcmp_render_outputs)
        pcmp_random_btn.click(
            fn=on_pcmp_random,
            inputs=[pcmp_exp_data, pcmp_task, pcmp_filtered_indices],
            outputs=pcmp_render_outputs)

        pcmp_filter.change(
            fn=on_pcmp_filter_change,
            inputs=[pcmp_exp_data, pcmp_task, pcmp_filter],
            outputs=pcmp_render_outputs + [pcmp_filtered_indices])

        prompt_cmp_tab.select(
            fn=lambda: gr.update(choices=_discover_comparison_experiments()),
            inputs=[], outputs=[pcmp_experiment])

        # === Archive Manager ===
        def _normalize_arch_task(task: str) -> str:
            """Convert archive dropdown display label to actual task key."""
            return "*" if task and task.startswith("*") else task

        def _on_arch_task_change(task):
            """Update archive variant dropdown when task changes (includes CSV-only variants)."""
            task = _normalize_arch_task(task)
            variants = [v for v in get_all_known_variants(task) if not is_archived(task, v)]
            return gr.Dropdown(choices=variants, value=variants[0] if variants else None)

        def _on_archive(task, variant, current_task, current_metric):
            """Archive a single task/variant experiment."""
            global EXPERIMENT_INDEX
            task = _normalize_arch_task(task)
            if not task or not variant:
                return gr.update(), gr.update(), "Select a task and variant.", gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
            _ARCHIVED_EXPERIMENTS.add((task, variant))
            _save_archive()
            EXPERIMENT_INDEX = load_experiments_csv()
            clear_all_caches()
            archived_labels = [f"{t} / {v}" for t, v in sorted(_ARCHIVED_EXPERIMENTS)]
            new_variants = get_active_variants(current_task, list(DATASET_INDEX.get(current_task, {}).keys()))
            remaining = [v for v in get_all_known_variants(task) if not is_archived(task, v)]
            # Re-render Training Monitor plot, table, and notes
            cp_table = create_checkpoint_table(current_task, variant=None, all_variants=True)
            plot = create_metrics_plot(current_task, variant=None, all_variants=True, metric=current_metric)
            notes = get_training_notes(current_task)
            return (
                gr.Dropdown(choices=remaining, value=remaining[0] if remaining else None),
                gr.Dropdown(choices=archived_labels, value=None),
                f"Archived **{task} / {variant}**.",
                gr.Dropdown(choices=new_variants, value=new_variants[0] if new_variants else None),
                gr.update(),
                cp_table,
                plot,
                notes,
            )

        def _on_restore(selection, current_task, current_metric):
            """Restore a previously archived experiment."""
            global EXPERIMENT_INDEX
            if not selection:
                return gr.update(), "Nothing selected.", gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
            parts = selection.split(" / ", 1)
            if len(parts) != 2:
                return gr.update(), "Invalid selection.", gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
            task, variant = parts
            _ARCHIVED_EXPERIMENTS.discard((task, variant))
            _save_archive()
            EXPERIMENT_INDEX = load_experiments_csv()
            clear_all_caches()
            archived_labels = [f"{t} / {v}" for t, v in sorted(_ARCHIVED_EXPERIMENTS)]
            new_variants = get_active_variants(current_task, list(DATASET_INDEX.get(current_task, {}).keys()))
            cp_table = create_checkpoint_table(current_task, variant=None, all_variants=True)
            plot = create_metrics_plot(current_task, variant=None, all_variants=True, metric=current_metric)
            notes = get_training_notes(current_task)
            return (
                gr.Dropdown(choices=archived_labels, value=None),
                f"Restored **{task} / {variant}**.",
                gr.Dropdown(choices=new_variants, value=new_variants[0] if new_variants else None),
                gr.update(),
                cp_table,
                plot,
                notes,
            )

        arch_task_dd.change(fn=_on_arch_task_change, inputs=[arch_task_dd], outputs=[arch_variant_dd])
        archive_btn.click(
            fn=_on_archive,
            inputs=[arch_task_dd, arch_variant_dd, task_dropdown, metric_selector],
            outputs=[arch_variant_dd, arch_restore_dd, archive_status, variant_dropdown, metric_selector,
                     checkpoint_table, metrics_plot, training_notes]
        )
        restore_btn.click(
            fn=_on_restore,
            inputs=[arch_restore_dd, task_dropdown, metric_selector],
            outputs=[arch_restore_dd, archive_status, variant_dropdown, metric_selector,
                     checkpoint_table, metrics_plot, training_notes]
        )

    return app


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point for the monitoring app."""
    parser = argparse.ArgumentParser(
        description="VLM Monitoring"
    )
    parser.add_argument("--port", type=int, default=7861, help="Port to run server on")
    parser.add_argument("--share", action="store_true", help="Create public share link")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Build indexes at startup
    global DATASET_INDEX, MODEL_INDEX, EXPERIMENT_INDEX, BENCHMARKS_INDEX, MODEL_TO_TASK_VARIANT, VALIDATOR_INDEX, CONFUSION_FLAGS

    print("=" * 80)
    print("VLM POSE ESTIMATION PIPELINE MONITOR")
    print("=" * 80)
    print()

    print("📈 Loading experiments tracking...")
    EXPERIMENT_INDEX = load_experiments_csv()
    if EXPERIMENT_INDEX is not None:
        print(f"✓ Loaded {len(EXPERIMENT_INDEX)} experiment records")
        print(f"✓ Built model-to-task-variant mapping for {len(MODEL_TO_TASK_VARIANT)} models")
    else:
        print("⚠ Experiments CSV not found")
    print()

    print("📊 Loading benchmark results...")
    BENCHMARKS_INDEX = load_benchmarks_index()
    if BENCHMARKS_INDEX:
        print(f"✓ Loaded IFEval: {len(BENCHMARKS_INDEX.get('ifeval', {}).get('models', {}))} models")
        print(f"✓ Loaded SIBench: {len(BENCHMARKS_INDEX.get('sibench', {}).get('models', {}))} models")
    else:
        print("⚠ Benchmark results not found")
    print()

    print("📊 Building dataset index...")
    DATASET_INDEX = build_dataset_index()
    _sync_task_names()
    print(f"✓ Found {len(DATASET_INDEX)} tasks with {sum(len(v) for v in DATASET_INDEX.values())} variants")
    print()

    print("📝 Building validator index...")
    VALIDATOR_INDEX = build_validator_index()
    print(f"✓ Found {len(VALIDATOR_INDEX)} validated image_ids")
    print()

    print("🔍 Building 2D/3D confusion flags index...")
    CONFUSION_FLAGS = build_confusion_flags_index()
    print(f"✓ Found {len(CONFUSION_FLAGS)} flagged samples")
    print()

    print("🎯 Building checkpoint index...")
    MODEL_INDEX = build_checkpoint_index()
    total_checkpoints = sum(len(cps) for task_variants in MODEL_INDEX.values() for cps in task_variants.values())
    print(f"✓ Found {total_checkpoints} checkpoints across {len(MODEL_INDEX)} tasks")
    print()

    print("🧠 Building reasoning traces index...")
    build_reasoning_index()
    reas_total = sum(info["count"] for t in REASONING_INDEX.values() for info in t.values())
    reas_hall = sum(len(s) for t in REASONING_HALLUCINATIONS.values() for s in t.values())
    print(f"✓ Found {reas_total} reasoning traces across {len(REASONING_INDEX)} tasks ({reas_hall} hallucinated)")
    print()

    print("=" * 80)
    print("🚀 Launching Gradio app...")
    print("=" * 80)

    # Build and launch app
    app = build_ui()

    # Allow access to dataset image directories for Gradio 6.0 security
    allowed_paths = [
        "/mnt/data/shared/vlm/data/output-train-dataset",  # Training dataset images
        "/mnt/data/shared/vlm/data",  # Parent directory for all data
    ]

    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        show_error=True,
        theme=custom_theme,
        css=custom_css,
        allowed_paths=allowed_paths
    )


if __name__ == "__main__":
    main()
