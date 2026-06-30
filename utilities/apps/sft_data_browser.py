#!/usr/bin/env python3
"""
Text SFT Dataset Monitoring App.

Browse, filter, and quality-check all text-based SFT datasets:
  - 5 new datasets from Thrive VLM Database v2
  - Existing auxiliary datasets (exercise_instructions, technical_tips)

Usage:
    cd /home/sgsilva/utilities/apps
    python sft_data_browser.py --port 7866
    # or: launch_app.sh sft-data
"""

import argparse
import json
import logging
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ["GRADIO_TEMP_DIR"] = os.path.expanduser("~/.gradio_temp")

import gradio as gr
import pandas as pd
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

APP_ROOT = Path(__file__).resolve().parent
HOME_ROOT = Path.home()
VLM_EVAL_ROOT = HOME_ROOT / "vlm-evaluation"
VIDEO_SFT_ROOT = HOME_ROOT / "video-sft-vlm"
# generate_text_sft_datasets (parse_excel/load_csv_names) was migrated from the
# retired sft-data-vlm repo into aux_tasks; this app is its only remaining caller.
GEN_TEXT_SFT_DIR = (
    HOME_ROOT / "vlm-post-training" / "aux_tasks" / "text_tasks" / "generation"
)

NEW_DATASETS_DIR = Path(
    os.environ.get("TEXT_SFT_DIR", "/mnt/data/shared/vlm/data/text_aux_datasets/_archive")
)
EXISTING_DATASETS_DIR = Path(
    os.environ.get("TEXT_AUX_DIR", "/mnt/data/shared/vlm/data/text_aux_datasets")
)
EVAL_RESULTS_DIR = Path(
    os.environ.get("TEXT_EVAL_DIR", str(VLM_EVAL_ROOT / "text-dataset" / "results" / "final"))
)

NEW_DATASET_NAMES = [
    "text_error_recognition",
    "text_phase_sequencing",
    "text_muscle_exercise",
    "text_clinical_reasoning",
    "text_error_correction",
]

# Numbered labels for the dataset selector (maps label -> internal name)
DATASET_LABELS = {
    "1. Error Recognition (MCQA)": "text_error_recognition",
    "2. Phase Sequencing (MCQA)": "text_phase_sequencing",
    "3. Muscle-Exercise (MCQA)": "text_muscle_exercise",
    "4. Clinical Reasoning (Open)": "text_clinical_reasoning",
    "5. Error Correction (MCQA)": "text_error_correction",
}
DATASET_LABEL_TO_NAME = DATASET_LABELS
DATASET_NAME_TO_LABEL = {v: k for k, v in DATASET_LABELS.items()}

# Dataset version directories (v1=easy distractors, v2=hardened cross-exercise)
DATASET_VERSIONS = ["v2 (current)", "v1 (easy)"]

EXISTING_DATASET_NAMES = [
    "exercise_instructions_1102",
    "exercise_instructions_natural_1102",
    "exercise_instructions_natural_reasoning_1102",
    "technical_tips_1102",
    "technical_tips_natural_1102",
    "technical_tips_natural_reasoning_1102",
]

# Exercise metadata (loaded from Excel for full field coverage)
_EXERCISE_EXCEL_PATH = Path(
    os.environ.get(
        "EXERCISE_EXCEL",
        "/mnt/data/sgsilva/Thrive VLM Database_v2.xlsx",
    )
)
_EXERCISE_CSV_PATH = Path(
    os.environ.get(
        "EXERCISE_CSV",
        str(VIDEO_SFT_ROOT / "training" / "exercise_metadata.csv"),
    )
)
_EXERCISE_DATA: Dict[str, Dict[str, str]] = {}  # code -> row dict


def _load_exercise_data() -> Dict[str, Dict[str, str]]:
    """Load exercise data from Excel (full fields) with CSV name corrections.

    L/R variant codes (not in Excel) are mapped to their bilateral parent.
    """
    global _EXERCISE_DATA
    if _EXERCISE_DATA:
        return _EXERCISE_DATA
    try:
        import sys
        sys.path.insert(0, str(GEN_TEXT_SFT_DIR))
        from generate_text_sft_datasets import parse_excel, load_csv_names
        exercises = parse_excel(str(_EXERCISE_EXCEL_PATH))
        csv_names = load_csv_names(str(_EXERCISE_CSV_PATH))
        for ex in exercises:
            code = ex.get("exercise_code")
            if code is None:
                continue
            if code in csv_names:
                ex["exercise_name"] = csv_names[code]
            key = str(code)
            if key in _EXERCISE_DATA:
                # Duplicate code (exercise on both sheets) — merge: keep non-empty
                prev = _EXERCISE_DATA[key]
                for field, val in ex.items():
                    if val:  # only overwrite with non-empty
                        prev[field] = val
            else:
                _EXERCISE_DATA[key] = ex
        # Merge CSV-only fields (primary_joints, min_rom_target)
        import csv as csv_mod
        if _EXERCISE_CSV_PATH.exists():
            with open(_EXERCISE_CSV_PATH, encoding="utf-8") as cf:
                for csv_row in csv_mod.DictReader(cf):
                    c = csv_row.get("exercise_code", "").strip()
                    if c in _EXERCISE_DATA:
                        for fld in ("primary_joints", "min_rom_target", "correctness_criteria"):
                            csv_val = csv_row.get(fld, "").strip()
                            if csv_val:
                                _EXERCISE_DATA[c][fld] = csv_val
        # Map L/R variant codes to their bilateral parent data
        for code, name in csv_names.items():
            if str(code) in _EXERCISE_DATA:
                continue
            # L/R variants are typically code-1 or code-2 from bilateral
            for offset in [1, 2, 3]:
                parent = str(code - offset)
                if parent in _EXERCISE_DATA:
                    variant = dict(_EXERCISE_DATA[parent])
                    variant["exercise_name"] = name
                    variant["exercise_code"] = code
                    _EXERCISE_DATA[str(code)] = variant
                    break
        logger.info("Loaded %d exercises from Excel", len(_EXERCISE_DATA))
    except Exception as e:
        logger.warning("Failed to load Excel data: %s", e)
    return _EXERCISE_DATA

# ---------------------------------------------------------------------------
# Methodology content (rendered as static markdown in the Methodology tab)
# ---------------------------------------------------------------------------

METHODOLOGY_MD = r"""
## Source Data

All datasets are generated from the **Thrive VLM Database v2** Excel file (275 exercises, 2 sheets:
lower-extremity/lower-body and upper-extremity). Exercise names use the CSV-authoritative L/R labels
(the Excel v2 has a known L/R swap regression for 152 exercises).

The **Physiotherapy Knowledge Graph (PKG)** provides muscle-exercise associations for Dataset 3.

**Train/test split**: 80/20 by base exercise name (L/R pairs always in same split), stratified by
body\_region and exercise\_type. Total: 2,867 train / 1,657 test.

---

## 1. Error Recognition MCQA (`text_error_recognition` — 2,072 samples)

**Purpose**: Match error categories to their descriptions, and vice versa.

**Source columns**: `Typical Movement Errors` (structured as category + bullet-point descriptions).

**Templates**:

| Variant | Question | Correct answer | Distractors |
|---------|----------|----------------|-------------|
| `error_cat_to_desc` | "Which description best matches the error category '{category}' in {exercise}?" | The actual description for that category | Other descriptions from the **same exercise's** errors |
| `error_desc_to_cat` | "A patient shows: '{description}'. Which error category does this describe?" | The category name | Other category names from the **same exercise** |

**Filter**: Exercise must have >= 4 error categories (need 3 distractors).

---

## 2. Phase Sequencing MCQA (`text_phase_sequencing` — 1,166 samples)

**Purpose**: Test knowledge of movement phase content at the individual element level.

**Source columns**: `Exercise Phases` (structured as phase name + bullet-point elements, in order).

**Templates**:

| Variant | Question | Correct answer | Distractors |
|---------|----------|----------------|-------------|
| `phase_content` | "Which description corresponds to a key element of the '{phase}' phase of {exercise}?" | A single bullet point from that phase | Individual bullets from **other phases** (intra-exercise + cross-exercise) |

**Filter**: Exercise must have >= 2 phases with bullet-point descriptions. Bullets that appear in multiple phases of the same exercise are excluded (ambiguous).

---

## 3. Muscle-Exercise Association MCQA (`text_muscle_exercise` — 786 samples)

**Purpose**: Test bidirectional muscle-exercise reasoning using the Physiotherapy Knowledge Graph (PKG).

### How Muscles Were Identified

The muscle-exercise mappings come from the **PKG** (`training/pkg_graph.json`), a structured knowledge graph built from the authoritative exercise metadata CSV.

**Input**: The `muscles` column in `exercise_metadata.csv`, where physiotherapists manually listed involved muscles for each of the 275 exercises (e.g., "Glutes, Hamstrings, Abdominals").

**Normalization pipeline** (`video-sft-vlm/scripts/build_pkg.py`):
1. Raw muscle strings (102 unique variants with typos, synonyms, compound names) are cleaned via a curated lookup table of **147 normalization rules** (e.g., "Glute medius and minimus" → `gluteus_medius` + `gluteus_minimus`, "Levateur scapulae" → `levator_scapulae`).
2. Canonical names are organized into **11 anatomical groups** (hip, core, shoulder, etc.).
3. For exercises with empty muscle fields, **37 code-based defaults** provide mappings (e.g., wrist exercises → `wrist_flexors`, `wrist_extensors`).

**Result**: 259 exercises mapped to **54 unique canonical muscles** via 809 `INVOLVES_MUSCLE` edges.

**No LLM was used for muscle identification** — all muscle-exercise mappings are deterministic, based on expert physiotherapist input and curated normalization rules.

### Exercise Similarity (SIMILAR\_TO edges)

**2,305 SIMILAR\_TO** pairs are computed between exercises using **Jaccard similarity** on their muscle sets. For each pair of exercises, the Jaccard index = |shared muscles| / |all muscles in either exercise|. Pairs with Jaccard ≥ 0.3 (30% overlap) are linked. This is purely deterministic — no LLM involved. These edges are used in the `muscle_shared` variant (V3).

### Muscle names in questions

Canonical names use underscores internally (e.g., `latissimus_dorsi`). These are automatically converted to title case for display in questions and choices (e.g., "Latissimus Dorsi").

**Templates**:

| Variant | Question | Correct answer | Distractors |
|---------|----------|----------------|-------------|
| `muscle_odd_one_out` | "Which muscle is NOT primarily involved in {exercise}?" | A muscle **not** involved (per PKG) | Muscles that **are** involved |
| `muscle_reverse_lookup` | "Which exercise primarily targets {muscle}?" | An exercise that involves that muscle | Exercises that do **not** involve it |
| `muscle_shared` | "{exercise\_A} and {exercise\_B} share a common muscle. Which one?" | A muscle involved in both | Muscles **not** shared between the pair |

**Filters**: V1 requires >= 2 involved + >= 1 uninvolved muscle. V2 requires >= 1 targeting exercise + >= 3 non-targeting.
V3 requires pairs with >= 1 shared + >= 3 non-shared muscles; capped at 300 pairs.

---

## 4. Clinical Reasoning Open-Ended (`text_clinical_reasoning` — 683 samples)

**Purpose**: Multi-column synthesis — why prescribed, correct execution, patient-friendly explanation.

**Source columns**: `Background/Reasoning`, `Correct Movement Definition`, `DT Description`,
`Clinical Tips Breathing`, `Clinical Tips Technique`.

**Templates**:

| Variant | Question | Answer source |
|---------|----------|---------------|
| `clinical_background` | "Why is '{exercise}' prescribed in physiotherapy?" | `background` column |
| `correct_execution` | "Describe the correct execution of '{exercise}'." | `correct_movement` column |
| `patient_explanation` | "Explain '{exercise}' to a patient in simple terms." | Concatenation of `dt_description` + `breathing` + `technique` |

**Filter**: Skipped if the required source field is empty. No distractors (open-ended format).

---

## 5. Error Correction Advice MCQA (`text_error_correction` — 217 samples)

**Purpose**: Given observed errors, select the appropriate corrective technique tip.

**Source columns**: `Typical Movement Errors`, `Clinical Tips Technique`, `body_region`.

**Template**:

| Variant | Question | Correct answer | Distractors |
|---------|----------|----------------|-------------|
| `error_to_technique` | "A patient performing {exercise} shows errors: {error\_summary}. Which technique tip addresses this?" | This exercise's `clinical_tips_technique` | Technique tips from **other exercises** in the same body\_region |

**Filters**: Requires >= 1 error + non-empty technique tip + >= 3 other exercises in same body region with tips.

---

## Distractor Strategy Summary

| Dataset | Distractor scope | Risk of trivial shortcuts |
|---------|-----------------|--------------------------|
| 1. Error Recognition | Same exercise, other errors | Low — all errors are plausible for the exercise |
| 2. Phase Sequencing | Same exercise, other phases | Low — similar granularity |
| 3. Muscle-Exercise | PKG graph edges | Medium — some muscles are obviously unrelated |
| 5. Error Correction | Same body\_region, other exercises | Medium — tips from different exercises may be distinguishable by specificity |

## System Prompt

All datasets use: *"You are an AI physical therapist assistant working for Sword Health."*

---

## V1 → V2 Changes (Difficulty Hardening)

V1 datasets were too easy for baseline models (Qwen3-VL-4B: **82.5% overall MCQA**). V2 introduces **cross-exercise distractors** using the PKG graph to eliminate surface-level shortcuts, targeting 40–60% baseline accuracy.

### Root Causes of V1 Easiness

| Dataset | V1 Accuracy | Problem |
|---------|-------------|---------|
| 1. Error Recognition | 91.5% | **Lexical leakage** — category names share keywords with correct descriptions; distractors only from same exercise (small pool of 4–13 errors) |
| 2. Phase Sequencing | 67.5% | **Structural keyword tells** — "Initial position" always has a static description; distractors only from same exercise |
| 3. Muscle-Exercise | 72.9% | **Cross-region giveaways** — e.g., "Forearm Pronators" as distractor for a Bridge exercise is obviously wrong |
| 5. Error Correction | 53.0% | Good difficulty but small N (66 test samples) due to strict filtering |

### V2 Distractor Strategy

**1. Error Recognition**: Mixed cross-exercise + intra-exercise distractors.
- V1 (cat\_to\_desc): Adds descriptions of the **same error category from other exercises** (eliminates keyword matching) + descriptions from different categories of SIMILAR\_TO exercises.
- V2 (desc\_to\_cat): Adds category names from SIMILAR\_TO exercises not present in the current exercise.
- Composition: prioritize 2 cross-exercise + 1 intra-exercise (fallback to intra when cross unavailable).

**2. Phase Sequencing**: Individual bullet-point elements instead of full descriptions.
- Each option is now a **single bullet** (e.g., "Maximum elbow flexion achieved comfortably") instead of the entire multi-sentence phase description.
- Distractors are individual bullets from other phases (intra-exercise + cross-exercise via PKG).
- Bullets appearing in multiple phases of the same exercise are excluded (ambiguous).
- Sample count increased from 419 → 1,166 (each bullet generates its own question).

**3. Muscle-Exercise**: Same-group muscle filtering.
- V1 (odd\_one\_out): The "NOT involved" muscle now comes from the **same muscle\_group** (e.g., `piriformis` for a Bridge exercise instead of `forearm_pronators`). Fallback: same body\_region → all muscles.
- V2 (reverse\_lookup): Exercise distractors filtered to **same body\_region** as the target exercise.
- V3 (muscle\_shared): Non-shared muscle distractors filtered to **same muscle\_group(s)** as the shared muscles.

**4. Error Correction**: Expanded distractor pool.
- Supplements same-region technique tips with tips from **SIMILAR\_TO exercises** (Jaccard ≥ 0.5).
- Lowers same-region threshold from 3 to 2, supplementing with SIMILAR\_TO tips.
- Preserves difficulty while enabling more exercises to pass filters.

### Data Leakage Guard

Cross-exercise distractors are filtered by **split\_codes** — train samples only use distractors from training exercises, test samples only from test exercises. The PKG indexes span all exercises but lookups are restricted per split.

### V2 Results

| Dataset | V1 (Qwen3-VL-4B) | V2 (Qwen3-VL-4B) | Change |
|---------|-------------------|-------------------|--------|
| 1. Error Recognition | 91.5% | 90.7% | −0.8pp |
| 2. Phase Sequencing | 67.5% | 69.8% | +2.3pp |
| 3. Muscle-Exercise | 72.9% | **41.9%** | **−31.0pp** |
| 5. Error Correction | 53.0% | 50.0% | −3.0pp |

Muscle-Exercise saw the largest improvement (within target range). Error Recognition remains high — the 77/109 shared error categories have identical descriptions across exercises, limiting the effectiveness of cross-exercise same-category distractors.
"""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> List[Dict]:
    """Load samples from a JSONL file."""
    samples = []
    if not path.exists():
        return samples
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def load_hf_dataset(path: Path) -> List[Dict]:
    """Load samples from a HuggingFace arrow dataset directory."""
    try:
        from datasets import load_from_disk
        ds = load_from_disk(str(path))
        samples = []
        for row in ds:
            sample = dict(row)
            if isinstance(sample.get("messages"), list):
                sample["messages"] = json.dumps(sample["messages"])
            samples.append(sample)
        return samples
    except Exception as e:
        logger.warning(f"Could not load HF dataset {path}: {e}")
        return []


def load_dataset(name: str, split: str = "train", version: str = "") -> List[Dict]:
    """Load a dataset by name, split, and optional version subdir (e.g. 'v1')."""
    if version:
        versioned_path = NEW_DATASETS_DIR / name / version / f"{split}.jsonl"
        if versioned_path.exists():
            return load_jsonl(versioned_path)
    jsonl_path = NEW_DATASETS_DIR / name / f"{split}.jsonl"
    if jsonl_path.exists():
        return load_jsonl(jsonl_path)
    hf_path = EXISTING_DATASETS_DIR / name / split
    if hf_path.exists():
        return load_hf_dataset(hf_path)
    return []


def get_available_datasets() -> List[str]:
    """Get list of all available dataset names."""
    available = []
    for name in NEW_DATASET_NAMES:
        if (NEW_DATASETS_DIR / name).exists():
            available.append(name)
    for name in EXISTING_DATASET_NAMES:
        if (EXISTING_DATASETS_DIR / name).exists():
            available.append(name)
    return available


def get_available_splits(name: str) -> List[str]:
    """Get available splits for a dataset."""
    splits = []
    for split in ["train", "test"]:
        jsonl = NEW_DATASETS_DIR / name / f"{split}.jsonl"
        hf = EXISTING_DATASETS_DIR / name / split
        if jsonl.exists() or hf.exists():
            splits.append(split)
    return splits or ["train"]


# ---------------------------------------------------------------------------
# Caching layer
# ---------------------------------------------------------------------------

_SAMPLE_CACHE: Dict[Tuple[str, str, str], List[Dict]] = {}


def _resolve_version(version_label: str) -> str:
    """Convert UI version label to subdir name. Empty string = root (current)."""
    if not version_label or "current" in version_label:
        return ""
    # Extract 'v1' from 'v1 (easy)' etc.
    return version_label.split()[0] if version_label else ""


def get_samples(dataset: str, split: str, version: str = "") -> List[Dict]:
    """Cached sample loader."""
    key = (dataset, split, version)
    if key not in _SAMPLE_CACHE:
        _SAMPLE_CACHE[key] = load_dataset(dataset, split, version=version)
    return _SAMPLE_CACHE[key]


def get_all_samples(dataset: str, version: str = "") -> List[Dict]:
    """Get train+test samples for a dataset, each tagged with _split."""
    results = []
    for split in get_available_splits(dataset):
        for s in get_samples(dataset, split, version=version):
            tagged = dict(s)
            tagged["_split"] = split
            results.append(tagged)
    return results


def get_variants(dataset: str, version: str = "") -> List[str]:
    """Get unique variant names for a dataset."""
    samples = get_all_samples(dataset, version=version)
    return sorted(set(s.get("variant", "") for s in samples if s.get("variant")))


# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------

def extract_messages(sample: Dict) -> List[Dict]:
    """Parse messages from a sample."""
    msgs = sample.get("messages", "[]")
    if isinstance(msgs, str):
        msgs = json.loads(msgs)
    return msgs


def extract_question(sample: Dict) -> str:
    """Extract user question text from sample."""
    msgs = extract_messages(sample)
    for m in msgs:
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


def extract_answer(sample: Dict) -> str:
    """Extract assistant answer from sample."""
    msgs = extract_messages(sample)
    for m in msgs:
        if m.get("role") == "assistant":
            content = m.get("content", "")
            if isinstance(content, list):
                return " ".join(
                    item.get("text", "") for item in content if isinstance(item, dict)
                )
            return content
    return ""


def truncate(text: str, max_len: int = 120) -> str:
    """Truncate text for table display."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


# ---------------------------------------------------------------------------
# Quality flags
# ---------------------------------------------------------------------------

def compute_sample_flags(sample: Dict) -> List[str]:
    """Return quality flags for a single sample."""
    flags = []

    # Message parse check
    try:
        msgs = extract_messages(sample)
        roles = [m.get("role") for m in msgs]
        if "user" not in roles:
            flags.append("NO_USER_MSG")
        if "assistant" not in roles:
            flags.append("NO_ASSISTANT_MSG")
    except Exception:
        flags.append("MSG_PARSE_ERROR")
        return flags

    answer = extract_answer(sample)

    # Very short answer for open-ended
    if "correct_answer" not in sample and len(answer) < 60:
        flags.append("SHORT_ANSWER")

    # MCQA checks
    if "correct_answer" in sample and "choices" in sample:
        try:
            choices = json.loads(sample["choices"]) if isinstance(sample["choices"], str) else sample["choices"]
            if sample["correct_answer"] not in choices:
                flags.append("ANSWER_NOT_IN_CHOICES")
            # Choice length variance
            lengths = [len(v) for v in choices.values()]
            if len(lengths) >= 2 and min(lengths) > 5 and max(lengths) > 3 * min(lengths):
                flags.append("HIGH_CHOICE_LEN_VAR")
        except (json.JSONDecodeError, TypeError):
            flags.append("CHOICES_PARSE_ERROR")

    return flags


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def filter_samples(
    dataset: str,
    variant: Optional[str] = None,
    body_region: Optional[str] = None,
    exercise_search: str = "",
    split: Optional[str] = None,
    version: str = "",
) -> List[Dict]:
    """Filter samples by criteria. Returns list of samples with _split tag."""
    if split:
        samples = [dict(s, _split=split) for s in get_samples(dataset, split, version=version)]
    else:
        samples = get_all_samples(dataset, version=version)

    if variant:
        samples = [s for s in samples if s.get("variant") == variant]
    if body_region:
        samples = [s for s in samples if s.get("body_region") == body_region]
    if exercise_search.strip():
        term = exercise_search.strip().lower()
        samples = [s for s in samples if term in s.get("exercise_name", "").lower()]
    return samples


# ---------------------------------------------------------------------------
# Sample display
# ---------------------------------------------------------------------------

_CHOICE_LINE_RE = re.compile(r"\n([A-D])\.\s")


def _format_question_text(content: str) -> str:
    """Make MCQA choice lines (A. B. C. D.) render on separate lines in markdown."""
    # Ensure each "A. ..." starts on its own line with a blank line before it
    content = _CHOICE_LINE_RE.sub(r"\n\n**\1.** ", content)
    return content


def format_sample_display(sample: Dict) -> str:
    """Format a single sample as markdown for display."""
    parts = []

    # Metadata block
    # Map variant → source Excel column(s) used to create the question
    variant_source_map = {
        "error_cat_to_desc": "Typical Movement Errors",
        "error_desc_to_cat": "Typical Movement Errors",
        "phase_content": "Exercise Phases",
        "muscle_odd_one_out": "muscles (CSV) + PKG graph",
        "muscle_reverse_lookup": "muscles (CSV) + PKG graph",
        "muscle_shared": "muscles (CSV) + PKG SIMILAR_TO edges",
        "clinical_background": "Background / Reasoning",
        "correct_execution": "Correct Movement Definition",
        "patient_explanation": "DT Description + Clinical Tips (Breathing, Technique)",
        "error_to_technique": "Typical Movement Errors + Clinical Tips Technique",
    }
    variant = sample.get("variant", "")
    source_col = variant_source_map.get(variant, "—")

    parts.append(
        f"| | |\n|---|---|\n"
        f"| **Exercise** | {sample.get('exercise_name', 'N/A')} |\n"
        f"| **Region** | {sample.get('body_region', 'N/A')} |\n"
        f"| **Variant** | {sample.get('variant', 'N/A')} |\n"
        f"| **Source Column** | {source_col} |\n"
        f"| **Split** | {sample.get('_split', '?')} |\n"
        f"| **ID** | {sample.get('exercise_id', 'N/A')} |\n"
        f"| **L/R Pair** | {sample.get('is_lr_pair', 'N/A')} |"
    )
    parts.append("")

    try:
        msgs = extract_messages(sample)
    except Exception:
        parts.append("*[Failed to parse messages]*")
        return "\n".join(parts)

    for m in msgs:
        role = m.get("role", "?")
        content = m.get("content", "")
        if isinstance(content, list):
            content = "\n".join(
                item.get("text", str(item)) for item in content if isinstance(item, dict)
            )
        if role == "system":
            parts.append(f"> *System: {truncate(content, 200)}*")
            parts.append("")
        elif role == "user":
            parts.append("---")
            parts.append(f"#### QUESTION\n\n{_format_question_text(content)}")
            parts.append("")
        elif role == "assistant":
            parts.append("---")
            # Ensure bullet lists render properly in markdown
            answer_text = re.sub(r"(?<!\n)\n- ", "\n\n- ", content)
            parts.append(f"#### ANSWER\n\n{answer_text}")
            parts.append("")

    # Source data for cross-checking (from Excel — all fields)
    ex_data = _load_exercise_data()
    if ex_data:
        # Build name→row lookup for multi-exercise questions
        data_by_name = {}
        for r in ex_data.values():
            name = r.get("exercise_name", "")
            if name:
                data_by_name[name] = r

        # Collect exercises to show: primary + any other mentioned in question
        exercises_to_show = []
        ex_code = str(sample.get("exercise_id", "")).strip()
        primary = ex_data.get(ex_code)
        if primary:
            exercises_to_show.append(primary)

        # Check for second exercise in question (muscle_shared variant)
        try:
            q_msgs = extract_messages(sample)
            q_text = next(
                (m["content"] for m in q_msgs if m.get("role") == "user"), ""
            )
            if isinstance(q_text, list):
                q_text = " ".join(
                    item.get("text", "") for item in q_text if isinstance(item, dict)
                )
            quoted = re.findall(r"'([^']+)'", q_text)
            for name in quoted:
                row2 = data_by_name.get(name)
                if row2 and str(row2.get("exercise_code")) != ex_code:
                    exercises_to_show.append(row2)
        except Exception:
            pass

        # Fields to display (all Excel columns, ordered by relevance)
        field_labels = [
            ("exercise_code", "Code"),
            ("exercise_name", "Name"),
            ("body_region", "Body Region"),
            ("position", "Position"),
            ("muscles", "Muscles"),
            ("primary_joints", "Primary Joints"),
            ("min_rom_target", "Min ROM Target"),
            ("background", "Background / Reasoning"),
            ("correct_movement", "Correct Movement Definition"),
            ("correctness_criteria", "Correctness Criteria"),
            ("errors", "Typical Movement Errors"),
            ("phases", "Exercise Phases"),
            ("long_desc", "Long Description"),
            ("brief_desc", "Brief Description"),
            ("dt_description", "DT Exercise Description"),
            ("clinical_tips_technique", "Clinical Tips — Technique"),
            ("clinical_tips_breathing", "Clinical Tips — Breathing"),
            ("clinical_tips_balance", "Clinical Tips — Balance"),
        ]

        def _cell(row, field):
            val = str(row.get(field, "") or "").strip()
            return val.replace("|", " \\| ").replace("\n", "<br>") if val else ""

        if exercises_to_show:
            parts.append("")
            parts.append("---")
            parts.append("#### SOURCE DATA\n")

            if len(exercises_to_show) == 1:
                # Single exercise: Field | Value
                row = exercises_to_show[0]
                parts.append("| Field | Value |")
                parts.append("|-------|-------|")
                for field, label in field_labels:
                    val = _cell(row, field)
                    if val:
                        parts.append(f"| **{label}** | {val} |")
            else:
                # Multiple exercises: side-by-side comparison
                names = [_cell(r, "exercise_name") or f"Exercise {i+1}"
                         for i, r in enumerate(exercises_to_show)]
                header = "| Field | " + " | ".join(f"**{n}**" for n in names) + " |"
                sep = "|-------|" + "|".join("-------|" for _ in names)
                parts.append(header)
                parts.append(sep)
                for field, label in field_labels:
                    if field == "exercise_name":
                        continue  # already in header
                    cells = [_cell(r, field) for r in exercises_to_show]
                    if not any(cells):
                        continue
                    parts.append(f"| **{label}** | " + " | ".join(cells) + " |")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Tab 1: Sample Browser callbacks
# ---------------------------------------------------------------------------

def _resolve_dataset(label_or_name: str) -> str:
    """Resolve a dataset label (e.g. '1. Error Recognition (MCQA)') to internal name."""
    return DATASET_LABEL_TO_NAME.get(label_or_name, label_or_name)


def on_dataset_change(dataset_label: str, version_label: str = ""):
    """Update variant dropdown when dataset changes."""
    dataset = _resolve_dataset(dataset_label)
    ver = _resolve_version(version_label)
    variants = get_variants(dataset, version=ver)
    return gr.update(choices=["all"] + variants, value="all")


def on_apply_filters(dataset_label, variant, split, region, exercise_text, version_label=""):
    """Apply filters and return first sample + state."""
    dataset = _resolve_dataset(dataset_label)
    ver = _resolve_version(version_label)
    v = None if variant == "all" else variant
    s = None if split == "all" else split
    r = None if region == "all" else region

    filtered = filter_samples(dataset, variant=v, body_region=r,
                              exercise_search=exercise_text, split=s, version=ver)

    total_in_ds = sum(len(get_samples(dataset, sp, version=ver)) for sp in get_available_splits(dataset))
    count_text = f"**{len(filtered)}** of {total_in_ds} samples match"

    if not filtered:
        state = {"samples": [], "pos": 0}
        return state, count_text, "0 / 0", "No samples match these filters.", ""

    state = {"samples": filtered, "pos": 0}
    sample_md = format_sample_display(filtered[0])
    flags = compute_sample_flags(filtered[0])
    flags_md = "  ".join(f"`{f}`" for f in flags) if flags else ""
    pos_text = f"1 / {len(filtered)}"

    return state, count_text, pos_text, sample_md, flags_md


def _navigate(state, direction):
    """Navigate within filtered results."""
    samples = state.get("samples", [])
    if not samples:
        return state, "0 / 0", "No samples.", ""

    pos = state["pos"]
    if direction == "next":
        pos = min(pos + 1, len(samples) - 1)
    elif direction == "prev":
        pos = max(pos - 1, 0)
    elif direction == "random":
        pos = random.randint(0, len(samples) - 1)

    state["pos"] = pos
    sample = samples[pos]
    flags = compute_sample_flags(sample)
    flags_md = "  ".join(f"`{f}`" for f in flags) if flags else ""
    return state, f"{pos + 1} / {len(samples)}", format_sample_display(sample), flags_md


# ---------------------------------------------------------------------------
# Tab 2: Quality Dashboard
# ---------------------------------------------------------------------------

def compute_quality_dashboard():
    """Compute quality metrics. Returns (mcqa_md, answer_dist_fig, length_fig, issues_df, variant_md)."""

    mcqa_datasets = [n for n in NEW_DATASET_NAMES if n != "text_clinical_reasoning"]

    # --- MCQA summary ---
    mcqa_lines = [
        "**Balance** = min(A,B,C,D) / max(A,B,C,D). Perfect = 1.0. "
        "**Correct=Longest** = % where the correct choice is also the longest text (length bias risk).\n",
        "| Dataset | Samples | A | B | C | D | Balance | Correct=Longest | High Len Var |",
        "|---------|---------|---|---|---|---|---------|-----------------|--------------|",
    ]
    all_letter_data = {}

    for ds_name in mcqa_datasets:
        samples = get_all_samples(ds_name)
        letter_counts = Counter()
        correct_is_longest = 0
        high_var_count = 0
        total = 0

        for s in samples:
            ca = s.get("correct_answer", "")
            if ca:
                letter_counts[ca] += 1
                total += 1

            if "choices" in s:
                try:
                    choices = json.loads(s["choices"]) if isinstance(s["choices"], str) else s["choices"]
                    lengths = {k: len(v) for k, v in choices.items()}
                    if ca and lengths:
                        longest_letter = max(lengths, key=lengths.get)
                        if longest_letter == ca:
                            correct_is_longest += 1
                    vals = list(lengths.values())
                    if len(vals) >= 2 and min(vals) > 5 and max(vals) > 3 * min(vals):
                        high_var_count += 1
                except (json.JSONDecodeError, TypeError):
                    pass

        all_letter_data[ds_name] = letter_counts
        a, b, c, d = letter_counts.get("A", 0), letter_counts.get("B", 0), letter_counts.get("C", 0), letter_counts.get("D", 0)
        counts = [a, b, c, d]
        balance = round(min(counts) / max(max(counts), 1), 2)
        longest_pct = f"{100 * correct_is_longest / max(total, 1):.1f}%"
        var_pct = f"{high_var_count} ({100 * high_var_count / max(total, 1):.1f}%)"

        short_name = ds_name.replace("text_", "")
        mcqa_lines.append(f"| {short_name} | {total} | {a} | {b} | {c} | {d} | {balance} | {longest_pct} | {var_pct} |")

    mcqa_md = "\n".join(mcqa_lines)

    # --- Per-variant breakdown ---
    variant_lines = [
        "| Dataset | Variant | Samples | Train | Test |",
        "|---------|---------|---------|-------|------|",
    ]
    for ds_name in NEW_DATASET_NAMES:
        short = ds_name.replace("text_", "")
        for split in get_available_splits(ds_name):
            samples = get_samples(ds_name, split)
            for s in samples:
                v = s.get("variant", "?")
                s["_split"] = split

        all_s = get_all_samples(ds_name)
        variant_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"train": 0, "test": 0})
        for s in all_s:
            v = s.get("variant", "?")
            sp = s.get("_split", "?")
            variant_counts[v][sp] += 1

        for v in sorted(variant_counts):
            tr = variant_counts[v].get("train", 0)
            te = variant_counts[v].get("test", 0)
            variant_lines.append(f"| {short} | {v} | {tr + te} | {tr} | {te} |")

    variant_md = "\n".join(variant_lines)

    # --- Answer letter distribution chart ---
    fig_dist = go.Figure()
    for letter in ["A", "B", "C", "D"]:
        fig_dist.add_trace(go.Bar(
            name=letter,
            x=[ds.replace("text_", "") for ds in mcqa_datasets],
            y=[all_letter_data.get(ds, {}).get(letter, 0) for ds in mcqa_datasets],
        ))
    fig_dist.update_layout(
        title="MCQA Answer Letter Distribution",
        barmode="group",
        xaxis_title="Dataset",
        yaxis_title="Count",
        height=350,
        margin=dict(t=40, b=40),
    )

    # --- Answer length distribution (box plot, all datasets) ---
    fig_len = go.Figure()
    for ds_name in NEW_DATASET_NAMES:
        samples = get_all_samples(ds_name)
        for variant in sorted(set(s.get("variant", "?") for s in samples)):
            lengths = [len(extract_answer(s)) for s in samples if s.get("variant") == variant]
            if lengths:
                short_ds = ds_name.replace("text_", "")
                fig_len.add_trace(go.Box(
                    y=lengths,
                    name=f"{short_ds}/{variant}",
                    boxmean=True,
                ))
    fig_len.update_layout(
        title="Answer Length Distribution (chars)",
        yaxis_title="Answer length",
        height=400,
        margin=dict(t=40, b=80),
        showlegend=False,
    )

    # --- Issues scan ---
    issues = find_all_issues()

    return mcqa_md, fig_dist, fig_len, issues, variant_md


def find_all_issues() -> pd.DataFrame:
    """Scan all new datasets for quality issues."""
    rows = []
    for ds_name in NEW_DATASET_NAMES:
        for split in get_available_splits(ds_name):
            for i, sample in enumerate(get_samples(ds_name, split)):
                flags = compute_sample_flags(sample)
                if flags:
                    rows.append({
                        "dataset": ds_name.replace("text_", ""),
                        "split": split,
                        "index": i + 1,
                        "exercise": sample.get("exercise_name", ""),
                        "flags": ", ".join(flags),
                    })
    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame(columns=["dataset", "split", "index", "exercise", "flags"])


# ---------------------------------------------------------------------------
# Tab 3: Dataset Overview
# ---------------------------------------------------------------------------

def compute_overview():
    """Compute dataset overview. Returns (total_md, ds_md, ex_md, type_md, df, audit_md)."""
    rows = []
    grand_total = 0
    all_exercises = set()
    mcqa_count = 0
    openended_count = 0

    for ds_name in NEW_DATASET_NAMES:
        for split in get_available_splits(ds_name):
            samples = get_samples(ds_name, split)
            variant_counts = Counter(s.get("variant", "?") for s in samples)
            exercises = set(s.get("exercise_name", "") for s in samples)
            is_mcqa = any("correct_answer" in s for s in samples)

            for variant, count in sorted(variant_counts.items()):
                rows.append({
                    "dataset": ds_name.replace("text_", ""),
                    "split": split,
                    "variant": variant,
                    "samples": count,
                    "type": "MCQA" if is_mcqa else "Open-ended",
                })

            grand_total += len(samples)
            all_exercises.update(exercises)
            if is_mcqa:
                mcqa_count += len(samples)
            else:
                openended_count += len(samples)

    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["dataset", "split", "variant", "samples", "type"]
    )

    total_md = f"### {grand_total:,}\nTotal samples"
    ds_md = f"### {len(NEW_DATASET_NAMES)}\nDatasets"
    ex_md = f"### {len(all_exercises)}\nExercises"
    type_md = f"### {mcqa_count:,} / {openended_count:,}\nMCQA / Open-ended"
    audit_md = load_split_info()

    # Build a combined summary
    summary_lines = [
        "### Dataset Summary\n",
        "Each dataset tests a different aspect of physiotherapy knowledge. "
        "4 of 5 datasets are MCQA (multiple-choice), 1 is open-ended.\n",
    ]
    ds_descriptions = {
        "error_recognition": "Match movement error categories to their descriptions (and vice versa).",
        "phase_sequencing": "Identify which description belongs to a specific exercise phase.",
        "muscle_exercise": "Muscle-exercise associations, causal error analysis, and exercise similarity (uses PKG).",
        "clinical_reasoning": "Open-ended: clinical background, correct execution, patient-friendly explanation.",
        "error_correction": "Given observed errors, select the appropriate corrective technique tip.",
    }
    for ds_name in NEW_DATASET_NAMES:
        short = ds_name.replace("text_", "")
        desc = ds_descriptions.get(short, "")
        ds_samples = sum(
            r["samples"] for _, r in df.iterrows()
            if r["dataset"] == short
        ) if not df.empty else 0
        summary_lines.append(f"**{short}** ({ds_samples:,} samples): {desc}\n")

    summary_md = "\n".join(summary_lines)

    return total_md, ds_md, ex_md, type_md, df, audit_md, summary_md


def load_split_info() -> str:
    """Load and format the split_info.json for audit display."""
    path = NEW_DATASETS_DIR / "split_info.json"
    if not path.exists():
        return "No split_info.json found. Regenerate datasets to create it."

    info = json.loads(path.read_text())
    train_n = info.get("train_count", "?")
    test_n = info.get("test_count", "?")

    lines = [
        "### Train/Test Split Audit\n",
        "Exercises are split 80/20 by **base exercise name** (L/R pairs always stay in the same split), "
        "stratified by body region and exercise type to ensure balanced representation.\n",
        f"**{train_n} base exercises** in train, **{test_n} base exercises** in test "
        f"(~{round(100 * train_n / (train_n + test_n)) if isinstance(train_n, int) and isinstance(test_n, int) else '?'}% / "
        f"{round(100 * test_n / (train_n + test_n)) if isinstance(train_n, int) and isinstance(test_n, int) else '?'}%)\n",
    ]

    strat = info.get("stratification", {})
    if strat:
        lines.append("#### Stratification Balance\n")
        lines.append("Each stratum is a `body_region × exercise_type` combination. "
                      "Balanced strata ensure the model sees proportional examples of each type.\n")
        lines.append("| Stratum | Train | Test | Ratio |")
        lines.append("|---------|-------|------|-------|")
        all_strata = set(list(strat.get("train", {}).keys()) + list(strat.get("test", {}).keys()))
        for s in sorted(all_strata):
            train_c = strat.get("train", {}).get(s, 0)
            test_c = strat.get("test", {}).get(s, 0)
            total = train_c + test_c
            ratio = f"{round(100 * train_c / total)}:{round(100 * test_c / total)}" if total else "—"
            lines.append(f"| {s} | {train_c} | {test_c} | {ratio} |")
        lines.append("")

    assignments = info.get("split_assignments", {})
    if assignments:
        train_exs = sorted(k for k, v in assignments.items() if v == "train")
        test_exs = sorted(k for k, v in assignments.items() if v == "test")
        lines.append("<details><summary>Exercise Assignments "
                      f"(train: {len(train_exs)}, test: {len(test_exs)})</summary>\n")
        lines.append(f"**Train**: {', '.join(train_exs)}\n")
        lines.append(f"**Test**: {', '.join(test_exs)}")
        lines.append("</details>")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Evaluation Results helpers
# ---------------------------------------------------------------------------

MCQA_DATASET_NAMES = {
    "text_error_recognition", "text_phase_sequencing",
    "text_muscle_exercise", "text_error_correction",
}


def list_eval_results() -> List[Tuple[str, str]]:
    """Return (label, filepath) pairs for all eval JSON files."""
    if not EVAL_RESULTS_DIR.exists():
        return []
    items = []
    for p in sorted(EVAL_RESULTS_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        if p.name.startswith("report_"):
            continue
        try:
            with open(p) as f:
                meta = json.load(f).get("metadata", {})
            ds = meta.get("dataset", "?")
            tag = meta.get("tag", "")
            split = meta.get("split", "?")
            ts = meta.get("timestamp", "")[:10]
            label = f"{ds} | {tag} | {split} | {ts}"
        except Exception:
            label = p.stem
        items.append((label, str(p)))
    return items


def load_eval_result(filepath: str) -> Optional[Dict]:
    """Load a single eval result JSON."""
    if not filepath:
        return None
    try:
        with open(filepath) as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load eval result: {e}")
        return None


def format_eval_summary(results: Dict) -> str:
    """Build summary markdown from eval results."""
    meta = results.get("metadata", {})
    ov = results.get("overall", {})
    lines = [
        f"**Model**: `{meta.get('model', '?')}`  \n"
        f"**Tag**: {meta.get('tag', '—')}  |  "
        f"**Split**: {meta.get('split', '?')}  |  "
        f"**Samples**: {meta.get('samples_evaluated', '?')}  |  "
        f"**Inference**: {meta.get('inference_time_s', '?')}s",
        "",
    ]
    if "accuracy" in ov:
        lines.append(f"### Accuracy: **{ov['accuracy']}%** "
                      f"({ov['correct']}/{ov['total']})")
        if ov.get("unparsed"):
            lines.append(f"Unparsed responses: {ov['unparsed']} ({ov['unparsed_pct']}%)")
        delta = round(ov["accuracy"] - 25.0, 1)
        lines.append(f"Random baseline (25%): {'+'if delta>0 else ''}{delta}pp")
    elif "avg_rouge_l" in ov:
        lines.append(f"### ROUGE-L: **{ov['avg_rouge_l']}**")
    return "\n".join(lines)


def build_eval_breakdown(results: Dict) -> pd.DataFrame:
    """Build per-variant + per-region breakdown table."""
    rows = []
    for label, section in [("by_template", "Variant"), ("by_region", "Region")]:
        data = results.get(label, {})
        for name, d in sorted(data.items()):
            acc = d.get("accuracy", d.get("avg_rouge_l", "—"))
            rows.append({
                "Group": section,
                "Name": name,
                "Correct": d.get("correct", "—"),
                "Total": d.get("total", "—"),
                "Accuracy %": acc,
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def build_confusion_heatmap(results: Dict) -> Optional[go.Figure]:
    """Build a Plotly heatmap with count + percentage annotations."""
    cm = results.get("confusion_matrix", {})
    if not cm:
        return None
    # Use only A-D labels (skip UNPARSED for cleaner display)
    labels = sorted(k for k in cm.keys() if k != "UNPARSED")
    z = [[cm.get(gt, {}).get(pred, 0) for pred in labels] for gt in labels]
    # Build text annotations: count + percentage of row
    text = []
    for i, gt in enumerate(labels):
        row_total = sum(z[i])
        row_text = []
        for j in range(len(labels)):
            pct = (z[i][j] / row_total * 100) if row_total > 0 else 0
            row_text.append(f"{z[i][j]}<br>({pct:.0f}%)")
        text.append(row_text)
    fig = go.Figure(data=go.Heatmap(
        z=z, x=labels, y=labels, text=text,
        texttemplate="%{text}", textfont=dict(size=12),
        colorscale="Blues",
        hovertemplate="GT: %{y}<br>Pred: %{x}<br>Count: %{z}<extra></extra>",
    ))
    fig.update_layout(
        xaxis_title="Predicted", yaxis_title="Ground Truth",
        yaxis_autorange="reversed",
        width=450, height=420, margin=dict(l=60, r=20, t=30, b=60),
    )
    return fig


def load_all_eval_results() -> List[Dict]:
    """Load all eval JSONs and return a list of dicts with key fields."""
    if not EVAL_RESULTS_DIR.exists():
        return []
    all_results = []
    for p in sorted(EVAL_RESULTS_DIR.glob("*.json")):
        if p.name.startswith("report_") or p.name.startswith("SUMMARY"):
            continue
        try:
            with open(p) as f:
                data = json.load(f)
            meta = data.get("metadata", {})
            ov = data.get("overall", {})
            tag = meta.get("tag", "")
            # Parse version from tag: *-v1 or *-v2
            version = "v2" if "-v2" in tag else "v1"
            # Parse model short name
            if "qwen3.5-4b-thinking" in tag:
                model = "qwen3.5-4b-thinking"
            elif "qwen3-vl-4b" in tag:
                model = "qwen3-vl-4b"
            elif "qwen3.5-4b" in tag:
                model = "qwen3.5-4b"
            else:
                model = tag.split("-baseline")[0]
            ds = meta.get("dataset", "")
            is_mcqa = ds in MCQA_DATASET_NAMES
            all_results.append({
                "filepath": str(p),
                "dataset": ds,
                "model": model,
                "version": version,
                "tag": tag,
                "accuracy": ov.get("accuracy") if is_mcqa else None,
                "rouge_l": ov.get("avg_rouge_l") if not is_mcqa else None,
                "total": ov.get("total", 0),
                "is_mcqa": is_mcqa,
                "by_template": data.get("by_template", {}),
                "by_exercise": data.get("by_exercise", {}),
                "gt_distribution": data.get("gt_distribution", {}),
                "pred_distribution": data.get("pred_distribution", {}),
                "confusion_matrix": data.get("confusion_matrix", {}),
            })
        except Exception as e:
            logger.warning(f"Failed to load {p.name}: {e}")
    return all_results


def build_comparison_table(all_results: List[Dict]) -> pd.DataFrame:
    """Build a cross-run comparison DataFrame."""
    rows = []
    for r in all_results:
        ds_label = DATASET_NAME_TO_LABEL.get(r["dataset"], r["dataset"])
        metric = f"{r['accuracy']:.1f}%" if r["accuracy"] is not None else (
            f"{r['rouge_l']:.3f}" if r["rouge_l"] is not None else "—"
        )
        rows.append({
            "Dataset": ds_label,
            "Model": r["model"],
            "Version": r["version"],
            "Metric": metric,
            "N": r["total"],
            "Tag": r["tag"],
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def build_accuracy_comparison_chart(all_results: List[Dict]) -> Optional[go.Figure]:
    """Grouped bar chart: accuracy per dataset, grouped by model+version."""
    mcqa = [r for r in all_results if r["is_mcqa"] and r["accuracy"] is not None]
    if not mcqa:
        return None
    # Group by run_label (model + version)
    run_labels = sorted(set(f"{r['model']} {r['version']}" for r in mcqa))
    datasets = sorted(set(r["dataset"] for r in mcqa))
    ds_labels = [DATASET_NAME_TO_LABEL.get(d, d).split(". ", 1)[-1] if ". " in DATASET_NAME_TO_LABEL.get(d, d) else d for d in datasets]
    fig = go.Figure()
    colors = ["#3b82f6", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6", "#ec4899"]
    for i, run in enumerate(run_labels):
        accs = []
        for ds in datasets:
            match = [r for r in mcqa if f"{r['model']} {r['version']}" == run and r["dataset"] == ds]
            accs.append(match[0]["accuracy"] if match else 0)
        fig.add_trace(go.Bar(
            name=run, x=ds_labels, y=accs,
            text=[f"{a:.1f}%" for a in accs], textposition="outside",
            marker_color=colors[i % len(colors)],
        ))
    fig.add_hline(y=25, line_dash="dash", line_color="gray",
                  annotation_text="Random (25%)", annotation_position="top left")
    fig.update_layout(
        barmode="group", yaxis_title="Accuracy (%)",
        yaxis_range=[0, 105],
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=50, r=20, t=50, b=80), height=400,
    )
    return fig


def build_variant_accuracy_chart(all_results: List[Dict],
                                  dataset_filter: str = "") -> Optional[go.Figure]:
    """Per-variant accuracy chart for a specific dataset across all runs."""
    filtered = [r for r in all_results if r["is_mcqa"]
                and (not dataset_filter or r["dataset"] == dataset_filter)]
    if not filtered:
        return None
    run_labels = sorted(set(f"{r['model']} {r['version']}" for r in filtered))
    # Collect all variant names (exclude deprecated variants)
    _DEPRECATED_VARIANTS = {"causal_error", "phase_ordering"}
    all_variants = sorted(set(
        v for r in filtered for v in r["by_template"].keys()
        if v not in _DEPRECATED_VARIANTS
    ))
    if not all_variants:
        return None
    fig = go.Figure()
    colors = ["#3b82f6", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6", "#ec4899"]
    for i, run in enumerate(run_labels):
        matches = [r for r in filtered if f"{r['model']} {r['version']}" == run]
        accs = []
        for v in all_variants:
            # Aggregate across datasets if no filter
            total_correct, total_n = 0, 0
            for r in matches:
                vdata = r["by_template"].get(v, {})
                total_correct += vdata.get("correct", 0)
                total_n += vdata.get("total", 0)
            accs.append(round(total_correct / total_n * 100, 1) if total_n > 0 else 0)
        fig.add_trace(go.Bar(
            name=run, x=all_variants, y=accs,
            text=[f"{a:.1f}%" for a in accs], textposition="outside",
            marker_color=colors[i % len(colors)],
        ))
    fig.add_hline(y=25, line_dash="dash", line_color="gray")
    fig.update_layout(
        barmode="group", yaxis_title="Accuracy (%)",
        yaxis_range=[0, 105],
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=50, r=20, t=50, b=100), height=400,
        xaxis_tickangle=-30,
    )
    return fig


def build_distribution_chart(results: Dict) -> Optional[go.Figure]:
    """GT vs Predicted distribution grouped bar chart."""
    gt = results.get("gt_distribution", {})
    pred = results.get("pred_distribution", {})
    if not gt:
        return None
    labels = sorted(set(list(gt.keys()) + list(pred.keys())))
    # Remove UNPARSED for cleaner display
    labels = [l for l in labels if l != "UNPARSED"]
    gt_vals = [gt.get(l, 0) for l in labels]
    pred_vals = [pred.get(l, 0) for l in labels]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Ground Truth", x=labels, y=gt_vals,
                         marker_color="#3b82f6", text=gt_vals, textposition="outside"))
    fig.add_trace(go.Bar(name="Predicted", x=labels, y=pred_vals,
                         marker_color="#f59e0b", text=pred_vals, textposition="outside"))
    max_val = max(gt_vals + pred_vals) if gt_vals + pred_vals else 100
    fig.update_layout(
        barmode="group", yaxis_title="Count", xaxis_title="Answer",
        yaxis_range=[0, max_val * 1.15],
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=50, r=20, t=50, b=50), height=350, width=450,
    )
    return fig


def build_exercise_accuracy_chart(results: Dict) -> Optional[go.Figure]:
    """Horizontal bar chart of per-exercise accuracy, sorted ascending."""
    by_ex = results.get("by_exercise", {})
    if not by_ex:
        return None
    items = [(name, d.get("accuracy", 0), d.get("total", 0))
             for name, d in by_ex.items()]
    items.sort(key=lambda x: x[1])  # ascending
    names = [f"{it[0]} (n={it[2]})" for it in items]
    accs = [it[1] for it in items]
    fig = go.Figure(go.Bar(
        x=accs, y=names, orientation="h",
        text=[f"{a:.0f}%" for a in accs], textposition="outside",
        marker_color=["#ef4444" if a < 40 else "#f59e0b" if a < 60 else "#10b981"
                       for a in accs],
    ))
    fig.add_vline(x=25, line_dash="dash", line_color="gray")
    chart_height = max(350, len(items) * 22 + 80)
    fig.update_layout(
        xaxis_title="Accuracy (%)", xaxis_range=[0, 105],
        margin=dict(l=200, r=40, t=20, b=40), height=chart_height,
    )
    return fig


def filter_eval_samples(results: Dict, filter_mode: str) -> List[Dict]:
    """Filter detailed_results by correctness."""
    detailed = results.get("detailed_results", [])
    if filter_mode == "Incorrect Only":
        return [d for d in detailed if not d.get("is_correct", True)]
    elif filter_mode == "Correct Only":
        return [d for d in detailed if d.get("is_correct", False)]
    return detailed


def format_eval_sample(sample: Dict, is_mcqa: bool) -> str:
    """Format a single eval sample for display."""
    parts = []
    ex = sample.get("exercise_name", "")
    variant = sample.get("variant", "")
    region = sample.get("body_region", "")
    parts.append(f"**Exercise**: {ex}  |  **Variant**: {variant}  |  **Region**: {region}")
    parts.append("")

    # Question — strip inline choices (A. ... B. ...) if present since we display them separately
    question = sample.get("question", "")
    if is_mcqa:
        # Remove everything from "\nA. " or "\n\nA. " onwards
        for sep in ["\n\nA. ", "\nA. ", "\nA) "]:
            idx = question.find(sep)
            if idx > 0:
                question = question[:idx].strip()
                break
    parts.append(f"**Question**: {question}")
    parts.append("")

    if is_mcqa:
        gt = sample.get("ground_truth", "")
        pred = sample.get("predicted", "")
        choices = sample.get("choices", {})
        correct = sample.get("is_correct", False)

        for letter in ["A", "B", "C", "D"]:
            text = choices.get(letter, "")
            if not text:
                continue
            marker = ""
            if letter == gt and letter == pred:
                marker = " ✓"
            elif letter == gt:
                marker = " ✓ (GT)"
            elif letter == pred:
                marker = " ✗ (predicted)"
            parts.append(f"**{letter}.** {text}{marker}")
            parts.append("")  # blank line forces Markdown line break
        parts.append("")
        status = "✅ Correct" if correct else "❌ Incorrect"
        parts.append(f"**Result**: {status}  (GT: {gt}, Predicted: {pred})")
    else:
        # Open-ended
        rouge = sample.get("rouge_l", "—")
        parts.append(f"**ROUGE-L**: {rouge}")
        parts.append("")
        parts.append(f"**Ground Truth**:\n> {sample.get('ground_truth', '')}")
        parts.append("")
        parts.append(f"**Model Response**:\n> {sample.get('response', '')}")
        return "\n".join(parts)

    parts.append("")
    parts.append(f"**Model Response**:\n> {sample.get('response', '')}")
    return "\n".join(parts)


def _eval_nav(state: Dict, direction: str) -> Tuple[Dict, str, str]:
    """Navigate eval samples. Returns (state, position_text, sample_md)."""
    filtered = state.get("filtered", [])
    if not filtered:
        return state, "0 / 0", "No samples loaded."
    pos = state.get("pos", 0)
    if direction == "prev":
        pos = max(0, pos - 1)
    elif direction == "next":
        pos = min(len(filtered) - 1, pos + 1)
    state["pos"] = pos
    is_mcqa = state.get("is_mcqa", True)
    sample_md = format_eval_sample(filtered[pos], is_mcqa)
    return state, f"{pos + 1} / {len(filtered)}", sample_md


# ---------------------------------------------------------------------------
# Build Gradio app
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    """Build the Gradio monitoring app."""
    # Build labeled choices: numbered labels for new datasets, raw names for existing
    available_raw = get_available_datasets()
    ds_choices = []
    for name in available_raw:
        label = DATASET_NAME_TO_LABEL.get(name)
        if label:
            ds_choices.append(label)
        else:
            ds_choices.append(name)
    default_ds = ds_choices[0] if ds_choices else ""

    with gr.Blocks(title="Text SFT Dataset Monitor") as app:
        gr.Markdown("# Text SFT Dataset Monitor")

        with gr.Tabs():
            # ================================================================
            # Tab 1: Sample Browser
            # ================================================================
            with gr.Tab("Sample Browser"):
                with gr.Row():
                    # -- Sidebar --
                    with gr.Column(scale=1, min_width=220):
                        ds_dropdown = gr.Dropdown(
                            choices=ds_choices, value=default_ds,
                            label="Dataset", interactive=True,
                        )
                        version_radio = gr.Radio(
                            choices=DATASET_VERSIONS,
                            value=DATASET_VERSIONS[0],
                            label="Version",
                        )
                        variant_dropdown = gr.Dropdown(
                            choices=["all"], value="all",
                            label="Variant", interactive=True,
                        )
                        split_radio = gr.Radio(
                            choices=["all", "train", "test"], value="all",
                            label="Split",
                        )
                        region_radio = gr.Radio(
                            choices=["all", "lower_body", "upper_body"], value="all",
                            label="Body Region",
                        )
                        exercise_search = gr.Textbox(
                            label="Exercise (search)", placeholder="type to filter...",
                        )
                        apply_btn = gr.Button("Apply Filters", variant="primary")

                    # -- Main content --
                    with gr.Column(scale=3):
                        match_count = gr.Markdown("Apply filters to browse samples.")
                        with gr.Row():
                            prev_btn = gr.Button("< Prev", scale=1)
                            sample_position = gr.Markdown("0 / 0", elem_classes=["center"])
                            next_btn = gr.Button("Next >", scale=1)
                            random_btn = gr.Button("Random", scale=1)
                        quality_flags = gr.Markdown("")
                        sample_display = gr.Markdown("Select a dataset and click **Apply Filters** to start browsing.")

                browser_state = gr.State({"samples": [], "pos": 0})

                # Wiring
                ds_dropdown.change(on_dataset_change, [ds_dropdown, version_radio], [variant_dropdown])
                version_radio.change(on_dataset_change, [ds_dropdown, version_radio], [variant_dropdown])

                apply_btn.click(
                    on_apply_filters,
                    [ds_dropdown, variant_dropdown, split_radio, region_radio, exercise_search, version_radio],
                    [browser_state, match_count, sample_position, sample_display, quality_flags],
                )

                prev_btn.click(
                    lambda s: _navigate(s, "prev"), [browser_state],
                    [browser_state, sample_position, sample_display, quality_flags],
                )
                next_btn.click(
                    lambda s: _navigate(s, "next"), [browser_state],
                    [browser_state, sample_position, sample_display, quality_flags],
                )
                random_btn.click(
                    lambda s: _navigate(s, "random"), [browser_state],
                    [browser_state, sample_position, sample_display, quality_flags],
                )

            # ================================================================
            # Tab 2: Quality Dashboard
            # ================================================================
            with gr.Tab("Quality Dashboard"):
                quality_btn = gr.Button("Compute Quality Analysis", variant="primary")

                gr.Markdown("### Per-Variant Breakdown")
                variant_breakdown = gr.Markdown()

                gr.Markdown("### MCQA Answer Bias Check")
                mcqa_summary = gr.Markdown()
                with gr.Row():
                    answer_dist_plot = gr.Plot(label="Answer Distribution")
                    length_plot = gr.Plot(label="Answer Length Distribution")

                gr.Markdown("### Issues")
                issues_table = gr.Dataframe(interactive=False, wrap=True, max_height=400)

                quality_btn.click(
                    compute_quality_dashboard, [],
                    [mcqa_summary, answer_dist_plot, length_plot, issues_table, variant_breakdown],
                )

            # ================================================================
            # Tab 3: Evaluation Results
            # ================================================================
            with gr.Tab("Evaluation Results"):
                eval_choices = list_eval_results()
                eval_dd_choices = [label for label, _ in eval_choices]
                eval_path_map = {label: path for label, path in eval_choices}

                # --- Section 1: Cross-Run Overview ---
                gr.Markdown("## Cross-Run Comparison")
                eval_overview_btn = gr.Button("Load All Results", variant="primary")
                eval_overview_table = gr.Dataframe(
                    interactive=False, wrap=True, max_height=300,
                    label="All Evaluation Runs",
                )
                eval_accuracy_chart = gr.Plot(label="Accuracy by Dataset & Model")

                # Dataset filter for variant chart
                with gr.Row():
                    eval_ds_filter = gr.Dropdown(
                        choices=["(all MCQA)"] + sorted(MCQA_DATASET_NAMES),
                        value="(all MCQA)", label="Dataset filter (variant chart)",
                        interactive=True, scale=2,
                    )
                eval_variant_chart = gr.Plot(label="Accuracy by Variant")

                # Store all_results in state for variant chart filtering
                eval_all_state = gr.State([])

                gr.Markdown("---")

                # --- Section 2: Single-Result Inspector ---
                gr.Markdown("## Single Result Inspector")
                with gr.Row():
                    eval_dropdown = gr.Dropdown(
                        choices=eval_dd_choices,
                        value=eval_dd_choices[0] if eval_dd_choices else None,
                        label="Result File", interactive=True, scale=4,
                    )
                    eval_load_btn = gr.Button("Load", variant="primary", scale=1)
                    eval_refresh_btn = gr.Button("Refresh", scale=1)

                eval_summary_md = gr.Markdown("Select a result file and click **Load**.")

                gr.Markdown("### Breakdown (by variant / region)")
                eval_breakdown_df = gr.Dataframe(interactive=False, wrap=True, max_height=300)

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Confusion Matrix")
                        eval_cm_plot = gr.Plot()
                    with gr.Column():
                        gr.Markdown("### GT vs Predicted Distribution")
                        eval_dist_plot = gr.Plot()

                gr.Markdown("### Per-Exercise Accuracy")
                eval_exercise_plot = gr.Plot()

                gr.Markdown("### Sample Explorer")
                with gr.Row():
                    eval_filter_radio = gr.Radio(
                        choices=["All", "Incorrect Only", "Correct Only"],
                        value="All", label="Filter",
                    )
                with gr.Row():
                    eval_prev_btn = gr.Button("< Prev", scale=1)
                    eval_pos_md = gr.Markdown("0 / 0")
                    eval_next_btn = gr.Button("Next >", scale=1)
                eval_sample_md = gr.Markdown("Load results to browse samples.")

                eval_state = gr.State({"filtered": [], "pos": 0, "results": None, "is_mcqa": True})

                # --- Callbacks ---
                def on_load_overview():
                    all_res = load_all_eval_results()
                    table = build_comparison_table(all_res)
                    acc_chart = build_accuracy_comparison_chart(all_res)
                    var_chart = build_variant_accuracy_chart(all_res)
                    return all_res, table, acc_chart, var_chart

                def on_ds_filter_change(ds_filter, all_res):
                    ds_val = "" if ds_filter == "(all MCQA)" else ds_filter
                    return build_variant_accuracy_chart(all_res, dataset_filter=ds_val)

                def on_refresh_eval():
                    fresh = list_eval_results()
                    eval_path_map.clear()
                    eval_path_map.update({label: path for label, path in fresh})
                    labels = [label for label, _ in fresh]
                    return gr.update(choices=labels, value=labels[0] if labels else None)

                def on_load_eval(label, state):
                    filepath = eval_path_map.get(label, "")
                    results = load_eval_result(filepath)
                    if not results:
                        return (state, "Failed to load results.", pd.DataFrame(),
                                None, None, None, "0 / 0", "No data.", state)
                    summary = format_eval_summary(results)
                    breakdown = build_eval_breakdown(results)
                    cm_fig = build_confusion_heatmap(results)
                    dist_fig = build_distribution_chart(results)
                    exercise_fig = build_exercise_accuracy_chart(results)
                    ds = results.get("metadata", {}).get("dataset", "")
                    is_mcqa = ds in MCQA_DATASET_NAMES
                    filtered = results.get("detailed_results", [])
                    new_state = {"filtered": filtered, "pos": 0,
                                 "results": results, "is_mcqa": is_mcqa}
                    pos_text = f"1 / {len(filtered)}" if filtered else "0 / 0"
                    sample_md = format_eval_sample(filtered[0], is_mcqa) if filtered else "No samples."
                    return (new_state, summary, breakdown, cm_fig,
                            dist_fig, exercise_fig, pos_text, sample_md, new_state)

                def on_eval_filter(filter_mode, state):
                    results = state.get("results")
                    if not results:
                        return state, "0 / 0", "No data."
                    filtered = filter_eval_samples(results, filter_mode)
                    state["filtered"] = filtered
                    state["pos"] = 0
                    if not filtered:
                        return state, "0 / 0", "No matching samples."
                    sample_md = format_eval_sample(filtered[0], state.get("is_mcqa", True))
                    return state, f"1 / {len(filtered)}", sample_md

                eval_overview_btn.click(
                    on_load_overview, [],
                    [eval_all_state, eval_overview_table, eval_accuracy_chart, eval_variant_chart],
                )
                eval_ds_filter.change(
                    on_ds_filter_change, [eval_ds_filter, eval_all_state],
                    [eval_variant_chart],
                )
                eval_refresh_btn.click(
                    on_refresh_eval, [], [eval_dropdown],
                )
                eval_load_btn.click(
                    on_load_eval, [eval_dropdown, eval_state],
                    [eval_state, eval_summary_md, eval_breakdown_df, eval_cm_plot,
                     eval_dist_plot, eval_exercise_plot,
                     eval_pos_md, eval_sample_md, eval_state],
                )
                eval_filter_radio.change(
                    on_eval_filter, [eval_filter_radio, eval_state],
                    [eval_state, eval_pos_md, eval_sample_md],
                )
                eval_prev_btn.click(
                    lambda s: _eval_nav(s, "prev"), [eval_state],
                    [eval_state, eval_pos_md, eval_sample_md],
                )
                eval_next_btn.click(
                    lambda s: _eval_nav(s, "next"), [eval_state],
                    [eval_state, eval_pos_md, eval_sample_md],
                )

            # ================================================================
            # Tab 4: Dataset Overview
            # ================================================================
            with gr.Tab("Dataset Overview"):
                overview_btn = gr.Button("Load Overview", variant="primary")

                with gr.Row():
                    total_card = gr.Markdown()
                    datasets_card = gr.Markdown()
                    exercises_card = gr.Markdown()
                    type_card = gr.Markdown()

                summary_md = gr.Markdown()
                overview_table = gr.Dataframe(interactive=False, wrap=True)

                gr.Markdown("---")
                split_audit_md = gr.Markdown()

                overview_btn.click(
                    compute_overview, [],
                    [total_card, datasets_card, exercises_card, type_card,
                     overview_table, split_audit_md, summary_md],
                )

            # ================================================================
            # Tab 5: Methodology
            # ================================================================
            with gr.Tab("Methodology"):
                gr.Markdown(METHODOLOGY_MD)

    return app


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Text SFT Dataset Monitoring App")
    parser.add_argument("--port", type=int, default=7866, help="Port (default: 7866)")
    parser.add_argument("--share", action="store_true", help="Create public link")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
