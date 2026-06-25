#!/usr/bin/env python3
"""
Cleanup script to keep only the best-performing checkpoint per training run.

This script:
1. Parses evaluation results to identify best checkpoints
2. Removes intermediate checkpoints from training runs
3. Keeps only the final/best checkpoint per run
4. Optionally removes corresponding HuggingFace exports in /mnt/data/sgsilva/models

Usage:
    # Dry run (preview what will be deleted)
    python cleanup_checkpoints.py --dry-run

    # Actually delete files
    python cleanup_checkpoints.py

    # Also cleanup /mnt/data/sgsilva/models directory
    python cleanup_checkpoints.py --cleanup-models
"""

import argparse
import csv
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


# Fallback mapping of training runs to their best checkpoints
# Used if auto-detection from evaluation reports fails
# Last manually updated: 2026-02-12
FALLBACK_BEST_CHECKPOINTS = {
    # === Task 1: Keypoint Coordinate Prediction ===
    "sft_vlm_megatron_4b_4epochs": "step_1292",  # Original baseline, keep final
    "sft_vlm_megatron_4b_4epochs_task1_original": "step_648",  # task1_original_v1 → step648 (Epoch 2)
    "sft_vlm_megatron_4b_4epochs_task1b_cropped": "step_320",  # task1b_cropped_v1 → step320 (Epoch 0)
    "sft_vlm_megatron_4b_4epochs_task1c_cropped": "step_630",  # task1c_cropped_v1 → step630 (Epoch 1, F1=97.7%)
    "sft_vlm_4b_4epochs_task1_cropped_v2": "step_1292",  # FAILED: mode collapse, keep final for reference
    "sft_vlm_megatron_4b_4epochs_mixed_balanced_v1": "step_1260",  # FAILED: learned format not precision

    # === Task 2: Keypoint Labeling ===
    "sft_vlm_megatron_4b_4epochs_task2": "step_969",  # task2_v2 → step969 (Epoch 3)
    "sft_vlm_megatron_4b_4epochs_task2_v4": "step_1328",  # task2_v4 → step1328 (Epoch 4, 33.4%)
    "sft_vlm_megatron_4b_4epochs_task2_v5": None,  # Eval pending

    # === Task 3: Error Detection ===
    "sft_vlm_megatron_4b_4epochs_task3a_high": "step_646",  # task3a → step646 (Epoch 2, F1=19.3%)
    "sft_vlm_megatron_4b_4epochs_task3b_low_missing": "step_338",  # Retrained with Title Case, eval pending
    "sft_vlm_megatron_4b_4epochs_task3c_background_displacement": "step_338",  # Retrained with Title Case, eval pending
    "sft_vlm_megatron_4b_4epochs_task3c_small_displacement": "step_338",  # Retrained with Title Case, eval pending
    "sft_vlm_megatron_4b_4epochs_task3d_mixed": None,  # New, eval pending

    # === Task 4: MCQA ===
    "sft_vlm_4b_4epochs_task4_mcqa": "step_1352",  # v1 MCQA → step1352
    "sft_vlm_4b_4epochs_task4_mcqa_v3": "step_656",  # v3 → keep final
    "sft_vlm_4b_4epochs_task4_mcqa_v5": "step_432",  # v5 → keep final
    "sft_vlm_4b_4epochs_task4_mcqa_v5.1": "step_1428",  # v5.1 → keep final
    "sft_vlm_4b_4epochs_task4_mcqa_v5.3": None,  # Eval pending
    "sft_vlm_4b_4epochs_task4_mcqa_v6.1.2": None,  # Eval pending
    "sft_vlm_4b_4epochs_task4_mcqa_v6.2": None,  # Eval pending (step730 best so far: 75.2%)
}

# Explicit best-step mapping for exported model directory prefixes in /mnt/data/sgsilva/models.
# This is especially useful for mixed-task runs whose evaluation artifacts do not
# map cleanly back to a single training run name.
MODEL_PREFIX_BEST_STEPS = {
    # 4B mixed runs
    "mixed_balanced_v2": "976",

    # Qwen mixed runs with checkpoint-comparison reports
    "qwen3-vl-4b-4epochs-mixed-final-a": "1125",
    "qwen3-vl-4b-4epochs-mixed-final-b": "1124",

    # Qwen mixed runs inferred from the available per-task final reports
    "qwen3-vl-4b-4epochs-mixed-v3": "4500",
}

CANONICAL_EVALS_ROOT_CANDIDATES = (
    Path("/mnt/data/sgsilva/vlm-post-training/aux_tasks/evals"),
    Path("/home/sgsilva/vlm-post-training/aux_tasks/evals"),
)

# --- NEW-ERA eval board (the `eval-vlm` skill / eval_all.sh + compile_eval_results.py) ---
# The legacy RUN_METADATA/eval_matrix tree above is retired (empty on disk as of 2026-06).
# Best-checkpoint truth now lives in the curated master board: each evaluated checkpoint
# appears (a) as a `Model Path` row in eval_master*.csv and (b) as a step-bearing `pattern`
# in master_models.json. A step named in EITHER source is a KEEPER for its run.
MASTER_CSV_DIR = Path("/mnt/data/sgsilva/results/master")
MASTER_CSV_FILES = ("eval_master_27b.csv", "eval_master_4b.csv", "eval_master_other.csv", "eval_master.csv")
MASTER_MODELS_JSON_CANDIDATES = (
    Path("/home/sgsilva/utilities/eval/master_models.json"),
    Path("/mnt/data/sgsilva/utilities/eval/master_models.json"),
)
MASTER_CSV_MODEL_PATH_COL = 1  # 0-based: row[1] == "Model Path" (after the leading display col)
TASK_GROUP_PRIMARY_DATASET = {
    "text_qa_postsession_2403": "patient_qa_mcqa_postsession_2403",
    "text_qa_patient_2403": "patient_qa_mcqa_2403",
}


def _extract_step_num(value: str | None) -> Optional[str]:
    if not value:
        return None
    match = re.search(r"step[_-]?(\d+)", value, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def _normalize_checkpoint_step(step_num: str | None) -> Optional[str]:
    return f"step_{step_num}" if step_num else None


def _model_basename_to_prefix(model_name: str) -> Optional[str]:
    step_num = _extract_step_num(model_name)
    if not step_num:
        return None
    return re.sub(r"[-_]?step[_-]?\d+$", "", model_name)


def _export_prefix_to_run_name(prefix: str) -> str:
    if prefix.startswith(("qwen35-", "qwen3-vl-")):
        prefix = "sft_" + prefix
    return prefix.replace("-", "_")


def _prefix_is_specific(prefix: str) -> bool:
    """
    True if an export prefix is specific enough to safely token-match a run dir.

    Keeps a prefix iff it carries a model-size token (e.g. '27b'/'4b') OR has
    >= 3 content tokens (after dropping a trailing 'sft'). Drops bare generic
    stems like 'sft' / 'grpo' that would otherwise wrongly match a real run dir.
    """
    if re.search(r"\d+b([-_]|$)", prefix, flags=re.IGNORECASE):
        return True
    tokens = [t for t in re.split(r"[-_.]", prefix) if t and t.lower() != "sft"]
    return len(tokens) >= 3


def _resolve_canonical_evals_root() -> Optional[Path]:
    for candidate in CANONICAL_EVALS_ROOT_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def _is_truthy_best_flag(value: object) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "best"}


def _load_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def detect_best_checkpoints_from_canonical_evals(
    evals_root: Optional[Path] = None,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Detect best checkpoints from the canonical eval tree.

    Returns:
      - run_best: training-run-name -> step_N
      - prefix_best: exported-model-prefix -> step number string
    """
    run_best: Dict[str, Tuple[float, float, str]] = {}
    prefix_best: Dict[str, Tuple[float, float, str]] = {}

    if evals_root is None:
        evals_root = _resolve_canonical_evals_root()

    if evals_root is None or not evals_root.exists():
        return {}, {}

    for metadata_path in evals_root.rglob("RUN_METADATA.json"):
        meta = _load_json(metadata_path)
        if not meta:
            continue

        eval_family = str(meta.get("eval_family", ""))
        if not eval_family.startswith("sft_"):
            continue

        model_ref = str(meta.get("model", ""))
        model_name = Path(model_ref).name
        step_num = _extract_step_num(model_name) or _extract_step_num(str(meta.get("run_dir", "")))
        if not step_num:
            continue

        export_prefix = _model_basename_to_prefix(model_name)
        if not export_prefix:
            continue

        run_name = _export_prefix_to_run_name(export_prefix)
        task_group = str(meta.get("task_group", ""))
        primary_dataset = TASK_GROUP_PRIMARY_DATASET.get(task_group)

        overall_correct = 0
        overall_total = 0
        primary_accuracy: Optional[float] = None

        for ds in meta.get("datasets", []):
            ds_name = ds.get("dataset")
            results_json = ds.get("results_json")
            if not ds_name or not results_json:
                continue
            payload = _load_json(Path(results_json))
            if not payload:
                continue
            overall = payload.get("overall", {})
            correct = overall.get("correct")
            total = overall.get("total")
            accuracy = overall.get("accuracy")
            if isinstance(correct, int) and isinstance(total, int):
                overall_correct += correct
                overall_total += total
            if ds_name == primary_dataset and isinstance(accuracy, (int, float)):
                primary_accuracy = float(accuracy)

        overall_accuracy = round(overall_correct / overall_total * 100, 1) if overall_total else -1.0
        primary_score = primary_accuracy if primary_accuracy is not None else overall_accuracy

        prev = run_best.get(run_name)
        if prev is None or (primary_score, overall_accuracy, int(step_num)) > (
            prev[0],
            prev[1],
            int(_extract_step_num(prev[2]) or 0),
        ):
            run_best[run_name] = (primary_score, overall_accuracy, _normalize_checkpoint_step(step_num))

        prev_prefix = prefix_best.get(export_prefix)
        if prev_prefix is None or (primary_score, overall_accuracy, int(step_num)) > (
            prev_prefix[0],
            prev_prefix[1],
            int(prev_prefix[2]),
        ):
            prefix_best[export_prefix] = (primary_score, overall_accuracy, step_num)

    return (
        {run_name: best_step for run_name, (_, _, best_step) in run_best.items()},
        {prefix: step for prefix, (_, _, step) in prefix_best.items()},
    )


def detect_best_checkpoints_from_eval_matrix(
    evals_root: Optional[Path] = None,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Detect best checkpoints from eval_matrix.csv rows marked as best.

    Returns:
      - run_best: training-run-name -> step_N
      - prefix_best: exported-model-prefix -> step number string
    """
    if evals_root is None:
        evals_root = _resolve_canonical_evals_root()

    if evals_root is None:
        return {}, {}

    eval_matrix_path = evals_root / "eval_matrix.csv"
    if not eval_matrix_path.exists():
        return {}, {}

    run_best: Dict[str, str] = {}
    prefix_best: Dict[str, str] = {}

    try:
        with eval_matrix_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if not _is_truthy_best_flag(row.get("best_step")):
                    continue

                model_ref = str(row.get("model", "")).strip()
                run_id = str(row.get("run_id", "")).strip()
                step_num = _extract_step_num(model_ref) or _extract_step_num(run_id)
                if not step_num:
                    continue

                export_prefix = _model_basename_to_prefix(Path(model_ref).name)
                if not export_prefix:
                    continue

                run_name = _export_prefix_to_run_name(export_prefix)
                run_best[run_name] = _normalize_checkpoint_step(step_num)
                prefix_best[export_prefix] = step_num
    except Exception:
        return {}, {}

    return run_best, prefix_best


def parse_best_checkpoint_from_report(report_path: Path) -> Optional[tuple]:
    """
    Parse evaluation report to find the best checkpoint.
    Returns: (task_name, step_number) or None
    """
    try:
        content = report_path.read_text()

        # Extract task name from filename
        # Format: checkpoint_comparison_4b_task1b_cropped_v1.txt
        match = re.search(r'checkpoint_comparison_\w+_(task\w+.*?)\.txt', report_path.name)
        if not match:
            return None

        task_name = match.group(1)

        # Find the line with ⭐ BEST marker
        for line in content.split('\n'):
            if '⭐ BEST' in line or '⭐ Best' in line:
                # Look for the step number in previous lines
                lines = content.split('\n')
                idx = lines.index(line)

                # Search backwards for step number
                for i in range(idx, max(0, idx-5), -1):
                    step_match = re.search(r'step_?(\d+)', lines[i], re.IGNORECASE)
                    if step_match:
                        step_num = step_match.group(1)
                        return (task_name, f"step_{step_num}")

        return None

    except Exception as e:
        # Silently fail - will use fallback
        return None


def auto_detect_best_checkpoints(evaluations_dir: str = "vlm-evaluation/results/evaluations") -> Dict[str, str]:
    """
    Automatically detect best checkpoints from evaluation reports.
    Returns dict mapping training run names to best checkpoint names.
    """
    eval_path = Path(evaluations_dir)
    if not eval_path.exists():
        return {}

    best_checkpoints = {}

    # Task name to training run name mapping
    task_to_run = {
        # Task 1
        'task1_cropped_v1': 'sft_vlm_megatron_4b_4epochs',
        'task1_original_v1': 'sft_vlm_megatron_4b_4epochs_task1_original',
        'task1b_cropped_v1': 'sft_vlm_megatron_4b_4epochs_task1b_cropped',
        'task1c_cropped_v1': 'sft_vlm_megatron_4b_4epochs_task1c_cropped',
        'task1_cropped_v2': 'sft_vlm_4b_4epochs_task1_cropped_v2',
        # Task 2
        'task2_visualized_cropped_v2': 'sft_vlm_megatron_4b_4epochs_task2',
        'task2_visualized_cropped_v4': 'sft_vlm_megatron_4b_4epochs_task2_v4',
        'task2_visualized_v5': 'sft_vlm_megatron_4b_4epochs_task2_v5',
        # Task 3
        'task3a_v1_high_error': 'sft_vlm_megatron_4b_4epochs_task3a_high',
        'task3b_v1_low_missing': 'sft_vlm_megatron_4b_4epochs_task3b_low_missing',
        'task3c_v1_background_displacement': 'sft_vlm_megatron_4b_4epochs_task3c_background_displacement',
        'task3c_v1_small_displacement': 'sft_vlm_megatron_4b_4epochs_task3c_small_displacement',
        'task3d_v1_mixed': 'sft_vlm_megatron_4b_4epochs_task3d_mixed',
        # Task 4 MCQA
        'task4_mcqa_v5.3': 'sft_vlm_4b_4epochs_task4_mcqa_v5.3',
        'task4_mcqa_v6.1.2': 'sft_vlm_4b_4epochs_task4_mcqa_v6.1.2',
        'task4_mcqa_v6.2': 'sft_vlm_4b_4epochs_task4_mcqa_v6.2',
    }

    # Scan all comparison reports
    for report_file in eval_path.glob("checkpoint_comparison_*.txt"):
        result = parse_best_checkpoint_from_report(report_file)
        if result:
            task_name, best_step = result

            # Map to training run name
            if task_name in task_to_run:
                run_name = task_to_run[task_name]
                best_checkpoints[run_name] = best_step

    return best_checkpoints


def _steps_from_master_models_json() -> Set[Tuple[str, str]]:
    """
    Extract (export_prefix, step_num) pairs from master_models.json patterns that name a step,
    e.g. 'qwen35-27b-oracle-obs-merged-1805-binary-aux12k-union-sft-step4127'
         -> ('qwen35-27b-oracle-obs-merged-1805-binary-aux12k-union-sft', '4127').
    A path-style pattern ('.../step_768') is handled too via the generic basename.
    """
    pairs: Set[Tuple[str, str]] = set()
    for cand in MASTER_MODELS_JSON_CANDIDATES:
        if not cand.exists():
            continue
        data = _load_json(cand)
        if not data:
            continue
        for entry in data.get("models", []):
            pattern = str(entry.get("pattern", ""))
            name = Path(pattern).name  # tolerate full paths
            step_num = _extract_step_num(name)
            if not step_num:
                continue
            prefix = _model_basename_to_prefix(name)
            if prefix:
                pairs.add((prefix, step_num))
        break  # first existing file wins
    return pairs


def _steps_from_master_csvs() -> Set[Tuple[str, str]]:
    """
    Extract (export_prefix, step_num) from the `Model Path` column of every eval_master*.csv.
    Each board row's Model Path points at the exact evaluated/kept checkpoint dir.
    """
    pairs: Set[Tuple[str, str]] = set()
    for fname in MASTER_CSV_FILES:
        path = MASTER_CSV_DIR / fname
        if not path.exists():
            continue
        try:
            with path.open(newline="") as handle:
                for row in csv.reader(handle):
                    if len(row) <= MASTER_CSV_MODEL_PATH_COL:
                        continue
                    model_path = row[MASTER_CSV_MODEL_PATH_COL].strip()
                    if not model_path or model_path == "Model Path":
                        continue
                    name = Path(model_path).name
                    step_num = _extract_step_num(name)
                    if not step_num:
                        continue
                    prefix = _model_basename_to_prefix(name)
                    if prefix:
                        pairs.add((prefix, step_num))
        except Exception:
            continue
    return pairs


def detect_best_checkpoints_from_master_board() -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    NEW-ERA detector: best checkpoints from the curated master board
    (eval_master*.csv + master_models.json). Returns (run_best, prefix_best).

    A checkpoint named in either source is the keeper for its run. If a run somehow has
    >1 board step, the highest step wins (boards list the chosen/final eval target).
    """
    pairs = _steps_from_master_csvs() | _steps_from_master_models_json()

    prefix_best: Dict[str, int] = {}
    for prefix, step_num in pairs:
        # Guard against junk prefixes that collapse to a generic stem (e.g. a pmartins
        # '.../sft_.../step_2812/hf' path yielding bare 'sft', or 'grpo'). Such a short,
        # non-specific key would falsely token-match a real run dir in _resolve_run_key
        # and assign it the WRONG best step.
        #
        # A prefix is KEPT iff it is specific enough to be unambiguous:
        #   - it carries a model-size token (e.g. '27b'/'4b'), OR
        #   - it has >= 3 content tokens (after dropping a trailing 'sft').
        # This drops the bare 'sft'/'grpo' stems while preserving legitimate
        # size-less stems that only appear in master_models.json (e.g.
        # 'oracle-obs-merged-1805-binary-aux12k-union-reasoning-sft' -> step4125,
        # which is NOT present in any eval_master*.csv).
        if not _prefix_is_specific(prefix):
            continue
        n = int(step_num)
        if prefix not in prefix_best or n > prefix_best[prefix]:
            prefix_best[prefix] = n

    run_best = {
        _export_prefix_to_run_name(prefix): _normalize_checkpoint_step(str(n))
        for prefix, n in prefix_best.items()
    }
    prefix_best_str = {prefix: str(n) for prefix, n in prefix_best.items()}
    return run_best, prefix_best_str


def get_best_checkpoints() -> Dict[str, str]:
    """
    Get best checkpoints, trying auto-detection first, falling back to hardcoded values.

    Detection precedence (later overrides earlier):
      1. FALLBACK_BEST_CHECKPOINTS (hand-maintained, oldest)
      2. legacy report parsing + retired canonical eval tree + eval_matrix.csv
      3. NEW-ERA master board (eval_master*.csv + master_models.json) — authoritative
    """
    canonical_run_best, canonical_prefix_best = detect_best_checkpoints_from_canonical_evals()
    matrix_run_best, matrix_prefix_best = detect_best_checkpoints_from_eval_matrix()
    master_run_best, master_prefix_best = detect_best_checkpoints_from_master_board()

    # Try auto-detection from legacy evaluation reports
    auto_detected = auto_detect_best_checkpoints()

    result = FALLBACK_BEST_CHECKPOINTS.copy()
    result.update(auto_detected)
    result.update(canonical_run_best)
    result.update(matrix_run_best)
    result.update(master_run_best)  # new-era board wins

    if master_run_best:
        print("✓ Auto-detected best checkpoints from NEW-ERA master board (eval_master*.csv + master_models.json)")
        print(f"  Found {len(master_run_best)} run(s)")
    if canonical_run_best:
        print("✓ Auto-detected best checkpoints from canonical eval tree")
        print(f"  Found {len(canonical_run_best)} run(s)")
    if matrix_run_best:
        print("✓ Auto-detected best checkpoints from eval_matrix.csv")
        print(f"  Found {len(matrix_run_best)} run(s)")
    if auto_detected:
        print("✓ Auto-detected best checkpoints from legacy evaluation reports")
        print(f"  Found {len(auto_detected)} checkpoint(s)")
    if not master_run_best and not canonical_run_best and not matrix_run_best and not auto_detected:
        print("⚠️  Auto-detection failed, using fallback values")

    # Store prefix best steps as an attribute for model cleanup (new-era board wins).
    get_best_checkpoints.prefix_best_steps = {
        **MODEL_PREFIX_BEST_STEPS,
        **canonical_prefix_best,
        **matrix_prefix_best,
        **master_prefix_best,
    }
    return result


def get_checkpoint_dirs(results_dir: Path) -> Dict[str, List[Path]]:
    """Get all checkpoint directories for each training run."""
    checkpoints = {}

    for run_dir in sorted(results_dir.iterdir() if results_dir.exists() else []):
        if not run_dir.is_dir():
            continue

        run_name = run_dir.name
        step_dirs = sorted(
            [d for d in run_dir.iterdir() if d.is_dir() and re.match(r"(tmp_)?step_\d+$", d.name)]
        )

        if step_dirs:
            checkpoints[run_name] = step_dirs

    return checkpoints


def _dir_size_bytes(path: Path) -> int:
    """
    Return directory size in bytes.

    Uses `du -sb` when available for speed, falling back to Python traversal.
    """
    try:
        proc = subprocess.run(
            ["du", "-sb", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
        # Format: "<bytes>\t<path>"
        return int(proc.stdout.split()[0])
    except Exception:
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


_FRESH_SUFFIX_RE = re.compile(r"__fresh_\d{8}_\d{6}$")
_TIMESTAMP_SUFFIX_RE = re.compile(r"__\d{8}_\d{6}$")


def _canonicalize_run_name(run_name: str) -> str:
    """
    Normalize run directory names that include timestamp suffixes.

    Examples:
      - foo__fresh_20260408_112044 -> foo
      - foo__20260408_112044 -> foo
    """
    run_name = _FRESH_SUFFIX_RE.sub("", run_name)
    run_name = _TIMESTAMP_SUFFIX_RE.sub("", run_name)
    return run_name


# Token aliases applied before set comparison so on-disk run dirs and board keys
# that spell the same concept differently still match (e.g. checkpoint dir uses
# 'mix_reas_12k' while the board export prefix uses 'mix-reasoning-12k').
_RUN_TOKEN_ALIASES = {"reas": "reasoning"}

# Pure-noise tokens carrying no run identity. 'sft' marks the family (every run
# here is an SFT run) and appears inconsistently — as a LEADING token on disk
# (sft_qwen35_27b_...) and a TRAILING token on board keys (..._union_sft) — so it
# must be dropped from BOTH ends, not just stripped once. The qwen family tokens
# are likewise dropped because master_models.json entries omit them while disk
# dirs and CSV entries include them. Model-SIZE tokens ('27b'/'4b') are NOT noise
# (they distinguish the 27b vs 4b '..._mix_12k_1506' runs) and are handled
# separately via a size-compatibility guard.
_RUN_NOISE_TOKENS = {"sft", "qwen35", "qwen3.5", "qwen3", "vl"}
_SIZE_TOKEN_RE = re.compile(r"\d+b", flags=re.IGNORECASE)


def _run_content_tokens(name: str) -> frozenset:
    """
    Content-token set for a run name / board key, used for SAFE matching.

    canonicalize (strip timestamps) -> alias (reas->reasoning) -> drop pure-noise
    family/sft tokens AND model-size tokens. Size is compared separately by
    _run_size_token so a 4b run can never match a 27b board key.
    """
    name = _canonicalize_run_name(name)
    tokens = (_RUN_TOKEN_ALIASES.get(t, t) for t in name.split("_"))
    return frozenset(
        t for t in tokens if t and t not in _RUN_NOISE_TOKENS and not _SIZE_TOKEN_RE.fullmatch(t)
    )


def _run_size_token(name: str) -> Optional[str]:
    """Return the model-size token ('27b'/'4b') if the name carries one, else None."""
    for t in _canonicalize_run_name(name).split("_"):
        if _SIZE_TOKEN_RE.fullmatch(t):
            return t.lower()
    return None


def _resolve_run_key(run_name: str, best_checkpoints: Dict[str, str]) -> Optional[str]:
    """
    Resolve a run_name to a key present in best_checkpoints — SAFELY.

    Strategy:
      1) Exact match.
      2) Canonicalized name match (strip known timestamp suffixes).
      3) Strict content-token-set EQUALITY, with a model-size guard:
         a board key matches iff its content tokens equal the run's content tokens
         AND their size tokens are compatible (equal, or one side omits size).
         If the matching keys all point at the SAME best step, return one of them;
         if they disagree (genuine ambiguity) or none match, return None.

    A None result is the SAFE outcome: the caller treats an unresolved run as
    "unknown — keep all checkpoints", so we never delete a sibling of a step we
    could not positively identify.

    The old longest-prefix `startswith` fallback was REMOVED: it wrongly matched
      'sft_qwen35_27b_oracle_obs_merged_1805_binary_aux12k_union'
    against the shorter, DIFFERENT key
      'sft_qwen35_27b_oracle_obs_merged_1805' (step_2558).
    Subset/superset matching was also rejected: a run carrying an EXTRA
    identity token the key lacks (e.g. '...union_reasoning' vs '...union') is a
    DIFFERENT run, so partial-containment must never resolve. Only full content
    equality is safe.
    """
    if run_name in best_checkpoints:
        return run_name

    canon = _canonicalize_run_name(run_name)
    if canon in best_checkpoints:
        return canon

    run_tokens = _run_content_tokens(run_name)
    if not run_tokens:
        return None
    run_size = _run_size_token(run_name)

    equal = []
    for key in best_checkpoints:
        if _run_content_tokens(key) != run_tokens:
            continue
        key_size = _run_size_token(key)
        if run_size is not None and key_size is not None and run_size != key_size:
            continue  # 27b run can't claim a 4b board key (or vice-versa)
        equal.append(key)

    if not equal:
        return None

    # Equal-content keys may be duplicates of the same run spelled two ways (one
    # with the qwen35-27b prefix, one without). Accept only if they agree on the
    # best step; otherwise the board is genuinely ambiguous -> None (safe).
    steps = {best_checkpoints.get(k) for k in equal}
    if len(steps) == 1:
        return equal[0]
    return None


def calculate_space_to_free(checkpoints: Dict[str, List[Path]], best_checkpoints: Dict[str, str]) -> int:
    """Calculate total space that will be freed."""
    total_size = 0

    for run_name, step_dirs in checkpoints.items():
        run_key = _resolve_run_key(run_name, best_checkpoints)
        best_step = best_checkpoints.get(run_key) if run_key else None

        for step_dir in step_dirs:
            step_name = step_dir.name

            # Skip if this is the best checkpoint
            if best_step and step_name == best_step:
                continue

            # Calculate size
            size = _dir_size_bytes(step_dir)
            total_size += size

    return total_size


def find_exported_model_for_keeper(
    run_name: str,
    keeper_step: str,
    models_dir: Path,
) -> Optional[Path]:
    """
    Return the exported HF model dir for a run's KEEPER step, or None if absent.

    A checkpoint step may only be deleted once we have positively confirmed its
    keeper survived as an HF export under models_dir (Sandra's hard rule:
    'never delete a checkpoint step unless its HF export is verified present').

    Matching is robust to prefix-spelling differences (export prefixes may or may
    not carry the 'qwen35-27b-' family, and use '-step<N>' vs '_step_<N>'):
    a models-dir entry qualifies iff
      - its embedded step number == the keeper's step number, AND
      - its prefix's content tokens equal the run's content tokens (same logic as
        _resolve_run_key), with a compatible model-size token.
    """
    keeper_num = _extract_step_num(keeper_step)
    if not keeper_num:
        return None
    if not models_dir.exists():
        return None

    run_tokens = _run_content_tokens(run_name)
    run_size = _run_size_token(run_name)

    for model_dir in sorted(models_dir.glob("*")):
        if not model_dir.is_dir():
            continue
        name = model_dir.name
        if _extract_step_num(name) != keeper_num:
            continue
        prefix = _model_basename_to_prefix(name)
        if not prefix:
            continue
        prefix_run = _export_prefix_to_run_name(prefix)
        if _run_content_tokens(prefix_run) != run_tokens:
            continue
        prefix_size = _run_size_token(prefix_run)
        if run_size is not None and prefix_size is not None and run_size != prefix_size:
            continue
        return model_dir
    return None


def cleanup_checkpoints(
    results_dir: Path,
    best_checkpoints: Dict[str, str],
    dry_run: bool = True,
    interactive: bool = False,
    exclude_patterns: Optional[List[str]] = None,
    unknown_policy: str = "keep-all",
    models_dir: Optional[Path] = None,
    skip_export_check: bool = False,
) -> Dict[str, any]:
    """Remove intermediate checkpoints, keeping only the best one per run.

    EXPORT-VERIFIED GUARD (default ON): before deleting ANY step of a run, the
    run's KEEPER step must have a corresponding HF export present under
    models_dir. If it is not found, the ENTIRE run is skipped (all checkpoints
    kept) — we must never delete a keeper's siblings when we cannot confirm the
    keeper itself survived as an export. Pass skip_export_check=True to bypass.
    """

    stats = {
        "deleted_dirs": [],
        "kept_dirs": [],
        "space_freed": 0,
        "errors": []
    }

    if models_dir is None:
        models_dir = Path("/mnt/data/sgsilva/models")

    checkpoints = get_checkpoint_dirs(results_dir)
    exclude_patterns = exclude_patterns or []

    print(f"Found {len(checkpoints)} training runs with checkpoints")
    print()

    for run_name, step_dirs in sorted(checkpoints.items()):
        # Check exclude patterns
        if any(pat in run_name for pat in exclude_patterns):
            print(f"📁 {run_name}")
            print(f"   ⏭️  Excluded by --exclude pattern")
            for step_dir in step_dirs:
                stats["kept_dirs"].append(str(step_dir))
            print()
            continue

        run_key = _resolve_run_key(run_name, best_checkpoints)
        best_step_known = run_key is not None
        best_step = best_checkpoints.get(run_key) if run_key else None

        print(f"📁 {run_name}")
        print(f"   Total checkpoints: {len(step_dirs)}")
        if best_step_known and run_key != run_name:
            print(f"   Matched run key: {run_key}")
        print(f"   Best checkpoint: {best_step}")

        if best_step is None and best_step_known:
            print(f"   ⏳ Eval pending — keeping all checkpoints")
            for step_dir in step_dirs:
                stats["kept_dirs"].append(str(step_dir))
            print()
            continue

        if best_step is None and not best_step_known:
            if unknown_policy == "keep-all":
                print(f"   ⚠️  Unknown run — keeping all checkpoints (use --unknown-policy keep-latest to prune)")
                for step_dir in step_dirs:
                    stats["kept_dirs"].append(str(step_dir))
                print()
                continue
            if unknown_policy == "keep-latest":
                print(f"   ⚠️  Unknown run — keeping latest checkpoint (highest step)")
                best_step = "__KEEP_LATEST__"
            else:
                raise ValueError(f"Unknown policy: {unknown_policy}")

        if not best_step:
            print(f"   ⚠️  WARNING: No best checkpoint defined, keeping all")
            for step_dir in step_dirs:
                stats["kept_dirs"].append(str(step_dir))
            print()
            continue

        # EXPORT-VERIFIED GUARD (Sandra's hard rule): never delete a step unless
        # the run's KEEPER step is verified present as an HF export. Applies only
        # when we have a real board keeper (not the keep-latest prune path, which
        # has no export to verify against).
        if not skip_export_check and best_step != "__KEEP_LATEST__":
            exported = find_exported_model_for_keeper(run_name, best_step, models_dir)
            if exported is None:
                print(
                    f"   ⚠️  export not verified — keeping all "
                    f"(no HF export for keeper {best_step} under {models_dir})"
                )
                for step_dir in step_dirs:
                    stats["kept_dirs"].append(str(step_dir))
                print()
                continue
            print(f"   ✅ export verified: {exported.name}")

        # Find the best checkpoint
        best_dir = None
        for step_dir in step_dirs:
            if step_dir.name == best_step:
                best_dir = step_dir
                break

        # If best checkpoint not found, use fallback (keep most recent/highest step)
        if not best_dir:
            if best_step != "__KEEP_LATEST__":
                print(f"   ⚠️  WARNING: Best checkpoint {best_step} not found!")
                available = [d.name for d in step_dirs]
                print(f"   Available: {', '.join(available)}")

            # Use the most recent checkpoint (highest step number) as fallback
            fallback_dir = max(step_dirs, key=lambda d: int(re.search(r'step_(\d+)', d.name).group(1)) if re.search(r'step_(\d+)', d.name) else 0)
            if best_step == "__KEEP_LATEST__":
                print(f"   📌 Keeping latest checkpoint: {fallback_dir.name}")
            else:
                print(f"   📌 Using fallback checkpoint: {fallback_dir.name}")
            best_dir = fallback_dir

        # Show all checkpoints, collect candidates for deletion
        to_delete = []
        for step_dir in step_dirs:
            if step_dir == best_dir:
                marker = "✓ Keeping (BEST)" if step_dir.name == best_step else "✓ Keeping (FALLBACK)"
                print(f"   {marker}: {step_dir.name}")
                stats["kept_dirs"].append(str(step_dir))
            else:
                size = _dir_size_bytes(step_dir)
                size_gb = size / (1024**3)
                print(f"   🗑️  Candidate: {step_dir.name} ({size_gb:.1f} GB)")
                to_delete.append((step_dir, size))

        # Process deletions for this run
        if to_delete:
            total_del_size = sum(s for _, s in to_delete)
            if dry_run:
                print(f"   → Would free {format_size(total_del_size)}")
                stats["space_freed"] += total_del_size
            elif interactive and len(step_dirs) > 1:
                answer = input(f"   → Delete {len(to_delete)} checkpoint(s) ({format_size(total_del_size)})? [y/N] ").strip().lower()
                if answer != 'y':
                    print(f"   ⏭️  Skipped all")
                    for step_dir, _ in to_delete:
                        stats["kept_dirs"].append(str(step_dir))
                else:
                    for step_dir, size in to_delete:
                        print(f"   🗑️  Deleting: {step_dir.name}")
                        try:
                            shutil.rmtree(step_dir)
                            stats["deleted_dirs"].append(str(step_dir))
                            stats["space_freed"] += size
                        except Exception as e:
                            stats["errors"].append(f"Failed to delete {step_dir}: {e}")
            else:
                for step_dir, size in to_delete:
                    print(f"   🗑️  Deleting: {step_dir.name}")
                    try:
                        shutil.rmtree(step_dir)
                        stats["deleted_dirs"].append(str(step_dir))
                        stats["space_freed"] += size
                    except Exception as e:
                        stats["errors"].append(f"Failed to delete {step_dir}: {e}")

        print()

    return stats


def cleanup_models_dir(
    models_dir: Path,
    best_checkpoints: Dict[str, str],
    dry_run: bool = True,
    interactive: bool = False,
    exclude_patterns: Optional[List[str]] = None
) -> Dict[str, any]:
    """Remove HuggingFace model exports that don't correspond to best checkpoints."""

    stats = {
        "deleted_dirs": [],
        "kept_dirs": [],
        "space_freed": 0,
        "errors": []
    }

    if not models_dir.exists():
        print(f"⚠️  Models directory not found: {models_dir}")
        return stats

    print("🔍 Scanning models directory...")
    print()

    prefix_best_steps = dict(MODEL_PREFIX_BEST_STEPS)
    prefix_best_steps.update(getattr(get_best_checkpoints, "prefix_best_steps", {}))

    print("Prefix-specific best steps:")
    for prefix, step in sorted(prefix_best_steps.items()):
        print(f"  {prefix} -> step{step}")
    print()

    exclude_patterns = exclude_patterns or []

    # Group model directories by run prefix (everything before -stepNNN)
    from collections import defaultdict
    groups = defaultdict(list)  # prefix -> [(model_dir, step_num)]
    ungrouped = []  # dirs with no step number

    for model_dir in sorted(models_dir.glob("*")):
        if not model_dir.is_dir():
            continue

        # Check exclude patterns
        if any(pat in model_dir.name for pat in exclude_patterns):
            print(f"   ⏭️  Excluded: {model_dir.name}")
            stats["kept_dirs"].append(str(model_dir))
            continue

        match = re.search(r'^(.*?)-?step(\d+)', model_dir.name)
        if not match:
            ungrouped.append(model_dir)
            continue

        prefix = match.group(1)
        step_num = match.group(2)
        groups[prefix].append((model_dir, step_num))

    # Show ungrouped dirs
    for model_dir in ungrouped:
        print(f"   ? Skipping (no step number): {model_dir.name}")
        stats["kept_dirs"].append(str(model_dir))

    # Process each group
    for prefix, entries in sorted(groups.items()):
        print(f"📁 {prefix}")
        to_delete = []
        kept_count = 0
        explicit_best_step = prefix_best_steps.get(prefix)

        for model_dir, step_num in sorted(entries, key=lambda e: int(e[1])):
            size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
            size_gb = size / (1024**3)

            keep_this = False
            if explicit_best_step is not None:
                keep_this = step_num == explicit_best_step

            if keep_this:
                print(f"   ✓ Keeping: {model_dir.name} ({size_gb:.1f} GB)")
                stats["kept_dirs"].append(str(model_dir))
                kept_count += 1
            else:
                to_delete.append((model_dir, size, size_gb))

        # If no model in this group is a known best, keep all
        if to_delete and kept_count == 0:
            print(f"   ⏳ No best checkpoint identified — keeping all")
            for model_dir, size, _ in to_delete:
                stats["kept_dirs"].append(str(model_dir))
            print()
            continue

        # Show candidates
        for model_dir, size, size_gb in to_delete:
            print(f"   🗑️  Candidate: {model_dir.name} ({size_gb:.1f} GB)")

        if to_delete:
            total_del_size = sum(s for _, s, _ in to_delete)
            if dry_run:
                print(f"   → Would free {format_size(total_del_size)}")
                stats["space_freed"] += total_del_size
            elif interactive and len(entries) > 1:
                answer = input(f"   → Delete {len(to_delete)} model(s) ({format_size(total_del_size)})? [y/N] ").strip().lower()
                if answer != 'y':
                    print(f"   ⏭️  Skipped all")
                    for model_dir, _, _ in to_delete:
                        stats["kept_dirs"].append(str(model_dir))
                else:
                    for model_dir, size, _ in to_delete:
                        print(f"   🗑️  Deleting: {model_dir.name}")
                        try:
                            shutil.rmtree(model_dir)
                            stats["deleted_dirs"].append(str(model_dir))
                            stats["space_freed"] += size
                        except Exception as e:
                            stats["errors"].append(f"Failed to delete {model_dir}: {e}")
            else:
                for model_dir, size, _ in to_delete:
                    print(f"   🗑️  Deleting: {model_dir.name}")
                    try:
                        shutil.rmtree(model_dir)
                        stats["deleted_dirs"].append(str(model_dir))
                        stats["space_freed"] += size
                    except Exception as e:
                        stats["errors"].append(f"Failed to delete {model_dir}: {e}")

        print()
    return stats


def format_size(size_bytes: int) -> str:
    """Format bytes as human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def main():
    parser = argparse.ArgumentParser(
        description="Cleanup checkpoints to keep only the best-performing ones",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview what will be deleted (dry run)
  python cleanup_checkpoints.py --dry-run

  # Actually delete checkpoints from results directory
  python cleanup_checkpoints.py

  # Also cleanup /mnt/data/sgsilva/models
  python cleanup_checkpoints.py --cleanup-models

  # Do everything
  python cleanup_checkpoints.py --cleanup-models
        """
    )

    parser.add_argument(
        '--results-dir',
        type=Path,
        default=Path('/mnt/data/sgsilva/checkpoints'),
        help='Path to checkpoint directory (default: /mnt/data/sgsilva/checkpoints)'
    )

    parser.add_argument(
        '--models-dir',
        type=Path,
        default=Path('/mnt/data/sgsilva/models'),
        help='Path to models directory (default: /mnt/data/sgsilva/models)'
    )

    parser.add_argument(
        '--cleanup-models',
        action='store_true',
        help='Also cleanup HuggingFace model exports in models directory'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview what will be deleted without actually deleting'
    )

    parser.add_argument(
        '--yes',
        action='store_true',
        help='Skip the top-level confirmation prompt (use with care)'
    )

    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Ask for confirmation before each deletion'
    )

    parser.add_argument(
        '--exclude',
        nargs='+',
        default=[],
        metavar='PATTERN',
        help='Exclude runs/models whose name contains any of these patterns'
    )

    parser.add_argument(
        '--unknown-policy',
        choices=['keep-all', 'keep-latest'],
        default='keep-all',
        help='For runs not present in best-checkpoint mapping: keep-all (safe) or keep-latest (prune)'
    )

    parser.add_argument(
        '--skip-export-check',
        action='store_true',
        help='Bypass the export-verified guard (DANGEROUS). By default a run is '
             'kept entirely unless its best/keeper step is verified present as an '
             'HF export under --models-dir.'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("CHECKPOINT CLEANUP SCRIPT")
    print("=" * 80)
    print()

    if args.dry_run:
        print("🔍 DRY RUN MODE - No files will be deleted")
    else:
        print("⚠️  LIVE MODE - Files will be permanently deleted!")
        if not args.yes:
            response = input("Continue? (yes/no): ")
            if response.lower() != 'yes':
                print("Aborted.")
                return

    print()

    # Cleanup results directory
    print("=" * 80)
    print("CLEANING UP RESULTS DIRECTORY")
    print("=" * 80)
    print()

    # Get best checkpoints (auto-detect or fallback)
    best_checkpoints = get_best_checkpoints()
    print()

    if args.exclude:
        print(f"Excluding patterns: {args.exclude}")
        print()

    results_stats = cleanup_checkpoints(
        args.results_dir,
        best_checkpoints,
        dry_run=args.dry_run,
        interactive=args.interactive,
        exclude_patterns=args.exclude,
        unknown_policy=args.unknown_policy,
        models_dir=args.models_dir,
        skip_export_check=args.skip_export_check,
    )

    # Cleanup models directory if requested
    models_stats = None
    if args.cleanup_models:
        print("=" * 80)
        print("CLEANING UP MODELS DIRECTORY")
        print("=" * 80)
        print()

        models_stats = cleanup_models_dir(
            args.models_dir,
            best_checkpoints,
            dry_run=args.dry_run,
            interactive=args.interactive,
            exclude_patterns=args.exclude
        )

    # Print summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()

    total_deleted = len(results_stats["deleted_dirs"])
    total_kept = len(results_stats["kept_dirs"])
    total_freed = results_stats["space_freed"]

    if models_stats:
        total_deleted += len(models_stats["deleted_dirs"])
        total_kept += len(models_stats["kept_dirs"])
        total_freed += models_stats["space_freed"]

    print(f"Directories deleted: {total_deleted}")
    print(f"Directories kept: {total_kept}")
    print(f"Space freed: {format_size(total_freed)}")

    if results_stats["errors"] or (models_stats and models_stats["errors"]):
        all_errors = results_stats["errors"] + (models_stats["errors"] if models_stats else [])
        print()
        print(f"⚠️  Errors: {len(all_errors)}")
        for error in all_errors:
            print(f"  - {error}")

    print()

    if args.dry_run:
        print("🔍 This was a dry run. Use without --dry-run to actually delete files.")
    else:
        print("✅ Cleanup complete!")

    print()


if __name__ == "__main__":
    main()
