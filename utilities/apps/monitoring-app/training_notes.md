# Training Notes

Notes for each task displayed in the Training Monitor tab.
Each section header (## task_key) maps to a task name used in the app.
Content under each header is displayed as-is (Markdown).

---

## task1

**cropped_v1** (25 kp, sequential numbering) — best variant, OKS 0.063 → 0.197

**cropped_v2** (canonical COCO indices) — FAILED. Mode collapse: 128 memorized templates vs v1's 2,703 unique. Loss plateau 0.412 vs v1's 0.327.

**cropped_confidence_v1** (with confidence scores) — early experiment, not trained. Prompt used canonical numbering (like v2), response missing confidence scores.

**cropped_confidence_v2** (v1 format + confidence scores) — v1's proven prompt format with confidence scores added. Forces model to reason about visibility. Config ready.

**original_v1** (full uncropped images) — regression, OKS 0.195 → 0.058 (step648 best). Full uncropped images hurt performance vs cropped.

## task1b

**cropped_v1** (17 COCO-17 kp, no hands/feet) — OKS 0.054 → 0.178

## task1c

**cropped_v1** (12 body-only kp) — OKS 0.059 → 0.277, fewer keypoints = easier

## task2

**visualized_cropped_v1** (basic numbered overlay) — early experiment, baseline only

**visualized_cropped_v2** (improved overlay) — Acc 4.7% → 33.9%

**visualized_cropped_v3** (larger numbers) — early experiment, baseline only

**visualized_cropped_v4** (color-coded labels) — Acc 6.0% → 33.4%. Plateaued after epoch 1. Root cause: images too small (median 256x330px, numbers 5-8px).

**visualized_v5** (uncropped 640px + enhanced rendering) — SFT complete, baseline 7.8%, loss 0.952→0.415. Eval running.

**visualized_cropped_v6** (cropped, larger images 282x640, all 25 kp) — config ready, not trained yet.

**visualized_cropped_v7** (cropped, sparse selection 5-8 kp per image) — config ready, not trained yet.

## task3a

**v1_high_error** (80% L/R swap rate) — F1 0.072 → 0.193

**v1_mixed** (combined error types) — baseline only

**BROKEN:** All checkpoints predict `has_errors=true` for 100% of samples (degenerate). Correct error *count* (avg 4.2) but random *indices* (F1 ~24%). Root causes: 5-step cross-modal reasoning too hard for 4B; 80/20 class imbalance; small images (5-8px numbers).

## task3b

**v1_low_missing** (30% missing rate) — F1 0.408 → 0.770

**v1_high_missing** (50% missing rate) — not trained

OKS corrected 2026-02-11 (~30% lower). Trained on snake_case data — retrain needed on Title Case.

## task3c

**v1_small_displacement** — F1 0.442 → 0.737

**v1_background_displacement** — F1 0.441 → 0.715

OKS corrected 2026-02-11 (~30% lower). Trained on snake_case data — retrain needed on Title Case.

## task3d

**v1_mixed** (missing + displaced) — baseline only, not trained

**v1_challenging** (harder error cases) — baseline only

## task4

**V1** (LLM desc + random distractors) — 84.2% → 95.2%, too easy

**V3** (single-call VLM) — 97.4% baseline, abandoned

**V4.2** (keypoint hints + consensus filter) — 94.0% baseline, too easy

**V4.3** (Qwen-only filter) — 91.3% baseline, too easy

**V4.4** (calibrated difficulty, Kimi+Qwen) — 71.4% baseline, pending

**V5** (geometric QA - tier 1 and 2) — 40.3% → 77.7%, best gain (+37pp)

**V5.1** (balanced all-tier) — 32.7% baseline

**V5.2** (improved thresholds) — superseded by V5.3

**V5.3** (final geometric + foreshortening) — 32.5% → 37.2% (step1095). Marginal +4.7pp — model struggles with pure geometric reasoning.

**V6.1.2** (LLM geo context) — 65.6% → 98.8%. **INFLATED**: distractors 90.9% similar to correct answer. Learned word patterns, not pose comprehension.

**V6.2** (template descriptions) — 48.2% → 75.2% (step730). Plateau: 74.6% (ep3), 74.9% (ep4).

## mixed

**mixed_balanced_v1** — 25% per task (T1+T2+T3+T4). T4 93.2%, T2 Acc 0.423, T3 F1 0.928, T1 OKS near-zero (0.010). High F1 but near-zero OKS = learned output FORMAT, not coordinate PRECISION.

**mixed_v2_phase1** — Two-phase curriculum (T1/T1b/T1c/T3b/T3c). SUPERSEDED — caused 2x L/R confusion (~30% vs single-task's ~15%).

**mixed_v3** — 7 tasks single-phase (T1c+T2+T3b+T3c+T3d+T4V5.3+T4V6.2). T4 matches single-task. Coordinate tasks degrade (~40% of single-task).

**mixed_v4_weighted** — Same v3 data + per-task loss weights (T1c=3.0, T4=0.5). Testing if weighting compensates for gradient interference.

**mixed_v5** — 4 tasks (T1c+T2+T3d+T4V6.2), LR 5e-6. FAILED — same pattern as v3 (T4 dominates).

**mixed_v6** — 3 tasks (T1c+T2+T3d, NO T4), LR 1e-5. Tests if T4 causes interference.

**mixed_v7** — 3 tasks (T1c+T2+T3d, NO T4), variant of v6.

**mixed_final_a** — 7 tasks (T1+T2+T3a+T3b+T3c+T3d+T4), v2 datasets. Latest main mixed experiment. Ongoing.

**mixed_final_b** — Same tasks as final_a, different training schedule. Ongoing.
