# Task & Variant Descriptions

Brief descriptions of each task and its variants, displayed in the Training Monitor.
Format: `## task_key` headers map to task names. Each paragraph starts with `**variant_name**`.

---

## task1

Task 1 evaluates keypoint coordinate prediction on cropped person images. The model receives a cropped image and must output (x, y) coordinates for each visible keypoint. Key metric: OKS (Object Keypoint Similarity).

**cropped_v1** (25 kp, sequential numbering) — Best single-task variant. OKS 0.063 -> 0.197. Uses simple sequential keypoint indices with cropped images.

**v2** — V2 dataset with 1-based numbering and updated prompts.

## task1b

Task 1b is a simplified variant of Task 1, predicting only COCO-17 keypoints (no hands/feet).

**cropped_v1** (17 kp) — OKS 0.054 -> 0.178.

**v2** — V2 dataset.

## task1c

Task 1c predicts 12 body-only keypoints (excluding extremities). Fewer keypoints = easier learning signal.

**cropped_v1** (12 kp) — OKS 0.059 -> 0.277. Best coordinate accuracy among all Task 1 variants.

**v2** — V2 dataset.

## task2

Task 2 evaluates per-keypoint accuracy identification on images with numbered keypoint overlays. The model sees numbered dots on the image and must identify which body part each number corresponds to. Metric: Per-keypoint accuracy.

**visualized_cropped_v2** — Acc 4.7% -> 33.9%.

**visualized_cropped_v4** — Acc 6.0% -> 33.4%. Plateaued after epoch 1 (images too small, numbers 5-8px).

**visualized_v5** — Uncropped 640px + enhanced rendering. Loss 0.952 -> 0.415.

**v2** — V2 dataset.

## task3a

Task 3a: Error detection — identify L/R swaps in keypoint annotations. The model sees an image and keypoint list, and must determine if any left/right labels are swapped. Key metric: F1.

**v1_high_error** (80% L/R swap rate) — F1 0.072 -> 0.193. All checkpoints predict has_errors=true for 100% of samples (degenerate classifier).

**v1_mixed** (combined error types) — Baseline only.

**v2** — V2 dataset with variable adulteration rates.

## task3b

Task 3b: Missing keypoint detection — identify which keypoints are missing from the annotation. Key metric: F1.

**v1_low_missing** (30% missing rate) — F1 0.408 -> 0.770.

**v1_high_missing** (50% missing rate) — Not trained.

**v2** — V2 dataset.

## task3c

Task 3c: Displaced keypoint detection — identify keypoints that have been shifted from their true position. Metric: F1.

**v1_small_displacement** — F1 0.442 -> 0.737.

**v1_background_displacement** — F1 0.441 -> 0.715.

**v2** — V2 dataset.

## task3d

Task 3d: Combined error detection (missing + displaced + swapped). The hardest error detection sub-task.

**v1_mixed** (missing + displaced) — Baseline only.

**v1_challenging** — Baseline only.

**v2_d1** — V2 dataset, difficulty tier 1.

**v2_d2** — V2 dataset, difficulty tier 2.

## task4

Task 4: Exercise description MCQA. The model sees a pose image and a multiple-choice question about which exercise is being performed. Key metric: Accuracy.

**mcqa_v1** (LLM desc + random distractors) — 84.2% -> 95.2%. Too easy due to weak distractors.

**mcqa_v5.3** (geometric + foreshortening QA) — 32.5% -> 37.2%. Model struggles with pure geometric reasoning.

**mcqa_v6.2** (template descriptions) — 48.2% -> 75.2% (step730). Best honest variant. Odd epochs exploit length shortcut.

**mcqa_v6.1.2** (LLM geo context) — 65.6% -> 98.8%. INFLATED: distractors 90.9% similar to correct answer.

## mixed

Mixed SFT: models trained on multiple tasks simultaneously, then evaluated per-task. Tests whether multi-task training causes gradient interference.

**mixed_balanced_v1** — 25% per task (T1+T2+T3+T4). T4 93.2%, T2 0.423, T3 F1 0.928, T1 OKS near-zero.

**mixed_v3** — 7 tasks single-phase. T4 matches single-task. Coordinate tasks degrade (~40% of single-task).

**mixed_final_a** — 7 tasks (T1+T2+T3a-d+T4), v2 datasets.

**mixed_final_b** — Same tasks as final_a, smaller training dataset size.
