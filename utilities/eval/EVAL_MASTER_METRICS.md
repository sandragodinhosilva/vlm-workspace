# `eval_master.csv` — metric dictionary (VO-focused)

What every column on the eval board means, where it comes from, and how it's computed. The
visual-obs (VO) section is the deep one — that's the axis under active development.

- **Board:** `/mnt/data/sgsilva/results/master/eval_master.csv` (+ `_4b.csv`, `_27b.csv` splits)
- **Built by:** `~/utilities/eval/compile_eval_results.py` (read-only union/join on the served checkpoint path)
- **Two-row header:** row 1 = band labels (group spans); row 2 = column names. `csv.DictReader` must
  skip row 1. Blank spacer columns separate groups.
- **Conventions:** all F1/accuracy/precision/recall/OKS values are **percentages** (×100, 2 dp).
  MAE / Pearson / Spearman are **raw** (not ×100). Blank cell = not evaluated / not applicable
  (distinct from a real `0.00`).

---

## 1. Identity & timing
| Column | Meaning |
|---|---|
| Display | Curated model name (from `master_models.json`). |
| Model Path | Served checkpoint path — the JOIN KEY across all 3 eval stages. |
| Model Created | Checkpoint export mtime (blank for bare baselines). |
| Owner | `sgsilva` / `<user> (external)` / blank, derived from the path. |
| Baseline? | yes = raw un-SFT'd Qwen3.5; no = an SFT/GRPO/merged checkpoint. |
| Trained w/ Reasoning | Curated: trained on reasoning data (yes/no/NA). |
| Eval Thinking | Serving mode at eval time (on/off) — same ckpt gives different numbers per mode. |
| Last Eval | Newest eval-artifact mtime feeding this row. |

## 2. General benchmarks
Three public multimodal benchmarks the model is held to, so VO/aux gains can't silently regress
general capability. All are multiple-choice; the cell is accuracy (%) over the split run here (NOT
the full upstream set). Per-row facts below: `Source` = where the compiler reads the number;
`Samples (here)` = the exact N this harness evaluates (verified from the run artifacts on disk);
`Modality`, `Released`, and the per-benchmark subcategory list follow.

| Column | Source | What's evaluated | Modality | Samples (here) | Released |
|---|---|---|---|---|---|
| MMMU-val | benchmarks `summary[_judge].csv` | College-level **multi-discipline understanding** — MMMU `MMMU_DEV_VAL` split, expert-level subject questions. | still image | **1050** (dev+val) | [arXiv 2311.16502](https://arxiv.org/abs/2311.16502), Nov 2023 (CVPR 2024) |
| Video-MME | same | General **video understanding** — VLMEvalKit Video-MME, answered from sampled video frames. | video (frames) | **2700** | [arXiv 2405.21075](https://arxiv.org/abs/2405.21075), May 2024 (CVPR 2025) |
| VSI-Bench | same | **Visual spatial reasoning** — the board runs **SIBench-VSR** (a meta-benchmark aggregating BLINK / 3DSRBench / SpatialBench / SITE / Omni3D-Bench / SPHERE-VLM etc.), restricted to 5 of its 23 task types. Closest axis to the VO/3D work. ⚠ Despite the column label this is **NOT** the NYU VSI-Bench paper. | mostly **image** (154 img / 24 video) | **178** | [arXiv 2509.18905](https://arxiv.org/abs/2509.18905), Sep 2025 |

**Subcategories present in each benchmark's data** (driver: `~/benchmarks/scripts/run_eval.py`):
- **MMMU-val** — 6 disciplines (result-file `category` col): Business · Tech & Engineering · Art &
  Design · Health & Medicine · Science · Humanities & Social Science. (Upstream: 30 subjects / 183
  subfields; only DEV_VAL is run.)
- **Video-MME** — sliced 3 ways: **Duration** short/medium/long · **Domain** (6): Knowledge ·
  Film & Television · Sports Competition · Artistic Performance · Life Record · Multilingual ·
  plus 30 `sub_category` and 12 `task_type` values (Counting Problem, Object Recognition, Action
  Reasoning, Temporal Perception, OCR Problems, …). The board reports only the overall accuracy.
- **VSI-Bench (SIBench subset)** — exactly the 5 tasks in `VSI_TASKS` (run_eval.py:45), with their
  on-disk per-task N and source benchmarks:
  | Task | N | input | source benchmark(s) |
  |---|---|---|---|
  | Counting | 18 | image | BLINK |
  | Height | 40 | image | 3DSRBench |
  | Existence | 40 | image | SpatialBench |
  | Object_Localization | 40 | 24 video / 16 image | BLINK, SITE |
  | Spatial_Relation | 40 | image | BLINK, Omni3D-Bench, SPHERE-VLM |
  | **Total** | **178** | 154 img / 24 video | |

Scored over **parsable answers only** — runaway/non-response generations are excluded from the
denominator. `Benchmark Scoring` records the method per cell: `raw` / `judged` / `parsable(-N)`.
The judge RESCUES right-but-unparsed answers (`\boxed{}`/prose); it can't fabricate (a degenerate
model stays ~chance).

---

## 3. Visual-obs (VO) — the core

### 3.0 The data shape (read this first)
The VO eval runs on the **1181-rep test split**. Each rep has **multiple exercise-specific
error-types** (e.g. "Side Trunk Lean", "Hip Rotation During Lift", …), each with a ground-truth
**severity integer 1–6** (1 = no error; ≥2 = error present). So the data is a grid of
**(rep × error-type) slots** — ~7060 slots total. "Positive" everywhere = **severity > 1**.

Every VO metric is one of: pool over **slots**, pool over **reps**, or exact-match over **slots**.

### 3.1 Three pipelines (NOT interchangeable — separate column sets)
| Pipeline | How severity is produced | Board columns | Source file |
|---|---|---|---|
| **single-stage** | model emits the severity dict DIRECTLY in one call (no obs step) | `VO Error-F1/Sample-F1/Severity Acc` (1st of each) | `*_singlestage_*.json` |
| **two-stage** | stage-1 obs → a stage-2 **reasoner** turns them into severity | `VO Error-F1/Sample-F1/Severity Acc` (2nd of each) | `stage2_*.json` |
| **agreement vs human GT** | model answers categorical obs → fixed clinical RULES → errors, scored vs human | `VO Agree-F1/Acc/Prec/Rec` | `agreement_*.json` |

> ⚠ "single-stage" ≠ "stage-1". **single-stage = the one-shot pipeline that COMPUTES severity
> directly.** "stage 1" is the obs-generation step of the *two-stage* pipeline (it produces MCQ
> answers, no severity). The board spells out "single-stage"/"two-stage" to avoid the "1-stage looks
> like stage-1" misread.

### 3.2 The headline metrics (report BOTH — they name different champions)
- **VO Error-F1 (single-stage)** ← `metrics.error_detection_f1`. Binarize every (rep×error-type)
  slot as present (`sev>1`) / absent, micro-F1 of the positive class over the whole ~7060-slot pool.
  Pred is matched to GT **BY ERROR NAME** → rewards detection AND output-format/name alignment.
- **VO Agree-F1 (vs GT)** ← `agreement_*.json` `error_relevant.vs_gt.a.overall.micro_f1`. Model
  (side **a**) only answers categorical obs; fixed RULES derive errors (threshold sev≥2); scored vs
  the **human** annotator. Confound-free (no naming credit) — the clinical signal; matches the
  formatted CSV's "agreement with human" ordering.

These rank models DIFFERENTLY (a format-trained reasoner can top error-F1 but not agree-F1). When
you say "the VO winner", **say which metric**. There is intentionally **no single ranker**.

### 3.3 Why F1 for detection but Acc for severity
- Detection (Error-F1, Sample-F1) is **binary and heavily imbalanced** (most slots are "no error").
  Plain accuracy is inflated by the easy negatives — a "say no-error everywhere" model scores high
  accuracy while catching zero errors. **F1 over the positive class** ignores the easy TNs.
- Severity is a **6-class ordinal label** — F1 (a binary metric) has no single positive class, so the
  natural metric is **exact-match accuracy**: did the model hit the precise level 1–6?

### 3.4 The full VO detail block (appended after the metadata columns)
Mirrors `visual_obs_sft_results_1105_formatted.csv`'s per-band layout, for single-stage AND
two-stage (two-stage mostly blank until the reasoner sweep fills it). All from the same result JSON's
`metrics.*`.

**Error Detection (error-based)** — pooled over **(rep × error-type) slots**, binary `sev>1`:
| Col | JSON key | Meaning |
|---|---|---|
| Acc | `error_detection_accuracy` | fraction of slots whose present/absent call is right (incl. easy TNs — misleading alone). |
| F1 Score | `error_detection_f1` | micro-F1 of positive class. **headline.** |
| Precision | `error_detection_precision` | of slots the model flagged, how many were real. |
| Recall | `error_detection_recall` | of real error slots, how many were caught. |

**Error Detection (sample-based)** — collapse each rep with `any(sev>1)`; **1181 decisions, one per rep**:
| Col | JSON key | Meaning |
|---|---|---|
| Acc | `sample_error_detection_accuracy` | per-rep good/flawed call accuracy. |
| F1 Score | `sample_error_detection_f1` | per-rep "any error" micro-F1 (coarser; always ≥ error-based F1). |
| Precision / Recall | `sample_error_detection_{precision,recall}` | per-rep. |

**Variability → Avg Dist Exercise** — mean over per-exercise categorical **index-distance** between
the model's obs answers and the human GT (from the agreement JSON's raw-answer block,
`per_exercise[*].categorical.mean_index_distance`, averaged across exercises — NOT from `metrics.*`).
**RAW distance, lower = better** (closer to GT). Single-stage only (two-stage has no raw obs answers);
blank for the oracle-ceiling row (it's the reference). NOTE: the historical formatted CSV left this
column blank (`build_formatted_csv.py` hardcodes `""`); the board now populates it.

**Error Severity** — exact ordinal match over **all slots** (incl. sev==1 "no error"):
| Col | JSON key | Meaning |
|---|---|---|
| Acc | `overall_severity_accuracy` | `mean(pred == gt)` over ALL slots. **Adjacent miss = full miss.** Rewards correct "no error" (1) calls too — a "say 1 everywhere" model scores high here. |
| Acc - within 1 | `overall_severity_within_1` | tolerant: `\|pred−gt\| ≤ 1`. |
| Acc (non-1) | `overall_severity_accuracy_non1` | exact acc **restricted to slots where GT > 1** (a real error). The "of actual errors, how often exact level" metric. |
| Acc (non-1) - within 1 | `overall_severity_within_1_non1` | non-1 slots, ±1 tolerant. |
| Acc - 1 … Acc - 5 (+ within 1) | `per_severity_level[N].{accuracy,within_1}` | per-GT-level breakdown: of slots whose GT is exactly N, how often pred is exactly N (and ±1). |
| MAE / MAE (non-1) | `overall_severity_mae{,_non1}` | mean abs level error (raw, all slots / error-only). |

> **Denominator note:** `Severity Acc` is over ALL slots, NOT only detected/error slots. A missed
> error (GT>1, model said no-error) is filled pred=0 and counts as a severity miss. e.g. GT=5,
> pred=3 → Acc miss (3≠5), within-1 miss (off by 2), MAE contribution 2; but it's a detection
> TRUE POSITIVE (both >1). Same slot: detection success, severity failure.

**Effectiveness Score / Injury Risk Score** — the rep's overall 1-N scores (separate from per-error severity):
| Col | JSON key (eff / inj) | Meaning |
|---|---|---|
| Score Acc | `{effectiveness,injury_risk}_exact_match_rate` | exact-match rate of the predicted score. |
| Score MAE | `{...}_mae` | mean abs error (raw). |
| Pearson Correlation | `{...}_correlation` | Pearson r vs GT (raw). |
| Spearman Correlation | `{...}_spearman_correlation` | Spearman ρ vs GT (raw). |

### 3.5 The Oracle ceiling row
`Oracle 397B visual-obs cat (ceiling)` is a **reference row, not a deployable model**: the 397B
ORACLE visual-obs scored vs human GT = the agreement **side=b** (b=oracle) constant
(F1 86.08 / Acc 91.77 / Prec 86.19 / Rec 85.97). It's the upper bound the SFT models distil toward.
Distinct from `Qwen3.5-397B (… plain obs)` which is **side=a** (the model run plain, agree-F1 33.07).
Config-driven via `master_models.json` (`oracle_ceiling: true` + `vo_agree_side` + `vo_agree_source`).
Benchmark/aux/detection columns are blank — it was never run on those.

### 3.5b Two-stage VO USAGE (grounding proxy) — trailing columns (2026-07-07)
Five columns at the END of the CSV, populated per `stage2_*` file from its own
`per_sample_results` via `vlm-post-training/visual_obs/measure_stage2_vo_usage.py`
(imported by the compiler; if that script is missing the columns stay blank, loudly):

| Column | Meaning |
|---|---|
| VO Usage Mean % | mean fraction of the fed VO answers the reasoning trace references (exact phrase or ≥60% content-token overlap) |
| VO Usage 100% Samples % | samples where EVERY fed observation is referenced |
| VO Usage 0% Samples % | samples referencing NONE (fully ungrounded reasoning) |
| VO Usage Shuffled Ctrl % | same matcher run against an UNRELATED sample's trace — paired by a DIFFERENT `exercise_id` (falls back to a different `session_id`), NOT shift-by-1: eval files are session-grouped, so a naive neighbor is usually the same person/exercise and its shared vocabulary would inflate the control |
| VO Usage Headroom (pp) | Mean − Ctrl — **the number to read**; the heuristic matcher fires on unrelated traces often (ctrl ≈ 35–55), so the raw Mean is inflated |

Only samples with a NON-EMPTY `reasoning_content` are measurable — a thinkoff reasoner's file
leaves the columns BLANK (never rendered as "0% usage"). Full method + caveats:
`~/.claude/reports/visual_observations/2026-07-07_stage2_vo_usage_metric.md`.

### 3.6 Pipeline / completeness provenance
`VO Source (single-stage)` / `VO Source (two-stage)` = the exact JSON each VO number came from.
`Stage2 Reasoner` (+ thinking) = the reasoner that produced the two-stage numbers (board policy:
ONLY the sft2812 reasoner; base-27B historical files excluded). `VO Eval N (eval/failed)` = sample
completeness (clean = 1181/0). `VO Test Set` = the test split path each eval ran on.

---

## 4. Aux tasks (domain multimodal test set, `testset_1506`)
| Column | Source | Meaning |
|---|---|---|
| Aux 3-Mod Weighted | eval_matrix `acc_weighted_3modalities` | equal-weight mean of video/text/image accuracy (NOT sample-weighted — name is loose). |
| Aux Video (all) | `acc_video` | combined video MCQA accuracy. |
| Aux Video 3D | run JSON `by_source[mcqa_video_3d_2705]` | 3D spatial MCQA (harder) subset accuracy. |
| Aux Video non-3D | `by_source[mcqa_video_1505]` | non-3D video MCQA subset accuracy. |
| Aux Text | `acc_text` | text MCQA accuracy. |
| Aux Image | `acc_image` | image-task composite accuracy. |
| Aux Image Dense OKS | `oks_image` (= `image_dense.metric_value_pct`) | dense keypoint OKS — a sample-weighted dense COMPOSITE (OKS for task1/1b/1c + per-keypoint Acc for task2 + per-sample F1 for task3a-d), labeled "OKS" by convention. NOT pure OKS. |
| Aux Image Task4 | `acc_task4a` (fallback `acc_task4b`) | Task-4 MCQA accuracy. |

## 5. Training & source provenance
`Train Group` / `Train Samples` (from eval_matrix) · `Benchmark Scoring` / `Benchmark Source` ·
`Aux Run TS` / `Aux Run ID` / `Aux Run Dir` / `Aux Source` (the run lineage feeding the aux cells).

---

## Mapping to the formatted CSV
`visual_obs_sft_results_1105_formatted.csv` is the historical reference, organized in BANDS:
"Single stage baseline" → `vo_s1_*`; "Two stages" → `vo_s2_*`; "agreement with human annotations"
→ `vo_agree_*`. The board's full detail block (§3.4) reproduces each band's metric columns. NOTE: the
board's two-stage uses the sft2812 reasoner while the formatted CSV's "Two stages" band used the
base-27B reasoner — so those numbers are NOT directly comparable across the two files.

## See also
- `EVAL_README.md` — the pipelines, how each is run/collected, the driver, the master CSV mechanics.
- `~/.claude/projects/-home-sgsilva/memory/projects/project_eval_v2_era.md` — the V2 era, audit
  findings (F1–F14), and decisions behind these columns.
