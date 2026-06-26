# VLM Pose Estimation Monitoring App - Complete Documentation

## Table of Contents
1. [Overview](#overview)
2. [Purpose & Use Cases](#purpose--use-cases)
3. [Architecture](#architecture)
4. [Data Sources (Inputs)](#data-sources-inputs)
5. [User Interface (Outputs)](#user-interface-outputs)
6. [Data Flow](#data-flow)
7. [Key Features](#key-features)
8. [Technical Details](#technical-details)
9. [Usage Guide](#usage-guide)

---

## Overview

The **VLM Pose Estimation Monitoring App** is a comprehensive Gradio web application for monitoring the complete machine learning pipeline for vision-language model (VLM) pose estimation tasks.

### What It Does
- **Monitors datasets**: Browse SFT training datasets with annotations
- **Tracks training**: View checkpoint progression and training metrics
- **Analyzes evaluation**: Compare model performance across benchmarks
- **Visualizes results**: Interactive charts, tables, and image overlays

### Key Technologies
- **Framework**: Gradio (Python web UI)
- **Visualization**: Plotly (interactive charts), OpenCV (image processing)
- **Data**: Pandas (tables), JSON/JSONL (datasets), CSV (experiments)

---

## Purpose & Use Cases

### Primary Use Cases

#### 1. **Dataset Quality Assurance**
- Browse training/test datasets
- Verify annotations are correct
- Check image quality
- Identify problematic samples

#### 2. **Training Monitoring**
- Track which checkpoints have been created
- See training progression (steps, epochs)
- Identify best-performing checkpoints
- View training configurations

#### 3. **Model Evaluation**
- Compare checkpoints against baseline
- Analyze per-metric performance
- Identify regression issues
- Track benchmark performance (IFEval, SIBench)

#### 4. **Debugging & Analysis**
- View model predictions vs ground truth
- Analyze per-sample errors
- Compare multiple models side-by-side
- Track lineage of training samples

---

## Architecture

### Design Pattern: Single-File Gradio Application

```
app.py (3,600+ lines)
├── Imports & Configuration (lines 1-120)
├── Global Variables & Indexes (lines 121-180)
├── Data Loading Functions (lines 696-830)
├── Visualization Functions (lines 1400-2000)
├── UI Layout (build_ui) (lines 2618-2910)
├── Event Handlers (lines 2910-3620)
└── Main Entry Point (lines 3628+)
```

### Three-Tier Caching Strategy

```python
# Tier 1: Startup Cache (loaded once at app start)
DATASET_INDEX = {}      # All datasets indexed by task/variant
MODEL_INDEX = {}        # All checkpoints indexed by task/variant
EXPERIMENT_INDEX = None # experiments-final.csv loaded into memory
BENCHMARKS_INDEX = None # IFEval/SIBench results

# Tier 2: Session Cache (@lru_cache)
@lru_cache(maxsize=100)
def load_evaluation_results(result_file: str) -> Dict
    # Caches loaded JSON results

@lru_cache(maxsize=50)
def load_dataset_samples(dataset_path: str, split: str) -> List[Dict]
    # Caches loaded JSONL samples

# Tier 3: User Interaction State (Gradio gr.State)
current_page = gr.State(value=0)        # Gallery pagination
current_sample_idx = gr.State(value=None) # Selected sample
```

### Why This Design?
- **Single file**: Easy to deploy, no complex imports
- **Three-tier cache**: Fast startup, responsive UI, minimal memory
- **Gradio**: Built-in reactivity, automatic API generation

---

## Data Sources (Inputs)

### 1. SFT Datasets

**Location**: `/mnt/data/shared/vlm/data/sft_datasets_v4/`

**Structure**:
```
sft_datasets_v4/
├── task1/
│   ├── cropped_v1/
│   │   ├── train.jsonl          # Training samples
│   │   ├── test.jsonl           # Test samples
│   │   ├── train_stats.json     # Dataset statistics
│   │   └── test_stats.json
│   └── original_v1/
│       └── ...
├── task2/
│   ├── visualized_cropped_v4/
│   │   └── ...
└── task3a/
    └── ...
```

**JSONL Format** (each line is a JSON object):
```json
{
  "image_id": "10003_331337_08092025072815_30930078-sauron_uncorrected_42281675309",
  "image_path": "/path/to/image.jpg",
  "conversations": [
    {"from": "human", "value": "Detect all keypoints..."},
    {"from": "gpt", "value": "0. Nose: <point>(500, 200)</point>\n1. Left Eye: ..."}
  ],
  "metadata": {
    "exercise": "squat",
    "dimensions": [1920, 1080],
    "num_keypoints": 25,
    "keypoint_subset": "coco25"
  }
}
```

**Stats JSON Format**:
```json
{
  "total_samples": 4323,
  "exercises": {"squat": 1200, "deadlift": 980, ...},
  "dimensions": {"1920x1080": 3500, "1280x720": 823},
  "avg_keypoints": 24.8
}
```

**How It's Used**:
- Dataset Explorer: Shows images in gallery
- Sample Details: Displays metadata and annotations
- Statistics Card: Shows dataset composition

---

### 2. Model Checkpoints

**Location**: `/mnt/data/sgsilva/models/`

**Structure**:
```
models/
├── qwen3-vl-4b-4epochs-task1-step646/
│   ├── config.yaml          # Training configuration
│   ├── training_info.json   # Metadata (step, epoch, samples)
│   └── model weights/       # (not used by monitoring app)
├── qwen3-vl-4b-4epochs-task2-v4-step1328/
└── ...
```

**config.yaml**:
```yaml
model_name: qwen3-vl-4b
task: task1
variant: cropped_v1
learning_rate: 1e-5
batch_size: 8
epochs: 4
```

**training_info.json**:
```json
{
  "step": 646,
  "epoch": 2,
  "consumed_samples": 2584,
  "timestamp": "2026-01-15T14:32:10",
  "status": "completed"
}
```

**Checkpoint Naming Convention**:
```
{model_size}_4epochs_{task}_{variant}-step{step}

Examples:
- qwen3-vl-4b-4epochs-task1-step646
- qwen3-vl-4b-4epochs-task2-v4-step1328
- 4b_4epochs_task1b_cropped-step320
```

**How It's Used**:
- Training Monitor: Lists all checkpoints in table
- Checkpoint Details: Shows config and training info
- Prediction Loading: Maps checkpoint to evaluation results

---

### 3. Evaluation Results

**Location**: `/mnt/data/sgsilva/vlm-evaluation/results/final/`

**Structure**:
```
results/final/
├── task1_cropped_v1_test_step646_20260115_143210_oks_updated.json
├── task1_original_v1_test_step1292_20260116_091234_oks_updated.json
├── task2_visualized_cropped_v4_test_step1328_20260117_102345_f1_updated.json
└── ...
```

**Naming Pattern**:
```
{task}_{variant}_test_step{step}_{timestamp}_{metric}_updated.json
```

**JSON Format** (Task 1 - Keypoint Detection):
```json
{
  "summary": {
    "aggregated_metrics": {
      "oks_score_mean": 0.7234,
      "f1_score_mean": 0.8156,
      "precision_mean": 0.8423,
      "recall_mean": 0.7912,
      "coordinate_mae_total_mean": 42.3,
      "pck_50_mean": 0.8834,
      "pck_100_mean": 0.9456,
      "hallucination_rate_mean": 0.023
    },
    "per_keypoint_aggregated": {
      "Nose": {"oks": 0.85, "pck_50": 0.92, ...},
      "Left Eye": {"oks": 0.81, ...},
      ...
    }
  },
  "samples": [
    {
      "sample_id": 0,
      "image_id": "10003_331337_...",
      "image_path": "/path/to/image.jpg",
      "ground_truth": "0. Nose: <point>(500, 200)</point>\n...",
      "prediction": "0. Nose: <point>(498, 203)</point>\n...",
      "metadata": {...},
      "metrics": {
        "oks_score": 0.7456,
        "f1_score": 0.8234,
        "coordinate_mae_total": 38.2,
        ...
      }
    },
    ...
  ]
}
```

**Task 2 Format** (Keypoint Labeling):
```json
{
  "summary": {
    "aggregated_metrics": {
      "per_keypoint_accuracy_mean": 0.8923,
      "exact_match_mean": 0.7234,
      "left_right_confusion_rate_mean": 0.034
    }
  },
  "samples": [...]
}
```

**How It's Used**:
- Metrics Plot: Shows OKS/F1/Precision/Recall over steps
- Comparison Table: Compares models side-by-side
- Per-Sample Analysis: Shows individual predictions
- Prediction Overlays: Draws keypoints on images

---

### 4. Experiments Tracking

**Location**: `/mnt/data/sgsilva/vlm-evaluation/experiments-final.csv`

**CSV Format**:
```csv
model,task,dataset_variant,result_file,oks_score,f1_score,precision,recall,...
Qwen__Qwen3-VL-4B-Instruct,task1,cropped_v1,/path/to/result.json,0.0,0.6234,0.7123,...
qwen3-vl-4b-4epochs-task1-step646,task1,cropped_v1,/path/to/result.json,0.7234,0.8156,...
```

**Columns**:
- `model`: Checkpoint name
- `task`: Task type (task1, task2, etc.)
- `dataset_variant`: Dataset variant (cropped_v1, etc.)
- `result_file`: Path to results JSON
- Metric columns: `oks_score`, `f1_score`, `precision`, `recall`, `accuracy`, etc.

**Special Model Naming**:
- Models starting with `Qwen__Qwen` or `gemini` are baseline models
- Baseline models shown at step 0 in plots
- App adds `[Baseline]` prefix in UI

**How It's Used**:
- Primary data source for Training Monitor tab
- Filters models by task/variant
- Identifies best checkpoint per metric
- Powers metrics progression plots

---

### 5. Benchmark Results

#### 5.1 IFEval Reports

**Location**: `/mnt/data/sgsilva/vlm-evaluation/results/reports/`

**Files**:
```
reports/
├── Qwen__Qwen3-VL-4B-Instruct_report.md  # Baseline
├── qwen3-vl-4b-4epochs-task1-step646_report.md
├── 4b_4epochs_task1b_cropped-step320_report.md
└── ...
```

**Report Format**:
```markdown
# Evaluation Report: qwen3-vl-4b-4epochs-task1-step646

## Executive Summary

**Text Instruction Following (IFEval)**
- Prompt-Level Strict: **48.24%**
- Instruction-Level Strict: **59.15%**
- vs Baseline: **-4.63%** (⚠️ DEGRADED)

## Text Instruction Following (IFEval)

### Metrics

| Metric | Score | Description |
|--------|-------|-------------|
| Prompt-Level Strict | 0.4824 (48.24%) | % of prompts where ALL instructions followed |
| Instruction-Level Strict | 0.5915 (59.15%) | % of individual instructions followed |

### Comparison with Baseline

| Metric | Model | Baseline | Delta |
|--------|-------|----------|-------|
| Prompt-Level Strict | 0.4824 | 0.5287 | -0.0463 |
| Instruction-Level Strict | 0.5915 | 0.6403 | -0.0488 |
```

**How It's Parsed**:
```python
# Extract metrics using regex
prompt_match = re.search(r'Prompt-Level\s+Strict.*?(\d+\.\d+)%', content)
instr_match = re.search(r'Instruction-Level\s+Strict.*?(\d+\.\d+)%', content)
```

---

#### 5.2 SIBench Results

**Location**: `/mnt/data/sgsilva/outputs/sibench/`

**Structure**:
```
sibench/
├── qwen3-vl-4b-baseline/
│   └── T20260121_G95b91480/
│       ├── qwen3-vl-4b-baseline_Counting.csv
│       ├── qwen3-vl-4b-baseline_Existence.csv
│       ├── qwen3-vl-4b-baseline_Spatial_Relation.csv
│       └── ... (12 task CSVs)
├── qwen3-vl-4b-sft-step646/
│   └── T20260121_G95b91480/
│       └── ...
└── report_20260121_131630.md  # Aggregated report
```

**Report Format**:
```markdown
# SIBench Evaluation Report

## Individual Model Results

### qwen3-vl-4b-baseline

**Run:** T20260121_G95b91480

| Task | Correct | Total | Accuracy |
|------|--------:|------:|---------:|
| Counting | 13 | 18 | 72.2% |
| Existence | 38 | 40 | 95.0% |
| Spatial_Relation | 24 | 40 | 60.0% |
| **OVERALL** | **259** | **458** | **56.6%** |

### qwen3-vl-4b-sft-step646

| Task | Correct | Total | Accuracy |
|------|--------:|------:|---------:|
| Counting | 10 | 18 | 55.6% |
| Existence | 34 | 40 | 85.0% |
| **OVERALL** | **230** | **458** | **50.2%** |
```

**How It's Parsed**:
```python
# Extract overall accuracy
overall_match = re.search(
    r'\|\s+\*\*OVERALL\*\*\s+\|[^|]+\|[^|]+\|\s+\*\*(\d+\.?\d*)%\*\*',
    section_content
)

# Extract per-task accuracies
task_matches = re.finditer(
    r'\|\s+([^|]+?)\s+\|\s+\d+\s+\|\s+\d+\s+\|\s+(\d+\.?\d*)%',
    section_content
)
```

---

#### 5.3 BENCHMARKS_TESTED.md

**Location**: `/mnt/data/sgsilva/vlm-evaluation/results/BENCHMARKS_TESTED.md`

**Content**: Summary of all benchmarks tested, key findings, and recommendations

**How It's Used**: Summary text shown in Benchmarks eval tab

---

## User Interface (Outputs)

### Layout Structure

```
┌─────────────────────────────────────────────────────────┐
│  VLM Pose Estimation Pipeline Monitor                  │
├─────────────────┬───────────────────────────────────────┤
│  SIDEBAR (1/3)  │  MAIN CONTENT (2/3)                   │
├─────────────────┼───────────────────────────────────────┤
│ Task Selector   │  ┌─────────────────────────────────┐ │
│ Variant Selector│  │  Tab 1: Dataset Explorer        │ │
│ Split (train/test) │  │  Tab 2: Training Monitor     │ │
│                 │  │  Tab 3: Evaluation Dashboard    │ │
│ Stats Card      │  │  Tab 4: Benchmarks eval         │ │
│ Checkpoints     │  └─────────────────────────────────┘ │
└─────────────────┴───────────────────────────────────────┘
```

---

### Tab 1: Dataset Explorer

**Purpose**: Browse and inspect training/test datasets

**Components**:

1. **Image Gallery** (4×3 grid, paginated)
   - Shows 50 images per page
   - Pagination controls (prev/next, page indicator)
   - Click image to view details

2. **Selected Image Details** (accordion)
   - **Annotated Image**: Ground truth keypoints overlaid in green
   - **Metadata**: JSON viewer with image info
   - **Ground Truth**: Text display of annotations
   - **Multi-Model Comparison**:
     - Toggle to show predictions
     - Select up to 4 checkpoints
     - Shows predictions in different colors (Red, Blue, Yellow, Purple)
     - Side-by-side: GT vs Predictions

3. **Prediction Metrics Table** (test split only)
   - Compares selected models on current sample
   - Shows OKS, PCK@0.5, MAE for each model

**Data Flow**:
```
User selects task/variant/split
    ↓
App loads JSONL file (paginated)
    ↓
Gallery displays images
    ↓
User clicks image
    ↓
App extracts ground truth from JSONL
    ↓
If predictions enabled:
    - Loads evaluation results for selected checkpoints
    - Finds matching sample by image_id
    - Overlays predictions on image
    ↓
Displays comparison view
```

**Visual Output Example**:
```
┌──────────────────┬──────────────────┐
│ Ground Truth     │ Predictions Only │
│ (Green skeleton) │ (Red/Blue/...)   │
└──────────────────┴──────────────────┘

Metrics Table:
| Color | Model       | OKS   | PCK@0.5 | MAE   |
|-------|-------------|-------|---------|-------|
| Red   | step-646    | 0.745 | 0.883   | 38.2  |
| Blue  | step-1292   | 0.689 | 0.856   | 45.1  |
```

---

### Tab 2: Training Monitor

**Purpose**: Track checkpoint creation and training progression

**Components**:

1. **Checkpoint Table** (sortable)
   - Columns: Checkpoint, Task, Variant, Step, Epoch, Consumed Samples, Date, Best Metrics
   - Best checkpoint per metric highlighted in green
   - Shows all checkpoints for selected task (across all variants)

2. **Metric Selector** (dropdown)
   - OKS, F1 Score, Precision, Recall, MAE, PCK@50
   - Changes based on task type

3. **Metrics Progression Plot** (Plotly line chart)
   - X-axis: Training step
   - Y-axis: Selected metric
   - Multiple traces:
     - Baseline models (dotted line at step 0)
     - Each variant as separate trace
   - Hover shows exact values
   - Legend shows variant names

4. **Checkpoint Details** (accordion)
   - Select checkpoint from dropdown
   - Shows `config.yaml` (training configuration)
   - Shows `training_info.json` (metadata)

**Data Flow**:
```
User selects task
    ↓
App filters EXPERIMENT_INDEX by task
    ↓
Aggregates baseline models by variant (avg at step 0)
    ↓
Groups checkpoints by variant
    ↓
Creates table showing all checkpoints
    ↓
User selects metric
    ↓
Creates plot with metric over steps
    ↓
Highlights best checkpoint per metric
```

**Visual Output Example**:
```
Checkpoint Table:
┌────────────────┬──────┬──────┬──────┬───────┬────────────────┐
│ Checkpoint     │ Step │ Epoch│ OKS  │ F1    │ Status         │
├────────────────┼──────┼──────┼──────┼───────┼────────────────┤
│ [Baseline] ... │ 0    │ -    │ 0.00 │ 0.623 │ ✅ Baseline    │
│ step-320       │ 320  │ 1    │ 0.68 │ 0.795 │                │
│ step-646       │ 646  │ 2    │ 0.72 │ 0.816 │ 🏆 Best OKS    │
│ step-1292      │ 1292 │ 4    │ 0.70 │ 0.821 │ 🏆 Best F1     │
└────────────────┴──────┴──────┴──────┴───────┴────────────────┘

Metrics Plot:
     OKS
     ↑
0.75 │         ●───●
     │        /     \
0.70 │   ●───●       ●
     │  /
0.65 │ ●
     │
0.00 ├─●─────────────────→ Step
     0 320  646  960 1292

     ● Baseline (step 0)
     ● cropped_v1 (steps 320-1292)
```

---

### Tab 3: Evaluation Dashboard

**Purpose**: Compare checkpoint performance

**Components**:

#### View Mode: Quick View (Pre-generated Report)

**What It Shows**: Pre-generated text comparison reports

**Source**: `/mnt/data/sgsilva/vlm-evaluation/results/evaluations/checkpoint_comparison_*.txt`

**Example Files**:
- `checkpoint_comparison_4b_task1_cropped_v1.txt`
- `checkpoint_comparison_4b_task2_visualized_cropped_v4.txt`

**Report Format**:
```
=== Checkpoint Comparison: Task 1, cropped_v1 ===

Baseline: Qwen3-VL-4B-Instruct
- OKS: 0.000
- F1: 0.623
- Precision: 0.712
- Recall: 0.556

Best Checkpoint: qwen3-vl-4b-4epochs-task1-step646
- OKS: 0.723 (+0.723 vs baseline)
- F1: 0.816 (+0.193 vs baseline)
- Precision: 0.842 (+0.130 vs baseline)
- Recall: 0.791 (+0.235 vs baseline)

Key Observations:
- Vision training significantly improved all metrics
- Best checkpoint achieved at step 646 (epoch 2)
- No significant degradation after step 646
```

---

#### View Mode: Custom Comparison

**What It Shows**: Dynamic comparison of user-selected checkpoints

**Components**:

1. **Checkpoint Selector** (checkbox group)
   - Select up to 5 checkpoints
   - Filtered by current task/variant

2. **Comparison Table**
   - Baseline + selected checkpoints
   - Columns: Model, Samples, OKS, F1, Precision, Recall, MAE, PCK@50/100
   - Δ columns showing difference from baseline
   - Color-coded: Green (improvement), Red (degradation)

3. **Radar Chart** (Plotly)
   - 5-6 metrics as axes
   - One trace per model
   - Easy visual comparison

4. **Improvement Summary** (auto-generated markdown)
   - Which metrics improved/degraded
   - Magnitude of changes
   - Recommendations

**Data Flow**:
```
User selects checkpoints
    ↓
App looks up each in EXPERIMENT_INDEX (filtered by task+variant)
    ↓
Extracts metrics from CSV
    ↓
Calculates deltas from baseline
    ↓
Generates table + radar chart + summary
```

**Visual Output Example**:
```
Comparison Table:
┌──────────────────┬─────────┬──────┬────────┬──────────┐
│ Model            │ Samples │ OKS  │ Δ OKS  │ F1       │
├──────────────────┼─────────┼──────┼────────┼──────────┤
│ [Baseline] Qwen  │ 1000    │ 0.00 │   -    │ 0.623    │
│ step-320         │ 1000    │ 0.68 │ +0.68  │ 0.795    │
│ step-646         │ 1000    │ 0.72 │ +0.72  │ 0.816    │
└──────────────────┴─────────┴──────┴────────┴──────────┘

Radar Chart:
        OKS
         ●
        /|\
       / | \
  F1  ●  |  ● Precision
       \ | /
        \|/
         ●
       Recall

  ── Baseline (Qwen)
  ── step-646

Improvement Summary:
✅ All metrics improved vs baseline
📊 OKS: +0.72 (+∞% relative, 0→0.72)
📊 F1: +0.193 (+31.0% relative)
🏆 Best improvements: Recall (+42%), OKS (+∞)
```

---

### Tab 4: Benchmarks eval

**Purpose**: Track IFEval and SIBench benchmark performance

**Components**:

1. **Benchmark Selector** (radio buttons)
   - IFEval
   - SIBench

2. **Summary Card**
   - Key findings from BENCHMARKS_TESTED.md
   - Number of models evaluated
   - Baseline performance

3. **Results Table**
   - **IFEval**: Model, Prompt Strict %, Δ Prompt, Instr Strict %, Δ Instr, Status
   - **SIBench**: Model, Overall Accuracy %, Δ from Baseline, Status
   - Status icons: ✅ Maintained, ⚠️ Degraded, 🔴 Severely Degraded
   - Sorted by worst degradation first

4. **Visualization Chart**
   - **IFEval**: Bar chart of delta from baseline
     - Green bars: Improved (≥0%)
     - Orange bars: Minor degradation (-5% to 0%)
     - Red bars: Severe degradation (<-5%)
   - **SIBench**: Grouped bar chart of per-task accuracies
     - Each model as separate colored bars
     - Tasks on x-axis
     - Accuracy % on y-axis

5. **Model Details** (accordion)
   - Select model from dropdown
   - Shows detailed metrics
   - Link to full report
   - **IFEval**: Prompt/Instruction metrics + deltas
   - **SIBench**: Overall + per-task breakdown

**Data Flow**:
```
App startup
    ↓
load_benchmarks_index()
    ↓
Parse IFEval reports (regex extraction)
    ↓
Parse SIBench markdown report (regex extraction)
    ↓
Identify baseline models
    ↓
Calculate deltas for all models
    ↓
Store in BENCHMARKS_INDEX
    ↓
User switches benchmark type
    ↓
on_benchmark_change() generates table + chart
    ↓
User selects model
    ↓
on_model_select() shows detailed metrics
```

**Visual Output Example**:

**IFEval Table**:
```
┌─────────────────────────────┬──────────────┬─────────┬────────┐
│ Model                       │ Prompt (%)   │ Δ       │ Status │
├─────────────────────────────┼──────────────┼─────────┼────────┤
│ Baseline (Qwen3-VL-4B)      │ 52.87        │ -       │ ✅     │
│ 4b_4epochs_task1-step646    │ 48.24        │ -4.63   │ ⚠️     │
│ 4b_4epochs_task4-step1352   │ 28.10        │ -24.77  │ 🔴     │
└─────────────────────────────┴──────────────┴─────────┴────────┘
```

**IFEval Chart**:
```
    Δ from Baseline (%)
     ↑
 +5  │     ▮
     │     ▮
  0  ├─────────────────────────
     │        ▮
 -5  │        ▮
     │          ▮▮▮▮▮▮▮▮▮
-10  │                    ▮
     │                    ▮
-25  │                         ▮
     └────────────────────────────→
        Models (sorted by delta)

    ▮ Green: Improved
    ▮ Orange: Minor degradation
    ▮ Red: Severe degradation
```

**SIBench Chart**:
```
    Accuracy (%)
     ↑
100  │
     │
 80  │ ▮▮  ▮▮      ▮▮
     │ ▮▮  ▮▮      ▮▮
 60  │ ▮▮  ▮▮  ▮▮  ▮▮  ▮▮
     │ ▮▮  ▮▮  ▮▮  ▮▮  ▮▮
 40  │ ▮▮  ▮▮  ▮▮  ▮▮  ▮▮  ▮▮
     │ ▮▮  ▮▮  ▮▮  ▮▮  ▮▮  ▮▮
 20  │ ▮▮  ▮▮  ▮▮  ▮▮  ▮▮  ▮▮
     └─────────────────────────────→
       Count Exist Geom Height Loc Shape ...

    ▮▮ baseline
    ▮▮ sft-step646
```

---

## Data Flow

### Startup Sequence

```
1. main() called
   ↓
2. Load global indexes:
   - build_dataset_index()
     - Scans /mnt/data/shared/vlm/data/sft_datasets_v4/
     - Loads *_stats.json for each task/variant
     - Stores in DATASET_INDEX

   - build_checkpoint_index()
     - Scans /mnt/data/sgsilva/models/
     - Parses checkpoint names (regex)
     - Loads training_info.json
     - Stores in MODEL_INDEX

   - load_experiments_csv()
     - Reads experiments-final.csv into pandas DataFrame
     - Stores in EXPERIMENT_INDEX

   - load_benchmarks_index()
     - Scans IFEval reports, parses with regex
     - Scans SIBench reports, parses with regex
     - Identifies baselines, calculates deltas
     - Stores in BENCHMARKS_INDEX
   ↓
3. build_ui()
   - Creates Gradio components
   - Wires event handlers
   ↓
4. app.launch(port=7861)
   - Starts Gradio server
   - Opens browser
```

### User Interaction Flow

```
User selects task/variant/split
    ↓
Event: task_dropdown.change()
    ↓
Handler: on_task_change()
    - Updates variant choices
    - Updates metric selector choices
    ↓
Event: variant_dropdown.change()
    ↓
Handler: on_variant_change()
    - Loads stats from DATASET_INDEX
    - Creates image gallery (paginated)
    - Loads checkpoints from MODEL_INDEX
    - Creates checkpoint table from EXPERIMENT_INDEX
    - Creates metrics plot
    - Updates all UI components
    ↓
User clicks image in gallery
    ↓
Event: image_gallery.select()
    ↓
Handler: on_gallery_select()
    - Loads JSONL sample
    - Extracts ground truth
    - If predictions enabled:
      - Calls find_prediction_for_sample()
      - Loads evaluation results
      - Finds matching sample by image_id
      - Visualizes keypoints with cv2
    - Updates image viewer + metadata
    ↓
User selects checkpoints for comparison
    ↓
Event: compare_btn.click()
    ↓
Handler: generate_custom_comparison()
    - Filters EXPERIMENT_INDEX by task+variant+models
    - Calculates deltas from baseline
    - Creates DataFrame + Plotly chart
    - Generates markdown summary
    - Returns all outputs
```

---

## Key Features

### 1. Pagination for Large Datasets
- Gallery loads 50 images at a time
- Lazy loading (only reads needed JSONL lines)
- Fast navigation (prev/next buttons)

### 2. Multi-Model Prediction Overlay
- Compare up to 4 models on same image
- Different colors per model (Red, Blue, Yellow, Purple)
- Side-by-side: GT (green) vs Predictions (colors)
- Per-model metrics table

### 3. Baseline Aggregation
- Multiple baseline models averaged per variant
- Shown as single point at step 0
- Avoids clutter in metrics plots

### 4. Dynamic Metric Selection
- Metric selector adapts to task type
- Task 1: OKS, F1, Precision, Recall, MAE, PCK@50
- Task 2: Accuracy, F1, Precision, Recall
- Task 3: F1, Precision, Recall, OKS, PCK@50
- Task 4: Accuracy, F1, Precision, Recall

### 5. Intelligent Checkpoint Filtering
- Training Monitor: Shows all variants for task
- Dataset Explorer: Shows only current variant
- Evaluation Dashboard: Filters by task+variant

### 6. Pre-generated Reports
- Quick View mode loads pre-generated text reports
- No computation needed
- Instant display

### 7. Benchmark Tracking
- IFEval: Text instruction following
- SIBench: Spatial intelligence
- Automatic baseline detection
- Delta calculations
- Status indicators

### 8. Error Handling
- Graceful degradation if files missing
- Try-except blocks in all event handlers
- User-friendly error messages
- Logging for debugging

---

## Technical Details

### Technology Stack

```python
# Core Framework
import gradio as gr              # Web UI framework

# Data Processing
import pandas as pd              # Tables, CSV processing
import numpy as np              # Numerical operations
import json                     # JSON parsing

# Visualization
import plotly.graph_objects as go  # Interactive charts
import cv2                      # Image processing
from PIL import Image           # Image loading

# Caching & Performance
from functools import lru_cache # Session-level caching
```

### Gradio Components Used

| Component | Purpose | Examples |
|-----------|---------|----------|
| `gr.Dropdown` | Task/variant/model selection | task_dropdown, variant_dropdown |
| `gr.Radio` | Split selection, benchmark type | split_radio, benchmark_selector |
| `gr.Gallery` | Image grid display | image_gallery |
| `gr.Image` | Single image display | gt_image, pred_image1-4 |
| `gr.Dataframe` | Tables | checkpoint_table, comparison_table |
| `gr.Plot` | Interactive charts | metrics_plot, radar_chart |
| `gr.Markdown` | Text display | stats_display, summary |
| `gr.JSON` | JSON viewer | metadata_json |
| `gr.Code` | Code/config display | config_display, gt_text |
| `gr.Checkbox` | Toggles | show_predictions |
| `gr.CheckboxGroup` | Multi-select | selected_checkpoints |
| `gr.State` | Hidden state | current_page, current_sample_idx |
| `gr.Accordion` | Collapsible sections | checkpoint_details, model_details |
| `gr.Tabs` | Tab navigation | Dataset/Training/Eval tabs |

### Performance Optimizations

1. **Paginated JSONL Reading**
   ```python
   def get_paginated_samples(dataset_path, split, page, page_size=50):
       offset = page * page_size
       # Only read needed lines, not entire file
       samples = []
       with open(jsonl_file) as f:
           for i, line in enumerate(f):
               if i < offset:
                   continue
               if i >= offset + page_size:
                   break
               samples.append(json.loads(line))
       return samples
   ```

2. **@lru_cache for Expensive Operations**
   ```python
   @lru_cache(maxsize=100)
   def load_evaluation_results(result_file: str) -> Dict:
       # Cache loaded JSON, avoid re-reading
       with open(result_file) as f:
           return json.load(f)
   ```

3. **Startup Index Building**
   - All datasets/checkpoints scanned once at startup
   - Stored in memory for fast access
   - No repeated file system scans

4. **Conditional UI Updates**
   - Only update changed components
   - Use `gr.update(visible=...)` for show/hide
   - Avoid full page reloads

### File Parsing Patterns

**JSONL** (datasets):
```python
def load_jsonl(path):
    samples = []
    with open(path) as f:
        for line in f:
            samples.append(json.loads(line))
    return samples
```

**CSV** (experiments):
```python
df = pd.read_csv(path)
filtered = df[(df['task'] == task) & (df['dataset_variant'] == variant)]
```

**JSON** (results):
```python
with open(path) as f:
    data = json.load(f)
summary = data['summary']['aggregated_metrics']
```

**Markdown** (reports):
```python
with open(path) as f:
    content = f.read()
match = re.search(r'Prompt-Level\s+Strict.*?(\d+\.\d+)%', content)
value = float(match.group(1))
```

---

## Usage Guide

### Starting the App

```bash
cd /mnt/data/sgsilva/monitoring-app
python app.py --port 7861 --share
```

**Arguments**:
- `--port`: Port to run on (default: 7861)
- `--share`: Create public Gradio share link
- `--debug`: Enable debug logging

**Access**: `http://localhost:7861`

### Common Workflows

#### Workflow 1: Verify Dataset Quality

1. Select task from dropdown (e.g., "Task 1: Keypoint Detection")
2. Select variant (e.g., "cropped_v1")
3. Select split (e.g., "test")
4. Browse image gallery
5. Click suspect images
6. Check if annotations look correct (green skeleton)
7. Verify metadata (dimensions, keypoint count)

#### Workflow 2: Find Best Checkpoint

1. Select task + variant
2. Go to "Training Monitor" tab
3. Look at checkpoint table
4. Identify rows with 🏆 (best per metric)
5. Select metric from dropdown (e.g., "OKS")
6. View progression plot
7. Identify peak performance step
8. Click checkpoint to view config

#### Workflow 3: Compare Checkpoints

1. Go to "Evaluation Dashboard" tab
2. Select "Custom Comparison" mode
3. Check boxes for checkpoints to compare
4. Click "Generate Comparison"
5. View table (sorted by performance)
6. View radar chart (visual comparison)
7. Read auto-generated summary

#### Workflow 4: Analyze Sample Predictions

1. Select task + variant, split = "test"
2. Go to "Dataset Explorer" tab
3. Enable "Show Model Predictions"
4. Select up to 4 checkpoints
5. Click an image
6. View predictions overlaid on image
7. Check metrics table for per-sample scores
8. Identify which models perform best

#### Workflow 5: Check Benchmark Performance

1. Go to "Benchmarks eval" tab
2. Select benchmark (IFEval or SIBench)
3. View table (sorted by degradation)
4. Identify severely degraded models (🔴)
5. View chart for visual comparison
6. Select model for detailed metrics
7. Click report link for full analysis

---

## Summary

The **VLM Pose Estimation Monitoring App** is a comprehensive tool that:

### Takes as Input:
1. **SFT Datasets** (JSONL + stats JSON)
2. **Model Checkpoints** (config YAML + training info JSON)
3. **Evaluation Results** (detailed JSON with predictions)
4. **Experiments CSV** (aggregated metrics)
5. **Benchmark Reports** (IFEval MD, SIBench MD)

### Produces as Output:
1. **Interactive Web UI** with 4 main tabs
2. **Visual Analytics**: Charts, plots, tables
3. **Image Overlays**: Keypoint visualizations
4. **Comparison Views**: Multi-model analysis
5. **Auto-Generated Summaries**: Performance insights

### Core Value:
- **Unified View**: All pipeline stages in one place
- **Interactive**: Click to drill down, compare on demand
- **Fast**: Paginated, cached, optimized
- **Comprehensive**: From raw data to benchmarks
- **Actionable**: Identifies best checkpoints, regression issues

### Typical Use:
1. **Dataset QA**: Verify annotations are correct
2. **Training Monitoring**: Track checkpoint progression
3. **Model Selection**: Identify best performers
4. **Debugging**: Analyze per-sample errors
5. **Benchmark Tracking**: Monitor text/spatial capabilities

---

**Last Updated**: 2026-02-04
**App Version**: v1.0 (with Benchmarks eval tab)
**Status**: ✅ Production Ready
