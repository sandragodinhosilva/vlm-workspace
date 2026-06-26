# VLM Pose Estimation Pipeline Monitoring App

Comprehensive Gradio web application for monitoring dataset creation, SFT training, and evaluation.

## Features

- **📊 Dataset Explorer**: Browse datasets, view annotations, compare multi-model predictions
- **🚀 Training Monitor**: Track checkpoints, metrics progression, identify best models
- **📈 Evaluation Dashboard**: View pre-generated comparison reports
- **📊 Benchmarks**: IFEval and SIBench results with baseline comparisons
- **🔄 Mixed vs Single**: Compare multi-task vs single-task training effectiveness
- **📝 MCQA Browser**: Navigate and inspect Task 4 multiple-choice QA samples

## Installation

### Option 1: Install Dependencies Globally

```bash
cd /home/sgsilva/utilities/apps/monitoring-app
pip3 install -r requirements.txt
```

### Option 2: Use Virtual Environment (Recommended)

```bash
cd /home/sgsilva/utilities/apps/monitoring-app
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

### Launch the App

```bash
cd /home/sgsilva/utilities/apps/monitoring-app
python3 app.py --port 7861
```

### Launch with Public Sharing

```bash
python3 app.py --port 7861 --share
```

### Enable Debug Logging

```bash
python3 app.py --port 7861 --debug
```

## Usage

1. **Select Task**: Choose from Task 1/1b/1c, 2, 3a/3b/3c/3d, 4, or Mixed Tasks
2. **Select Variant**: Pick the dataset variant (e.g., cropped_v1, mcqa_v1)
3. **Choose Split**: Train or test dataset
4. **Explore**: Navigate through the six main tabs

### Tab 1: 📊 Dataset Explorer

- **Gallery**: Browse paginated image gallery (50 images/page, Previous/Next navigation)
- **Search**: Find images by ID with keyword matching
- **Multi-Model Comparison**: Select up to 4 checkpoints to compare predictions
- **Per-Image Metrics**: View OKS, F1, Precision for each model (task-specific)
- **Visualizations**: Ground truth keypoint overlays (Task 1/1b/1c)
- **Metadata**: Full sample JSON, image dimensions, keypoint counts

### Tab 2: 🚀 Training Monitor

- **Checkpoint Table**: All evaluated checkpoints with metrics (from experiments-final.csv)
- **Best Model Highlighting**: Green row indicates best checkpoint per task
- **Metrics Plot**: Interactive Plotly chart showing metric progression over training steps
- **Metric Selector**: Task-specific metrics (OKS, F1, Accuracy, Error Detection, etc.)
- **Checkpoint Details**: Training configs and metadata in expandable accordion
- **Baseline Comparison**: Baseline shown at step 0, including baseline-only variants (no SFT yet)
- **Synthetic Baselines**: Mixed-task variants auto-inherit instruct-model baselines

### Tab 3: 📈 Evaluation Dashboard

**Quick View Mode** (Fully Implemented):
- Pre-generated comparison reports from evaluation scripts
- Baseline vs checkpoint performance tables
- Key observations and recommendations
- Auto-loads report for selected task/variant

**Custom Comparison Mode** (UI Only):
- Interface created but comparison logic not implemented
- Use Quick View for current comparisons

### Tab 4: 📊 Benchmarks

- **IFEval Results**: Text instruction-following metrics (Prompt-Level, Instruction-Level Strict)
- **SIBench Results**: Visual spatial reasoning across 12 tasks
- **Baseline Comparison**: Delta calculations vs baseline model
- **Filters**: By task and variant
- **Model Details**: Direct links to evaluation reports

### Tab 5: 🔄 Mixed vs Single Task Comparison

- **Fixed 40-Checkpoint Analysis**: Mixed-task (8 ckpts) vs single-task (31 ckpts + baseline)
- **Key Findings**: Hardcoded comparison results for quick reference
  - Task 1: Mixed achieves 24.61% OKS vs single 19.70% (+24.9%)
  - Task 3: Mixed shows 38.93% OKS
- **Metrics Plot**: Training progression comparison
- **Model Selector**: View details for any checkpoint

### Tab 6: 📝 MCQA Browser (Task 4)

- **Navigation**: Previous/Next buttons, Jump to sample number, Random sample button
- **Question Display**: Full question text with image
- **Image Path**: Full filesystem path displayed below image
- **Answer Choices**: A/B/C/D options with correct answer highlighted
- **Validator Filter**: Filter by consensus status (CORRECT = all validators agree, INCORRECT = any disagrees)
- **Validator Results**: Qwen/Gemini validation judgments (variant-aware: V4.3 shows Qwen only)
- **Sample Counter**: "Sample X of Y" progress indicator
- **Metadata**: Full sample JSON in expandable accordion
- **Performance**: O(1) sample access via cached JSONL lines

## Directory Structure

```
monitoring-app/
├── app.py                              # Main application (~5,200 lines)
├── requirements.txt                    # Python dependencies
├── README.md                          # This file
├── CHANGELOG.md                       # Version history and changes
├── APP_READY.md                       # App readiness status
├── METRIC_EXPLANATIONS_UPDATE.md      # Metric documentation
├── TASK_STRUCTURE_UPDATE.md           # Task structure reference
└── venv/                              # Virtual environment (optional)
```

## Implementation Status

### ✅ Fully Implemented (Production Ready)

**Core Features**:
- ✅ **Dataset Explorer**: Gallery, search, multi-model comparison, keypoint visualization
- ✅ **Training Monitor**: Checkpoint table, metrics plot, best model selection
- ✅ **Evaluation Dashboard**: Quick View mode with pre-generated reports
- ✅ **Benchmarks Tab**: IFEval and SIBench results with filtering
- ✅ **Mixed vs Single Tab**: 40-checkpoint comparison analysis
- ✅ **MCQA Browser**: Full navigation and sample inspection

**Data Integration**:
- ✅ experiments-final.csv loading with full metrics
- ✅ Evaluation JSON parsing (all task types)
- ✅ Checkpoint metadata and config loading
- ✅ Benchmark report parsing (IFEval, SIBench)

**Performance**:
- ✅ Three-tier caching (startup, session, user state)
- ✅ O(1) JSONL random access via `_load_jsonl_lines()` cache
- ✅ LRU cache for checkpoint table, metrics plot, evaluation results
- ✅ Efficient image gallery pagination

**Code Quality**:
- ✅ Task-specific metric handling (6 task types)
- ✅ Graceful error handling and fallbacks
- ✅ Dead code removed (2026-02-06)
- ✅ Comprehensive logging

### ⚠️ Partially Implemented

- ⚠️ **Custom Comparison**: UI exists but backend logic not implemented
  - Current: Use Quick View for comparisons
  - Future: Implement custom multi-checkpoint comparison generation

### 📋 Known Limitations

1. **Per-Image Metrics**: Shows per-sample metrics, not summary aggregations
   - This is correct for sample-level analysis
   - Summary metrics available in checkpoint table

2. **Benchmark Baselines**: Hardcoded fallback values (IFEval: 52.87%/64.03%)
   - Serves as safety net if report files missing
   - Values verified to match actual baseline reports

3. **Image Loading**: Requires read access to dataset directories
   - Configured in: `DATASETS_BASE_PATH`, `ALLOWED_IMAGE_DIRS`

## Data Sources

- **Datasets**: `/mnt/data/shared/vlm/data/sft_datasets_v4/`
- **Checkpoints**: `/mnt/data/sgsilva/models/`
- **Results**: `/mnt/data/sgsilva/vlm-evaluation/results/final/`
- **Reports**: `/mnt/data/sgsilva/vlm-evaluation/results/evaluations/`
- **Experiments**: `/mnt/data/sgsilva/vlm-evaluation/experiments-final.csv`

## Architecture

- **Single-file app**: `app.py` (~5,200 lines)
- **Three-tier caching**: Startup cache, session cache (@lru_cache), user state
- **Theme**: Soft/warm aesthetic
- **Layout**: Two-column (1:2 ratio) - sidebar controls + main content

## Troubleshooting

### Module Not Found Errors

Install missing dependencies:
```bash
pip3 install gradio pandas numpy opencv-python plotly pillow
```

### Port Already in Use

Try a different port:
```bash
python3 app.py --port 7862
```

### Datasets Not Found

Verify paths in `app.py` match your setup:
- `DATASETS_BASE_PATH`
- `MODELS_BASE_PATH`
- `RESULTS_BASE_PATH`

## Recent Updates

### 2026-02-09: MCQA Browser & Metrics Plot Improvements
- **Added**: Validator display in MCQA tab (variant-aware: V4.3 Qwen-only, V4.2 Qwen+Gemini)
- **Added**: Full image path display in MCQA tab
- **Added**: V4.2/V4.3 variant detection in `parse_checkpoint_name()`
- **Added**: Baseline-only variants shown as dots at step 0 in metrics plot
- **Added**: Synthetic baselines for mixed models missing their own baseline row
- **Fixed**: `metric_map` UnboundLocalError (forward reference)
- **Fixed**: MCQA choice parser stripping "Select the letter..." suffix
- **Fixed**: `create_stats_card()` fallback to JSONL line count when stats files missing
- **Filtered**: `_filtered` and `_hf` intermediate datasets hidden from variant dropdown

### 2026-02-06: Comprehensive Metrics Revision
- **Fixed**: Task 4 column name bugs (4 instances)
- **Fixed**: Metric selector choices for all tasks
- **Fixed**: Per-image metrics table to match evaluation JSONs
- **Improved**: MCQA sample counting performance (caching)
- **Removed**: 63 lines of dead code
- **Updated**: experiments-final.csv with Task 4 P/R/F1 metrics

See [CHANGELOG.md](CHANGELOG.md) for complete details.

## Development

### Architecture
- **Single-file design**: All functionality in `app.py` (~5,200 lines)
- **Three-tier caching**:
  - Tier 1 (Startup): `DATASET_INDEX`, `MODEL_INDEX`, `EXPERIMENT_INDEX`, `BENCHMARKS_INDEX`
  - Tier 2 (Session): `@lru_cache` decorators (maxsize=50)
  - Tier 3 (User): Gradio `gr.State()` for pagination/selections
- **Task-aware logic**: Separate handling for 6 task types (1, 1b, 1c, 2, 3a-3d, 4)
- **Metrics mapping**: Dynamic column name resolution per task

### Contributing

When making changes:
1. Update [CHANGELOG.md](CHANGELOG.md) with all modifications
2. Test all 6 tabs after changes
3. Verify metrics match CSV/JSON data
4. Check console for errors
5. Update README if new features added

### Testing Checklist

- [ ] App launches without errors (`python3 app.py --port 7861`)
- [ ] All 6 tabs visible and functional
- [ ] Task dropdown populates correctly
- [ ] Metrics plot shows task-specific metrics
- [ ] Per-image metrics display correctly
- [ ] MCQA navigation works smoothly
- [ ] Benchmark tab shows IFEval/SIBench data
- [ ] No console errors during normal operation

---

**Created**: 2026-02-02
**Last Updated**: 2026-02-09
**Status**: Production Ready (5/6 tabs fully functional)
**Lines of Code**: ~5,200 (app.py)
**Supported Tasks**: 6 (Task 1, 1b, 1c, 2, 3a-3d, 4, Mixed)
