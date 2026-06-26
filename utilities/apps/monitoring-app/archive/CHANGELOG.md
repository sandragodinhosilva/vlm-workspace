# Changelog - VLM Monitoring App

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [2026-02-06b] - Task 1 Metrics Overhaul, MCQA Tab Fixes & Gradio Patch

### 🎯 Overview
Follow-up revision addressing misleading Task 1 precision metrics, broken MCQA tab for keypoint_qa datasets, empty gallery bug for Task 4, and Gradio 6.5.1 crash on empty gallery clicks.

### 🐛 Fixed

#### Gradio 6.5.1 Gallery Crash (KeyError: 'value')
- **Root cause**: Clicking empty Gallery component fires `SelectData` event without `value` key
- **Fix**: Monkey-patched `gr.SelectData.__init__` to default missing `value` to `None`
- **Guard**: Added `evt.value is None` check in `on_gallery_select` for graceful handling

#### MCQA Tab Not Showing keypoint_qa Datasets
- **Root cause**: `is_mcqa_variant()` checked for `'mcqa' in variant.lower()` — excluded `keypoint_qa_*` variants
- **Fix**: Changed to `return variant is not None` — all Task 4 variants are MCQA by definition

#### Task 4 Gallery Showing No Images (train or test)
- **Root cause**: No `train_stats.json` in any Task 4 directory → `train_samples = 0` → gallery pagination never loads
- **Fix**: Fall back to JSONL line count when stats files missing; also added general fallback for all tasks

### ✨ Changed

#### Task 1 Metrics: Replaced Misleading Precision/Recall with PCK
- **Problem**: Precision/Recall for Task 1 were name-match only (did the model output the right keypoint name?) — trivially high (~96%) and misleading next to low OKS (~27%)
- **Root cause**: `calculate_classification_metrics` (distance-based) silently fails, falls back to name matching
- **Metric selector**: `["OKS", "PCK@50", "PCK@100", "PCK@150", "MAE", "Name Match (F1)"]`
- **Checkpoint table**: Columns now `OKS | PCK@50 | PCK@100 | MAE | Name F1`
- **Per-image metrics**: Headers `OKS | Matched | Missing` (shows matched keypoint count and missing count)
- **Metric map**: Added `PCK@100` → `pck_100`, `PCK@150` → `pck_150`, `Name Match (F1)` → `f1_score`
- Task 3/4 precision/recall left unchanged (meaningful classification metrics there)

#### Dataset Index Filtering for Task 4
- Skip `cache`, `validation`, `archive_*` directories
- Skip `*_backup_*` and `*_test_*` directories
- Skip directories without `train.jsonl` or `train_stats.json`
- **Result**: Task 4 dropdown reduced from 20 junk entries to 4 valid datasets

---

## [2026-02-06] - Comprehensive Metrics Revision

### 🎯 Overview
Comprehensive audit and revision of monitoring app focusing on metrics accuracy, data integrity, and code quality. All metrics now correctly match CSV data and evaluation JSON structures.

### ✨ Added

#### Metric Selector Options
- **Task 2**: Added "L/R Confusion" and "Exact Match" metrics
- **Task 3a**: Added "Error Detection Acc" metric
- **Task 3b/3c/3d**: Added "Error Detection Acc" and "MAE" metrics

#### Metric Mappings
- "L/R Confusion" → `left_right_confusion`
- "Exact Match" → `exact_match`
- "Error Detection Acc" → `error_detection_acc`

#### Performance Optimizations
- **MCQA Sample Counting Cache**: Task 4 sample counts now cached in `DATASET_INDEX` during startup
  - Eliminates repeated linear scans of JSONL files
  - Significant performance improvement for MCQA navigation

### 🔧 Fixed

#### Task 4 Column Name Bugs (4 instances)
1. **Line 1694**: `create_checkpoint_table()` - Mixed-task table
   - Changed: `per_keypoint_accuracy` → `accuracy`
   - Impact: Task 4 accuracy now displays in mixed-task checkpoint table

2. **Lines 1735-1741**: `create_checkpoint_table()` - Best checkpoint selection
   - Changed: Split Task 2 and Task 4 logic (previously combined)
   - Impact: Correct "Best" checkpoint identification for both tasks

3. **Line 1790**: `create_checkpoint_table()` - Task 4 table row
   - Changed: `per_keypoint_accuracy` → `accuracy`
   - Impact: Task 4 metrics display correctly in checkpoint table

4. **Line 2026**: `create_metrics_plot()` - Metric mapping
   - Changed: Made "Accuracy" mapping task-aware
   - Logic: Task 4 uses `accuracy`, Task 2 uses `per_keypoint_accuracy`
   - Impact: Metrics plot works correctly for both tasks

#### Metric Selector Choices
Updated `get_metric_choices()` to match actual CSV data:

- **Task 2**:
  - Before: `[Accuracy, F1, Precision, Recall]`
  - After: `[Accuracy, L/R Confusion, Exact Match]`
  - Reason: F1/P/R not populated for Task 2 in CSV

- **Task 3a**:
  - Before: `[F1, Precision, Recall, OKS, PCK@50]`
  - After: `[Error Detection Acc, F1, Precision, Recall]`
  - Reason: OKS/PCK not populated for Task 3a in CSV

- **Task 3b/3c/3d**:
  - Before: `[F1, Precision, Recall, OKS, PCK@50]`
  - After: `[Error Detection Acc, F1, Precision, Recall, MAE, OKS]`
  - Reason: Added error detection and MAE, removed PCK@50 (not populated)

#### Per-Image Metrics Table
Fixed metric extraction to use actual per-sample metrics from evaluation JSONs:

**Task 1/1b/1c**:
- Headers: `[Color, Model, OKS, PCK@0.5, MAE]` → `[Color, Model, OKS, F1, Precision]`
- Reason: PCK and MAE not available in individual sample metrics

**Task 2**:
- Fixed: `left_right_confusion` → `left_right_confusion_rate`
- Reason: Metric name mismatch with evaluation JSON

**Task 3a**:
- Headers: `[Color, Model, OKS, PCK, MAE]` → `[Color, Model, F1, Precision, Recall]`
- Reason: Task 3a doesn't have position metrics, only detection metrics

**Task 3b/3c/3d**:
- Headers: `[Color, Model, OKS, PCK, MAE]` → `[Color, Model, F1, Precision, MAE (px)]`
- Fixed: `mae_total` → `mae_total_mean_corrected`
- Reason: Correct metric name for position error in corrected keypoints

**Task 4**:
- Headers: `[Color, Model, Accuracy, F1, Parse Rate]` → `[Color, Model, Correct, Predicted, Parsed]`
- Changed to show per-sample metrics: `correct` (0/1), `predicted_answer`, `parse_success` (0/1)
- Reason: Accuracy/F1 are summary-level metrics, not per-sample

### 🗑️ Removed

#### Dead Code
- **Deleted**: `_analyze_sample_old()` function (63 lines, lines 3198-3260)
- Reason: Function never called, contained unimplemented TODO
- Impact: Cleaner codebase, removed misleading code

### 🔄 Changed

#### Data Processing
- **Applied fix_task4_metrics.py**: Computed precision/recall/f1 for 5 mixed-task Task 4 evaluations
- **Updated experiments-final.csv**: Task 4 metrics now complete for mixed models
- **Backup created**: `experiments-final_backup.csv`

#### Code Organization
- `count_mcqa_samples()`: Now checks cache before performing linear scan
- `build_dataset_index()`: Caches Task 4 MCQA sample counts during startup

### 📋 Data Changes

**experiments-final.csv - Before**:
```csv
mixed_balanced_v1-step315,task4,,,,0.891239
                                ^f1 ^pr ^rec (EMPTY!)
```

**experiments-final.csv - After**:
```csv
mixed_balanced_v1-step315,task4,0.8907,0.8925,0.8907,0.891239
                                ^f1   ^pr    ^rec   (POPULATED!)
```

**Models Updated**: 5 mixed-task checkpoints now have complete Task 4 metrics

### ✅ Verification

All changes verified against:
- ✅ CSV column structure (`experiments-final.csv`)
- ✅ Evaluation JSON structure (sample files for each task)
- ✅ Actual baseline reports (IFEval, SIBench)
- ✅ Function call graphs (dead code analysis)

### 📊 Impact Summary

| Component | Before | After | Status |
|-----------|--------|-------|--------|
| Task 4 Accuracy Display | Empty ("-") | Populated | ✅ Fixed |
| Metric Selector Options | Generic | Task-specific | ✅ Improved |
| Per-Image Metrics | Incorrect extraction | Correct per-sample | ✅ Fixed |
| MCQA Sample Counting | Repeated linear scans | Cached | ✅ Optimized |
| Code Quality | 63 lines dead code | Removed | ✅ Cleaned |
| CSV Completeness | Missing Task 4 P/R/F1 | Complete | ✅ Fixed |

---

## [2026-02-05] - Previous Updates

See:
- [APP_READY.md](APP_READY.md) - App ready status
- [METRIC_EXPLANATIONS_UPDATE.md](METRIC_EXPLANATIONS_UPDATE.md) - Metric explanations
- [TASK_STRUCTURE_UPDATE.md](TASK_STRUCTURE_UPDATE.md) - Task structure updates

---

## Development Notes

### Testing Checklist

Before deploying, verify:
- [ ] Training Monitor tab shows Task 4 metrics for mixed models
- [ ] Metric selector shows task-specific options
- [ ] Per-image metrics table displays correct values
- [ ] MCQA browser loads quickly (cache working)
- [ ] All tabs render without errors

### Known Limitations

1. **Benchmark Tab**: Hardcoded IFEval baseline values (52.87%, 64.03%) serve as fallback if reports missing
   - This is intentional - provides graceful degradation
   - Values verified to match actual baseline reports

2. **Per-Image Metrics**: Show per-sample metrics, not summary aggregations
   - This is correct behavior for sample-level analysis
   - Summary metrics available in checkpoint table

### Future Enhancements

Consider adding:
- [ ] Custom Comparison implementation (currently Quick View only)
- [ ] Additional SIBench baseline fallback (currently uses report auto-detection)
- [ ] Metric explanations in tooltips
- [ ] Export comparison reports as PDF

---

## Contributors

- Claude Sonnet 4.5 (Comprehensive audit and fixes)
- User (Metrics verification and testing)
