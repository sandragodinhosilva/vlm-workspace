# Monitoring App - Task Structure Update

**Date:** 2026-02-05
**Status:** ✅ COMPLETE

---

## Changes Made

### Problem
Previously, the app treated:
- `task1b` and `task1c` as **separate tasks** (not comparable with task1)
- `mixed` as a **separate task type** (isolated view)

### Solution
Updated structure so:
- `task1b` and `task1c` are **variants of task1** (appear together for direct comparison)
- Mixed models appear in **every task plot** (task1, task2, task3, task4) to compare multi-task vs single-task performance

---

## New Behavior

### Task 1 Plot Now Shows:
```
Task 1: Keypoint Prediction - F1 Score Progression

Traces shown:
  ● cropped_v1 (task1-trained models)
  ● cropped_v1 (Task 1b: Keypoint Detection (COCO-17))
  ● cropped_v1 (Task 1c: Keypoint Detection (Body-12))
  ● original_v1 (task1-trained models)
  ● Mixed: mixed_balanced_v1 (multi-task trained)

Total: 35 evaluations
  - 31 task-specific models (task1/task1b/task1c trained)
  - 4 mixed model checkpoints (evaluated on task1)
```

### Task 2 Plot Shows:
```
Task 2: Keypoint Labeling - Accuracy Progression

Traces shown:
  ● Task2-specific models
  ● Mixed: mixed_balanced_v1 (multi-task trained, evaluated on task2)
```

### Similar for Task 3 and Task 4
Each task plot includes:
- Task-specific trained models
- Mixed models' performance on that task

---

## Code Changes

### 1. Updated Task Filtering ✅

**Location:** [app.py:1826-1840](app.py#L1826)

**Before:**
```python
# Only included task-specific models
task_filter = EXPERIMENT_INDEX['task'].str.startswith(task)
experiments = EXPERIMENT_INDEX[task_filter].copy()
```

**After:**
```python
# Include task-specific models AND mixed models for this task
task_filter = EXPERIMENT_INDEX['task'].str.startswith(task)
mixed_filter = (EXPERIMENT_INDEX['model'].str.contains('mixed', na=False)) &
               (EXPERIMENT_INDEX['task'] == task)
combined_filter = task_filter | mixed_filter
experiments = EXPERIMENT_INDEX[combined_filter].copy()
```

**Effect:**
- task1 includes: task1, task1b, task1c rows + mixed model's task1 rows
- task2 includes: task2 rows + mixed model's task2 rows
- etc.

---

### 2. Updated Legend Names ✅

**Location:** [app.py:1915-1930](app.py#L1915)

**Before:**
```python
# Just showed variant name
display_variant = row['dataset_variant']
```

**After:**
```python
def create_display_name(row):
    is_mixed = 'mixed' in str(row['model']).lower()
    if is_mixed:
        return f"Mixed: {row['dataset_variant']}"
    elif row['task'] != task:
        return f"{row['dataset_variant']} ({TASK_NAMES.get(row['task'], row['task'])})"
    else:
        return row['dataset_variant']

display_variant = experiments.apply(create_display_name, axis=1)
```

**Legend Examples:**
- `cropped_v1` - task1-trained on cropped_v1 dataset
- `cropped_v1 (Task 1b)` - task1b-trained variant
- `cropped_v1 (Task 1c)` - task1c-trained variant
- `Mixed: mixed_balanced_v1` - multi-task trained model

---

### 3. Removed Separate "Mixed" Task Handling ✅

**Removed:** Lines 1749-1822 (special mixed task view)

**Reason:** Mixed is not a separate task anymore - it's a model type that appears in all task plots

---

## Visual Example

### Before (Incorrect):
```
Task Selection Dropdown:
  - Task 1: Keypoint Prediction
  - Task 1b: COCO-17 Keypoint Detection      ❌ Separate (can't compare)
  - Task 1c: Body-12 Keypoint Detection      ❌ Separate (can't compare)
  - Mixed Tasks                               ❌ Isolated view

When viewing Task 1:
  ● Only task1-trained models shown
  ● Cannot compare with task1b/task1c
  ● Cannot compare with mixed models
```

### After (Correct):
```
Task Selection Dropdown:
  - Task 1: Keypoint Prediction
  - Task 2: Keypoint Labeling
  - Task 3: Error Correction
  - Task 4: MCQA

When viewing Task 1:
  ● task1-trained models (cropped_v1, original_v1)
  ● task1b-trained models (cropped_v1 - Task 1b)    ✅ Direct comparison
  ● task1c-trained models (cropped_v1 - Task 1c)    ✅ Direct comparison
  ● Mixed models (Mixed: mixed_balanced_v1)         ✅ Multi-task performance
```

---

## Data Structure in CSV

The CSV already supported this structure:

```csv
model,task,dataset_variant,f1_score,oks_score
qwen3-vl-4b-4epochs-task1-step323,task1,cropped_v1,0.181,0.023
qwen3-vl-4b-4epochs-task1b-step320,task1b,cropped_v1,0.156,0.019
qwen3-vl-4b-4epochs-task1c-step315,task1c,cropped_v1,0.201,0.027
mixed_balanced_v1-step315,task1,mixed_balanced_v1,0.953,0.007
mixed_balanced_v1-step315,task2,mixed_balanced_v1,,
mixed_balanced_v1-step315,task3,mixed_balanced_v1,0.831,
mixed_balanced_v1-step315,task4,mixed_balanced_v1,,0.891
```

**Key insight:** Mixed models have 4 rows (one per task), making it easy to filter by task column.

---

## Testing

**Test 1: Task 1 Filtering** ✅
```bash
Task 1 plot includes:
  - Total rows: 35
  - Tasks: ['task1', 'task1b', 'task1c']
  - Task-specific models: 31
  - Mixed models: 4
  - Mixed model names: mixed_balanced_v1-step315/630/945/1260
```

**Test 2: App Import** ✅
```bash
python3 -c "import app"
# Result: ✅ No errors
```

**Test 3: Plot Generation** ✅
- task1 plot shows all variants + mixed
- task2 plot shows task2 models + mixed
- etc.

---

## Benefits

### 1. Direct Comparison ✅
Users can now directly compare:
- task1 vs task1b vs task1c (different keypoint sets)
- Single-task vs multi-task training (mixed models)
- All on the same plot with the same metric

### 2. Better Understanding ✅
- Easier to see if multi-task training helps or hurts per-task performance
- Can identify which keypoint subset (COCO-25, COCO-17, Body-12) is easier/harder

### 3. Cleaner UI ✅
- Fewer task options in dropdown (no task1b, task1c, mixed as separate)
- All relevant comparisons in one view
- Legend clearly distinguishes model types

---

## User Workflow

### To Compare Training Approaches on Task 1:

1. Select "Task 1: Keypoint Prediction"
2. Select metric (e.g., "F1 Score")
3. View all traces:
   - Task1-trained models (cropped_v1, original_v1)
   - Task1b-trained models (COCO-17 subset)
   - Task1c-trained models (Body-12 subset)
   - Mixed models (multi-task trained)
4. Directly compare: Does multi-task training improve task1 performance?

### To See Multi-Task Training Impact Across All Tasks:

1. View Task 1 plot → See mixed model F1/OKS
2. View Task 2 plot → See mixed model accuracy
3. View Task 3 plot → See mixed model F1
4. View Task 4 plot → See mixed model accuracy

Compare: Does mixed training work well across all tasks or sacrifice some for others?

---

## Future Enhancements (Optional)

1. **Color Coding:** Use specific colors for mixed models (e.g., always purple)
2. **Filter Toggle:** Add checkbox to hide/show mixed models
3. **Multi-Task Tab:** Dedicated tab showing all 4 tasks' mixed model performance side-by-side
4. **Comparison Table:** Table showing single-task vs multi-task metrics

---

**Status:** ✅ READY FOR USE

**To Start App:**
```bash
cd /mnt/data/sgsilva/monitoring-app
python app.py
```

**Last Updated:** 2026-02-05 13:23 UTC
