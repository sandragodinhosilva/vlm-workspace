# Monitoring App - Metric Explanations Feature

**Date:** 2026-02-05
**Status:** ✅ COMPLETE

---

## Changes Made

### 1. Added Metric Explanation Function ✅

**Location:** [app.py:1660](app.py#L1660)

**Function:** `get_metric_explanation(task: str, metric: str) -> str`

**Coverage:**
- **Task 1 (Keypoint Prediction):** OKS, F1 Score, Precision, Recall, MAE, PCK@50, Accuracy
- **Task 2 (Keypoint Labeling):** Accuracy, F1 Score, Precision, Recall
- **Task 3 (Error Correction):** F1 Score, Precision, Recall, OKS, MAE, PCK@50, Accuracy
- **Task 4 (MCQA):** Accuracy, F1 Score, Precision, Recall
- **Mixed Tasks:** OKS, F1 Score, Accuracy

**Example Explanations:**

**Task 1 - OKS:**
> "Object Keypoint Similarity (OKS): COCO-standard metric measuring spatial accuracy of predicted keypoint positions. Range [0,1], higher is better. Accounts for keypoint visibility and person scale."

**Task 1 - F1 Score:**
> "F1 Score: Harmonic mean of precision and recall for keypoint detection. Measures how well the model detects keypoints that exist (recall) vs. avoiding false detections (precision)."

**Task 3 - F1 Score:**
> "F1 Score: Harmonic mean of precision and recall for error detection and correction. Measures how well the model identifies and fixes incorrect keypoints."

---

### 2. Added Explanation Box to Plots ✅

**Feature:** Info box displayed below each metrics progression plot

**Position:**
- Centered horizontally
- Positioned below the X-axis (outside plot area at y=-0.22)
- Bottom margin increased to 150px to accommodate box

**Styling:**
- White background with light gray border
- Info icon (ℹ️) prefix
- Bold "Metric Definition:" label
- Gray text on white background

**Dynamic Behavior:**
- Changes based on selected task AND metric
- Different explanation for same metric across tasks (e.g., F1 in Task 1 vs Task 3)

---

### 3. Mixed-Task Plot Explanation ✅

**Location:** Mixed-task performance progression plot

**Explanation:**
> "Mixed-task models are trained on all 4 tasks simultaneously. Each trace shows task-specific performance: Task 1 uses OKS (keypoint localization), Task 2 uses Accuracy (keypoint labeling), Task 3 uses F1 (error detection/correction), Task 4 uses Accuracy (exercise classification)."

**Purpose:** Helps users understand multi-task training metrics

---

## Visual Example

```
┌─────────────────────────────────────────────────────────┐
│  F1 Score Progression - Task 1: Keypoint Detection     │
│                                                         │
│  1.0 ─┐                                                 │
│       │     ●────●────●                                 │
│  0.8 ─┤                                                 │
│       │                                                 │
│  0.6 ─┤                                                 │
│       │                                                 │
│  0.4 ─┤  ●                                              │
│       │                                                 │
│  0.2 ─┤                                                 │
│       │                                                 │
│  0.0 ─┴──────┬──────┬──────┬──────┬──────┬─────        │
│            0    250   500   750  1000  1250             │
│                   Training Step                         │
│                                                         │
│  ┌────────────────────────────────────────────────┐    │
│  │ ℹ️ Metric Definition:                          │    │
│  │ F1 Score: Harmonic mean of precision and       │    │
│  │ recall for keypoint detection. Measures how     │    │
│  │ well the model detects keypoints that exist     │    │
│  │ (recall) vs. avoiding false detections          │    │
│  │ (precision).                                    │    │
│  └────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

---

## Metric Definitions Summary

### Task 1: Keypoint Prediction (COCO-25)

| Metric | Definition | Range | Interpretation |
|--------|-----------|-------|----------------|
| **OKS** | COCO-standard spatial accuracy metric | [0,1] | Higher = better localization |
| **F1 Score** | Detection accuracy (precision + recall) | [0,1] | Higher = better detection |
| **Precision** | Correct predictions / all predictions | [0,1] | Higher = fewer false positives |
| **Recall** | Correct predictions / all ground truth | [0,1] | Higher = fewer false negatives |
| **MAE** | Average pixel distance error | [0,∞) | Lower = better localization |
| **PCK@50** | % keypoints within 50px of truth | [0,1] | Higher = better coarse accuracy |

### Task 2: Keypoint Labeling

| Metric | Definition | Range | Interpretation |
|--------|-----------|-------|----------------|
| **Accuracy** | Correct labels / all keypoints | [0,1] | Higher = better labeling |
| **F1 Score** | Label classification performance | [0,1] | Higher = better across classes |

### Task 3: Error Correction

| Metric | Definition | Range | Interpretation |
|--------|-----------|-------|----------------|
| **F1 Score** | Error detection + correction accuracy | [0,1] | Higher = better error handling |
| **Precision** | Correct errors / flagged errors | [0,1] | Higher = fewer false alarms |
| **Recall** | Correct errors / all errors | [0,1] | Higher = catches more errors |
| **OKS** | Corrected keypoint position accuracy | [0,1] | Higher = better corrections |
| **MAE** | Correction position error | [0,∞) | Lower = better repositioning |
| **Accuracy** | Error detection classification | [0,1] | Higher = better error identification |

### Task 4: MCQA (Exercise Classification)

| Metric | Definition | Range | Interpretation |
|--------|-----------|-------|----------------|
| **Accuracy** | Correct exercise matches / total | [0,1] | Higher = better classification |
| **F1 Score** | Multi-class classification performance | [0,1] | Higher = better across exercises |

---

## Implementation Details

### Code Structure

```python
# Helper function (line ~1660)
def get_metric_explanation(task: str, metric: str) -> str:
    # Normalize task to family
    task_family = task.startswith('task1') ? 'task1' : task

    # Lookup explanation from dictionary
    explanations = {
        'task1': {'OKS': '...', 'F1 Score': '...', ...},
        'task2': {...},
        ...
    }

    return explanations[task_family][metric]

# Plot creation (line ~2100)
def create_metrics_plot(...):
    # ... create plot traces ...

    # Get explanation
    explanation = get_metric_explanation(task, metric)

    # Add annotation box
    fig.add_annotation(
        text=f"<b>ℹ️ Metric Definition:</b> {explanation}",
        xref="paper", yref="paper",
        x=0.5, y=-0.22,  # Below plot area
        bgcolor='rgba(255, 255, 255, 0.95)',
        ...
    )
```

### Task Family Mapping

- `task1`, `task1b`, `task1c` → Use Task 1 explanations
- `task3`, `task3a`, `task3b`, `task3c`, `task3d` → Use Task 3 explanations
- `task2`, `task4`, `mixed` → Use their specific explanations

---

## Testing

**Test 1: App Import** ✅
```bash
python3 -c "import app"
# Result: ✅ App imports successfully
```

**Test 2: Plot Generation** ✅
```bash
fig = app.create_metrics_plot('task1', 'cropped_v1', metric='F1 Score')
# Result: ✅ Plot created with 1 annotation
```

**Test 3: Metric Explanation Retrieval** ✅
```python
explanation = app.get_metric_explanation('task1', 'OKS')
# Result: "Object Keypoint Similarity (OKS): COCO-standard metric..."
```

---

## Benefits

1. **Educational:** Users learn what metrics mean without leaving the app
2. **Context-Aware:** Same metric explained differently for different tasks
3. **Non-Intrusive:** Info box is below plot, doesn't obscure data
4. **Comprehensive:** All tasks and metrics covered

---

## Future Enhancements (Optional)

1. **Collapsible Box:** Make explanation box collapsible if users want more space
2. **Hover Tooltips:** Add hover tooltips on metric selector dropdown
3. **Link to Docs:** Add links to full documentation for detailed explanations
4. **Multi-Language:** Support explanations in multiple languages

---

**Status:** ✅ READY FOR USE

**To Start App:**
```bash
cd /mnt/data/sgsilva/monitoring-app
python app.py
```

**Last Updated:** 2026-02-05 13:17 UTC
