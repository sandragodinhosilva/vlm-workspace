# Monitoring App - Ready for Use

**Date:** 2026-02-05
**Status:** ✅ PRODUCTION READY

---

## Verification Results

### CSV Data Loading ✅
```python
✅ App imports successfully
✅ CSV loaded: 108 rows, 29 columns
✅ Standardized columns present: oks_score, mae_total, mae_x, mae_y, pck_50, task
✅ Built model mapping for 53 models
```

### Task Comparison Working ✅
**Task 1 Comparison Example:**
- **Total task1 rows:** 23
- **Mixed models:** 4 rows (mixed_balanced_v1-step315/630/945/1260)
- **Task1-specific models:** 8 rows (qwen3-vl-4b-4epochs-task1-step*)
- **Baselines:** Included (qwen3-vl-4b-baseline, etc.)

**Sample Metrics (verified populated):**
```
Model: mixed_balanced_v1-step315
Task: task1
OKS: 0.007337
MAE Total: 227.568361
PCK@50: 0.121348
```

### Updates Applied ✅
1. **Backward compatibility** for metric names (line 3737)
   ```python
   f"{metrics.get('mae_total', metrics.get('coordinate_mae_total', 0)):.2f}"
   ```

2. **CSV path configured** correctly
   ```python
   EXPERIMENTS_CSV_PATH = Path("/mnt/data/sgsilva/vlm-evaluation/experiments-final.csv")
   ```

3. **Standardized metric names** used throughout:
   - `oks_score` (not `mean_oks`)
   - `mae_total` (not `coordinate_mae_total`)
   - `pck_50` (standardized format)

---

## How to Use

### Start the App
```bash
cd /mnt/data/sgsilva/monitoring-app
python app.py
```

### Access Dashboard
Open browser to the URL shown in terminal (typically `http://localhost:7860`)

### Features Available

**1. View All Checkpoints**
- See all 108 evaluations from experiments-final.csv
- Sort by any metric column
- Filter by task, model, dataset

**2. Compare Models**
- Task1: Compare all task1-trained + mixed models' task1 performance
- Task2-4: Similar multi-model comparisons
- Side-by-side metric comparison

**3. Analyze Training Progress**
- View checkpoint progression over training steps
- Track metric improvements across epochs
- Identify best performing checkpoints

**4. Mixed-Task Analysis**
- Each mixed model shows 4 rows (one per task)
- Easy comparison of multi-task vs single-task approaches
- Metrics properly populated for each task

**5. Multi-Model Comparison (Image Gallery)** ✨ NEW
- **View sample predictions** with skeleton overlays (keypoint tasks only)
- **Compare up to 4 checkpoints** side-by-side on the same image
- **Metrics comparison table** shows OKS, PCK@50, MAE for each model

**How to use:**
1. Navigate to "Browse Dataset" → "Image Gallery" tab
2. Select a keypoint task (task1, task1b, or task1c)
3. Select "test" split
4. **✓ Check "Show Model Predictions" checkbox** ← REQUIRED
5. **✓ Select 1-4 checkpoints** from the list
6. Click any image in the gallery
7. View:
   - **Left**: Ground truth (green skeleton)
   - **Right**: Predictions from selected checkpoints (colored skeletons)
   - **Below**: Metrics comparison table

**Note**: If predictions don't appear, verify the checkbox is checked and checkpoints are selected. The app will show a helpful message if predictions are missing.

---

## Data Source

**CSV File:** `/mnt/data/sgsilva/vlm-evaluation/experiments-final.csv`

**Last Updated:** 2026-02-05 12:26

**Contents:**
- 108 evaluation results
- 29 metric columns
- 53 unique models
- Tasks: task1, task1b, task1c, task2, task3, task3a, task3b, task3c, task3d, task4

**Update Command:**
```bash
cd /mnt/data/sgsilva/vlm-evaluation
python scripts/analyze_results.py results/final --export-csv experiments-final.csv
```

---

## Metrics Available in Dashboard

### Task 1: Keypoint Prediction
- OKS Score
- F1, Precision, Recall
- MAE Total, MAE X, MAE Y
- PCK@50, PCK@100, PCK@150

### Task 2: Keypoint Labeling
- Per-keypoint Accuracy
- Left-Right Confusion Rate
- Exact Match Rate

### Task 3: Error Correction
- F1, Precision, Recall
- Error Detection Accuracy
- Exact Match Rate

### Task 4: MCQA
- Accuracy
- Parse Rate

---

## Example Queries in App

**Q: Which model performs best on Task 1?**
1. Filter by task: "task1"
2. Sort by OKS or PCK@50 descending
3. See top performers (both task1-trained and mixed models)

**Q: How does mixed_balanced_v1 perform across all tasks?**
1. Filter by model: "mixed_balanced_v1"
2. View 4 rows showing performance on each task
3. Compare metrics across tasks

**Q: What's the training progression for task1_cropped_v1?**
1. Filter by model contains: "task1_cropped_v1"
2. Sort by step number
3. See metric improvements: step323 → step646 → step969 → step1292

---

## Troubleshooting

### CSV Not Found
```bash
# Regenerate CSV
cd /mnt/data/sgsilva/vlm-evaluation
python scripts/analyze_results.py results/final --export-csv experiments-final.csv
```

### Metrics Showing Zero
- Check if JSON files have been standardized
- Verify metric names match: `oks_score`, `mae_total`, etc.
- Check backup compatibility is working

### Multi-Model Comparison Not Showing Predictions
**Symptom**: Right side image is blank, metrics table shows "Enable 'Show Model Predictions'..."

**Solution**:
1. ✓ Verify "Show Model Predictions" checkbox is **CHECKED**
2. ✓ Verify you've **selected 1-4 checkpoints** from the list
3. ✓ Verify you're viewing **test split** (images only show for test)
4. ✓ Verify task is **task1, task1b, or task1c** (keypoint tasks only)

**Symptom**: Right side image shows "No predictions found"

**Solution**:
- Check that selected checkpoint names match available results files
- Verify predictions exist for the selected image
- Check logs for errors: `find_prediction_for_sample()` should show checkpoint lookups

### App Won't Start
```bash
# Check dependencies
pip install gradio pandas numpy

# Check logs
tail -50 /mnt/data/sgsilva/monitoring-app/logs/app_*.log
```

---

## Next Steps

**Ready to use immediately:**
1. Start app: `python app.py`
2. Open browser to dashboard URL
3. Explore results, compare models, analyze training

**To update with new results:**
1. Run new evaluations (generate JSON files)
2. Regenerate CSV: `python scripts/analyze_results.py results/final --export-csv experiments-final.csv`
3. Restart app to reload data

---

## Validation Summary

✅ **App loads successfully**
✅ **CSV parsing working (108 rows, 29 columns)**
✅ **Standardized metrics present and populated**
✅ **Task comparison working (mixed + single-task models)**
✅ **Backward compatibility implemented**
✅ **All 53 models mapped correctly**

**Status:** READY FOR PRODUCTION USE

---

**Last Verified:** 2026-02-05 13:05 UTC
