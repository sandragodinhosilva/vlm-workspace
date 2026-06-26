# Data Validation Checks - Monitoring App

## Overview
This document describes the comprehensive validation checks added to ensure that metrics and predictions displayed in the monitoring app correctly match the selected models and images.

## Critical Bug Fixes

### 1. Gallery Pagination Bug (Fixed)
**Issue**: When clicking gallery images on pages > 1, wrong samples were loaded
**Root Cause**: `evt.index` was used directly instead of calculating absolute sample index
**Fix**: Calculate absolute index: `sample_idx = page * 50 + evt.index`
**Location**: `app.py:3554`

### 2. Loose Image ID Matching (Fixed)
**Issue**: Predictions from wrong images were displayed with mismatched OKS scores
**Root Cause**: Fallback matching only compared first part of image_id (e.g., "10012")
**Example**: Image `10012_491938_...812` matched predictions for `10012_491938_...797`
**Fix**: Use exact image_id matching only (removed fallback logic)
**Location**: `app.py:2866`

## Validation Checks Implemented

### A. Image-to-Prediction Matching
**Location**: `find_prediction_for_sample()` (app.py:2832-2896)

**Checks**:
1. ✅ **Exact Image ID Match**: Only returns predictions with exact matching image_id
2. ✅ **Metadata Validation**: Logs checkpoint/task/variant from results file
3. ✅ **OKS Range Validation**: Verifies OKS scores are between 0 and 1
4. ✅ **Missing Data Warnings**: Warns when evaluation results file not found
5. ✅ **Empty Results Warnings**: Warns when detailed results missing from file

**Logging Added**:
- `⚠️  No evaluation results file found for checkpoint: {name}`
- `⚠️  Image {id} NOT FOUND in {file} (has {N} samples)`
- `❌ Invalid OKS score {oks} for image {id}`
- `✓ Found prediction for image {id} (OKS: {oks})`

### B. Gallery Selection Validation
**Location**: `on_gallery_select()` (app.py:3549-3810)

**Single Checkpoint Case** (app.py:3676-3714):
1. ✅ **Image ID Verification**: Confirms loaded prediction matches requested image_id
2. ✅ **Metric Range Check**: Validates OKS is in [0, 1] range
3. ✅ **Data Invalidation**: Sets `pred_data = None` if validation fails

**Multiple Checkpoints Case** (app.py:3723-3773):
1. ✅ **Per-Checkpoint Image ID Verification**: Validates each checkpoint's predictions
2. ✅ **Skip on Mismatch**: Uses `continue` to skip invalid predictions
3. ✅ **Metric Validation**: Checks OKS range for each checkpoint
4. ✅ **Result File Logging**: Logs which result file was used

**Logging Added**:
- `❌ IMAGE ID MISMATCH! Requested: {id1}, Got: {id2}`
- `❌ Invalid OKS {oks} for {checkpoint}`
- `✓ Validated: {checkpoint} → OKS={oks} for image {id}`
- `✓ Loaded predictions from: {filename}`

### C. Comparison Generation Validation
**Location**: `generate_custom_comparison()` (app.py:2286-2550)

**Checks**:
1. ✅ **Checkpoint Logging**: Logs all checkpoints being compared
2. ✅ **Metric Range Validation**: Validates OKS, F1, Precision, Recall in [0, 1]
3. ✅ **Missing Checkpoint Warnings**: Warns when checkpoint not in experiments CSV
4. ✅ **Task/Variant Logging**: Logs task and variant being compared

**Logging Added**:
- `📊 Generating comparison for {N} checkpoints:`
- `❌ Invalid OKS {oks} for {checkpoint} in comparison`
- `❌ Invalid F1 {f1} for {checkpoint} in comparison`
- `⚠️  Checkpoint not found in experiments: {name}`

## Testing Recommendations

### 1. Image ID Matching Test
**Purpose**: Verify predictions match selected images
**Steps**:
1. Navigate to Dataset Explorer → task1 → any variant
2. Select a sample from gallery (note the Image ID)
3. Enable "Show Model Predictions" and select 2+ checkpoints
4. Check logs for: `✓ Validated: {checkpoint} → OKS={oks}`
5. Verify OKS scores match values in evaluation results JSON files

**Expected**: Log shows validated predictions with matching image IDs

### 2. Pagination Test
**Purpose**: Verify correct samples load on all pages
**Steps**:
1. Navigate to page 2 or 3 of gallery
2. Click first image on the page
3. Check Image ID and Sample Index match
4. Verify Ground Truth and Predictions show same person/image

**Expected**: Image ID, Ground Truth, and Predictions all match

### 3. Cross-Reference Test
**Purpose**: Manually verify displayed metrics match source data
**Steps**:
1. Note checkpoint name and image_id from UI
2. Find the evaluation results JSON file (check logs for filename)
3. Search JSON for the image_id
4. Compare OKS score in JSON vs. displayed in UI

**Expected**: Scores match exactly (within 0.001 precision)

### 4. Invalid Data Test
**Purpose**: Verify warnings appear for missing/invalid data
**Steps**:
1. Select a checkpoint that doesn't have results for current variant
2. Check logs for warning messages
3. Verify UI doesn't show incorrect data from other variants

**Expected**: Warning logs appear, no predictions shown

## Log Interpretation Guide

### Success Indicators
- `✓ Found prediction for image` - Prediction successfully loaded and validated
- `✓ Validated: {checkpoint} → OKS={oks}` - Metrics validated for display
- `✓ Loaded predictions from: {file}` - Confirms source file

### Warning Indicators
- `⚠️  No evaluation results file` - Results not available (expected for some checkpoints)
- `⚠️  Image {id} NOT FOUND` - Prediction missing for specific image
- `⚠️  Checkpoint not found` - Checkpoint not in experiments CSV

### Error Indicators (CRITICAL - Should Not Occur)
- `❌ IMAGE ID MISMATCH!` - Wrong prediction loaded (BUG)
- `❌ Invalid OKS score` - Metric out of range (DATA CORRUPTION or BUG)
- `❌ Invalid F1/Precision/Recall` - Metric out of range (DATA CORRUPTION or BUG)

**If you see ❌ errors**: Check the evaluation results JSON file for data corruption or code bugs.

## Monitoring Best Practices

1. **Always check logs** when viewing predictions to confirm validation passed
2. **Cross-reference** critical metrics with source JSON files
3. **Report mismatches** immediately if ❌ errors appear in logs
4. **Review validation logs** before trusting displayed metrics for research/papers

## Files Modified

- `app.py:2832-2896` - Enhanced `find_prediction_for_sample()` with validation
- `app.py:3549-3810` - Added validation to `on_gallery_select()`
- `app.py:2286-2550` - Added validation to `generate_custom_comparison()`

## Future Enhancements

1. **UI Validation Warnings**: Show validation status in the UI (not just logs)
2. **Automatic Cross-Check**: Periodically verify random samples against source data
3. **Validation Report**: Generate summary report of validation checks on app startup
4. **Metric Checksums**: Add checksums to results files to detect corruption
