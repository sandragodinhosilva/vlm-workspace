# Utilities Changelog

## 2026-02-12: Full Training Run Inventory Update

### Changes to `cleanup_checkpoints.py`

#### 1. Added all training runs (21 total)
Updated `FALLBACK_BEST_CHECKPOINTS` with all current runs organized by task:
- Task 1: 6 runs (original, 1b_cropped, 1c_cropped, cropped_v2, mixed_balanced_v1, task1_original)
- Task 2: 3 runs (v2, v4, v5)
- Task 3: 5 runs (3a_high, 3b_low_missing, 3c_background, 3c_small, 3d_mixed)
- Task 4: 7 runs (v1, v3, v5, v5.1, v5.3, v6.1.2, v6.2)

Runs with `None` best checkpoint = eval pending, keeps all checkpoints.

#### 2. Fixed glob pattern
Script now scans both `sft_vlm_megatron_4b_4epochs*` (old naming) and `sft_vlm_4b_4epochs*` (new naming). Previously missed all Task 4 MCQA and task1_cropped_v2 runs.

#### 3. Updated task_to_run mapping
Added 7 new entries: task1_cropped_v2, task2_v5, task3d_v1_mixed, task4_mcqa_v5.3, task4_mcqa_v6.1.2, task4_mcqa_v6.2.

#### 4. Added eval-pending handling
Runs with `best_step=None` now print "Eval pending" instead of a generic warning, and keep all checkpoints until evaluation determines the best.

---

## 2026-02-02: Checkpoint Cleanup Updates

### Changes to `cleanup_checkpoints.py`

#### 1. Updated Best Checkpoints (Based on Evaluation Results)
Updated the `BEST_CHECKPOINTS` dictionary based on latest evaluation reports from `/mnt/data/sgsilva/vlm-evaluation/results/evaluations` while preserving the current utilities paths for checkpoints at `/mnt/data/sgsilva/checkpoints` and exports at `/mnt/data/sgsilva/models`:

- **task3b_low_missing**: Changed from `step_338` → `step_1131` (Epoch 3, F1=77.0%)
- **task3c_small_displacement**: Changed from `step_338` → `step_1352` (Epoch 4, F1=73.7%)

Other checkpoints remain unchanged:
- task1_original: `step_648` (Epoch 2)
- task2_v2: `step_969` (Epoch 3)
- task2_v4: `step_1328` (Epoch 4)
- task3a_high: `step_646` (Epoch 2, F1=19.3%)
- task3c_background_displacement: `step_338` (Epoch 1)

#### 2. Added Fallback Logic
When the best checkpoint is not found, the script now:
1. Shows a warning message
2. Automatically selects the most recent checkpoint (highest step number) as a fallback
3. Keeps the fallback checkpoint instead of keeping ALL checkpoints
4. Marks the kept checkpoint as "(FALLBACK)" instead of "(BEST)"

**Example output:**
```
⚠️  WARNING: Best checkpoint step_1131 not found!
Available: step_338
📌 Using fallback checkpoint: step_338
✓ Keeping (FALLBACK): step_338
```

This ensures:
- At least one checkpoint is always preserved (no loss of training work)
- Disk space is still saved by deleting other intermediate checkpoints
- Clear indication when fallback logic is used

### Files Modified
- `/mnt/data/sgsilva/utilities/cleanup_checkpoints.py`

### Files Not Changed
- `cleanup_qwen_checkpoints.py` - Already has fallback logic (keeps final checkpoint)
- `cleanup_all.py` - Doesn't reference BEST_CHECKPOINTS dictionary

### Testing
Verified with dry-run mode:
```bash
python cleanup_checkpoints.py --dry-run
```

Results:
- ✅ Fallback logic correctly activates for task3b and task3c
- ✅ Existing checkpoints are preserved when best isn't available
- ✅ Script outputs clear warnings about fallback usage
