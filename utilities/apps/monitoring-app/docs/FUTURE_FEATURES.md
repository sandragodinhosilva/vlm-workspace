# Future Features / TODO

## Removed Features (To Be Implemented)

### Per-Sample Analysis
**Status**: UI removed, partial backend exists
**Location**: Evaluation Dashboard → Would be a new view mode

**Description**: Analyze individual sample predictions in detail
- Select a checkpoint and sample ID
- View the image with prediction overlay
- See side-by-side comparison of prediction vs ground truth
- Display per-sample metrics

**Backend Functions Available**:
- `_analyze_sample_old()` - needs updating
- `find_prediction_for_sample()` - working

**Implementation Needed**:
- [ ] Update analyze_sample function for all task types
- [ ] Create UI for sample navigation (slider/input)
- [ ] Add sample image display with predictions
- [ ] Format prediction vs GT comparison text
- [ ] Show per-sample metrics table

---

### Ablation Study
**Status**: UI removed, backend function exists
**Location**: Evaluation Dashboard → Would be a new view mode

**Description**: Compare different variants and configurations for the same task
- Select a task
- Automatically compare all variants (v1, v2, v3, v4, etc.)
- Show comparison table with best metrics per variant
- Visualize differences in bar/radar chart

**Backend Functions Available**:
- `generate_ablation_study()` - implemented but untested

**Implementation Needed**:
- [ ] Test generate_ablation_study() function
- [ ] Create clean UI for ablation comparison
- [ ] Add variant selection/filtering
- [ ] Improve visualization (maybe stacked bars)
- [ ] Add statistical significance indicators

---

## Enhancement Ideas

### Training Monitor
- [ ] Add download button for metrics data (CSV export)
- [ ] Show confidence intervals if multiple runs
- [ ] Add vertical line to mark best checkpoint
- [ ] Toggle to show/hide baseline in plot
- [ ] Add hover tooltips showing exact metric values

### Dataset Explorer
- [ ] Batch image download
- [ ] Filter samples by metadata (exercise type, dimensions, etc.)
- [ ] Search samples by image ID
- [ ] Show prediction confidence scores
- [ ] Highlight samples with largest errors

### Evaluation Dashboard
- [ ] Statistical significance tests between models
- [ ] Error analysis by category (per exercise, per keypoint)
- [ ] Export comparison reports as PDF
- [ ] Add more chart types (box plots, violin plots)
- [ ] Confusion matrix for task2 (L/R confusion visualization)

### General
- [ ] Dark mode support
- [ ] Keyboard shortcuts for navigation
- [ ] Save/load custom views
- [ ] User notes/annotations on samples
- [ ] Integration with TensorBoard logs
- [ ] Real-time monitoring during training

---

## Technical Debt

- [ ] Complete error handling for all data loading functions
- [ ] Add input validation for all user inputs
- [ ] Optimize caching strategy (currently using lru_cache)
- [ ] Add comprehensive logging
- [ ] Write unit tests for core functions
- [ ] Document all functions with docstrings
- [ ] Refactor large functions (esp. on_gallery_select)

---

**Last Updated**: 2026-02-03
