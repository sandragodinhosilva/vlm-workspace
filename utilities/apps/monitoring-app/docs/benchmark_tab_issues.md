# Benchmark Tab Implementation - Issues & Bugs Analysis

## 🔴 CRITICAL ISSUES

### 1. **SIBench Baseline Data Mismatch**
**Location**: `load_benchmarks_index()` line 735-737
**Issue**: Hardcoded baseline value (38.86%) doesn't match actual report data (56.6%)
```python
# Current (WRONG):
index['sibench']['baseline'] = {'overall': 38.86}

# Actual data from report shows: 56.6%
```
**Impact**: All SIBench delta calculations will be incorrect
**Fix**: Don't hardcode baseline - extract it from the actual report or use the value set at line 819

---

### 2. **Baseline Overwrite Race Condition**
**Location**: `load_benchmarks_index()` line 817-819
**Issue**: If multiple models contain 'baseline' in their name, the last one overwrites previous
```python
if 'baseline' in model_name.lower():
    index['sibench']['baseline']['overall'] = overall
```
**Impact**: Unpredictable baseline selection if multiple baseline models exist
**Fix**: Be more specific (exact match) or use first/best match logic

---

### 3. **Event Handlers Not Wired When No Data Exists**
**Location**: `build_ui()` line 3601-3620
**Issue**: Event handlers only wired if `BENCHMARKS_INDEX` has data
```python
if BENCHMARKS_INDEX and BENCHMARKS_INDEX.get('ifeval', {}).get('models'):
    # Wire event handlers
```
**Impact**: If app starts without benchmark data, adding data later won't enable functionality
**Fix**: Always wire event handlers, just handle empty data in the handler functions

---

### 4. **app.load() Won't Execute Without Data**
**Location**: `build_ui()` line 3616-3620
**Issue**: Initial data load is inside the conditional block
```python
if BENCHMARKS_INDEX and BENCHMARKS_INDEX.get('ifeval', {}).get('models'):
    app.load(fn=lambda: on_benchmark_change("IFEval"), ...)
```
**Impact**: UI won't populate on initial load if no data exists
**Fix**: Move app.load() outside the conditional, let handler deal with empty data

---

### 5. **Missing Error Handling in Event Handlers**
**Location**: `on_benchmark_change()` and `on_model_select()` (lines 3419-3504)
**Issue**: No try-except blocks around data access
```python
def on_benchmark_change(benchmark_type):
    # Direct dictionary access without error handling
    model_choices = list(BENCHMARKS_INDEX['ifeval']['models'].keys())
```
**Impact**: Any data inconsistency will crash the UI with KeyError
**Fix**: Wrap in try-except and return graceful error states

---

### 6. **Hardcoded Baseline Ignored**
**Location**: `load_benchmarks_index()` lines 728-737
**Issue**: Hardcoded IFEval and SIBench baselines never get overwritten by actual data
```python
# These are set at the beginning:
index['ifeval']['baseline'] = {'prompt_strict': 52.87, 'instr_strict': 64.03}
index['sibench']['baseline'] = {'overall': 38.86}

# But IFEval baseline model is also parsed from reports
# SIBench baseline is overwritten at line 819, but IFEval baseline never is
```
**Impact**: If baseline model name changes or metrics update, hardcoded values will be stale
**Fix**: Parse baseline from actual report files instead of hardcoding

---

## ⚠️ MEDIUM ISSUES

### 7. **Baseline Detection Too Broad**
**Location**: Multiple places (e.g., line 818)
**Issue**: `'baseline' in model_name.lower()` matches too broadly
```python
# Would match all of these:
# - "qwen3-vl-4b-baseline"
# - "baseline-v2-experimental"  ✗ (unintended)
# - "non-baseline-test"  ✗ (unintended)
# - "my_baseline_model"  ✗ (unintended)
```
**Fix**: Use stricter matching or exact name comparison

---

### 8. **Missing Data Validation**
**Location**: `create_sibench_table()` line 1844-1859
**Issue**: Assumes baseline dict exists and has 'overall' key
```python
baseline = BENCHMARKS_INDEX['sibench']['baseline']
# No check if baseline is empty {}
baseline['overall']  # KeyError if baseline wasn't set
```
**Fix**: Add defensive checks before accessing nested dicts

---

### 9. **Empty SIBench Chart**
**Location**: `on_benchmark_change()` line 3441
**Issue**: SIBench returns empty Figure instead of meaningful visualization
```python
chart = go.Figure()  # No chart for SIBench yet
```
**Impact**: Users see blank chart when selecting SIBench
**Fix**: Create a per-task comparison bar chart or radar chart for SIBench

---

### 10. **Status Logic Edge Case**
**Location**: `on_model_select()` line 3472
**Issue**: Complex ternary logic with potential edge case at -15%
```python
status = ('✅ Maintained' if abs(delta) <= 5
          else '⚠️ Degraded' if delta < -5 and delta > -15  # What about delta == -15?
          else '🔴 Severely Degraded')
```
**Fix**: Simplify to if-elif-else for clarity

---

### 11. **No Validation of model_data**
**Location**: `on_model_select()` lines 3458, 3482
**Issue**: Doesn't validate model exists before accessing
```python
model_data = BENCHMARKS_INDEX['ifeval']['models'].get(model_name, {})
# Returns {} if not found, then:
model_data.get('prompt_strict', 0)  # Returns 0 - misleading!
```
**Impact**: Shows "0.00%" for missing models instead of error message
**Fix**: Check if model_data is empty and return appropriate message

---

### 12. **Missing per_task Validation**
**Location**: `on_model_select()` line 3494
**Issue**: Assumes per_task exists
```python
per_task = model_data.get('per_task', {})
for task, score in sorted(per_task.items()):  # Works, but...
```
**Concern**: If per_task parsing failed, this silently shows no tasks
**Fix**: Add message if per_task is empty

---

### 13. **Inconsistent Sorting**
**Location**: `create_sibench_table()` line 1858
**Issue**: SIBench sorts by overall (descending), IFEval sorts by delta (ascending)
```python
# IFEval: sorted by worst degradation first (line 1755)
sorted_models = sorted(models.items(), key=lambda x: x[1]['delta_prompt'])

# SIBench: sorted by best performance first (line 1858)
for model_name, metrics in sorted(models.items(), key=lambda x: x[1]['overall'], reverse=True):
```
**Impact**: Inconsistent user experience
**Fix**: Decide on consistent sorting strategy (e.g., always worst first)

---

### 14. **Regex Import Inside Function**
**Location**: `load_benchmarks_index()` line 749
**Issue**: `import re` inside function, but re is used throughout
```python
def load_benchmarks_index():
    # ... code ...
    import re  # Line 749
    prompt_match = re.search(...)  # Line 750
    # ... more re usage at lines 780, 790, 796 ...
```
**Fix**: Move `import re` to top of file with other imports

---

## 📝 MINOR ISSUES

### 15. **Hardcoded Model Names in Tables**
**Location**: `create_ifeval_table()` line 1746, `create_sibench_table()` line 1851
**Issue**: Baseline row shows hardcoded names that might not match actual model
```python
'Model': 'Baseline (Qwen3-VL-4B-Instruct)',  # IFEval
'Model': 'Baseline (qwen3-vl-4b)',  # SIBench
```
**Fix**: Extract actual baseline model name from data

---

### 16. **Long Model Names in Chart**
**Location**: `create_ifeval_chart()` line 1832
**Issue**: X-axis with long model names at -45° might still be hard to read
```python
xaxis=dict(tickangle=-45)
```
**Suggestion**: Consider abbreviating names or using horizontal bar chart

---

### 17. **No Logging in Visualization Functions**
**Location**: All `create_*` functions
**Issue**: If chart creation fails, no logging to debug
**Fix**: Add logging for chart creation steps

---

### 18. **Summary Loads Entire BENCHMARKS_TESTED.md**
**Location**: `load_benchmarks_index()` line 724-726
**Issue**: Loads entire file (300+ lines) into summary field
```python
with open(benchmarks_file, 'r') as f:
    content = f.read()
    index['summary'] = content  # All 300+ lines
```
**Impact**: Clutters UI if displayed directly
**Fix**: Extract only key findings section or summary paragraph

---

### 19. **No Caching for Visualization Functions**
**Location**: All `create_*` functions
**Issue**: Recreate DataFrames on every call, but data is static
**Suggestion**: Could use @lru_cache if beneficial for performance

---

### 20. **No Refresh Mechanism**
**Issue**: If benchmark reports are updated, must restart app to reload
**Suggestion**: Add a "Refresh Benchmark Data" button to reload BENCHMARKS_INDEX

---

### 21. **File Path Assumptions**
**Location**: Throughout `load_benchmarks_index()`
**Issue**: Hardcoded paths like `/mnt/data/sgsilva/vlm-evaluation/results/reports`
**Suggestion**: Use config file or environment variables for paths

---

### 22. **No Dropdown Pagination**
**Location**: `on_benchmark_change()` line 3432, 3443
**Issue**: If 50+ models exist, dropdown will be very long
**Suggestion**: Consider table-based selection or searchable dropdown

---

### 23. **Inconsistent Error Returns**
**Location**: Various functions
**Issue**: Some return None, others return empty DataFrame, others return empty dict
**Fix**: Standardize error return values across all functions

---

### 24. **Missing Type Hints**
**Location**: `on_benchmark_change()` and `on_model_select()`
**Issue**: Event handler functions don't have type hints
**Fix**: Add type hints for consistency with other functions

---

## 🐛 POTENTIAL RUNTIME BUGS

### 25. **Race Condition in Report Parsing**
**Location**: `load_benchmarks_index()` line 771
**Issue**: Uses `sorted(..., reverse=True)` to get latest report, but what if multiple reports have same timestamp?
```python
report_files = sorted(sibench_dir.glob("report_*.md"), reverse=True)
```
**Impact**: Non-deterministic behavior
**Fix**: Add tiebreaker or use modification time

---

### 26. **Regex Pattern Fragility**
**Location**: Multiple regex patterns in `load_benchmarks_index()`
**Issue**: Patterns assume specific formatting
```python
# Line 750: Assumes "Prompt-Level Strict" with specific spacing
prompt_match = re.search(r'Prompt-Level\s+Strict.*?(\d+\.\d+)%', report_content)

# Line 790: Assumes specific table format with exact spacing
overall_match = re.search(r'\|\s+\*\*OVERALL\*\*\s+\|[^|]+\|[^|]+\|\s+\*\*(\d+\.?\d*)%\*\*', section_content)
```
**Impact**: Any format change breaks parsing
**Fix**: Make patterns more flexible or add validation

---

### 27. **Missing Boundary Check**
**Location**: `load_benchmarks_index()` line 782-784
**Issue**: `for i in range(0, len(model_sections), 2)` assumes even-length array
```python
for i in range(0, len(model_sections), 2):
    if i + 1 >= len(model_sections):  # Check exists, good!
        break
```
**Status**: Actually handled correctly, but worth noting

---

## ✅ THINGS NOT IMPLEMENTED (From Plan)

### 28. **SIBench Radar Chart**
**Plan**: Phase 3.4 - create_sibench_radar()
**Status**: Not implemented - only table and summary exist
**Impact**: Less visual insight into per-task performance

---

### 29. **Key Findings Accordion**
**Plan**: Phase 2.3 - Accordion with recommendations from BENCHMARKS_TESTED.md
**Status**: Not implemented in UI
**Current**: Summary text is shown but not in accordion format

---

### 30. **Statistical Significance Indicators**
**Plan**: FUTURE_FEATURES.md mentioned significance tests
**Status**: Not implemented - no significance testing between models

---

## 🎯 RECOMMENDATIONS

### Priority 1 (Fix Immediately):
1. Fix SIBench baseline mismatch (#1)
2. Wire event handlers unconditionally (#3)
3. Add error handling to event handlers (#5)
4. Move app.load() outside conditional (#4)

### Priority 2 (Fix Soon):
5. Don't hardcode baselines - parse from reports (#6)
6. Improve baseline detection logic (#7, #2)
7. Add data validation in visualization functions (#8, #11)
8. Create SIBench visualization chart (#9)

### Priority 3 (Nice to Have):
9. Move re import to top of file (#14)
10. Add logging to visualization functions (#17)
11. Extract only summary section from BENCHMARKS_TESTED.md (#18)
12. Add refresh button for benchmark data (#20)

### Future Enhancements:
- Implement SIBench radar chart
- Add statistical significance tests
- Add searchable/paginated model selector
- Make paths configurable
