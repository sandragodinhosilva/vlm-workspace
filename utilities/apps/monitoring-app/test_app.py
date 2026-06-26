#!/usr/bin/env python3
"""
Smoke tests for the monitoring app.

Catches the class of bugs we keep hitting:
- NameErrors from stale references after refactoring
- Broken function signatures (wrong arg count / missing params)
- Invalid CONFIG keys
- Malformed external data files (training_notes.md, archived_experiments.json)
- Gradio wiring mismatches (output count != component count)

Run:  python test_app.py
      python -m pytest test_app.py -v
"""

import ast
import json
import re
import sys
from pathlib import Path

APP_PATH = Path(__file__).parent / "app.py"
APP_SOURCE = APP_PATH.read_text()
APP_AST = ast.parse(APP_SOURCE)

# ---------------------------------------------------------------------------
# 1. Syntax & Import Sanity
# ---------------------------------------------------------------------------

def test_syntax_valid():
    """app.py parses without SyntaxError."""
    ast.parse(APP_SOURCE)  # would raise on failure


def test_no_undefined_name_patterns():
    """Catch stale references like _empty_fig that should be empty_figure."""
    # Known patterns that have caused NameErrors
    dangerous = [
        r'\b_empty_fig\b(?!\s*[=:(])',   # _empty_fig used but not defined/param
        r'\breas_split_radio\b',          # should be split_radio after refactor
    ]
    for pattern in dangerous:
        matches = re.findall(pattern, APP_SOURCE)
        # Filter out: function parameter definitions, comments
        real_matches = []
        for m in re.finditer(pattern, APP_SOURCE):
            line_start = APP_SOURCE.rfind('\n', 0, m.start()) + 1
            line = APP_SOURCE[line_start:APP_SOURCE.find('\n', m.end())]
            # Skip if it's a function parameter def or a comment
            if 'def ' in line and m.group() in line.split('def ')[0]:
                continue
            if line.strip().startswith('#'):
                continue
            # _empty_fig as a parameter name is OK
            if f'{m.group()}' in line and ('def ' in line or f'{m.group()}):' in line or f'{m.group()})' in line or f'{m.group()},' in line):
                # Check if it's actually a parameter definition
                if 'def ' in line:
                    continue
            real_matches.append((APP_SOURCE.count('\n', 0, m.start()) + 1, line.strip()))
        # _empty_fig is allowed as a function parameter name
        if pattern == r'\b_empty_fig\b(?!\s*[=:(])':
            real_matches = [(ln, l) for ln, l in real_matches if 'def ' not in l and '_empty_fig)' not in l and '_empty_fig,' not in l]
        assert not real_matches, f"Stale reference '{pattern}' found at: {real_matches}"


def test_no_emoji_in_tab_names():
    """Tab names and section headers should be emoji-free."""
    emoji_pattern = re.compile(
        "[\U0001F300-\U0001F9FF\U00002700-\U000027BF\U0001FA00-\U0001FA6F]"
    )
    # Find gr.Tab("...") calls
    tab_pattern = re.compile(r'gr\.Tab\(\s*"([^"]+)"')
    for m in tab_pattern.finditer(APP_SOURCE):
        tab_name = m.group(1)
        assert not emoji_pattern.search(tab_name), \
            f"Emoji found in tab name: '{tab_name}' at line {APP_SOURCE.count(chr(10), 0, m.start()) + 1}"

    # Find gr.Markdown("### ...") section headers
    header_pattern = re.compile(r'gr\.Markdown\(\s*"###\s*([^"]+)"')
    for m in header_pattern.finditer(APP_SOURCE):
        header = m.group(1)
        assert not emoji_pattern.search(header), \
            f"Emoji found in section header: '{header}'"


# ---------------------------------------------------------------------------
# 2. CONFIG Block
# ---------------------------------------------------------------------------

def test_config_has_required_keys():
    """CONFIG dict has all expected keys."""
    required = [
        "datasets_base_path", "models_base_path", "results_base_path",
        "evaluations_path", "experiments_csv_path", "reasoning_data_path",
        "show_evaluation_dashboard", "show_benchmarks_eval", "show_mixed_vs_single",
        "gallery_page_size", "server_port",
    ]
    # Extract CONFIG dict from source using AST
    config_node = None
    for node in ast.walk(APP_AST):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == 'CONFIG':
                    config_node = node
                    break
    assert config_node is not None, "CONFIG dict not found in app.py"
    # Check keys exist in source text around CONFIG
    config_text = APP_SOURCE[config_node.col_offset:].split('\n}')[0]
    for key in required:
        assert f'"{key}"' in config_text, f"Missing CONFIG key: {key}"


def test_config_feature_flags_are_bool():
    """Feature flags should be boolean values."""
    flag_pattern = re.compile(r'"(show_\w+)":\s*(True|False)')
    flags_found = flag_pattern.findall(APP_SOURCE)
    assert len(flags_found) >= 3, f"Expected >=3 feature flags, found {len(flags_found)}"
    for name, val in flags_found:
        assert val in ('True', 'False'), f"Feature flag {name} has non-bool value: {val}"


# ---------------------------------------------------------------------------
# 3. Cache Management
# ---------------------------------------------------------------------------

def test_no_raw_lru_cache():
    """All caches should use @cacheable, not raw @lru_cache."""
    lines = APP_SOURCE.splitlines()
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith('@lru_cache'):
            # Check context — is it inside the cacheable() definition itself?
            context = '\n'.join(lines[max(0, i-5):i+3])
            if 'def cacheable' in context:
                continue  # This is the implementation, not a usage
            assert False, f"Raw @lru_cache at line {i} — use @cacheable instead"


def test_cacheable_decorator_exists():
    """cacheable() decorator and clear_all_caches() must exist."""
    assert 'def cacheable(' in APP_SOURCE
    assert 'def clear_all_caches(' in APP_SOURCE
    assert '_ALL_CACHES' in APP_SOURCE


# ---------------------------------------------------------------------------
# 4. empty_figure() Consistency
# ---------------------------------------------------------------------------

def test_empty_figure_defined():
    """Global empty_figure() helper must exist."""
    assert 'def empty_figure(' in APP_SOURCE


def test_no_inline_empty_figures():
    """No inline go.Figure() + add_annotation patterns — use empty_figure()."""
    # Pattern: fig = go.Figure() followed by fig.add_annotation(text=...
    # within a few lines, with "No data" or "No results" style message
    lines = APP_SOURCE.splitlines()
    violations = []
    for i, line in enumerate(lines):
        if 'go.Figure()' in line and 'def empty_figure' not in line:
            # Check next 5 lines for add_annotation with a "no data" message
            context = '\n'.join(lines[i:i+6]).lower()
            if 'add_annotation' in context and any(w in context for w in ['no data', 'no result', 'no eval', 'not available']):
                violations.append(i + 1)
    assert not violations, \
        f"Inline empty figure patterns at lines {violations} — use empty_figure() instead"


# ---------------------------------------------------------------------------
# 5. External Data Files
# ---------------------------------------------------------------------------

def test_archived_experiments_json_valid():
    """archived_experiments.json is valid JSON with expected structure."""
    archive_path = Path(__file__).parent / "archived_experiments.json"
    if not archive_path.exists():
        return  # File is optional (created on first archive)
    data = json.loads(archive_path.read_text())
    assert "archived" in data, "Missing 'archived' key"
    assert isinstance(data["archived"], list), "'archived' must be a list"
    for entry in data["archived"]:
        assert "task" in entry and "variant" in entry, \
            f"Archive entry missing task/variant: {entry}"


def test_training_notes_md_parseable():
    """training_notes.md parses into task sections correctly."""
    notes_path = Path(__file__).parent / "training_notes.md"
    assert notes_path.exists(), "training_notes.md not found"
    content = notes_path.read_text()
    # Must have at least some ## headers
    headers = re.findall(r'^## (\S+)', content, re.MULTILINE)
    assert len(headers) >= 5, f"Expected >=5 task sections, found {len(headers)}: {headers}"
    # Known tasks that must be present
    for task in ['task1', 'task2', 'task4']:
        assert task in headers, f"Missing section for {task} in training_notes.md"


# ---------------------------------------------------------------------------
# 6. Gradio Wiring Safety
# ---------------------------------------------------------------------------

def test_dynamic_dropdowns_allow_custom_value():
    """Dropdowns that get repopulated via gr.update() must have allow_custom_value=True
    to avoid 'Value not in choices' errors during task/variant transitions."""
    # Find dropdown variable names that appear in gr.update(choices=...) outputs
    # These are the ones at risk of stale values
    update_pattern = re.compile(r'gr\.update\(choices=\w+.*?\)\s*,?\s*#\s*(\w+)')
    dynamic_dropdowns = set()
    for m in update_pattern.finditer(APP_SOURCE):
        name = m.group(1).strip()
        dynamic_dropdowns.add(name)

    # Now check that each of these dropdowns has allow_custom_value=True
    # We look for the gr.Dropdown definition that assigns to that variable
    missing = []
    for dd_name in dynamic_dropdowns:
        # Find the assignment: dd_name = gr.Dropdown(...)
        dd_def_pattern = re.compile(
            rf'{dd_name}\s*=\s*gr\.Dropdown\((.*?)\)',
            re.DOTALL
        )
        match = dd_def_pattern.search(APP_SOURCE)
        if match:
            dd_args = match.group(1)
            if 'allow_custom_value=True' not in dd_args and 'allow_custom_value' not in dd_args:
                line_num = APP_SOURCE.count('\n', 0, match.start()) + 1
                missing.append(f"{dd_name} (line ~{line_num})")

    assert not missing, \
        f"Dynamic dropdowns missing allow_custom_value=True: {missing}"


def test_on_variant_change_output_count():
    """on_variant_change return tuple must match _variant_change_outputs length."""
    # Find _variant_change_outputs list
    outputs_match = re.search(
        r'_variant_change_outputs\s*=\s*\[(.*?)\]',
        APP_SOURCE, re.DOTALL
    )
    assert outputs_match, "_variant_change_outputs not found"
    output_items = [x.strip() for x in outputs_match.group(1).split(',') if x.strip()]
    expected_count = len(output_items)

    # Find the return ( ... ) in on_variant_change and count comma-separated items.
    # Extract the tuple content, split by top-level commas, handle trailing comma.
    fn_start = APP_SOURCE.index('def on_variant_change(')
    ret_keyword = APP_SOURCE.index('return (', fn_start)
    paren_pos = APP_SOURCE.index('(', ret_keyword)
    # Extract text inside outermost parens
    depth = 0
    end_pos = paren_pos
    pos = paren_pos
    while pos < len(APP_SOURCE):
        ch = APP_SOURCE[pos]
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth == 0:
                end_pos = pos
                break
        pos += 1
    inner = APP_SOURCE[paren_pos + 1:end_pos]
    # Count top-level commas (skip nested parens/brackets)
    depth = 0
    commas = 0
    for ch in inner:
        if ch in '([{':
            depth += 1
        elif ch in ')]}':
            depth -= 1
        elif ch == ',' and depth == 0:
            commas += 1
    # Check for trailing comma: if last non-whitespace/comment before ) is a comma
    stripped = inner.rstrip()
    # Remove trailing comment if any
    last_line = stripped.rsplit('\n', 1)[-1]
    if '#' in last_line:
        last_line = last_line[:last_line.index('#')]
    has_trailing_comma = last_line.rstrip().endswith(',')
    return_items = commas if has_trailing_comma else commas + 1

    assert return_items == expected_count, \
        f"on_variant_change returns {return_items} items but _variant_change_outputs has {expected_count}"


def test_mcqa_filter_handler_exists():
    """Consolidated _on_mcqa_filter_change must exist (replaces 5 separate handlers)."""
    assert 'def _on_mcqa_filter_change(' in APP_SOURCE


def test_no_duplicate_keypoint_subsets():
    """KEYPOINT_SUBSETS should only be defined once."""
    matches = re.findall(r'^KEYPOINT_SUBSETS\s*=\s*\{', APP_SOURCE, re.MULTILINE)
    assert len(matches) == 1, \
        f"KEYPOINT_SUBSETS defined {len(matches)} times (expected 1)"


# ---------------------------------------------------------------------------
# 7. Archive Manager
# ---------------------------------------------------------------------------

def test_archive_functions_exist():
    """All archive management functions must exist."""
    for fn in ['_load_archive', '_save_archive', 'is_archived', 'get_active_variants']:
        assert f'def {fn}(' in APP_SOURCE, f"Missing archive function: {fn}"


def test_archive_filtering_in_on_task_change():
    """on_task_change must filter archived variants via get_active_variants."""
    fn_match = re.search(r'def on_task_change\(.*?\):.*?(?=\n        def |\n    return )', APP_SOURCE, re.DOTALL)
    assert fn_match, "on_task_change not found"
    fn_body = fn_match.group()
    assert 'get_active_variants' in fn_body, \
        "on_task_change does not filter archived variants"


# ---------------------------------------------------------------------------
# 8. Mixed Model Result Lookup
# ---------------------------------------------------------------------------

def test_mixed_model_fallback_in_find_result_file():
    """find_result_file must have a fallback for mixed model checkpoint names."""
    fn_match = re.search(
        r'def find_result_file\(.*?\):(.*?)(?=\ndef )',
        APP_SOURCE, re.DOTALL
    )
    assert fn_match, "find_result_file not found"
    fn_body = fn_match.group(1)
    assert "'mixed'" in fn_body and 'mixed_pattern' in fn_body, \
        "find_result_file is missing mixed model fallback logic"


# ---------------------------------------------------------------------------
# 9. Tab Auto-Selection
# ---------------------------------------------------------------------------

def test_mcqa_tab_selects_task4():
    """MCQA tab select handler must set task4 and update variant + metric dropdowns."""
    fn_match = re.search(
        r'def _select_task4_for_mcqa\(.*?\):(.*?)(?=\n        \w)',
        APP_SOURCE, re.DOTALL
    )
    assert fn_match, "_select_task4_for_mcqa not found"
    fn_body = fn_match.group(1)
    assert "'task4'" in fn_body, "MCQA tab handler doesn't select task4"
    # Verify it's wired to mcqa_tab.select with variant_dropdown in outputs
    wire_match = re.search(r'mcqa_tab\.select\(.*?outputs=\[.*?variant_dropdown', APP_SOURCE, re.DOTALL)
    assert wire_match, "mcqa_tab.select not wired with variant_dropdown output"


# ---------------------------------------------------------------------------
# 10. Function Signature Sanity
# ---------------------------------------------------------------------------

def test_key_functions_exist():
    """All key functions from the refactoring must exist."""
    required_functions = [
        'empty_figure', 'cacheable', 'clear_all_caches',
        '_on_mcqa_filter_change', '_sync_task_names',
        '_load_archive', '_save_archive', 'is_archived', 'get_active_variants',
        'get_training_notes', '_load_training_notes',
        'find_result_file', 'parse_checkpoint_name',
        'create_metrics_plot', 'create_checkpoint_table',
    ]
    for fn in required_functions:
        assert f'def {fn}(' in APP_SOURCE, f"Missing function: {fn}"


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import traceback

    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    passed = failed = 0
    failures = []

    for test_fn in tests:
        name = test_fn.__name__
        try:
            test_fn()
            passed += 1
            print(f"  PASS  {name}")
        except Exception as e:
            failed += 1
            failures.append((name, e))
            print(f"  FAIL  {name}: {e}")

    print(f"\n{'='*60}")
    print(f"  {passed} passed, {failed} failed, {passed + failed} total")
    if failures:
        print(f"\nFailures:")
        for name, err in failures:
            print(f"  - {name}: {err}")
        sys.exit(1)
    else:
        print("  All tests passed!")
