# Monitoring App Documentation

This directory contains comprehensive documentation for the VLM Pose Estimation Monitoring App.

## 📚 Documentation Files

### 1. [monitoring_app_documentation.md](monitoring_app_documentation.md)
**Complete User & Developer Guide** (10,000+ words)

Comprehensive documentation covering:
- Overview & Purpose
- Architecture & Design Patterns
- **Data Sources (Inputs)**:
  - SFT Datasets format
  - Model Checkpoints structure
  - Evaluation Results JSON
  - Experiments CSV
  - Benchmark Reports (IFEval, SIBench)
- **User Interface (Outputs)**:
  - Tab 1: Dataset Explorer
  - Tab 2: Training Monitor
  - Tab 3: Evaluation Dashboard
  - Tab 4: Benchmarks eval
  - Tab 5: Mixed vs Single Comparison
- Data Flow & Event Handlers
- Key Features & Optimizations
- Technical Stack Details
- **Usage Guide** with common workflows

**Start here** if you're new to the app or need to understand how it works.

---

### 2. [app_architecture_diagram.txt](app_architecture_diagram.txt)
**System Architecture & Data Flow Diagrams**

ASCII diagrams showing:
- System overview
- Data flow (inputs → caching → UI outputs)
- Event handler flow
- File organization (3,600 lines mapped)
- 3-tier caching strategy
- Memory usage profile (~210 MB typical)
- Performance characteristics

**Quick visual reference** for understanding the system architecture.

---

### 3. [benchmark_tab_issues.md](benchmark_tab_issues.md)
**Bug Analysis Report**

Detailed analysis of issues found during code review:
- 6 Critical issues identified
- 9 Medium issues identified
- 9 Minor issues identified
- 6 Not implemented features
- Each issue includes:
  - Location (file + line number)
  - Problem description
  - Impact analysis
  - Recommended fix

**For developers** working on bug fixes or code improvements.

---

### 4. [fixes_applied.md](fixes_applied.md)
**Fix Summary & Changelog**

Documents all critical and medium fixes applied:
- ✅ SIBench baseline parsing (56.6% instead of hardcoded 38.86%)
- ✅ IFEval baseline parsing from reports
- ✅ Improved baseline detection logic
- ✅ Error handling in event handlers
- ✅ Event handler wiring (always active)
- ✅ Data validation in visualization functions
- ✅ SIBench visualization chart created
- ✅ Simplified status logic

Includes before/after code examples and test results.

**Reference for changes made** and their impact.

---

## 🚀 Quick Start

### Running the App

**Option 1: Direct Python**
```bash
cd /mnt/data/sgsilva/monitoring-app
python app.py --port 7861 --share
```

**Option 2: Startup Script** (recommended)
```bash
cd /mnt/data/sgsilva/monitoring-app
./start_app.sh
```
The startup script includes:
- Cache clearing
- Logs directory setup
- Debug logging enabled by default

Access at: `http://localhost:7861`

**Arguments**:
- `--port`: Port to run on (default: 7861)
- `--share`: Create public Gradio share link
- `--debug`: Enable verbose debug logging

**Logging**:
- All logs written to: `logs/app_YYYYMMDD_HHMMSS.log`
- Console output also displayed in terminal
- Logs directory created automatically on startup

### Common Use Cases

1. **Verify Dataset Quality**
   - Select task + variant + split
   - Browse image gallery
   - Click images to verify annotations

2. **Find Best Checkpoint**
   - Go to Training Monitor tab
   - Look for 🏆 in checkpoint table
   - View metrics progression plot

3. **Compare Models**
   - Go to Evaluation Dashboard
   - Select "Custom Comparison"
   - Choose checkpoints to compare

4. **Check Benchmarks**
   - Go to Benchmarks eval tab
   - Select IFEval or SIBench
   - View performance comparison

5. **Compare Mixed vs Single-Task Training**
   - Go to Mixed vs Single Comparison tab
   - View comprehensive analysis of 40+ checkpoints
   - See visualizations comparing training approaches

---

## 📊 Data Requirements

The app expects the following directory structure:

```
/mnt/data/shared/vlm/data/sft_datasets_v4/
├── task1/cropped_v1/
│   ├── train.jsonl
│   ├── test.jsonl
│   └── train_stats.json
└── ...

/mnt/data/sgsilva/models/
├── qwen3-vl-4b-4epochs-task1-step646/
│   ├── config.yaml
│   └── training_info.json
└── ...

/mnt/data/sgsilva/vlm-evaluation/
├── experiments-final.csv
├── results/final/*.json
├── results/reports/*_report.md
└── ...

/mnt/data/sgsilva/outputs/sibench/
├── qwen3-vl-4b-baseline/
└── report_*.md
```

---

## 🔧 Technical Details

- **Framework**: Gradio 4.x
- **Language**: Python 3.10+
- **Dependencies**: pandas, plotly, opencv-python, PIL
- **Memory**: ~210 MB typical, ~400 MB peak
- **Performance**: <2 sec interactions, <3 sec startup
- **Caching**: 3-tier strategy (startup → session → interaction)

---

## 📝 Documentation Status

| Document | Status | Last Updated |
|----------|--------|--------------|
| README.md | ✅ Complete | 2026-02-05 |
| monitoring_app_documentation.md | ⚠️ Needs Update | 2026-02-04 |
| app_architecture_diagram.txt | ⚠️ Needs Update | 2026-02-04 |
| benchmark_tab_issues.md | ✅ Complete | 2026-02-04 |
| fixes_applied.md | ⚠️ Needs Update | 2026-02-04 |

---

## 🆘 Support

For questions or issues:
1. Check [monitoring_app_documentation.md](monitoring_app_documentation.md) first
2. Review [benchmark_tab_issues.md](benchmark_tab_issues.md) for known issues
3. Check app logs for error messages: `logs/app_YYYYMMDD_HHMMSS.log`
   - Find latest log: `ls -t logs/app_*.log | head -1`
   - Tail live log: `tail -f logs/app_*.log`

---

## 📋 Recent Changes (2026-02-05)

- ✅ Added Tab 5: Mixed vs Single Comparison (40+ checkpoint analysis)
- ✅ Added mixed tasks support across all tabs
- ✅ Reduced plot sizes in comparison tab (400px height)
- ✅ Fixed task1c_cropped_v1 OKS data display issues
- ✅ Improved dataset variant filtering

---

**Last Updated**: 2026-02-05
**App Version**: v1.1 (with Mixed Tasks & Comparison Analysis)
**Maintainer**: Generated with Claude Code
