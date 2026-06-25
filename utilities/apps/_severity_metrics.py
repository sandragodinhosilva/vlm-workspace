#!/usr/bin/env python3
"""Canonical severity-metrics helper for vibe_test.py.

The app venv lacks sklearn (needed by eval.evaluate). This runs in the eval venv
and reuses eval.compute_severity_metrics so the app shows the SAME metrics as the
board (Error-F1/P/R/Acc, severity acc exact/within-1/non-1, eff/injury exact+MAE).

stdin  : JSON list of {gt: "<bracket answer>", pred: "<model answer>"}
stdout : JSON of the metric dict (subset, board-named)
Run with: /home/sgsilva/vlm-post-training-home-venv/bin/python _severity_metrics.py
"""
import sys, json, re
from pathlib import Path

REPO = Path("/home/sgsilva/vlm-post-training")
sys.path.insert(0, str(REPO / "eval"))


def parse(text):
    errs = {}
    eblk = text.split("[ERRORS]", 1)[-1].split("[SCORES]", 1)[0] if "[ERRORS]" in text else ""
    for line in eblk.strip().splitlines():
        if ":" in line:
            k, v = line.rsplit(":", 1)
            m = re.search(r"\d+", v)
            if m:
                errs[k.strip()] = int(m.group())
    sblk = text.split("[SCORES]", 1)[-1].split("[FEEDBACK]", 1)[0] if "[SCORES]" in text else ""
    eff = re.search(r"Effectiveness:\s*(\d+)", sblk, re.I)
    inj = re.search(r"Injury Risk:\s*(\d+)", sblk, re.I)
    return errs, (int(eff.group(1)) if eff else None), (int(inj.group(1)) if inj else None)


def main():
    pairs = json.load(sys.stdin)
    import evaluate as E
    results = []
    for p in pairs:
        ge, gef, gin = parse(p["gt"])
        pe, pef, pin = parse(p["pred"])
        results.append({
            "gt_severity_scores": ge, "pred_severity_scores": pe,
            "gt_effectiveness": gef, "pred_effectiveness": pef,
            "gt_injury_risk": gin, "pred_injury_risk": pin,
        })
    m = E.compute_severity_metrics(results)
    keys = ["error_detection_f1", "error_detection_precision", "error_detection_recall",
            "error_detection_accuracy", "sample_error_detection_f1",
            "overall_severity_accuracy", "overall_severity_within_1",
            "overall_severity_accuracy_non1",
            "effectiveness_exact_match_rate", "effectiveness_mae",
            "injury_risk_exact_match_rate", "injury_risk_mae"]
    print(json.dumps({"n": len(results), **{k: m.get(k) for k in keys}}))


if __name__ == "__main__":
    main()
