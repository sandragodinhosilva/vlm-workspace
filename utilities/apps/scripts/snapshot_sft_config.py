#!/usr/bin/env python3
"""Snapshot an SFT config into its run's log dir, so the SFT dashboard's config
panel shows the EXACT config-as-run (not a best-effort name match).

Call this once at launch, right before/after `sbatch`. It:
  - reads the config YAML,
  - derives the run name from `checkpointing.checkpoint_dir` basename (the
    canonical run id the tensorboard logs use: nemo-rl-vlm/logs/<run>/),
  - copies the config verbatim to nemo-rl-vlm/logs/<run>/config_snapshot.yaml
    (+ a timestamped copy so a relaunch with a changed config keeps history).

Idempotent: re-running with the same content is a no-op. A changed config
overwrites config_snapshot.yaml and adds a new timestamped copy.

Usage (fully literal — no shell vars, per CLAUDE.md):
    /home/sgsilva/nemo-rl-vlm/.venv/bin/python \
        /home/sgsilva/utilities/apps/scripts/snapshot_sft_config.py \
        --config /home/sgsilva/nemo-rl-vlm/examples/configs/sft_vlm_qwen35_27b_mix_12k_1506_megatron.yaml
"""
import argparse
import datetime
import os
import shutil
import sys

import yaml

LOGS_ROOT = "/home/sgsilva/nemo-rl-vlm/logs"


def derive_run_name(cfg, config_path):
    ckpt = (cfg.get("checkpointing") or {}).get("checkpoint_dir")
    if ckpt:
        return os.path.basename(os.path.normpath(ckpt))
    # fallback: config stem sft_vlm_<x>_megatron.yaml -> sft_<x>
    stem = os.path.splitext(os.path.basename(config_path))[0]
    stem = stem.replace("sft_vlm_", "").replace("_megatron", "")
    return f"sft_{stem}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--logs-root", default=LOGS_ROOT)
    ap.add_argument("--run-name", default=None,
                    help="override the derived run name (rarely needed)")
    args = ap.parse_args()

    if not os.path.exists(args.config):
        sys.exit(f"[snapshot] config not found: {args.config}")
    cfg = yaml.safe_load(open(args.config))

    run = args.run_name or derive_run_name(cfg, args.config)
    run_dir = os.path.join(args.logs_root, run)
    os.makedirs(run_dir, exist_ok=True)

    dst = os.path.join(run_dir, "config_snapshot.yaml")
    shutil.copyfile(args.config, dst)
    # timestamped history copy (survives relaunch with a changed config)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    shutil.copyfile(args.config, os.path.join(run_dir, f"config_snapshot_{ts}.yaml"))

    print(f"[snapshot] run={run}\n[snapshot] wrote {dst}")


if __name__ == "__main__":
    main()
