#!/bin/bash
#SBATCH --job-name=expb-gateb-eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=48
#SBATCH --mem=400G
#SBATCH --exclude=worker-30,worker-31
#SBATCH --output=/mnt/data/sgsilva/logs/eval/slurm-expb-gateb-%j.out
#SBATCH --error=/mnt/data/sgsilva/logs/eval/slurm-expb-gateb-%j.err

# EXP-B GATE-B GT-obs eval, one checkpoint per job (campaign one-off, 202607).
# Serves the EXP-B stage-2 reasoner thinkON (its SFT targets carry real <think>) on 2 GPUs,
# waits for health, runs the two-stage severity eval on the 1806 test split with the 2906
# categorical_k5majority GT observations as stage-1 (--precomputed-visual-obs skips the
# stage-1 model call), then tears the server down so the node frees.
#   CKPT=<exported HF dir> sbatch <this>
# Prompt identity: --stage2-stance ondemand renders the trained user-turn byte-identically
# (verified 2,157/2,157; include-options is auto-forced OFF by evaluate.py).
# GATE B target: beat fixed-sft2812 2-call vo_s2 = 55.1 on GT-obs.

set -uo pipefail
CKPT="${CKPT:?set CKPT=/mnt/data/sgsilva/models/qwen35-27b-expb-stage2-ondemand-sft-stepNNN}"
STEP=$(basename "$CKPT" | grep -oE 'step[0-9]+')
PORT=$(( 8300 + (SLURM_JOB_ID % 100) ))
PY=/home/sgsilva/vlm-post-training-home-venv/bin/python
VPT=/home/sgsilva/vlm-post-training
# GT-obs arm v2 (2026-07-06): run SINGLE-STAGE on the EXP-B test BUILD, whose messages[1] IS the
# byte-exact trained prompt (ondemand template + 2906 GT obs inlined by build_stage2_from_raw).
# The first attempt used --two-stage --stage2-stance + --precomputed-visual-obs on repetitions_test:
# obs block + template matched, but the exercise DESCRIPTION was extracted from the original
# repetitions_test prompt, whose layout differs from the builder's build_ref_desc (~1.7k chars,
# 'Correct Movement Criteria:' placement) → prompts were NOT train-identical (outputs kept as
# *_gtobs_DESCDRIFT_thinkon.json). Single-stage on the build uses the stored prompt verbatim —
# no reconstruction, no drift. (The two-stage stance path is still needed for the MODEL-obs arm;
# its description source must be fixed to build_ref_desc before that arm runs.)
OUT=/mnt/data/sgsilva/results/visual_obs_runs/stage2_expb_stage2_ondemand_${STEP}_gtobsbuild_thinkon.json
TESTDIR=/mnt/data/sgsilva/datasets/1806/expb_stage2_from_raw_ondemand_test_flagkeep
EXPECTED_N=2157

echo "=== EXP-B GATE-B eval: $CKPT (thinkON, port $PORT, node $(hostname -s)) ==="

# 1. Serve thinkON (real <think> targets -> ENABLE_THINKING=1, no --disable-thinking)
ENABLE_THINKING=1 /home/sgsilva/utilities/serve/start_vllm_server.sh "$CKPT" 2 131072 "$PORT" &
SERVE_PID=$!
cleanup() { echo "=== teardown: killing vLLM (pid $SERVE_PID) ==="; kill "$SERVE_PID" 2>/dev/null; sleep 5; pkill -P "$SERVE_PID" 2>/dev/null; }
trap cleanup EXIT

# 2. Health wait (up to 30 min for 27B load)
for i in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then echo "server healthy after ~$((i*10))s"; break; fi
    if ! kill -0 "$SERVE_PID" 2>/dev/null; then echo "ERROR: vLLM exited during startup"; exit 1; fi
    [ "$i" -eq 180 ] && { echo "ERROR: server not healthy after 30 min"; exit 1; }
    sleep 10
done

# 3. Eval (two passes: --resume tops off stragglers)
cd "$VPT"
for pass in 1 2; do
    echo "=== eval pass $pass ==="
    "$PY" eval/evaluate.py \
      --test-dataset-dir "$TESTDIR" \
      --model "$CKPT" \
      --server-url "http://127.0.0.1:${PORT}/v1" \
      --max-tokens 32768 \
      --max-workers 16 \
      --output-file "$OUT" \
      --resume
done

# 4. Completeness check (never trust the summary line alone)
N=$("$PY" -c "import json;print(json.load(open('$OUT')).get('metadata',{}).get('evaluated_samples',0))")
echo "=== evaluated_samples: $N / $EXPECTED_N ==="
[ "$N" -ge $(( EXPECTED_N * 99 / 100 )) ] || { echo "ERROR: incomplete eval ($N < 99% of $EXPECTED_N)"; exit 1; }
echo "=== DONE: $OUT ==="
