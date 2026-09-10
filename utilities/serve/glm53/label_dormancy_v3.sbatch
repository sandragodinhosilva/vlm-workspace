#!/usr/bin/env bash
# Dormancy v3 scenario labelling as a SLURM job instead of a detached process.
#
# WHY sbatch: on 2026-09-09 a Claude session restart killed every `setsid` labeller, screen driver and
# waiter chain at once (the SLURM serve jobs survived) -- the same reason serve_glm53.sbatch exists.
# A labelling pass is many hours; it must outlive the session that starts it.
#
# CPU-only: it just drives the GLM endpoint that serve_glm53.sbatch already hosts. No --gres.
#   sbatch --job-name=label-v3-b2 label_dormancy_v3.sbatch batch2_tierA_knownconv 40
#   sbatch --job-name=label-v3-b3 label_dormancy_v3.sbatch batch3_tierBC_rest 24
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/data/sgsilva/logs/judge/slurm-label-v3-%j.out
set -uo pipefail
B="$1"; C="${2:-24}"
R=/mnt/data/sgsilva/3wc/failure_modes/results/dormancy_0409
PY=/home/sgsilva/dawn-research-3wc-home-venv/bin/python
PROMPT=exploration/prompt_inspect/scenarios/prompts/_archive/label_turn_scenario_prompt_v5.md
GLM_ENV=/home/sgsilva/utilities/serve/glm53/run/endpoint.env
source /home/sgsilva/utilities/logs-utils/log_run.sh
log_start judge "dormancy_v3_${B}_labels_sbatch_c${C}"
cd /home/sgsilva/dawn-research/3wc; export ONLY_UNIT=All
# the GLM node is whatever serve_glm53.sbatch published -- never hardcode it
NODE=$(grep '^NODE=' "$GLM_ENV" 2>/dev/null | cut -d= -f2); NODE="${NODE:-worker-29}"
echo "[$(date -Is)] labelling $B at c=$C against glm-5.3@$NODE:9000 (endpoint.env)"
curl -sf -m10 "http://$NODE:9000/v1/models" >/dev/null || { echo "[$(date -Is)] GLM unreachable at $NODE:9000 -- nothing run"; log_end; exit 0; }
# resume is by turn_id: rows already labelled are never re-billed
$PY failure_scenarios/scenarios/label_scenarios.py --prompt "$PROMPT" --turn-ids "$R/$B/prescreen_label_turnids.txt" \
  --from-corpus --context-turns 10 --model glm-5.3 --host "$NODE" --port 9000 --concurrency "$C" \
  --out "$R/$B/scenlabels_v5.jsonl" --io-dir "$R/$B/scenlabels_v5_io"
echo "[$(date -Is)] done: $(grep -c . "$R/$B/scenlabels_v5.jsonl") label rows in $R/$B/scenlabels_v5.jsonl"
log_end
