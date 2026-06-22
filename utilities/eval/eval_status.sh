#!/usr/bin/env bash
# eval_status.sh — at-a-glance stage of every running eval_all.sh job (+ today's finished ones).
# Reads SLURM (squeue/sacct) + the dated run-logs under logs/eval/<date>/, parses the latest
# stage marker per job. No args. Read-only.
#
#   /home/sgsilva/utilities/eval/eval_status.sh
#   watch -n 30 /home/sgsilva/utilities/eval/eval_status.sh   # live
#
# Stage is parsed from the run-log: serving → preflight → aux → visualobs → (evaluate_vo) →
# agreement → benchmarks → DONE, plus the live sub-step (aux leg / benchmark name / rep count).

set -uo pipefail
USER_ID="${USER:-sgsilva}"
LOG_ROOT="/mnt/data/sgsilva/logs/eval"

# newest run-log for a given SLURM jobid (search today + yesterday; long runs cross midnight)
_logfor() {
  local jid="$1" d
  for d in "$(date -u +%F)" "$(date -u -d yesterday +%F 2>/dev/null)"; do
    ls -t "$LOG_ROOT/$d"/eval_all_*__j${jid}__*.log 2>/dev/null | head -1 && return
  done
}

# current stage KEY from a run-log (machine-readable; latest reached wins)
_stage_key() {
  local log="$1"
  [[ -z "$log" || ! -f "$log" ]] && { echo "no-log"; return; }
  grep -q "==== RUN END ===="          "$log" 2>/dev/null && { echo DONE; return; }
  grep -q "PREFLIGHT FAIL"             "$log" 2>/dev/null && { echo PREFLIGHT-FAIL; return; }
  local s="serving"
  grep -q "PREFLIGHT PASS"             "$log" 2>/dev/null && s="preflight"
  grep -q ">>> STAGE: aux"             "$log" 2>/dev/null && s="aux"
  grep -q ">>> STAGE: visualobs"       "$log" 2>/dev/null && s="visualobs"
  grep -q ">>> STAGE: agreement"       "$log" 2>/dev/null && s="agreement"
  grep -q ">>> STAGE: benchmarks"      "$log" 2>/dev/null && s="benchmarks"
  echo "$s"
}

# live sub-step text for the current stage (for display)
_substep() {
  local log="$1" stage="$2"
  case "$stage" in
    aux)        grep -oE 'Running (video|text|image)[ a-zA-Z]*' "$log" 2>/dev/null | tail -1;;
    benchmarks) grep -oE '\[(RUN|SKIP)\] (VSI-Bench|MMMU|Video-MME)[^]]*|MMMU_DEV_VAL\] Sample [0-9]+|Video-MME' "$log" 2>/dev/null | tail -1;;
    agreement)  grep -oE 'step [12]/2|Processing: [^ ]+' "$log" 2>/dev/null | tail -1;;
  esac
}

# ETA = sum of REMAINING-stage typical minutes (empirical, 2026-06-19), minus time already
# spent in the current stage isn't tracked → conservative (whole-stage budgets). Scaled by
# thinkON (aux ~5x slower: 4B 2h43m / 27B 4h19m vs thinkoff ~10-15m) + base-model + Video-MME.
# Budgets are coarse on purpose — an order-of-magnitude "minutes vs hours" signal, not a clock.
_eta_min() {
  local log="$1" stage="$2" base="$3" think="$4"
  local order=(serving preflight aux visualobs agreement benchmarks DONE)
  # per-stage typical minutes [thinkoff]:  aux scales hugely with thinkON
  local m_serve=4 m_pre=1 m_vo=8 m_agr=18 m_bench_novmme=45
  local m_aux; if [[ "$think" == on ]]; then m_aux=$([[ "$base" == *27b* ]] && echo 250 || echo 165); else m_aux=$([[ "$base" == *27b* ]] && echo 12 || echo 10); fi
  # Video-MME present? (only known once benchmarks logs it; assume present unless SKIP seen)
  local vmme=60; grep -q "SKIP] Video-MME" "$log" 2>/dev/null && vmme=0
  local m_bench=$(( m_bench_novmme + vmme ))
  declare -A dur=( [serving]=$m_serve [preflight]=$m_pre [aux]=$m_aux [visualobs]=$m_vo [agreement]=$m_agr [benchmarks]=$m_bench )
  # Only count stages this run actually requested (--stages). visualobs implies agreement
  # (eval_all runs agreement after visualobs). serving/preflight always run.
  local req; req="$(grep -oE -- '--stages [^ ]+' "$log" 2>/dev/null | head -1 | awk '{print $2}')"
  local -A want=( [serving]=1 [preflight]=1 )
  case ",$req," in *,aux,*) want[aux]=1;; esac
  case ",$req," in *,visualobs,*) want[visualobs]=1; want[agreement]=1;; esac
  case ",$req," in *,benchmarks,*) want[benchmarks]=1;; esac
  [[ -z "$req" ]] && want=( [serving]=1 [preflight]=1 [aux]=1 [visualobs]=1 [agreement]=1 [benchmarks]=1 )  # unknown → all
  # sum remaining REQUESTED stages strictly AFTER the current one (current counted half — mid-flight)
  local seen=0 total=0 s
  for s in "${order[@]}"; do
    [[ "$s" == "$stage" ]] && { seen=1; [[ -n "${want[$s]:-}" ]] && total=$(( total + ${dur[$s]:-0}/2 )); continue; }
    [[ "$seen" == 1 && -n "${want[$s]:-}" && -n "${dur[$s]:-}" ]] && total=$(( total + dur[$s] ))
  done
  echo "$total"
}

# human ETA string + stall flag (log not written in >5min while RUNNING = suspect)
_fmt_eta() {
  local min="$1" log="$2"
  local stall=""
  if [[ -f "$log" ]]; then
    local age=$(( $(date +%s) - $(stat -c %Y "$log" 2>/dev/null || echo 0) ))
    (( age > 300 )) && stall=" ⚠STALE${age}s"
  fi
  if   (( min <= 0 ));  then echo "~done${stall}"
  elif (( min < 60 ));  then echo "~${min}m${stall}"
  else echo "~$(( min/60 ))h$(( min%60 ))m${stall}"; fi
}

echo "===================== eval status @ $(date '+%F %H:%M:%S') ====================="
printf '%-8s %-26s %-8s %-9s %-9s %-4s %-8s %s\n' JOBID NAME STATE ELAPSED NODE GPUS ETA STAGE
printf '%-8s %-26s %-8s %-9s %-9s %-4s %-8s %s\n' ----- ---- ----- ------- ---- ---- --- -----

# RUNNING/PENDING eval jobs (%b = TRES_PER_NODE, e.g. 'gres/gpu:2')
squeue -u "$USER_ID" -h -o '%i|%j|%T|%M|%N|%b|%R' 2>/dev/null | grep -iE 'eval|397b' | while IFS='|' read -r jid name state tm node tres reason; do
  gpus="${tres##*:}"; [[ "$gpus" =~ ^[0-9]+$ ]] || gpus="?"
  eta="—"; stg="$reason"
  if [[ "$state" == RUNNING ]]; then
    log="$(_logfor "$jid")"
    key="$(_stage_key "$log")"
    if [[ "$key" == no-log ]]; then
      stg="(no eval-log: interactive/srun?)"
    else
      sub="$(_substep "$log" "$key")"
      stg="$key${sub:+  ($sub)}"
      # ETA only for true eval_all runs in a real stage
      case "$key" in DONE) eta="~done";; PREFLIGHT-FAIL) eta="—";; *)
        base="$(grep -oE -- '--base-model [^ ]+' "$log" 2>/dev/null | head -1 | awk '{print $2}')"
        think="$(grep -oE -- '--thinking (on|off)' "$log" 2>/dev/null | head -1 | awk '{print $2}')"
        eta="$(_fmt_eta "$(_eta_min "$log" "$key" "${base:-}" "${think:-off}")" "$log")";;
      esac
    fi
  fi
  printf '%-8s %-26.26s %-8s %-9s %-9s %-4s %-8s %s\n' "$jid" "$name" "$state" "$tm" "${node:-—}" "$gpus" "$eta" "$stg"
done

# today's FINISHED eval jobs (terminal state)
echo
echo "--- finished today (sacct) ---"
sacct -S "$(date -u +%F)" -u "$USER_ID" --format=JobID,JobName%30,State,Elapsed -X -n 2>/dev/null \
  | grep -iE 'eval|397b' | grep -vE 'RUNNING|PENDING' \
  | awk '{printf "  %-8s %-30s %-12s %s\n", $1, $2, $3, $4}'
