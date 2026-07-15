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

# SLURM .out for a given jobid — carries the WITH_VO_S2 post-script markers that the RUN-log
# does NOT (the vo_s2 FIXED sweep runs in the sbatch wrapper AFTER eval_all.sh's own RUN END).
# Dated subdir (current wrapper) or flat (older). Empty if the job wasn't an sbatch (srun).
_slurmout() {
  local jid="$1" d p
  for d in "$(date -u +%F)" "$(date -u -d yesterday +%F 2>/dev/null)"; do
    p="$LOG_ROOT/slurm/$d/eval_all_slurm-${jid}.out"; [[ -f "$p" ]] && { echo "$p"; return; }
  done
  p="$LOG_ROOT/slurm/eval_all_slurm-${jid}.out"; [[ -f "$p" ]] && { echo "$p"; return; }
}

# SLURM .out for a STANDALONE reasoner_sweep_node.sbatch job (serve sft2812 TP8 + sweep an ONLY
# list; NOT an eval_all run, so it writes no eval_all run-log — flat path only). Empty if absent.
_sweepout() {
  local p="$LOG_ROOT/slurm/reasoner_sweep_tp8_slurm-${1}.out"; [[ -f "$p" ]] && echo "$p"
}

# stage of a standalone reasoner_sweep job from its .out: serving → sweep → DONE.
_sweep_stage() {
  local so="$1"
  [[ -z "$so" || ! -f "$so" ]] && return
  grep -qE "reasoner_sweep_node COMPLETE|sweep exited \(rc=" "$so" 2>/dev/null && { echo DONE; return; }
  grep -qE "\[FAIL\]" "$so" 2>/dev/null && { echo SWEEP-FAIL; return; }
  grep -q "launching sweep" "$so" 2>/dev/null && { echo reasoner-sweep; return; }
  grep -qE "WARM after|serving \(TP8\)" "$so" 2>/dev/null && { echo serving; return; }
  echo serving
}

# current stage KEY from a run-log (machine-readable; latest reached wins).
# $2 = jobid (optional) → after RUN END, check the SLURM .out for an in-flight WITH_VO_S2 sweep,
# which runs in the sbatch wrapper AFTER eval_all.sh's RUN END and is invisible to the run-log.
_stage_key() {
  local log="$1" jid="${2:-}"
  [[ -z "$log" || ! -f "$log" ]] && { echo "no-log"; return; }
  if grep -q "==== RUN END ====" "$log" 2>/dev/null; then
    # eval_all.sh finished — but a vo_s2 FIXED sweep may still be running in the wrapper.
    local so; so="$(_slurmout "$jid")"
    if [[ -n "$so" ]] && grep -q "WITH_VO_S2:" "$so" 2>/dev/null \
         && ! grep -qE "WITH_VO_S2: reasoner_sweep rc=|vo_s2 SKIPPED" "$so" 2>/dev/null; then
      echo vo_s2; return   # FIXED reasoner sweep still in flight → NOT done
    fi
    echo DONE; return
  fi
  grep -q "PREFLIGHT FAIL"             "$log" 2>/dev/null && { echo PREFLIGHT-FAIL; return; }
  local s="serving"
  grep -q "PREFLIGHT PASS"             "$log" 2>/dev/null && s="preflight"
  grep -q ">>> STAGE: aux"             "$log" 2>/dev/null && s="aux"
  grep -q ">>> STAGE: visualobs"       "$log" 2>/dev/null && s="visualobs"
  grep -q ">>> STAGE: agreement"       "$log" 2>/dev/null && s="agreement"
  grep -q ">>> STAGE: benchmarks"      "$log" 2>/dev/null && s="benchmarks"
  echo "$s"
}

# live sub-step text for the current stage (for display). $3 = jobid (for vo_s2 → SLURM .out).
_substep() {
  local log="$1" stage="$2" jid="${3:-}"
  case "$stage" in
    vo_s2)
      # FIXED reasoner sweep rep count lives in the SLURM .out, not the run-log
      local so; so="$(_slurmout "$jid")"
      # -a: the SLURM .out is a 'data' file (tqdm \r bytes); without it grep can print
      # "Binary file matches" and tail grabs a stale value instead of the live rep count.
      [[ -n "$so" ]] && grep -aoE 'Evaluating samples: *[0-9]+/[0-9]+|Processed: [0-9]+/[0-9]+' "$so" 2>/dev/null | tail -1;;
    aux)        grep -oE 'Running (video|text|image)[ a-zA-Z]*' "$log" 2>/dev/null | tail -1;;
    benchmarks)
      # Prefer the latest [RUN] line (the benchmark actually executing) so a [SKIP] line
      # (e.g. "[SKIP] Video-MME" on an IFBench-only run) never masquerades as the live sub-step.
      local sub
      sub="$(grep -oE '\[RUN\] (VSI-Bench|MMMU|Video-MME|IFBench)[^]]*' "$log" 2>/dev/null | tail -1)"
      [[ -z "$sub" ]] && sub="$(grep -oE 'MMMU_DEV_VAL\] Sample [0-9]+|IFBench prompt-level' "$log" 2>/dev/null | tail -1)"
      [[ -z "$sub" ]] && sub="$(grep -oE '\[SKIP\] (VSI-Bench|MMMU|Video-MME|IFBench)[^]]*' "$log" 2>/dev/null | tail -1)"
      echo "$sub";;
    agreement)
      # step 1/2 (obs generation) emits "... Complete | N/2157"; show that as a Processed count,
      # tagged with the current step phase. step 2/2 (GT scoring) has no per-rep counter (fast).
      local ph cnt
      ph="$(grep -aoE 'step [12]/2' "$log" 2>/dev/null | tail -1)"
      cnt="$(grep -aoE 'Complete \| [0-9]+/[0-9]+' "$log" 2>/dev/null | tail -1 | grep -oE '[0-9]+/[0-9]+')"
      if [[ -n "$cnt" ]]; then echo "${ph:+$ph }Processed: $cnt"; else echo "$ph"; fi;;
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
  local m_serve=4 m_pre=1 m_vo=8 m_agr=18
  local m_aux; if [[ "$think" == on ]]; then m_aux=$([[ "$base" == *27b* ]] && echo 250 || echo 165); else m_aux=$([[ "$base" == *27b* ]] && echo 12 || echo 10); fi
  # benchmarks budget = sum of the benchmarks that will ACTUALLY run. Each is subtracted when
  # SKIPPED (via --skip-* / [SKIP] in the log) — so an IFBench-only run (3 skips) budgets ~5min,
  # NOT the full ~75min. Typical thinkoff mins: VSI 12, MMMU 15, Video-MME 60, IFBench 5.
  local b_vsi=12 b_mmmu=15 b_vmme=60 b_ifb=5
  grep -qE "SKIP] VSI-Bench|--skip-vsibench" "$log" 2>/dev/null && b_vsi=0
  grep -qE "SKIP] MMMU|--skip-mmmu"          "$log" 2>/dev/null && b_mmmu=0
  grep -qE "SKIP] Video-MME|--skip-videomme" "$log" 2>/dev/null && b_vmme=0
  grep -qE "SKIP] IFBench|--skip-ifbench"    "$log" 2>/dev/null && b_ifb=0
  local m_bench=$(( b_vsi + b_mmmu + b_vmme + b_ifb ))
  # Authoritative override: VLMEvalKit's run_api logs "Total datasets: N" = how many benchmarks it
  # ACTUALLY loaded (after --skip-*). N=1 with an "IFBench" inference line ⇒ an IFBench-only run, so
  # budget ~5min even when the cmd-line skip flags are absent from the log (pre-fix LOG_CMD). NOTE:
  # the stage BANNER always lists all 4 names, so we must NOT use a negative VSI/MMMU/VideoMME match
  # here — "Total datasets: 1" + an IFBench inference/pipeline line is the unambiguous signal.
  if grep -q "Total datasets: 1" "$log" 2>/dev/null \
     && grep -qE "(-+ IFBench -+|\[IFBench\]|IFBench: (Pending|Running))" "$log" 2>/dev/null; then
    m_bench=$b_ifb
  fi
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
squeue -u "$USER_ID" -h -o '%i|%j|%T|%M|%N|%b|%R' 2>/dev/null | grep -iE 'eval|397b|^[0-9]+\|vo[-_]' | while IFS='|' read -r jid name state tm node tres reason; do
  gpus="${tres##*:}"; [[ "$gpus" =~ ^[0-9]+$ ]] || gpus="?"
  eta="—"; stg="$reason"
  if [[ "$state" == RUNNING ]]; then
    log="$(_logfor "$jid")"
    key="$(_stage_key "$log" "$jid")"
    if [[ "$key" == no-log ]]; then
      # not an eval_all job — could be a STANDALONE reasoner_sweep_node.sbatch (serve+sweep).
      swo="$(_sweepout "$jid")"
      if [[ -n "$swo" ]]; then
        skey="$(_sweep_stage "$swo")"
        case "$skey" in
          reasoner-sweep)
            cnt="$(grep -aoE 'Processed: [0-9]+/[0-9]+|Evaluating samples: *[0-9]+/[0-9]+' "$swo" 2>/dev/null | tail -1)"
            stg="reasoner-sweep${cnt:+  ($cnt)}"; eta="$(_fmt_eta 30 "$swo")";;
          serving)     stg="serving (reasoner TP8)"; eta="$(_fmt_eta 40 "$swo")";;
          DONE)        stg="DONE"; eta="~done";;
          SWEEP-FAIL)  stg="SWEEP-FAIL"; eta="—";;
        esac
      else
        stg="(no eval-log: interactive/srun?)"
      fi
    else
      sub="$(_substep "$log" "$key" "$jid")"
      stg="$key${sub:+  ($sub)}"
      # ETA only for true eval_all runs in a real stage
      case "$key" in DONE) eta="~done";; PREFLIGHT-FAIL) eta="—";; vo_s2) eta="~15m";; *)
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
  | grep -iE 'eval|397b|[[:space:]]vo[-_]' | grep -vE 'RUNNING|PENDING' \
  | awk '{printf "  %-8s %-30s %-12s %s\n", $1, $2, $3, $4}'
