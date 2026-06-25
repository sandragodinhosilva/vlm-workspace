#!/usr/bin/env bash
#
# cleanup_home.sh — reclaim space in /home/sgsilva.
#
# Home is the small/precious volume; /mnt/data/sgsilva is the big scratch volume.
# This script (1) deletes regenerable caches/temp dirs and (2) moves heavy
# data/output dirs to /mnt, leaving a symlink behind so paths keep working.
#
# SAFE BY DEFAULT: prints what it WOULD do and exits. Nothing is touched
# until you re-run with --run.
#
#   ./cleanup_home.sh                 # dry-run, show the plan + savings
#   ./cleanup_home.sh --run           # actually do it
#   ./cleanup_home.sh --run --yes     # skip the per-section confirmation prompt
#   VLLM_KEEP_DAYS=14 ./cleanup_home.sh --run   # keep vllm compiles newer than N days (default 21)
#
# SECTION SELECTION: by default ALL sections run. Pass any of the flags below
# to run ONLY the selected sections (combine freely):
#   --caches   [1] regenerable caches & temp (.gradio_temp, .nv, .triton, uv, ...)
#   --vllm     [2] vLLM compile-cache prune (older than VLLM_KEEP_DAYS)
#   --logs     [3] cold log archive delete
#   --move     [4] move heavy data dirs to /mnt (+ symlink back)
# e.g.  ./cleanup_home.sh --run --vllm --move   # only prune vllm + move data dirs
#
set -euo pipefail
source /home/sgsilva/utilities/logs-utils/log_run.sh

HOME_DIR="/home/sgsilva"
MNT_DIR="/mnt/data/sgsilva"
RESULTS_DIR="/mnt/data/sgsilva/results"     # per project convention: outputs live here
VLLM_KEEP_DAYS="${VLLM_KEEP_DAYS:-21}"      # vllm compile cache entries older than this are pruned

DRY_RUN=1
ASSUME_YES=0
declare -A WANT=()                          # populated by section flags; empty => run all
for arg in "$@"; do
  case "$arg" in
    --run) DRY_RUN=0 ;;
    --yes|-y) ASSUME_YES=1 ;;
    --caches) WANT[caches]=1 ;;
    --vllm)   WANT[vllm]=1 ;;
    --logs)   WANT[logs]=1 ;;
    --move)   WANT[move]=1 ;;
    -h|--help) sed -n '2,26p' "$0"; exit 0 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

# run_section <key> -> 0 (yes) if no section flags given, or this one was selected
run_section() { [ "${#WANT[@]}" -eq 0 ] || [ -n "${WANT[$1]:-}" ]; }

if [ "$DRY_RUN" -eq 0 ]; then
    _CLEANUP_LOG=$(log_start misc "cleanup_home")
    exec > >(tee -a "$_CLEANUP_LOG") 2>&1
fi
if [ "$DRY_RUN" -eq 1 ]; then
  echo "=== DRY RUN — nothing will be changed. Re-run with --run to apply. ==="
else
  echo "=== LIVE RUN — changes will be applied. ==="
fi
echo

sz() { du -sh "$1" 2>/dev/null | cut -f1; }      # human size, empty if missing
exists() { [ -e "$1" ] || [ -L "$1" ]; }

confirm() {
  # confirm "<section description>"  -> returns 0 to proceed
  [ "$DRY_RUN" -eq 1 ] && return 1            # dry-run never proceeds
  [ "$ASSUME_YES" -eq 1 ] && return 0
  read -r -p "  Proceed with: $1 ? [y/N] " ans
  [[ "$ans" == [yY] ]]
}

# delete_dir <path> <label>
delete_dir() {
  local path="$1" label="$2"
  exists "$path" || { echo "  [skip] $label — not present"; return; }
  local s; s=$(sz "$path")
  echo "  DELETE  $path  ($s)  — $label"
  if confirm "delete $path ($s)"; then
    rm -rf "$path"
    echo "    deleted."
  fi
}

# move_dir <src> <dest-parent>  — moves src into dest-parent and symlinks back
move_dir() {
  local src="$1" destparent="$2"
  exists "$src" || { echo "  [skip] $src — not present"; return; }
  if [ -L "$src" ]; then echo "  [skip] $src — already a symlink"; return; fi
  local s; s=$(sz "$src")
  local name; name=$(basename "$src")
  local dest="$destparent/$name"
  echo "  MOVE    $src  ($s)  ->  $dest  (+ symlink back)"
  if confirm "move $src ($s) to $dest"; then
    mkdir -p "$destparent"
    if [ -e "$dest" ]; then
      echo "    ERROR: $dest already exists — skipping to avoid clobber." >&2
      return
    fi
    mv "$src" "$dest"
    ln -s "$dest" "$src"
    echo "    moved + symlinked."
  fi
}

echo "Current /home/sgsilva total: $(sz "$HOME_DIR")"
echo

# ──────────────────────────────────────────────────────────────────────────
if run_section caches; then
echo "[1] Regenerable caches & temp — safe to delete (rebuilt automatically)"
# ──────────────────────────────────────────────────────────────────────────
delete_dir "$HOME_DIR/.gradio_temp"      "Gradio temp files"
delete_dir "$HOME_DIR/.vlm_video_cache"  "VLM video cache"
delete_dir "$HOME_DIR/.nv"               "NVIDIA GPU/JIT cache"
delete_dir "$HOME_DIR/.triton"           "Triton kernel cache"
delete_dir "$HOME_DIR/.cache/uv"         "uv package cache"
delete_dir "$HOME_DIR/.cache/flashinfer" "FlashInfer cache"
delete_dir "$HOME_DIR/.cache/torch"      "torch cache"
echo
fi

# ──────────────────────────────────────────────────────────────────────────
if run_section vllm; then
echo "[2] vLLM compile cache — prune entries older than ${VLLM_KEEP_DAYS} days"
echo "    (vLLM recompiles on demand; recent entries kept warm)"
# ──────────────────────────────────────────────────────────────────────────
VLLM_CACHE="$HOME_DIR/.cache/vllm/torch_compile_cache"
if [ -d "$VLLM_CACHE" ]; then
  echo "  vllm cache total: $(sz "$VLLM_CACHE")"
  mapfile -t OLD < <(find "$VLLM_CACHE" -mindepth 1 -maxdepth 1 -type d -mtime +"$VLLM_KEEP_DAYS" 2>/dev/null)
  if [ "${#OLD[@]}" -eq 0 ]; then
    echo "  no entries older than ${VLLM_KEEP_DAYS} days."
  else
    for d in "${OLD[@]}"; do echo "    stale: $(sz "$d")  $(basename "$d")  ($(stat -c '%y' "$d" | cut -d' ' -f1))"; done
    if confirm "delete ${#OLD[@]} stale vllm entries above"; then
      for d in "${OLD[@]}"; do rm -rf "$d"; done
      echo "    pruned ${#OLD[@]} entries. vllm cache now: $(sz "$VLLM_CACHE")"
    fi
  fi
else
  echo "  [skip] no vllm compile cache present"
fi
echo
fi

# ──────────────────────────────────────────────────────────────────────────
if run_section logs; then
echo "[3] Log archive — cold-archived pre-reorganization logs (safe to delete)"
# ──────────────────────────────────────────────────────────────────────────
LOG_ARCHIVE="/mnt/data/sgsilva/logs/_archive"
if [ -d "$LOG_ARCHIVE" ]; then
  echo "  Total: $(sz "$LOG_ARCHIVE")"
  for sub in "$LOG_ARCHIVE"/*/; do
    [ -d "$sub" ] && echo "    $(sz "$sub")  $(basename "$sub")"
  done
  delete_dir "$LOG_ARCHIVE" "pre-reorganization log archive (cold, git-independent)"
else
  echo "  [skip] $LOG_ARCHIVE — not present"
fi
echo
fi

# ──────────────────────────────────────────────────────────────────────────
if run_section move; then
echo "[4] Heavy data/output dirs — MOVE to $MNT_DIR (symlink left behind)"
echo "    Code stays in /home; only large data/results relocate."
# ──────────────────────────────────────────────────────────────────────────
# Model outputs -> results dir, per project convention.
move_dir "$HOME_DIR/vlm-evaluation/results" "$RESULTS_DIR/vlm-evaluation"
# Bulky generated data / archives -> general scratch under matching repo name.
move_dir "$HOME_DIR/vlm-post-training/aux_tasks" "$MNT_DIR/vlm-post-training"
move_dir "$HOME_DIR/vlm-post-training/archive"   "$MNT_DIR/vlm-post-training"
move_dir "$HOME_DIR/benchmarks"                  "$MNT_DIR"
echo
fi

echo "Done. /home/sgsilva total now: $(sz "$HOME_DIR")"
if [ "$DRY_RUN" -eq 1 ]; then
  echo
  echo "This was a DRY RUN. Re-run with --run to apply, or --run --yes to skip prompts."
fi
