#!/usr/bin/env bash
# 3WC sync — pull the 4 SWORD repos, push ONLY Sandra's 3wc/ work to her personal fork.
#
#   ./sync_3wc.sh pull     # update all 4 SWORD repos (read-only)
#   ./sync_3wc.sh push     # publish dawn-research/3wc/ -> sandragodinhosilva/3wc-sgsilva
#   ./sync_3wc.sh status   # show state of everything, touch nothing
#   ./sync_3wc.sh          # = status
#
# ⛔ HARD RULE: never push to a SWORDHealth remote. Enforced in assert_not_sword()
#    below, which is called before EVERY push. Do not remove it.
set -euo pipefail

MONO="/home/sgsilva/dawn-research"
FORK_REMOTE="3wc-sgsilva"
FORK_URL="git@github.com:sandragodinhosilva/3wc-sgsilva.git"
FORK_BRANCH="main"
PREFIX="3wc"

# The 3 read-only reference clones living inside dawn-research/3wc/ (git-ignored).
REFS="ai-services ai-documentation phoenix-gym"

c_ok=$'\033[32m'; c_warn=$'\033[33m'; c_err=$'\033[31m'; c_dim=$'\033[2m'; c_off=$'\033[0m'
say()  { printf '%s\n' "$*"; }
ok()   { printf '%s✓%s %s\n' "$c_ok"   "$c_off" "$*"; }
warn() { printf '%s!%s %s\n' "$c_warn" "$c_off" "$*"; }
die()  { printf '%s✗%s %s\n' "$c_err"  "$c_off" "$*" >&2; exit 1; }
hdr()  { printf '\n%s== %s ==%s\n' "$c_dim" "$*" "$c_off"; }

# ── the guard ────────────────────────────────────────────────────────────────
# Refuse to push anywhere that looks like SWORDHealth. Called before every push.
assert_not_sword() {
  local remote="$1" url
  url="$(git remote get-url --push "$remote" 2>/dev/null || true)"
  [ -n "$url" ] || die "remote '$remote' has no push URL"
  case "$url" in
    *SWORDHealth*|*swordhealth*)
      die "REFUSING to push to SWORD remote: $url" ;;
  esac
  case "$url" in
    *sandragodinhosilva*) : ;;
    *) die "push target is not a sandragodinhosilva remote: $url" ;;
  esac
}

need_repo() { [ -d "$MONO/.git" ] || die "not a git repo: $MONO"; }

# ── pull ─────────────────────────────────────────────────────────────────────
do_pull() {
  need_repo
  hdr "dawn-research (Dawn upstream — fetch only)"
  git -C "$MONO" fetch origin --prune
  local br; br="$(git -C "$MONO" rev-parse --abbrev-ref HEAD)"
  if git -C "$MONO" rev-parse --verify -q "origin/$br" >/dev/null; then
    local behind; behind="$(git -C "$MONO" rev-list --count "HEAD..origin/$br")"
    if [ "$behind" -gt 0 ]; then
      warn "branch '$br' is $behind commit(s) behind origin/$br"
      say  "  merge yourself when ready:  git -C $MONO merge --ff-only origin/$br"
    else
      ok "branch '$br' up to date with origin/$br"
    fi
  else
    warn "branch '$br' has no origin/$br tracking ref (local-only branch)"
  fi

  for r in $REFS; do
    hdr "$r (read-only reference clone)"
    local d="$MONO/$PREFIX/$r"
    if [ ! -d "$d/.git" ]; then
      warn "missing — clone with:"
      say  "  git clone --depth 1 --branch master git@github.com:SWORDHealth/$r.git $d"
      continue
    fi
    if [ -n "$(git -C "$d" status --porcelain)" ]; then
      warn "$r has local modifications — skipping pull (these are reference clones; keep them pristine)"
      continue
    fi
    git -C "$d" pull --ff-only --quiet 2>&1 | tail -2 || warn "pull failed for $r"
    ok "$r @ $(git -C "$d" log -1 --format='%h %ad' --date=short)"
  done
}

# ── push ─────────────────────────────────────────────────────────────────────
do_push() {
  need_repo
  hdr "publish $PREFIX/ -> $FORK_REMOTE/$FORK_BRANCH"

  git -C "$MONO" remote get-url "$FORK_REMOTE" >/dev/null 2>&1 \
    || git -C "$MONO" remote add "$FORK_REMOTE" "$FORK_URL"
  ( cd "$MONO" && assert_not_sword "$FORK_REMOTE" )
  ok "push target verified non-SWORD: $(git -C "$MONO" remote get-url --push "$FORK_REMOTE")"

  # Uncommitted work under 3wc/ would silently not be published.
  if [ -n "$(git -C "$MONO" status --porcelain -- "$PREFIX")" ]; then
    warn "uncommitted changes under $PREFIX/ — commit them first or they won't be published:"
    git -C "$MONO" status --short -- "$PREFIX"
    die "aborting (nothing pushed)"
  fi

  # Secret scan on exactly what would be published.
  local hits
  hits="$(git -C "$MONO" grep -lE '(sk-[a-zA-Z0-9]{20,}|AIza[a-zA-Z0-9]{20,}|-----BEGIN .*PRIVATE KEY|ghp_[a-zA-Z0-9]{20,})' -- "$PREFIX" 2>/dev/null || true)"
  [ -z "$hits" ] || { say "$hits"; die "possible secrets in $PREFIX/ — aborting"; }
  ok "no secret patterns in $PREFIX/"

  local tmp="3wc-export-$$"
  git -C "$MONO" subtree split --prefix="$PREFIX" -b "$tmp" >/dev/null 2>&1 \
    || die "subtree split failed"
  local n; n="$(git -C "$MONO" ls-tree -r --name-only "$tmp" | wc -l)"
  ok "split: $n files, $(git -C "$MONO" rev-list --count "$tmp") commits (authorship preserved)"

  if git -C "$MONO" push "$FORK_REMOTE" "$tmp:$FORK_BRANCH" 2>&1 | tail -3; then
    ok "pushed to $FORK_REMOTE/$FORK_BRANCH"
  else
    git -C "$MONO" branch -D "$tmp" >/dev/null 2>&1 || true
    die "push failed (history diverged? see: git push $FORK_REMOTE $tmp:$FORK_BRANCH --force-with-lease)"
  fi
  git -C "$MONO" branch -D "$tmp" >/dev/null 2>&1 || true
}

# ── status ───────────────────────────────────────────────────────────────────
do_status() {
  need_repo
  hdr "dawn-research"
  say "  branch:  $(git -C "$MONO" rev-parse --abbrev-ref HEAD)"
  say "  origin:  $(git -C "$MONO" remote get-url origin)  ${c_dim}(SWORD — never push)${c_off}"
  local dirty; dirty="$(git -C "$MONO" status --porcelain | wc -l)"
  [ "$dirty" -eq 0 ] && ok "working tree clean" || warn "$dirty uncommitted change(s)"

  hdr "$PREFIX/ -> personal fork"
  if git -C "$MONO" remote get-url "$FORK_REMOTE" >/dev/null 2>&1; then
    say "  remote:  $(git -C "$MONO" remote get-url "$FORK_REMOTE")"
    local rhead; rhead="$(git ls-remote "$FORK_URL" "refs/heads/$FORK_BRANCH" 2>/dev/null | cut -f1 | head -c 8)"
    [ -n "$rhead" ] && say "  $FORK_BRANCH:    $rhead" || warn "  $FORK_BRANCH not published yet"
  else
    warn "  remote '$FORK_REMOTE' not configured (run: $0 push)"
  fi
  say "  tracked: $(git -C "$MONO" ls-files "$PREFIX" | wc -l) files"

  hdr "reference clones"
  for r in $REFS; do
    local d="$MONO/$PREFIX/$r"
    if [ -d "$d/.git" ]; then
      printf '  %-18s %s\n' "$r" "$(git -C "$d" log -1 --format='%h %ad' --date=short)"
    else
      printf '  %-18s %s\n' "$r" "${c_warn}missing${c_off}"
    fi
  done
}

case "${1:-status}" in
  pull)   do_pull ;;
  push)   do_push ;;
  status) do_status ;;
  both)   do_pull; do_push ;;
  *) die "usage: $0 [pull|push|status|both]" ;;
esac
