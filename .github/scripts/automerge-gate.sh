#!/usr/bin/env bash
# Auto-merge gate. Executed by .github/workflows/auto-merge.yml, which runs on
# the DEFAULT branch's definition when a review agent workflow completes — so
# neither this script nor the token that arms the merge ever comes from a PR's
# own head. The review workflows themselves hold no merge token.
#
# Both review agents submit their PR reviews as github-actions[bot], so
# GitHub's own review decision only reflects whichever agent reviewed LAST.
# This gate arms auto-merge only when the latest review of every agent this PR
# requires is an approval of the PR's CURRENT head, and DISARMS whenever that
# is not the case — an arming from an earlier head must not survive a push.
#
# Whether the security agent is required is derived from facts a PR cannot
# influence from its branch: the base branch's security-review.yml path filter
# matched against the PR's full changed-file list; any Security Review Agent
# run for the head; a standing security REQUEST_CHANGES on the PR. Every probe
# fails CLOSED (required and unsatisfied, with a ::warning::).
#
# Residual, stated: agent identity is inferred from the review body header the
# Submit step writes.
#
# Env: GH_TOKEN (PAT with pull-requests: write), PR, REPO, DRY_RUN=1 to print
# the decision without acting.
set -uo pipefail

PR="${PR:?}"; REPO="${REPO:?}"; DRY_RUN="${DRY_RUN:-0}"
say() { printf '%s\n' "$*" | tr -d '\000-\010\013-\037'; }   # never let a PR-controlled string emit a workflow command

arm() {
  if [ "$DRY_RUN" = "1" ]; then say "DRY_RUN: would arm auto-merge for #$PR"; return 0; fi
  for ATTEMPT in 1 2 3; do
    if gh pr merge "$PR" --repo "$REPO" --auto --squash 2>&1; then say "Auto-merge queued"; return 0; fi
    say "::warning::Auto-merge attempt $ATTEMPT/3 failed, retrying in 10s..."; sleep 10
  done
  say "::warning::Auto-merge could not be queued after 3 attempts"
}
disarm() {
  if [ "$DRY_RUN" = "1" ]; then say "DRY_RUN: would disarm auto-merge for #$PR ($1)"; return 0; fi
  gh pr merge "$PR" --repo "$REPO" --disable-auto >/dev/null 2>&1 || true
  say "Auto-merge disarmed: $1"
}

PRJSON=$(gh pr view "$PR" --repo "$REPO" --json headRefOid,baseRefName,state 2>/dev/null) || { say "::warning::could not read PR #$PR; nothing armed"; exit 0; }
[ "$(printf '%s' "$PRJSON" | jq -r .state)" = "OPEN" ] || { say "PR #$PR is not open; nothing to do"; exit 0; }
HEAD_SHA=$(printf '%s' "$PRJSON" | jq -r .headRefOid)
BASE_REF=$(printf '%s' "$PRJSON" | jq -r .baseRefName)

# ── the reviews at head ─────────────────────────────────────────────────────
REVIEWS=$(gh api "repos/$REPO/pulls/$PR/reviews" --paginate \
  --jq '.[] | select(.user.login == "github-actions[bot]")
        | [ ((.body // "") | if startswith("## Security Review Agent") then "security"
                     elif startswith("## Code Review Agent") then "code"
                     else "other" end), .state, .commit_id ] | @tsv') || { disarm "could not read the PR reviews"; exit 0; }
latest() { printf '%s\n' "$REVIEWS" | awk -F'\t' -v k="$1" '$1 == k { line = $0 } END { print line }'; }
for KIND in code security; do
  L=$(latest "$KIND")
  if [ -n "$L" ] && [ "$(printf '%s' "$L" | cut -f2)" = "CHANGES_REQUESTED" ] && [ "$(printf '%s' "$L" | cut -f3)" = "$HEAD_SHA" ]; then
    disarm "$KIND review requested changes on the current head"; exit 0
  fi
done

# ── is the security agent required? ────────────────────────────────────────
SECURITY_REQUIRED=0
BASE_WF=$(gh api "repos/$REPO/contents/.github/workflows/security-review.yml?ref=$BASE_REF" --jq .content 2>/dev/null | base64 -d 2>/dev/null) || BASE_WF="__FETCH_FAILED__"
if [ "$BASE_WF" = "__FETCH_FAILED__" ]; then
  LISTING=$(gh api "repos/$REPO/contents/.github/workflows?ref=$BASE_REF" --jq '.[].name' 2>/dev/null) || { say "::warning::could not list the base workflows; security review treated as required"; SECURITY_REQUIRED=1; LISTING=""; }
  if printf '%s\n' "$LISTING" | grep -qx 'security-review.yml'; then say "::warning::could not fetch the base security-review.yml; security review treated as required"; SECURITY_REQUIRED=1
  elif [ -n "$LISTING" ]; then say "Base branch has no security-review workflow; only the code review is required"; fi
else
  CHANGED=$(gh api "repos/$REPO/pulls/$PR/files" --paginate --jq '.[].filename' 2>/dev/null) || { say "::warning::could not list the PR files; security review treated as required"; SECURITY_REQUIRED=1; CHANGED=""; }
  if [ "$SECURITY_REQUIRED" = "0" ]; then
    MATCH=$(BASE_WF_TEXT="$BASE_WF" CHANGED_FILES="$CHANGED" python3 - <<'PY'
import os, re, sys
text = os.environ.get("BASE_WF_TEXT", "")
paths = None
try:
    import yaml  # present on ubuntu-latest; the line parser below is the fallback
    doc = yaml.safe_load(text) or {}
    on = doc.get("on") or doc.get(True) or {}
    pr = on.get("pull_request") if isinstance(on, dict) else None
    if pr is None:
        print("__NO_PULL_REQUEST_TRIGGER__"); sys.exit(0)
    paths = pr.get("paths") if isinstance(pr, dict) else None
    paths = list(paths) if paths else []
except Exception:
    paths = None
if paths is None:
    paths, in_pr, in_paths, pr_indent, paths_indent = [], False, False, None, None
    for line in text.splitlines():
        stripped = line.strip(); indent = len(line) - len(line.lstrip(" "))
        if not stripped or stripped.startswith("#"): continue
        if in_paths:
            if indent > paths_indent and stripped.startswith("- "):
                paths.append(stripped[2:].strip().strip('"').strip("'")); continue
            in_paths = False
        if in_pr and pr_indent is not None and indent <= pr_indent and not stripped.startswith("- "): in_pr = False
        if stripped.startswith("pull_request:"): in_pr, pr_indent = True, indent; continue
        if in_pr and stripped.startswith("paths:"): in_paths, paths_indent = True, indent; continue
if not paths:
    print("all"); sys.exit(0)          # no filter: the security agent runs on every PR
def glob_re(g):
    if "[" in g or "]" in g or "{" in g:
        return None                    # character classes are not modelled: fail closed
    out, i = "", 0
    while i < len(g):
        if g.startswith("**/", i): out += "(?:.*/)?"; i += 3
        elif g.startswith("**", i): out += ".*"; i += 2
        elif g[i] == "*": out += "[^/]*"; i += 1
        elif g[i] == "?": out += "[^/]"; i += 1
        else: out += re.escape(g[i]); i += 1
    return re.compile("^" + out + "$")
res = []
for p in paths:
    if p.startswith("!"): continue     # negations ignored in the safe direction
    r = glob_re(p)
    if r is None:
        print("__UNMODELLED_PATTERN__"); sys.exit(0)
    res.append(r)
for f in os.environ.get("CHANGED_FILES", "").split("\n"):
    if f and any(r.match(f) for r in res):
        print("match"); sys.exit(0)
print("none")
PY
) || MATCH="__PY_FAILED__"
    case "$MATCH" in
      all|match) SECURITY_REQUIRED=1; say "Security review required by the base path filter ($MATCH)";;
      none) say "No changed file matches the base security path filter";;
      *) say "::warning::path filter evaluation: $MATCH; security review treated as required"; SECURITY_REQUIRED=1;;
    esac
  fi
fi
RUNS=$(gh api "repos/$REPO/actions/workflows/security-review.yml/runs?head_sha=$HEAD_SHA&per_page=100" --paginate --jq '.workflow_runs | length' 2>/dev/null) || RUNS="__FAILED__"
if [ "$RUNS" = "__FAILED__" ]; then
  [ "$BASE_WF" != "__FETCH_FAILED__" ] && { say "::warning::could not query the security workflow runs; security review treated as required"; SECURITY_REQUIRED=1; }
elif [ "$(printf '%s\n' "$RUNS" | awk '{s+=$1} END {print s+0}')" -gt 0 ]; then
  SECURITY_REQUIRED=1; say "A Security Review Agent run exists for $HEAD_SHA"
fi
SEC_LATEST=$(latest security)
if [ -n "$SEC_LATEST" ] && [ "$(printf '%s' "$SEC_LATEST" | cut -f2)" = "CHANGES_REQUESTED" ]; then
  SECURITY_REQUIRED=1; say "A security REQUEST_CHANGES is standing; a newer security approval at head is required"
fi

# ── decide ──────────────────────────────────────────────────────────────────
REQUIRED="code"; [ "$SECURITY_REQUIRED" = "1" ] && REQUIRED="code security"
for KIND in $REQUIRED; do
  L=$(latest "$KIND"); STATE=$(printf '%s' "$L" | cut -f2); SHA=$(printf '%s' "$L" | cut -f3)
  if [ "$STATE" != "APPROVED" ] || [ "$SHA" != "$HEAD_SHA" ]; then
    disarm "latest $KIND review is '${STATE:-none}' on '${SHA:-none}', head is $HEAD_SHA"; exit 0
  fi
done
say "Every required agent ($REQUIRED) approved $HEAD_SHA; queuing auto-merge for PR #$PR"
arm
