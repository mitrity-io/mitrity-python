#!/usr/bin/env bash
# Auto-merge gate. Executed by .github/workflows/auto-merge.yml, which runs on
# the DEFAULT branch's definition whenever one of the PR's workflows completes
# — so neither this script nor the merge token ever comes from a PR's own
# head. The review workflows themselves hold no merge token.
#
# Nothing is ever ARMED: GitHub's auto-merge is not bound to a head, so an
# arming made for one head would merge whatever head the PR has when its
# checks turn green. This gate instead re-evaluates on every completion and
# merges only with the head SHA pinned (--match-head-commit): a merge happens
# only for the exact commit whose reviews it verified, and a push in between
# makes the merge fail closed. If required checks are still pending the
# attempt is refused by GitHub and the next completion re-evaluates.
#
# Both review agents submit their PR reviews as github-actions[bot], so
# GitHub's own review decision only reflects whichever agent reviewed LAST;
# the gate requires the latest review of every agent this PR needs to be an
# approval of the current head.
#
# The reviews only mean something when everything that shaped them is the
# base branch's: a pull_request workflow runs the PR head's definition, and
# the review agents read the head's instruction files. So the head's
# .github/, .claude/ (root or nested), CLAUDE.md and AGENTS.md (root or
# nested), .mcp.json and root action.yml must not be changed by the PR at
# all; otherwise the gate refuses and the PR is merged by hand. What keeps a PR from forging its own
# approvals is the split of duties among workflows: the review workflows hold
# pull-requests: write to post their reviews but execute nothing from the
# head (Read/Grep/Glob only); every job that executes head code runs with a
# read-only token; a job that posts a comment from head-produced output
# checks out nothing.
#
# Dependabot PRs are merged by this gate too, never on a model verdict alone:
# the Dependency Review Agent must approve the head AND the PR title must
# describe a single-package PATCH bump; anything else is merged by hand.
#
# Threat model, stated: the gate defends against the CONTENT of one pull
# request (code an agent was steered into committing). It does not defend
# against whoever holds push rights: a second, attacker-authored PR carrying
# a workflow with pull-requests: write could post forged approvals on any PR,
# and that actor could equally push to main. Per-agent App identities and
# protected environments would close that; both are outside this gate.
#
# Stacked PRs (base other than the default branch) are never merged by the
# gate; they are retargeted by hand once their base has landed.
#
# Merging is verified, not assumed: the PR's merge state must be CLEAN (every
# required check green, no conflict) at the verified head, and every status
# check that branch protection requires is read back green from the workflow
# jobs behind it, through the Actions API. The checks come from the Actions
# API because the merge token cannot read the check-run rollup: it is a
# fine-grained token with contents, pull-request and Actions (read) rights,
# GitHub offers it no Checks permission, and GraphQL's statusCheckRollup
# fails on it deterministically (revision 8's diagnostic line, 2026-09-20),
# while the runs and jobs at the head are Actions reads. Only pull_request
# runs count, the one event auto-merge.yml itself trusts: a push or
# workflow_dispatch run at the same head, whose jobs may carry the same
# names as ci.yml's (deploy-dev.yml does), can neither vouch for a required
# context nor outrank a CI job of the same name. A required context that is
# not a pull_request Actions job (a commit status posted by an app; a job of
# a workflow_run or pull_request_target workflow, whose run has another
# head) has no job at the head, reads as absent and the gate refuses: that
# is the fail-closed direction, and such a PR is merged by hand. The token
# has no
# administration permission; the gate does not rely on that, it verifies.
# Dependabot PRs are left to their own gate.
#
# Whether the security agent is required is derived from facts a PR cannot
# influence from its branch: the base branch's security-review.yml path filter
# matched against the PR's full changed-file list; any Security Review Agent
# run for the head; a standing security REQUEST_CHANGES on the PR. Every probe
# fails CLOSED (required and unsatisfied, with a ::warning::).
#
# Residual, stated: an agent's identity is the review body header its Submit
# step writes; see the threat model above for what that does and does not
# cover.
#
# Reads on the merge path are retried before they refuse. A CLEAN merge state
# means every check has completed, so the completion that reaches the merge
# path is the PR's last: a read that fails there is not re-evaluated by a
# later completion and the PR sits unmerged until the gate is rerun by hand
# (#456, 2026-09-20). Three attempts, 5 s apart, and then the refusal stands
# as before, with the last error in the log so the line says whether the
# read was transient or the token cannot see the resource.
#
# Gate revision 10.
#
# Env: GH_TOKEN (PAT with merge rights), PR, REPO, DRY_RUN=1 to print the
# decision without acting.
set -uo pipefail

PR="${PR:?}"; REPO="${REPO:?}"; DRY_RUN="${DRY_RUN:-0}"
say() { printf '%s\n' "$*" | tr -d '\000-\010\013-\037'; }          # script-authored lines only
clean() { printf '%s' "$*" | tr -d '\000-\037' | cut -c1-200; }     # PR-influenced values: no newline can start a workflow command

# Retried read for the merge path (see the header). Usage:
#   retry_read VAR cmd args...
# On the first attempt that exits 0, VAR holds the command's stdout and the
# function returns 0. After three failures it returns 1 and READ_ERR holds
# the last attempt's stderr as one line, sanitized the way clean() sanitizes
# PR-influenced values (control characters removed, so no newline can start
# a workflow command; capped at 300) and with anything shaped like a GitHub
# token redacted, so the refusal can print it as its reason. stderr goes
# through a file: a command substitution captures one stream, and the
# caller's shell has to end up holding both.
retry_read() {
  local _var=$1 _out _err _n; shift
  READ_ERR=""
  _err=$(mktemp) || { READ_ERR="could not create a temporary file for stderr"; return 1; }
  for _n in 1 2 3; do
    if _out=$("$@" 2>"$_err"); then rm -f "$_err"; printf -v "$_var" '%s' "$_out"; return 0; fi
    [ "$_n" -lt 3 ] && sleep 5
  done
  READ_ERR=$(tr -d '\000-\037' <"$_err" | sed -E 's/(gh[pousr]|github_pat)_[A-Za-z0-9_]{8,}/[redacted]/g' | cut -c1-300)
  rm -f "$_err"
  [ -n "$READ_ERR" ] || READ_ERR="no error output"
  return 1
}

# The workflow jobs at the head, read through the Actions API (see the
# header): every pull_request workflow run whose head is HEAD_SHA, then
# every job of every attempt of each run (filter=all, so a re-run leaves the
# superseded attempt visible and the evaluation can pick the latest). Both
# listings page. On success JOBS_JSON holds one JSON array of
# {run_id, run_attempt, id, name, status, conclusion}; on failure the
# function returns 1 with READ_ERR set the way retry_read sets it, prefixed
# with the read that failed. The head SHA and the run ids are interpolated
# into URLs, so a value of the wrong shape is a refusal, not a request; the
# ids are read line by line, never word-split or glob-expanded.
read_head_jobs() {
  local _run _ids _jobs _all=""
  JOBS_JSON=""
  case "$HEAD_SHA" in ''|*[!0-9a-f]*) READ_ERR="unexpected head SHA '$(clean "$HEAD_SHA")'"; return 1;; esac
  retry_read _ids gh api "repos/$REPO/actions/runs?head_sha=$HEAD_SHA&event=pull_request&per_page=100" --paginate --jq '.workflow_runs[].id' || { READ_ERR="listing the runs: $READ_ERR"; return 1; }
  while IFS= read -r _run; do
    [ -n "$_run" ] || continue
    case "$_run" in *[!0-9]*) READ_ERR="unexpected run id '$(clean "$_run")' in the run listing"; return 1;; esac
    retry_read _jobs gh api "repos/$REPO/actions/runs/$_run/jobs?filter=all&per_page=100" --paginate --jq '.jobs[] | {run_id, run_attempt, id, name, status, conclusion}' </dev/null || { READ_ERR="run $_run: $READ_ERR"; return 1; }
    _all="$_all$_jobs"$'\n'
  done <<EOF_RUNS
$_ids
EOF_RUNS
  JOBS_JSON=$(printf '%s' "$_all" | jq -sc '.') || { READ_ERR="the job listing did not parse as JSON"; JOBS_JSON=""; return 1; }
}

# The latest job of a name decides for that name: highest run_id, then
# run_attempt, then id, so a re-run of a failed job supersedes the failure
# and a newer run supersedes an older one. Conclusions are the Actions API's
# (lowercase). success, skipped and neutral are green on purpose: skipped and
# neutral are GitHub's own reading of a required context whose job-level
# `if:` was false (a review workflow that skips Dependabot PRs); the
# review-identity check, not this rule, is what requires the agents'
# approvals. A name with no job at the head (never run, or a commit status
# rather than an Actions job) is "absent"; a job still running has no
# conclusion and shows its status instead. Everything else (failure,
# cancelled, timed_out, action_required, ...) is its conclusion, verbatim.
# Two jq definitions, prepended to every program that applies the rule.
JOB_RULE='def latest: sort_by([.run_id, .run_attempt, .id]) | last;
  def verdict: if . == null then "absent"
    elif (.conclusion // "") == "success" or (.conclusion // "") == "skipped" or (.conclusion // "") == "neutral" then "ok"
    else (.conclusion // .status // "pending") end;'

merge_now() {
  # The merge state is GitHub's own verdict on required checks, conflicts and
  # review requirements; it can be UNKNOWN for a moment after a push. It is
  # re-read immediately before every merge attempt so that a push or a check
  # that landed while the reviews were being evaluated is never merged over.
  for ATTEMPT in 1 2 3; do
    retry_read MS gh pr view "$PR" --repo "$REPO" --json mergeStateStatus,headRefOid --jq '"\(.mergeStateStatus) \(.headRefOid)"' || { say "::warning::Not merged: could not read the merge state of #$PR after 3 attempts: $READ_ERR; the next completion re-evaluates"; return 0; }
    [ -n "$MS" ] || { say "::warning::Not merged: the merge state of #$PR came back empty; the next completion re-evaluates"; return 0; }
    STATE=${MS%% *}; NOW_SHA=${MS##* }
    [ "$NOW_SHA" = "$HEAD_SHA" ] || { say "Not merged: the head moved from $HEAD_SHA to $(clean "$NOW_SHA") while evaluating"; return 0; }
    if [ "$STATE" = "UNKNOWN" ]; then [ "$ATTEMPT" -lt 3 ] && sleep 15; continue; fi
    if [ "$STATE" != "CLEAN" ]; then
      # Explanation only (the state already refused): every job name whose
      # latest job at the head is not green, from the same Actions data.
      if read_head_jobs; then
        NOT_GREEN=$(printf '%s' "$JOBS_JSON" | jq -r "$JOB_RULE"' [group_by(.name)[] | latest | verdict as $v | select($v != "ok") | "\(.name)=\($v)"] | join(", ")') || NOT_GREEN="the job listing did not evaluate"
      else NOT_GREEN="workflow jobs unreadable after 3 attempts: $READ_ERR"; fi
      say "::notice::Not merged: merge state is $(clean "$STATE") at $HEAD_SHA ($(clean "${NOT_GREEN:-no failing or pending jobs listed}")); the next completion re-evaluates"
      return 0
    fi
    # The merge state is not the only witness: every status check that branch
    # protection requires must be green at this head, read from the workflow
    # jobs behind it (read_head_jobs; see the header). The required contexts
    # come from the branch object, which read access can see (the
    # branch-protection endpoints need administration rights). Every degraded
    # case is a refusal: an API error, a protection block that is absent or
    # not enabled (a token that cannot see it, or a branch governed by
    # rulesets, which never appear here), a protection that names no required
    # check at all, and runs or jobs that cannot be read.
    retry_read BRANCH_JSON gh api "repos/$REPO/branches/$BASE_REF" || { say "::warning::Not merged: could not read branch $BASE_REF after 3 attempts: $READ_ERR"; return 0; }
    PROT_ENABLED=$(printf '%s' "$BRANCH_JSON" | jq -r '.protection.enabled // false' 2>/dev/null) || PROT_ENABLED=""
    [ "$PROT_ENABLED" = "true" ] || { say "::warning::Not merged: branch protection on $BASE_REF is not visible to the merge token (enabled=$(clean "${PROT_ENABLED:-unreadable}"))"; return 0; }
    # Both spellings of the required list: checks[].context (current) and the
    # contexts mirror (kept for compatibility); a check named in either counts.
    REQUIRED_CTX=$(printf '%s' "$BRANCH_JSON" | jq -r '.protection.required_status_checks | select(type == "object") | (((.checks // []) | map(.context)) + (.contexts // [])) | map(select(type == "string" and length > 0)) | unique | .[]' 2>/dev/null) || REQUIRED_CTX=""
    [ -n "$REQUIRED_CTX" ] || { say "::warning::Not merged: branch protection on $BASE_REF names no required status check"; return 0; }
    read_head_jobs || { say "::warning::Not merged: could not read the workflow jobs after 3 attempts: $READ_ERR"; return 0; }
    MISSING=""
    # One verdict per required context, by JOB_RULE; a verdict that cannot be
    # computed is a refusal too.
    while IFS= read -r CTX; do
      [ -n "$CTX" ] || continue
      OK=$(printf '%s' "$JOBS_JSON" | jq -r --arg c "$CTX" "$JOB_RULE"' [.[] | select(.name == $c)] | latest | verdict') || OK=unreadable
      [ "$OK" = "ok" ] || MISSING="$MISSING $CTX=$OK"
    done <<EOF_CTX
$REQUIRED_CTX
EOF_CTX
    [ -z "$MISSING" ] || { say "::notice::Not merged: required checks not green at $HEAD_SHA:$(clean "$MISSING"); the next completion re-evaluates"; return 0; }
    if [ "$DRY_RUN" = "1" ]; then say "DRY_RUN: would merge #$PR at $HEAD_SHA (merge state CLEAN, required checks green)"; return 0; fi
    OUT=$(gh pr merge "$PR" --repo "$REPO" --squash --match-head-commit "$HEAD_SHA" 2>&1) && { say "Merged #$PR at $HEAD_SHA"; return 0; }
    say "::notice::merge attempt $ATTEMPT/3 refused: $(clean "$OUT")"
    [ "$ATTEMPT" -lt 3 ] && sleep 15
  done
  say "::warning::PR #$PR was not merged; the next completion re-evaluates"
}
disarm() {
  # nothing is armed by this gate; disarming covers an arming made by hand or by an older gate
  if [ "$DRY_RUN" = "1" ]; then say "DRY_RUN: would not merge #$PR ($1)"; return 0; fi
  gh pr merge "$PR" --repo "$REPO" --disable-auto >/dev/null 2>&1 || true
  say "Not merged: $1"
}

PRJSON=$(gh pr view "$PR" --repo "$REPO" --json headRefOid,baseRefName,state,changedFiles,author 2>/dev/null) || { disarm "could not read PR #$PR"; exit 0; }
[ "$(printf '%s' "$PRJSON" | jq -r .state)" = "OPEN" ] || { say "PR #$PR is not open; nothing to do"; exit 0; }
HEAD_SHA=$(printf '%s' "$PRJSON" | jq -r .headRefOid)
BASE_REF=$(printf '%s' "$PRJSON" | jq -r .baseRefName)
# A PR merges into its base. The gate only ever merges into the default
# branch: a stacked PR (base = another feature branch) would otherwise be
# squashed into its parent branch the moment both agents approve it.
DEFAULT_BRANCH=$(gh api "repos/$REPO" --jq .default_branch 2>/dev/null) || { disarm "could not read the default branch"; exit 0; }
if [ "$BASE_REF" != "$DEFAULT_BRANCH" ]; then
  disarm "PR #$PR is stacked on '$(clean "$BASE_REF")', not on $DEFAULT_BRANCH; retarget it after its base merges"; exit 0
fi
N_FILES=$(printf '%s' "$PRJSON" | jq -r '.changedFiles // 0')
AUTHOR=$(printf '%s' "$PRJSON" | jq -r '.author.login // ""')
DEPENDABOT=0
case "$AUTHOR" in dependabot\[bot\]|app/dependabot) DEPENDABOT=1;; esac
case "$N_FILES" in ''|*[!0-9]*) disarm "could not determine the PR's file count ('$(clean "$N_FILES")')"; exit 0;; esac

# ── the PR's files: a PR-controlled workflow definition is never armed ─────
if [ "$N_FILES" -gt 3000 ]; then disarm "PR changes $N_FILES files, more than the files API lists; manual merge required"; exit 0; fi
# Both names of a renamed file count: moving a workflow out of .github/ is a
# change to the workflow surface, and the new name alone would not show it.
CHANGED=$(gh api "repos/$REPO/pulls/$PR/files" --paginate --jq '.[] | .filename, (.previous_filename // empty)' 2>/dev/null) || { disarm "could not list the PR files"; exit 0; }
# An empty list is a refusal too: a PR always changes at least one file, so an
# empty answer is a partial or failed listing, and everything below (the
# surface check, the security path filter) would otherwise read it as "nothing".
[ -n "$(printf '%s' "$CHANGED" | tr -d '[:space:]')" ] || { disarm "the PR file list came back empty"; exit 0; }
SURFACE_RE='^\.github/|(^|/)CLAUDE\.md$|(^|/)AGENTS\.md$|(^|/)\.claude/|^\.mcp\.json$|^action\.ya?ml$'
WF_CHANGE=$(printf '%s\n' "$CHANGED" | grep -m1 -E "$SURFACE_RE" || true)
if [ -n "$WF_CHANGE" ]; then
  disarm "PR changes the workflow or reviewer-instruction surface ($(clean "$WF_CHANGE")); reviews produced under a PR-controlled definition are not trusted, manual merge required"; exit 0
fi
# A PR that leaves the surface untouched ran its reviews under the surface of
# the base commit it branched from — an older revision of main's, trusted at
# the time. Refusing every PR that predates a change to .github/ would force a
# rebase and a full re-review of every open PR after each such change; where
# that strictness is wanted, branch protection's "require branches to be up
# to date" provides it mechanically (other MITRITY repositories use it).

# ── the reviews at head ─────────────────────────────────────────────────────
REVIEWS=$(gh api "repos/$REPO/pulls/$PR/reviews" --paginate \
  --jq '.[] | select(.user.login == "github-actions[bot]")
        | [ ((.body // "") | if startswith("## Security Review Agent") then "security"
                     elif startswith("## Code Review Agent") then "code"
                     elif startswith("## Dependency Review Agent") then "dependency"
                     else "other" end), .state, .commit_id ] | @tsv') || { disarm "could not read the PR reviews"; exit 0; }
latest() { printf '%s\n' "$REVIEWS" | awk -F'\t' -v k="$1" '$1 == k { line = $0 } END { print line }'; }
for KIND in code security dependency; do
  L=$(latest "$KIND")
  if [ -n "$L" ] && [ "$(printf '%s' "$L" | cut -f2)" = "CHANGES_REQUESTED" ] && [ "$(printf '%s' "$L" | cut -f3)" = "$HEAD_SHA" ]; then
    disarm "$KIND review requested changes on the current head"; exit 0
  fi
done

# ── Dependabot: the deterministic semver gate lives here now ───────────────
if [ "$DEPENDABOT" = "1" ]; then
  TITLE=$(gh pr view "$PR" --repo "$REPO" --json title --jq .title 2>/dev/null) || { disarm "could not read the PR title"; exit 0; }
  TIER=unknown
  # Same rule as dependency-review.yml's classifier: a full X.Y.Z on both sides,
  # anything else is unknown and merged by hand.
  # The subject is the title without Dependabot's directory suffix (" in /dir"),
  # so the guard and the two version extractions read the same string.
  SUBJECT=$(printf '%s' "$TITLE" | sed -E 's/ in [^ ]+$//')
  if printf '%s' "$SUBJECT" | grep -qE '(^|: )[Bb]ump [^ ]+ from [0-9]+\.[0-9]+\.[0-9]+ to [0-9]+\.[0-9]+\.[0-9]+$'; then
    OLD=$(printf '%s' "$SUBJECT" | sed -E 's/.* from ([0-9]+\.[0-9]+\.[0-9]+) to [0-9]+\.[0-9]+\.[0-9]+$/\1/')
    NEW=$(printf '%s' "$SUBJECT" | sed -E 's/.* to ([0-9]+\.[0-9]+\.[0-9]+)$/\1/')
    if [ "$(printf '%s' "$OLD" | cut -d. -f1)" != "$(printf '%s' "$NEW" | cut -d. -f1)" ]; then TIER=major
    elif [ "$(printf '%s' "$OLD" | cut -d. -f2)" != "$(printf '%s' "$NEW" | cut -d. -f2)" ]; then TIER=minor
    else TIER=patch; fi
  fi
  if [ "$TIER" != "patch" ]; then disarm "Dependabot PR is a $(clean "$TIER") bump ($(clean "$TITLE")); only single-package patch bumps merge unattended"; exit 0; fi
  L=$(latest dependency); STATE=$(clean "$(printf '%s' "$L" | cut -f2)"); SHA=$(clean "$(printf '%s' "$L" | cut -f3)")
  if [ "$STATE" != "APPROVED" ] || [ "$SHA" != "$HEAD_SHA" ]; then
    disarm "latest dependency review is '${STATE:-none}' on '${SHA:-none}', head is $HEAD_SHA"; exit 0
  fi
  say "Dependabot patch bump approved by the Dependency Review Agent at $HEAD_SHA; merging PR #$PR at that head"
  merge_now; exit 0
fi

# ── is the security agent required? ────────────────────────────────────────
SECURITY_REQUIRED=0
BASE_WF=$(gh api "repos/$REPO/contents/.github/workflows/security-review.yml?ref=$BASE_REF" --jq .content 2>/dev/null | base64 -d 2>/dev/null) || BASE_WF="__FETCH_FAILED__"
if [ "$BASE_WF" = "__FETCH_FAILED__" ]; then
  LISTING=$(gh api "repos/$REPO/contents/.github/workflows?ref=$BASE_REF" --jq '.[].name' 2>/dev/null) || { say "::warning::could not list the base workflows; security review treated as required"; SECURITY_REQUIRED=1; LISTING=""; }
  if printf '%s\n' "$LISTING" | grep -qx 'security-review.yml'; then say "::warning::could not fetch the base security-review.yml; security review treated as required"; SECURITY_REQUIRED=1
  elif [ -n "$LISTING" ]; then say "Base branch has no security-review workflow; only the code review is required"; fi
else
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
        return None                    # character classes are not modeled: fail closed
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
        print("__UNMODELED_PATTERN__"); sys.exit(0)
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
  L=$(latest "$KIND"); STATE=$(clean "$(printf '%s' "$L" | cut -f2)"); SHA=$(clean "$(printf '%s' "$L" | cut -f3)")
  if [ "$STATE" != "APPROVED" ] || [ "$SHA" != "$HEAD_SHA" ]; then
    disarm "latest $KIND review is '${STATE:-none}' on '${SHA:-none}', head is $HEAD_SHA"; exit 0
  fi
done
say "Every required agent ($REQUIRED) approved $HEAD_SHA; merging PR #$PR at that head"
merge_now
