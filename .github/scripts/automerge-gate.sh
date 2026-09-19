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
# required check green, no conflict) at the verified head, because the merge
# token belongs to an administrator and branch protection does not bind
# administrators here. Dependabot PRs are left to their own gate.
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
# Env: GH_TOKEN (PAT with merge rights), PR, REPO, DRY_RUN=1 to print the
# decision without acting.
set -uo pipefail

PR="${PR:?}"; REPO="${REPO:?}"; DRY_RUN="${DRY_RUN:-0}"
say() { printf '%s\n' "$*" | tr -d '\000-\010\013-\037'; }          # script-authored lines only
clean() { printf '%s' "$*" | tr -d '\000-\037' | cut -c1-200; }     # PR-influenced values: no newline can start a workflow command

merge_now() {
  # The merge state is GitHub's own verdict on required checks, conflicts and
  # review requirements; it can be UNKNOWN for a moment after a push. It is
  # re-read immediately before every merge attempt, because the merge token
  # is an administrator's and branch protection would not re-check for it.
  for ATTEMPT in 1 2 3; do
    MS=$(gh pr view "$PR" --repo "$REPO" --json mergeStateStatus,headRefOid --jq '"\(.mergeStateStatus) \(.headRefOid)"' 2>/dev/null) || MS=""
    [ -n "$MS" ] || { say "::warning::Not merged: could not read the merge state of #$PR (API failure); the next completion re-evaluates"; return 0; }
    STATE=${MS%% *}; NOW_SHA=${MS##* }
    [ "$NOW_SHA" = "$HEAD_SHA" ] || { say "Not merged: the head moved from $HEAD_SHA to $(clean "$NOW_SHA") while evaluating"; return 0; }
    if [ "$STATE" = "UNKNOWN" ]; then [ "$ATTEMPT" -lt 3 ] && sleep 15; continue; fi
    if [ "$STATE" != "CLEAN" ]; then
      ROLLUP=$(gh pr view "$PR" --repo "$REPO" --json statusCheckRollup --jq '[.statusCheckRollup[] | select((.conclusion // "") != "SUCCESS" and (.conclusion // "") != "SKIPPED" and (.conclusion // "") != "NEUTRAL") | "\(.name // .context)=\(.conclusion // .status // "pending")"] | join(", ")' 2>/dev/null || true)
      say "::notice::Not merged: merge state is $(clean "$STATE") at $HEAD_SHA ($(clean "${ROLLUP:-no failing or pending checks listed}")); the next completion re-evaluates"
      return 0
    fi
    # The merge state is not the only witness: every status check that branch
    # protection requires must be green at this head, read from the checks
    # themselves, because the merge token is an administrator's.
    REQUIRED_CTX=$(gh api "repos/$REPO/branches/$BASE_REF/protection/required_status_checks" --jq '.contexts[]' 2>/dev/null) || { say "::warning::Not merged: could not read the required status checks"; return 0; }
    ROLLUP_JSON=$(gh pr view "$PR" --repo "$REPO" --json statusCheckRollup --jq '.statusCheckRollup' 2>/dev/null) || { say "::warning::Not merged: could not read the checks"; return 0; }
    MISSING=""
    while IFS= read -r CTX; do
      [ -n "$CTX" ] || continue
      OK=$(printf '%s' "$ROLLUP_JSON" | jq -r --arg c "$CTX" '[.[] | select((.name // .context) == $c) | (.conclusion // .state // "")] | if length == 0 then "absent" elif all(. == "SUCCESS" or . == "SKIPPED" or . == "NEUTRAL") then "ok" else join(",") end')
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
# to date" provides it mechanically (iag-infra and iag-agents use it).

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
  if printf '%s' "$TITLE" | grep -qE '[Bb]ump [^ ]+( [^ ]+)* from [0-9]+\.[0-9]+\.[0-9]+ to [0-9]+\.[0-9]+\.[0-9]+$'; then
    OLD=$(printf '%s' "$TITLE" | sed -E 's/.* from ([0-9]+\.[0-9]+\.[0-9]+) to [0-9]+\.[0-9]+\.[0-9]+$/\1/')
    NEW=$(printf '%s' "$TITLE" | sed -E 's/.* to ([0-9]+\.[0-9]+\.[0-9]+)$/\1/')
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
