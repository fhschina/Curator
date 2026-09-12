# V0.6.2.29 credential recovery and stopped online preflight

Date: 2026-09-10 UTC. This follows the [offline implementation checkpoint](v06229_implementation_report.md).
The credential-loading blocker was resolved without replacing the existing key. The candidate then failed its
online engineering preflight; neither critic nor the formal 50-pair repeats ran. It remains an experiment, not a release.

## Credential-loading diagnosis

The existing `NVIDIA_API_KEY` is stored in `/raid/hfang/workspaces/Curator/.env`, owned by the current user with
permissions `600`. The current worktree `/raid/hfang/codex-home/worktrees/faab/Curator` has no `.env`, and its
execution process initially had no exported `NVIDIA_API_KEY`.

The established loader, `eval.dedup.cli._load_repository_env`, can accept an explicit dotenv path and preserves an
already-exported value. The new standalone diagnostic entry reads the process environment but does not call that
loader. The earlier claim that the user needed to configure the key again was premature: the original configuration
already existed, but this launch had not loaded it. This establishes the present loading gap, not the exact shell
command used by every historical run.

Recovery reused the existing loader inside the evaluation process, before invoking the unchanged frozen runner:

```python
from pathlib import Path
from eval.dedup.cli import _load_repository_env

_load_repository_env(Path("/raid/hfang/workspaces/Curator/.env"))
# Invoke the selected diagnostic in this SAME Python process after loading.
```

The check changed from credential absent to present, and actual inference subsequently returned HTTP 200 on all
12 requests. No new key was requested, generated, copied to this worktree or printed. The source `.env` was not
modified. Future launches must explicitly load this same file again; loading it in one child process does not export
it into other independently launched processes. There is no need to paste the key into a command, source file or chat.

For a read-only loading check in this workspace, the existing environment can run:

```bash
/raid/hfang/llm_judge_env_pr2324_latest/bin/python - <<'PY'
import os
from pathlib import Path
from eval.dedup.cli import _load_repository_env

_load_repository_env(Path("/raid/hfang/workspaces/Curator/.env"))
print("NVIDIA_API_KEY loaded:", bool(os.environ.get("NVIDIA_API_KEY", "").strip()))
PY
```

This prints only a boolean. It does not make a network request or validate a key remotely.

## Actual online result

Run root: `/raid/hfang/ihb/runs/v0.6.2.29-composite-diagnostic`.

Unchanged contract digest: `67376d0367224af13927ab29ee2f71fde758f7495560ecb9a308ce96070cb788`.

| Main preflight measure | Result |
| --- | --- |
| Requested pairs | 8 |
| Finally valid strict outputs | 7 |
| Terminal errors | 1 (`cookie_empty_anchor:ab`) |
| Pairs entering outer retry | 2 (`missing_record_field:ab`, `cookie_empty_anchor:ab`) |
| Recorded native correction | 1, for `missing_record_field:ab` on outer attempt 2 |
| Union of judge-retried pairs | 2/8 (25%) |
| External requests | 12, all HTTP 200 |
| Critic calls / formal repeat calls | 0 / 0 |

The first attempt's strict raw parser rejected `missing_record_field:ab` and `cookie_empty_anchor:ab`. Their
`UNCOVERED` side objects also emitted `harmless_unique_ids` and `opposite_support_ids`, which belong only to the
`COVERED` branch. The missing-field case recovered after retry; the cookie case repeated the invalid branch fields
on all three outer attempts. These are original-response schema errors, not authentication failures, transport-only
null padding or evidence of a passed semantic threshold. The strict parser was not relaxed and no fields were
deleted to turn those outputs into valid judgments.

The frozen runner stopped with `COMPOSITE_MAIN_COMPLETION` before calling critic. Both final completion and the
zero-correction preflight requirement fail. Saved `preflight/main_operations.json`, per-attempt raw traces,
`validation_errors.json`, `terminal_pair_ids.json`, `stopped.json` and `assessment_partial.json` retain the failure.
The seven accepted rows were strictly replayed; an incomplete main pass is not a semantic ranking.

The dedicated CPU Ray cluster was shut down after this bounded preflight; no shared `ray stop` was used. Historical
`.27`/`.28` freezes, `.29` frozen source/config/inputs, reference labels and old caches remain unchanged. No 75% pass,
full-development result, holdout result or 20,000-pair evaluation is claimed. Do not restart this observed run or edit
its frozen candidate in place. A subsequent correction needs its own frozen candidate/run; that correction was not
implemented as part of credential recovery.
