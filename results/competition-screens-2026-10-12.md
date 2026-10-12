# Bounded local competition screens - 12 October 2026 UTC

Two inexpensive checks use original training/validation evidence only. Neither
reads official scores or challenge labels, fits a new model, changes the retained
model, or uploads an artifact. Their scope differs from the purged predeparture
MAE board, so they receive no comparable leaderboard rank.

## Signed-clock fitting audit

Inspect the first six 2025 training files, restricted to actual UTC movement
times from 1 January through 30 June. There are 1,005,519 departures and 220
negative labels, with minimum -12 seconds. On origin-matched rows where both
NM clock proxies are negative, agree within 120 seconds and stay above -172,800,
only one of nine rows has a negative target; eight have nonnegative targets.
Using the signed actual-clock proxy gives squared error 23,151,695, versus
8,410,239 for zero on those nine rows. Reject the shortcut without holdout
search, native fitting or upload. These small negative labels do not support a
claim that clipping explains the current official gap.

## One fixed protected equal mixture

Before calculating metrics, freeze one candidate: equal weights on the saved
long-following ensemble and standalone NM model, outside the original specialist
and clock scopes. Preserve all 429 protected validation rows exactly. Require
at least one second of whole RMSE improvement, improvement in both months, and
at least 40 improved days. Do not search weights.

The 344,419 saved validation rows align exactly by movement ID, schedule and
target; all 80 negative labels remain. Their historical January/July reuse is
explicit and does not establish future-month or predeparture generalization.

| Local diagnostic | Retained reference | Fixed equal mixture |
| --- | ---: | ---: |
| Whole RMSE, seconds | 312.470399515 | 316.333514561 |
| January RMSE | 338.696316351 | 341.727515348 |
| July RMSE | 289.610268590 | 294.276649027 |

Gain is **-3.863115046 seconds**, and **0 / 62 days improve**. All three
predeclared checks fail. Reject the screen; no fresh native validation,
refitting or submission follows. The three-second CPU check and its frozen
source/input hashes remain in private runtime evidence. Retain ranking v14,
continue its distinct final native verification and observe official rank changes.

The complete source ZIP now includes all 22 preserved competition modules with
their original exact bytes, plus their hash manifest, GPL licence and dependencies.
The actual independently extracted archive passes **168 tests**; all forecast
training-source digests also match the public worktree. Public CI passes **168
tests / 43 strict maintained modules**. Runtime data, native weights and credentials
are excluded. Source archive SHA256 is
`e74f267f4cab8cca151b9e7f59db57da9334842cb1547f359b6bb0b64f2e9757`.
