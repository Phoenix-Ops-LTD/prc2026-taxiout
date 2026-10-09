# GPU continuation and live progress — 9 October 2026 UTC

The owner explicitly authorizes existing GPU capacity. Read-only discovery finds
an existing RTX 4090 with 3,824 MiB free. An isolated PRC virtual environment uses
the fixed forecast allowlist and purged September/October/November folds. It does
not mutate existing model services, infrastructure, instance rental or billing.
Prepared inputs contain 1,919,383 departures before December; no 2026 or December
rows are sent to the worker. The reusable preparer reproduces both prepared input
SHA-256 digests exactly. Historical later diagnostics have already been observed;
no claim of a historically untouched validation set is made.

| Fixed GPU candidate | Trees | Depth | Pooled forward RMSE (s) | Mean fold MAE (s) |
| --- | ---: | ---: | ---: | ---: |
| Deep, no demand, novel guard | 3,000 | 8 | 429.404316157 | 226.402205279 |
| Deep, demand, novel guard | 3,000 | 8 | 423.580446423 | 224.833852124 |
| Smooth, no demand, novel guard | 5,000 | 6 | 427.420856451 | 227.885684378 |

All nine fits finish and the native saved model reproduces every validation row
exactly on CPU. GPU training itself is nondeterministic. Separate fit/inference
timings, actual device, source bindings and commit provenance are retained in the
live leaderboard. Native artifact digests and environment receipts remain private.

An initial transfer compared an event digest with a later mutable ledger revision
and failed closed. Five completed fits are preserved; continuation verifies their
saved predictions and completes the remaining four with unchanged fit code and
parameters. Each continued experiment has an immutable prediction/ledger snapshot
whose hashes are checked before local publication.

All nine fixed equal pairs among the three GPU candidates and the two incumbent
components are evaluated. The best new pair, deep LightGBM with novel fallback +
GPU deep CatBoost with demand, scores 409.109835880 seconds pooled forward RMSE.
It fails the registered promotion rule against 403.769789931 for the incumbent;
the existing selected model and all-row-verified submission stay selected.
Official feedback and later diagnostics never enter these comparisons.

The continuously maintained board now ranks 326 completed candidates, with three
older partial runs unranked. Its MAE CHAMPION remains the guarded LightGBM +
airport LightGBM with novel fallback, at 218.202389046 seconds mean fold MAE.
Nine new completed folds and all nine new fixed blends are retained.

`prc_dashboard.py` serves <http://127.0.0.1:8786/> only on loopback. It displays
the local CHAMPION, selected forward RMSE, verified GPU fold results and official
rank movement with actual observation times. It reads all public API pages,
rejects repeated cursors or duplicate submission IDs, retains the last good
snapshot on errors and marks it stale. Pagination takes minutes; the UI refreshes
every five seconds and the observer waits three minutes between complete checks.
Official snapshots are ignored runtime files; selection never reads them. Official
and forward-validation RMSE cover different datasets. No upload is performed.

Desktop and 390-pixel mobile browser verification finds no script errors or page
overflow. API tests cover tied ranks, unscored teams, missing teams, incomplete
pagination, stale observation retention, private-path denial and partial GPU CV.
See [LIVE_PROGRESS.md](../LIVE_PROGRESS.md) for reproducible commands.
