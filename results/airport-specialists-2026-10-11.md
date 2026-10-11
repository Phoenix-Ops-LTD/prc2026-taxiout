# Airport specialist continuation ? 11 October 2026 UTC

Continue the owner-authorized first-place objective in isolated PRC research.
The existing RTX 4090 has 3,824 MiB free before training; memory uses the prior
35% free-memory cap, with at least 2,600 MiB required before each sequential fit.
The unchanged prepared inputs contain only departures before December 2025.
No official feedback or later diagnostic labels enter this experiment.

Freeze three candidates before fitting: 1,200-tree/depth-6 heads without and with
demand, and 800-tree/depth-8 heads with demand. Learning rate is 0.04, leaf
regularization 15 and seed 20261011. Every airport with at least 10,000 fitting
rows gets a specialist. Predictions mix the matching frozen global checkpoint
and local specialist equally. Novel-aircraft and unseen-airport fallback stays
unchanged. Original input, source, fold checkpoint and checkpoint-ledger hashes
are bound in the frozen plan.

All 90 airport fits and nine forward model evaluations complete. Each native
model exactly reloads on every validation row. Native artifact hashes and separate
fit/inference clocks are retained privately. Reported total training time includes
measured global fitting and incremental airport fitting; the reused contribution
is explicit. The deployment loader rejects the new policy until an explicitly
compatible loader is used, preventing silent loss of airport models.

| Model | Pooled forward RMSE (s) | Mean fold MAE (s) |
| --- | ---: | ---: |
| gpu_airport_no_demand_novel_guarded | 415.781698875 | 222.538340271 |
| gpu_airport_demand_novel_guarded | 414.350578257 | 221.000829665 |
| gpu_airport_deep_demand_novel_guarded | 414.136862159 | 220.664732174 |

All nine fixed equal pairs with the incumbent components are also evaluated.
The best new pair is deep LightGBM with novel fallback + deep GPU airport CatBoost,
with pooled RMSE 405.156613311 s, compared with 403.769789931 s for the incumbent.
Its September/October/November scores are 427.065576448 / 342.234246939 /
444.163712479 s; the incumbent is better on every fold. No promotion occurs.
The previous selected model and fully verified submissions remain selected.

The live board contains 338 ranked validated candidates, three older unranked
partials, six diagnostics and two full refit records. Its MAE CHAMPION remains
218.202389046 s mean fold MAE. The dashboard now automatically displays the newest
frozen GPU output batch while excluding preparation directories. It shows the
completed specialist results and comparison against the incumbent. Official
ranking history remains observation-only in ignored runtime files.

Validation: 153 research tests and 41 maintained strict Python modules pass;
repository lint/types and all 986 tests pass. Native roundtrip, unseen-input
fallback, corrupted specialist artifacts and discovery of new GPU batches are
covered. Desktop and 390-pixel mobile checks show no script errors or overflow.
The enlarged leaderboard also passes a scoped formatter check with a 4 MiB limit.
See [LIVE_PROGRESS.md](../LIVE_PROGRESS.md) for reproduction commands.
