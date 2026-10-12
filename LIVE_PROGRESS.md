# PRC live progress

Start the local dashboard from this research directory:

```sh
python prc_dashboard.py --port 8786
```

Open <http://127.0.0.1:8786/> on the same computer. The server binds only to
loopback, serves the dashboard and its status API, and has no upload or mutation
actions. Use `--no-official` for local-only observation.

The dashboard also reads the private `runs/latest-ranking-submission.json` receipt
to show the latest upload and its scored RMSE separately from the team's best
official standing. Only safe aggregate fields are exposed; private paths and
bucket contents are not served. It remains a read-only observer.

Local data refreshes every five seconds. The independent leaderboard monitor
validates source-bound prediction/ledger pairs before updating `LEADERBOARD.md`
and `leaderboard.json`. Completed September, October and November folds are
required for a candidate to receive a rank. Mean fold MAE determines the CHAMPION.
Worst-fold, worst-decile and catastrophic errors remain visible. GPU candidates
show each completed fold, measured fit time and their full-CV RMSE gain against
the selected incumbent. Partial CV gains are withheld.

The official observer reads every page of the public ranking API, keeps each
team's best scored submission and handles ties as competition ranks. It records
the observation time, our rank, score, gap to first and changed-snapshot history.
Complete pagination can take several minutes; the next check starts three minutes
after the previous check finishes. The last complete snapshot remains visible
while fetching. API failures mark it stale, retaining its original observation
time. Duplicate submission IDs or repeated cursors fail closed. A changing
upstream ranking is an observation, not an atomic contest-side database snapshot.

Official observations are stored only in ignored `runs/live-official-*.json`
files. Neither the local leaderboard nor model-selection code reads them.
Local validation and official scoring cover different datasets. Official rank
moves only when public scoring or competitors' standings change; the dashboard
does not submit files or predict an official rank from local error.

## GPU continuation

`forecast_gpu_search.py` uses fixed 3,000/3,000/5,000-tree CatBoost candidates,
unchanged forecast features, fit-only novel-input fallback and identical purged
forward folds. Prepared inputs exclude December and 2026 rows. It records source
and input digests, exact native-model CPU reload parity and separate fit/inference
clocks. Training is nondeterministic on GPU; saved-model inference is checked
exactly. GPU memory uses 35% of available free memory and each fit requires at
least 2,600 MiB free. Existing services and jobs must remain available.

An interrupted batch can resume with the same frozen inputs:

```sh
python forecast_gpu_resume.py --inputs /private/prepared-inputs --output /private/existing-gpu-output
```

Resume verifies the original source/input/parameter bindings, preserves completed
experiments and checks their full-fold saved-model predictions before fitting
remaining candidates. Each new completed experiment gets an immutable snapshot
for transfer. Transfer clients verify the snapshot digests and publish prediction
files before the ledger. This avoids comparing a completed event with a later
mutable ledger revision. An unrecorded native model stops continuation for audit.

GPU discoveries use only complete forward-validation results. The registered
promotion rule requires at least one second of pooled RMSE improvement and an
improvement on every fold over the incumbent. Official feedback and later
diagnostics never select or retune a model.

Prepare inputs from the audited base feature cache and its frozen CPU search plan:

```sh
python forecast_gpu_prepare.py --data /private/data --feature-run runs/generalization-v3 --reference-search runs/generalization-search-v1 --output runs/new-gpu-inputs
python forecast_gpu_search.py --inputs runs/new-gpu-inputs --output runs/new-gpu-output
python forecast_gpu_compare.py --gpu runs/new-gpu-output --incumbent runs/generalization-final-v3 --output runs/new-gpu-comparison
```

Transfer only the prepared directory to an authorized isolated GPU worker; keep
native models and competition rows private. Comparison evaluates each GPU single
and all nine fixed equal pairs among the three GPU candidates and the two frozen
incumbent components. It never changes the prior selected artifact. Include the
comparison directory in the leaderboard monitor's `--selections` argument to
record every completed fixed blend with its validation evidence.

## Airport specialist continuation

The live dashboard discovers the latest frozen GPU output batch automatically,
excluding prepared-input directories. The leaderboard discovers an experiment
only when forward prediction files exist; native archive receipts are excluded. The airport batch reuses the preserved
matching forward-fold global GPU checkpoints, then fits specialists for every
airport with at least 10,000 fitting rows. Global and local predictions have a
fixed 50/50 mix. No airport-specific mixture weights are selected. Novel-input
fallbacks remain unchanged. Native specialist artifacts use an explicit policy;
older loaders reject them instead of silently dropping the specialists.

On an isolated authorized worker containing the original prepared inputs and
completed global GPU batch, copy `forecast_gpu_airport.py` and
`forecast_gpu_compare.py` into the prepared input directory, then run:

```sh
python forecast_gpu_airport.py freeze --inputs /private/prepared-inputs --checkpoints /private/global-gpu-output --output /private/new-airport-plan.json --git-commit COMMIT
python forecast_gpu_airport.py train --inputs /private/prepared-inputs --checkpoints /private/global-gpu-output --plan /private/new-airport-plan.json --output /private/new-airport-output
python forecast_gpu_airport.py compare --gpu /private/new-airport-output --incumbent /private/incumbent-forward-run --output /private/new-airport-comparison
```

The plan binds original input, source and global checkpoint hashes before fits.
The three candidates use 1,200 trees/depth 6 without and with demand, and 800
trees/depth 8 with demand; learning rate 0.04, leaf regularization 15, seed 20261011,
the original bounded GPU memory settings and fixed 50% local prediction weight.
All head fits are sequential. Each completed model passes exact saved-model
inference on every fold row and publishes an immutable transfer snapshot.
Recorded training time includes measured original global fitting plus incremental
specialist fitting; both contributions remain explicit. Add the completed
comparison directory to the leaderboard monitor's `--selections` argument.

## Unique final submission

The current retained best is competition v14 / 272.8483 RMSE. Exact reconstruction
of every original ranking stage passes. The distinct final file requires two
complete native runs on all 670,790 rows. A local deadline guard reserves the
unique conditional final upload for 14 October 08:00 UTC, two hours before the
deadline; it rechecks the authoritative best score and every artifact binding.
See [FINAL_DELIVERY.md](FINAL_DELIVERY.md). Its private status is
`runs/final-deadline-guard.json`. The computer and guard process must remain running.
The dashboard observes ranking changes and has no upload action.
