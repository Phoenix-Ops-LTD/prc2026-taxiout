# PRC live progress

Start the local dashboard from this research directory:

```sh
python prc_dashboard.py --port 8786
```

Open <http://127.0.0.1:8786/> on the same computer. The server binds only to
loopback, serves the dashboard and its status API, and has no upload or mutation
actions. Use `--no-official` for local-only observation.

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
