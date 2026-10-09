# Generalization-first forecast, 9 October 2026

This is a separate predeparture research pipeline. Existing retrospective competition
modules, models, submissions and rejected experiments are preserved. The dedicated
`worker/prc-final` worktree changes no production application or customer code.

## Historical evidence

The documented official incumbent v12 scored 273.5557 seconds RMSE on 344,841
January/July 2026 departures. Its 183-feature retrospective ensemble uses supplied
operational observations and fixed component replacement weights. This is a dated
official result, not the new forecast's validation score.

The selected but unsubmitted v13 ensemble scored 312.470399515 seconds on 344,419
reused January/July 2025 labels, including 80 negative labels. Its longer LightGBM
component narrowly passed a predeclared 0.5-second gain threshold. Neither January
nor July was a strictly future-month holdout in that experiment.

An earlier NM residual model scored 267.997965419 on July–December 2025 after
training January–June. That result uses actual operational clocks and an unmatched
LIRF correction. It cannot be compared directly with the seasonal score or called
a predeparture forecast. The historical inventory retains 38 report objects; other
run artifacts, source modules, tests and documentation remain available unchanged.

## Inputs and audit

Supply all twelve official 2025 movement files and authorized evaluation templates.
The source has six movements 2–4 seconds before their filename's month boundary.
They remain in the dataset. Timestamp-based validation avoids file-month assumptions.
The audit checks schemas, UTC year, IDs, phases, finite labels, missing fields,
airport/aircraft distributions and monthly drift. Runway and stand drift is measured
with airport-scoped unseen rates, even though these fields are excluded as predictors.

The actual final files downloaded on 9 October contain **670,790 departures in all
four requested months**: January 152,719; February 144,158; June 181,791; July 192,122.
This differs from the public data page's description of two additional months.
January/July ranking retains 344,841 departures. Hidden departure targets must all
be blank. Predictions preserve each template's exact ID dtype, order and row count.

All ten training airports are present in the final data. There are 29 departure
rows with previously unseen aircraft types, four with unseen airport/runway pairs
and 3,034 with unseen airport/stand pairs. Aircraft distribution total variation
is 0.05954. The supplementary all-column audit confirms shared January/July
schedule and categorical values are unchanged, while actual movement clocks vary
by up to 12 seconds and many arrival labels change. Those observations do not enter
forecast features. Training retains 369 negative labels for evaluation; the maximum
positive label is 131,167 seconds. Extreme positive labels are never removed.

## Feature availability and leakage

The explicit predictor allowlist contains origin, destination, aircraft type,
flight rule and operator prefix from the published flight number. Time features are
UTC hour, weekday, month, weekend and cyclical daily/yearly calendar coordinates.
No private holiday database or external weather data is used.

All Network Manager `_flt` fields, last-known off-block time, actual movement and
block timestamps, final observed runway/stand assignments, movement/flight IDs and
taxi labels are excluded. In particular, supplied LOBT is not established to have
been known before departure. Recorded actual runway and stand are not assumed to be
planned assignments. This availability decision is stricter than the historical
retrospective challenge models.

Schedule congestion counts departures and arrivals within ±5, ±10, ±15, ±30 and
±60 minutes, plus strictly preceding 30-minute windows. Arrival context uses its
destination airport. A departure's own scheduled record is removed. Same-timestamp
events are excluded from strictly preceding counts. No observed queue is claimed.
Count calculation has an independent brute-force test and is invariant to inference
chunking when the same timetable is supplied.

**As-of limitation:** PRC publishes schedule fields without historical schedule
snapshots. The pipeline assumes those timetable values were known at prediction
time. The files cannot establish that assumption; leakage status explicitly retains
it. Operational use requires the caller to provide only its known timetable snapshot.
The no-demand model measures the effect of removing schedule context.

## Frozen validation and model selection

Models train on all earlier scheduled departure timestamps and validate complete
September, October and November 2025 separately. Every repeated non-null flight ID
in the held-out month is purged from fitting. Negative labels are excluded from fit
only; all validation labels and positive outliers remain. No random CV is used.

The fixed model list is global mean, median, airport/hour mean with airport/global
fallback, LightGBM (600 trees), the same model without demand, CatBoost (400 trees),
HistGradientBoosting (120 trees) and global plus airport LightGBM (400 trees).
Airport specialists require 10,000 fitting rows and use fixed equal shrinkage with
the global model. HGB uses deterministic category hash bins to respect its category
limit. Category vocabularies are built only from fitting rows; unseen values map to
UNKNOWN. Unknown airports use the fitting global mean.

The selection pool consists of each single candidate and fixed 50/50 pairs of the
four learned model families. Lowest pooled September–November RMSE wins, breaking
ties by fewer components and then name. No leaderboard information, adjustable
blend optimization or scored-month early stopping enters selection.

An independent fixed depth search runs in parallel: LightGBM depth five with and
without demand, and depth nine without demand, each with 1,000 trees and learning
rate 0.035. It reuses hash-verified features and identical purged forward folds,
with three CPU threads. Its plan is frozen before December evaluation.
`forecast_finalize.py` combines the eleven single candidates and fixed equal
learned-model pairs. A pair is eligible only if pooled RMSE improves by at least one
second over the best single and **all three** selection months improve. Final
selection uses no December or seasonal diagnostic feedback.

December is scored only after selection is frozen. February, June and July are
additional forward diagnostics trained on earlier 2025 timestamps; they cannot
change selection. December is a **winter proxy** for January, not a January calendar
score. True January forward validation is unavailable with only one training year.
All 2025 data has been used in previous research: the protocol is new, but December
must not be described as a historically untouched dataset.

Complete airport/time stress tests hold out November at the busiest and least
represented fitting airports. The selected model trains without that entire airport
and uses its documented global fallback. Metrics report counts, RMSE, mean signed
bias, MAE, p90 absolute error, worst-decile RMSE and target/prediction quantiles,
including per-airport and per-month evidence. No rows are dropped for scoring.

## Reproduction

The continuous forward leaderboard is maintained in `LEADERBOARD.md` and
`leaderboard.json` by `forward_leaderboard.py`. Only source-bound completed
September/October/November runs receive a rank; lower equal-weight mean fold MAE
wins. Completed individual folds remain recorded and incomplete full runs stay
unranked. Official challenge feedback and December diagnostics never enter this
board. Its MAE champion is distinct from the frozen RMSE submission selection.
Worst-decile MAE uses the largest ceil(10% of pooled rows) absolute errors;
catastrophic errors exceed 3,600 seconds. Novelty cohorts are derived separately
from each purged fitting set. Observed stand is a retrospective subgroup only.
Existing timer values include fit/predict/metrics together, so the board records
their combined elapsed time and marks separate training/inference times unavailable.

```sh
python forward_leaderboard.py --data /private/prc2026/data --runs runs/generalization-new runs/search-new --watch --interval 5
```

When using guarded candidates, include `runs/final-new` in `--runs`, and add
`--selections runs/generalization-new runs/final-new` to include every recorded
fixed ensemble. The final plan identifies guarded-only leaderboard entries to
avoid duplicating the native single-model experiments.

Use `leaderboard_monitor.py` for the full continuous lifecycle, including locked
diagnostics and completed refits. These are visible but have no rank and cannot
affect selection. It updates new diagnostics without recomputing unchanged CV.
Missing stress tail/cohort metrics are explicitly null with the reason that their
row-level predictions were not retained; no values are inferred from RMSE.
By default it discovers new `runs/*` experiments with the same three forward folds
and approved feature columns. Incompatible months and unreviewed feature scopes
are excluded. `--no-discover` limits it to explicitly supplied run directories.

```sh
python leaderboard_monitor.py --data /private/prc2026/data --runs runs/generalization-new runs/search-new runs/final-new --selections runs/generalization-new runs/final-new --final runs/final-new --interval 5
```

Use Python 3.12 and the pinned `requirements.lock.txt`. Store restricted data and
credentials outside public source. Commands run inside this independent directory:

```sh
python -m pip install -r requirements.lock.txt
python -m pytest -q
python -m mypy forecast_features.py forecast_model.py forecast_run.py
python forecast_run.py run --data /private/prc2026/data --final-data /private/prc2026/final --output runs/generalization-new --permission-ref "YOUR_DATED_PARTICIPATION_PERMISSION" --threads 6
```

Once the base audit/feature cache are complete, run the independent search. The
finalizer freezes its plan immediately, waits for both complete forward prediction
sets, then evaluates the selected model on December and proxies:

```sh
python forecast_search.py --data /private/prc2026/data --feature-run runs/generalization-new --output runs/search-new
python forecast_novel_finalize.py --base runs/generalization-new --search runs/search-new --data /private/prc2026/data --final-data /private/prc2026/final --output runs/final-new
```

`runs/final-new` is the final delivery. The base independently completes its smaller
pool; preserve both forward selections. In this session, stop its duplicate
post-selection refit after completed CV so final-v3 owns the selected ensemble's
diagnostics and full fit. Its December result is not input to final selection.
The final plan records whether that other pipeline had already scored December.

For this session, `forecast_finish.py` completes the locked selection after seasonal
diagnostics. All selected learners explicitly overwrite an entirely unseen airport
with the fitting global mean; the novel guard returns the same mean there. Compute
the two airport holdouts exactly from that rule, with native CatBoost/LightGBM
regression checks, rather than perform two irrelevant native fits. Save their
row-level predictions, tail/cohort MAEs and catastrophic rates. No parameter,
selection weight or scored-month rule changes after December. Stop the original
finalizer only after all four fixed prediction files and seasonal reports complete.

```sh
python forecast_finish.py --source runs/final-new --base runs/generalization-new --search runs/search-new --data /private/prc2026/data --final-data /private/prc2026/final --output runs/delivery-new
```

`runs/delivery-new` is the completed model/submission folder. Its phase plan binds
the original selection, component plans, audited cache and source hashes. The
full-fit ledger records training and inference timers separately. Use that folder
with the delivery verifier and portable interface. To monitor both phases, use
`--final runs/delivery-new --diagnostic-sources runs/final-new`; prediction hashes
deduplicate copied diagnostics.

Before December results, the live MAE board exposed severe unseen-aircraft errors
on 22 forward rows. Register two fixed guard policies for the eight learned models:
the first routes unknown/missing aircraft to the same fitting set's airport/hour
mean (at least 50 rows), then airport mean, then global mean. Forward testing rejects
this blanket missing-category fallback where RMSE worsens. The narrower policy
preserves missing categories observed during fitting and only falls back for truly
novel categories. Its observed-missing flag is stored and verified on native reload.
Known categories retain native predictions. A guard is selection-eligible only if pooled RMSE improves
by at least one second and none of the three fold RMSEs increases. Equal pairs
retain the existing one-second/all-fold-improvement rule. All 27 single candidates
and recorded fixed pairs remain visible in the MAE leaderboard even when the
separate RMSE stability rule rejects them. No official feedback or December labels
inform the policy. Guarded native inference uses `forecast_novel_guard.load_model` and
the generic `taxiout.TaxiOutPredictor`, which reads the saved policy metadata.
The original `forecast_finalize.py` remains available for the initial unguarded
11-candidate protocol. Its waiting run and the blanket-guard-only waiting run were
superseded before model selection. Their files and completed guard CV are retained.

The run directory must not exist. The plan is written before data inspection and
training. Receipts retain input, source, environment and saved model hashes, all
candidate results and timings. CPU fixed seeds and deterministic LightGBM settings
are used. Saved/native inference must agree exactly on every December feature row.
Different CPU/library builds may have numerical differences; compare evidence and
record them rather than promise cross-platform binary-identical training.

The run produces `audit.json`, `feature-schema.json`, `experiment-ledger.json`,
`selection.json`, forward predictions, `diagnostics.json`, `airport-stress.json`,
saved models, both submission parquet files and `final-report.json`. It never uploads
implicitly. Training on all nonnegative 2025 labels happens after model selection.
All candidate settings, intervals and model weights stay fixed during final fitting.

For new authorized evaluation files, use:

```sh
python forecast_run.py predict --run runs/generalization-new/model --ranking /private/final_ranking.parquet --template /private/final_submitting.parquet --output runs/new-final_submission.parquet
```

That legacy prediction command supports unguarded bundles. For a guarded bundle,
use the `make_submission` function in `forecast_novel_finalize.py` or the portable
typed interface; saved-policy loading is required to reproduce the fallback.

Verify fresh saved-model inference across every submitted row and build a public
source archive without restricted data or weights:

```sh
python forecast_delivery.py audit --ranking /private/ranking.parquet --final /private/final_ranking.parquet --output runs/evaluation-audit.json
python forecast_delivery.py verify --run runs/generalization-new --ranking /private/final_ranking.parquet --template /private/final_submitting.parquet --submission runs/generalization-new/submissions/final_submission.parquet --output runs/generalization-new/final-verification.json
python forecast_delivery.py export --output runs/public-source.zip
python forecast_delivery.py export --output runs/taxiout_inference_source.zip --inference-only
```

The export scanner runs before writing, validates archive integrity and records
every included source hash. Test the extracted archive in a directory with no parent
repository imports. The expanded preflight archive passes all 135 independent
tests. Completion coverage brings local validation to 137 tests and 35 strict
modules; regenerate and independently check the final archive after final reporting.

## Portable inference and product boundary

`forecast_model.TaxiOut(run).predict(PredictInput(...), known_schedule)` returns
`predictionSeconds`, `confidence`, `range`, `modelVersion` and `reasonCodes`.
Confidence is null: the December p90 absolute residual gives an empirical symmetric
range, not a calibrated probability or promised future coverage. Reason codes expose
unseen inputs, ignored runway/stand values and restricted competition weights.

The request's schedule snapshot timestamp must be UTC and no later than departure.
The caller is responsible for actual snapshot provenance. When predicting one
departure, remove that departure from the supplied timetable; batch submission
inference has actual movement identity and removes self automatically.

The adapter and feature implementation are reusable GPLv3 source. The existing
organizer clarification limits these competition data and weights to participation.
A commercial model needs separately licensed training data or explicit model-use
rights. No trained weights or dataset are installed in PhoenixAI Aviation, and no
production coupling is introduced. Public export includes original source, tests,
licence, pinned requirements and methodology; private data, model weights and run
artifacts are excluded.

Sources: [challenge deadline](https://prc-data-challenge-2026.netlify.app/),
[published schema](https://prc-data-challenge-2026.netlify.app/data.html),
[conditions](https://prc-data-challenge-2026.netlify.app/eligibility.html),
and the privately retained 7 September organizer clarification summarized in DATA_USE.md.
