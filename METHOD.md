# PRC 2026: retrospective NM residual model

Team: **zestful-fountain**. Original independent GPL-3.0-only implementation.

## Official result

Submission `zestful-fountain_v2.parquet`: **297.5883 seconds RMSE**, all **344,841** pairs scored, processed **2026-09-07T17:52:51Z**. At 17:53 UTC the team was **13th of 66**. The leading score was 263.4623. Rankings change; this is a dated snapshot, not a winning claim. The previous official submission scored 461.4635. Source: [official leaderboard API](https://datacomp.opensky-network.org/api/competitions/bb3693e1-26bc-4a9e-8619-4fe78b4eab0c/leaderboard), [ranking rules](https://prc-data-challenge-2026.netlify.app/ranking.html).

## Data and permitted context

Only organizer-supplied training/ranking tables are used. No external datasets, product/customer records or hidden ranking labels enter this model. The 2025 files contain arrivals and departures; fitting uses departure labels. Airport `BLOCK_TIME_UTC_mvt`, `TAXITIME_SEC_mvt` and movement IDs are never predictors.

The [official schema](https://prc-data-challenge-2026.netlify.app/data.html) retains Network Manager initial, estimated, actual and last-known timestamps in the ranking data. The separate contest model uses those supplied fields explicitly. They are retrospective observations, unsuitable for claiming a predeparture operational forecast. The original `pipeline.py` predeparture and challenge-context modes remain separate. Competition data and trained models are restricted to the challenge; no commercial-product integration is included or implied. A future open-data release must be checked against its published licence.

## Features and estimator

65 features: airport/aircraft/operator/runway/stand categories and interactions; UTC schedule and movement calendars; movement/schedule deltas; trailing arrival/departure counts at airport and runway level over 15/30/60 minutes; headways; initial/estimated/actual NM time deltas, missingness and airport-match indicators. Exact-time events are excluded from trailing counts. Arrivals contribute traffic context through their destination airport only.

A continuous baseline uses valid matching-airport NM actual off-block time, falling back to last-known and first-filed estimates, then 1,000 seconds. For unmatched LIRF records with missing initial NM time, it uses the nonnegative movement/schedule delta. This exception arose from inspection of the January–June 2025 fitting partition: some ground-source records encode very long delays that tree leaves cannot extrapolate. The CatBoost regressor learns the remaining residual. Labels are not rewritten. Positive outliers remain; negative labels are excluded from fitting but retained in validation scoring. Final predictions are floored at zero.

CatBoost RMSE, depth 8, learning rate 0.055, seed 20260907, maximum 2,200 iterations, patience 120. Seasonal validation selected 2,199 iterations; final fitting uses all 2,084,678 nonnegative departure labels. GPU execution uses an NVIDIA RTX 5060 Laptop (8 GB). GPU floating-point reduction is not bitwise deterministic; record model hashes and numerical differences when reproducing. The locked environment uses Python 3.12.

## Validation and rejected changes

These are 2025 holdouts, not leaderboard estimates. The July–December split trains on the first six months. The January/July seasonal split trains on the other ten months and is explicitly **not** a chronological forecast. It checks the seasonal mixture represented by the ranking months without reading ranking labels.

| Candidate | 2025 holdout | RMSE seconds |
|---|---|---:|
| Original context, 500 trees | July–December | 389.207 |
| Context, 1,000 trees | July–December | 382.284 |
| Traffic features with capped labels (rejected) | July–December | 434.145 |
| Planned-time residual | July–December | 355.680 |
| Supplied NM records, residual | July–December | 342.865 |
| NM residual with unmatched LIRF fallback | July–December | **267.998** |
| NM residual with same fallback | January and July | **331.284** |

The seasonal test is harder; scores across those two splits are not directly comparable. The local July–December result does not imply a 268-second official score. No changes were selected by probing individual ranking targets or exploiting the scoring service. Only one improved submission was made on 7 September before this report. The earlier CPU traffic run was cancelled and has no result.

## Reproduce the submitted method

Obtain authorized dataset access independently. Store all twelve 2025 monthly files, `ranking.parquet` and `submitting.parquet` privately in `runs/data/`. Nothing is downloaded or uploaded implicitly by the training command.

```sh
python -m pip install -r requirements.lock.txt
python -m pytest -q
python -m mypy pipeline.py buckets.py traffic_features.py contest.py
python contest.py train --data runs/data --output runs/reproduction-v2 --device GPU --validation seasonal --permission-ref "PRC2026 registered participant, challenge-only"
python contest.py predict --run runs/reproduction-v2 --ranking runs/data/ranking.parquet --template runs/data/submitting.parquet --output runs/submissions/zestful-fountain_v3.parquet --permission-ref "PRC2026 registered participant, challenge-only"
```

The example uses a new version number to avoid overwriting the scored submission. Uploading is a separate command in `buckets.py` and should respect the daily quota. The predictor also accepts the archived original v2 report/model. It verifies the model hash, feature order and exact template IDs/dtypes/order, and requires finite nonnegative values. Input, model, template and output hashes are retained privately in run manifests. `experiments.py` preserves the earlier ablation harness and its options; use fresh caches after any feature-code change. `contest.py` computes features directly and has no cache dependency.

## Remaining competitive work

Investigate LIRF heavy-tail errors using training-only folds, validate airport-specific residual models and blending on identical holdouts, and test robustness to missing NM matches. Preserve untouched folds for final model selection. Additional live data access is a separate OpenSky licensing process and is not required for this submission. Winning remains an objective, not an achieved result.

## Carrier model and unmatched LIRF specialist (v3 candidate)

`carrier_contest.py` adds five predictors to the same 65-feature matrix: movement airline prefix, airport/prefix pair, scheduled day of month and movement/schedule seconds. Full flight numbers and identifiers are excluded. The global residual model uses depth 9, learning rate 0.045, L2 regularization 5 and 2,980 trees. A separate CPU CatBoost model replaces its residual only for LIRF rows with missing `IOBT_flt`: depth 5, learning rate 0.04, L2 regularization 10, 1,014 trees. Both use seed 20260907 and the same continuous baseline. The specialist is fitted on 1,488 nonnegative labels in its full fit.

On the same January/July 2025 holdout, the global carrier model scored 327.340 seconds, and the combined model scored **324.065**, compared with 331.284 for v2. These parameters were selected using this holdout; it is **not an untouched test**, and these values are not official scores. Positive outliers remain unchanged. A larger unmatched-LIRF residual specialist was not retained. A matched-LIRF LightGBM experiment was rejected: residual RMSE 395.047 versus 393.384 for the existing global model on that scope; direct-target fitting was worse at 629.654. A capped-residual global experiment was stopped early because its validation result deteriorated. No leaderboard quota was used for these rejected runs.

Reproduce the selected carrier/specialist method with authorized data:

```sh
python carrier_contest.py train --data runs/data --output runs/reproduction-carrier --device GPU --permission-ref "PRC2026 registered participant, challenge-only"
python carrier_contest.py predict --global-run runs/reproduction-carrier/global --specialist-run runs/reproduction-carrier/specialist --ranking runs/data/ranking.parquet --template runs/data/submitting.parquet --output runs/submissions/zestful-fountain_v3.parquet --permission-ref "PRC2026 registered participant, challenge-only"
```

Choose a fresh version number if v3 already exists. The command fits validation models at the selected fixed tree counts and then refits each model on all its authorized nonnegative training rows. Hashes, feature schemas and counts are checked before prediction. The final full global fit was restarted after an interrupted local process; it completed 2,980 iterations, with recovery snapshots enabled. Snapshot files, logs and fitted models remain private. GPU training is not bitwise deterministic; the saved model digest identifies the exact submitted fit.
