# PRC 2026: retrospective NM residual model

Team: **zestful-fountain**. Original independent GPL-3.0-only implementation.

## Official result

Latest v5: **290.0659 seconds RMSE**, **19 of 89** at **2026-09-09T18:40:47Z**. All 344,841 pairs scored; leader 246.3605. [Dated snapshot](results/leaderboard-2026-09-09.md). The first-place-through-deadline goal remains incomplete.

Earlier official scores: v4 291.4829, v3 292.4043, v2 297.5883, v1 461.4635 seconds RMSE. Historical ranks and processing times remain in the [8 September](results/leaderboard-2026-09-08.md) and [7 September](results/leaderboard-2026-09-07.md) snapshots. Rankings change; these are dated measurements.

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

## Fixed LightGBM blend candidate

The LightGBM residual model uses the same 70 features, 63 leaves, learning rate 0.035, minimum child size 80, L2 regularization 10, column fraction 0.9 and 679 trees (seed 20260907). A fixed 25% LightGBM / 75% global CatBoost blend retains the unchanged unmatched-LIRF specialist. Predictions from each constituent are floored at zero before blending.

January/July holdout RMSE is **323.377**, versus **324.065** for v3, with improvements in both months and 54 of 62 UTC days. A paired day-cluster bootstrap (2,000 resamples, seed 20260908) gives a 95% interval of [-580.97, -301.10] seconds squared for the change in mean squared error. This is a conditional diagnostic on the reused tuning holdout, not independent evidence of generalization. The 50/50 blend was weaker and worsened July. A non-LIRF unmatched specialist gave only 0.012 seconds overall improvement while worsening most airports; it was rejected.

```sh
python ensemble_contest.py train --data runs/data --output runs/reproduction-light --permission-ref "PRC2026 registered participant, challenge-only"
python ensemble_contest.py predict --global-run runs/reproduction-carrier/global --specialist-run runs/reproduction-carrier/specialist --light-run runs/reproduction-light --ranking runs/data/ranking.parquet --template runs/data/submitting.parquet --output runs/submissions/zestful-fountain_v4.parquet --permission-ref "PRC2026 registered participant, challenge-only"
```

Use fresh run directories and an unused increasing submission version. Local validation does not establish an official score; only an organizer score receipt does.

The fixed blend was submitted as v4 and scored **291.4829 seconds RMSE** officially (2026-09-08T04:25:34Z), an improvement over v3. The larger 83-feature match-context model scored 324.998 with the LIRF specialist locally, so it was rejected.

## Duration-context submission v5 (9 September 2026)

`duration_contest.py` extends the carrier matrix to 93 features with normalized record-agreement indicators, route/aircraft/runway interactions, planned and actual NM flight-duration differences, duration ratios/plausibility and off-block/schedule deltas. These derive exclusively from organizer-supplied retrospective predictors; departure targets, airport block timestamps and identifiers remain excluded. Duration differences use the existing capped NM deltas. Missing NM observations remain missing.

The candidate uses CatBoost depth 9, learning rate 0.05, L2 regularization 7, 254 borders and seed 20260907. A 5,000-tree January/July validation run selected 4,998 trees. The raw model scored 324.864 seconds RMSE; retaining the existing unmatched-LIRF specialist gives 321.832. A coarse fixed blend comparison evaluated duration weights 0, 0.25, 0.5, 0.75 and 1 on identical IDs and labels. The selected 75% duration / 25% v4 blend scored **321.589**, versus **323.377** for v4. January improves from 346.045 to 344.710; July from 303.881 to 301.668; 55 of 62 UTC days improve. The specialist predictions remain exactly unchanged. This is a reused tuning holdout, not an untouched test or an official score.

Full fitting uses all 2,084,678 nonnegative 2025 departure labels. Recovery validates source hashes, configuration and feature schema before resuming snapshots. GPU fitting is not bitwise deterministic; save the exact model hash. Reproduction commands, using fresh output paths:

```sh
python duration_contest.py train --data runs/data --output runs/duration-validation --validation-only --permission-ref "PRC2026 registered participant, challenge-only"
python duration_contest.py train --data runs/data --output runs/duration-full --permission-ref "PRC2026 registered participant, challenge-only"
python duration_contest.py predict --run runs/duration-full --global-run runs/reproduction-carrier/global --specialist-run runs/reproduction-carrier/specialist --light-run runs/reproduction-light --ranking runs/data/ranking.parquet --template runs/data/submitting.parquet --output runs/submissions/zestful-fountain_v5.parquet --permission-ref "PRC2026 registered participant, challenge-only"
```

The validation command uses the selected fixed tree count, rather than repeating early stopping. Produce the carrier/specialist and LightGBM runs with the earlier commands. The predictor recomputes v4 from its constituent models, checks model hashes and feature schemas, blends aligned predictions, preserves the specialist and validates the exact official template. Training/prediction never uploads automatically. `--resume` resumes an interrupted training command with identical arguments and inputs; completed runs cannot be overwritten.

Further recovered experiments were rejected: airport-specific weights selected across the two months scored 323.712, worse than the fixed v4 blend; the normalized-carrier specialist scored 323.623 and worsened January. Arrival-to-departure NM linkage remains an unvalidated research possibility, with no submission based on it.

The duration blend was submitted as v5 and officially scored **290.0659 seconds RMSE**, rank **19 of 89** at 18:40 UTC. All 344,841 values exactly match independent recomputation against the saved duration model and the scored v4 predictions, including exact preservation of all 383 specialist values. [Dated official result](results/leaderboard-2026-09-09.md). The full model has 4,998 trees and uses all 2,084,678 nonnegative labels. This is the first submission on 9 September.

Post-v5 local checks: the duration LightGBM candidate at 523 trees scored 327.526 with the specialist; adding it at 10% to v5 gives 321.371 on the reused holdout. A separate 119-feature NM interval/forward-traffic LightGBM candidate scored 328.146 with the specialist and 321.376 at a 10% addition. Neither was submitted. Both gains are small and need further validation. Three mixture-of-baselines classifiers for unmatched LIRF worsened both months; rejected. Arrival NM record recovery also worsened both months and was rejected. A new recurring-flight-service category experiment is local only and unscored; it uses recurring service strings as categories, not movement/NM record IDs.

## Specialist seed ensemble v6 (10 September 2026)

`specialist_ensemble.py` averages the existing specialist with four additional CPU CatBoost fits. Every fit uses the existing 70-feature matrix, depth 5, 1,014 trees, learning rate 0.04 and L2 regularization 10. Seeds 20260907 through 20260911 were declared together and receive equal 20% weights; no seed was selected or discarded according to its score. Each constituent is floored at zero before averaging. All predictions outside LIRF with missing `IOBT_flt` remain exactly equal to v5.

The fixed comparison improved the specialist on all four validation months:

| Held-out month | Existing specialist RMSE | Five-seed RMSE |
| --- | ---: | ---: |
| January 2025 | 3618.550 | 3488.457 |
| July 2025 | 3655.346 | 3625.347 |
| February 2025 | 3048.315 | 2905.968 |
| August 2025 | 3508.444 | 3456.604 |

January/July fitting uses the other ten months (1,090 specialist fitting rows; 398 validation rows); February/August likewise uses the other ten months (1,234 fitting rows; 254 validation rows). All original evaluation labels, including extreme values, remain unchanged. January/July overall RMSE with the rest of v5 unchanged improves from **321.589 to 321.005**. February/August is an additional specialist comparison, not validation of the full v5 ensemble. These are local research results, not an official score.

The original seed's predictions are retained from the validated v5 baseline file. The command below fits the other four experts on all 1,488 nonnegative specialist labels. A fresh immutable output is required; inference verifies baseline/input/model digests, feature order, exact template alignment, finite values and scope preservation.

```sh
python specialist_ensemble.py train --data runs/data --output runs/specialist-bagging-full --permission-ref "PRC2026 registered participant, challenge-only"
python specialist_ensemble.py predict --run runs/specialist-bagging-full --baseline runs/submissions/zestful-fountain_v5.parquet --ranking runs/data/ranking.parquet --template runs/data/submitting.parquet --output runs/submissions/zestful-fountain_v6.parquet --permission-ref "PRC2026 registered participant, challenge-only"
```

Training and prediction do not submit automatically. Use an unused increasing version and obtain the organizer's score receipt separately. Source checks cover exact scope/order preservation, invalid seeds, nonfinite/negative expert values, model tampering and a synthetic inference round trip.

The ensemble was submitted as v6 and officially scored **288.3714 seconds RMSE** across all **344,841 pairs**, improving v5 by 1.6945 seconds. At 05:09 UTC on 10 September the complete 668-submission snapshot ranks the team 22nd of 95, with the leader at 245.2901. Independent recomputation matches every prediction exactly, including exact preservation of the 344,458 rows outside the specialist. [Dated result](results/leaderboard-2026-09-10.md). First place remains unachieved.

## Completed-arrival candidate v7

The organizers explicitly retain arrival taxi-in and in-block measurements in the ranking dataset; only departure taxi-out and block-time fields are blanked. `arrival_features.py` filters to ARR rows before reading those fields. It derives 26 features from arrivals completed strictly before the queried departure movement: airport/runway counts, taxi-in mean and standard deviation over 15/30/60 minutes, last-completion age/taxi-in, and same-stand age/taxi-in/aircraft/carrier agreement. Simultaneous airport/runway completions are averaged; ambiguous same-stand ties remain missing. This is retrospective challenge context, with no live forecasting claim. [Organizer schema](https://prc-data-challenge-2026.netlify.app/data.html#the-ranking-dataset).

The global LightGBM retains the existing fixed 679-tree configuration and adds these observations to 97 features (duration93 plus normalized recurring movement/NM service names and their airport interactions). The standalone fixed feature comparison improves January/July RMSE from 332.674 to 326.346 and February/August from 272.272 to 267.592, improving all four months. Service names are recurring categories; unique movement/flight identifiers are never model predictors.

The arrival specialist uses the original carrier70 plus the same 26 features, depth 5, 1,014 trees, learning rate 0.04 and L2 10. All five seeds 20260907–20260911 receive equal weight. Its scope RMSE improves from 3605.043 to 3470.237 on January/July and 3338.878 to 3192.807 on February/August. January alone worsens from 3488.457 to 3752.507; the other three months improve. No seed is selected or discarded.

V7 uses 25% global arrival / 75% v6 outside the unmatched-LIRF scope, and 50% arrival specialist / 50% v6 inside that scope. The complete candidate improves reused January/July RMSE from **321.005 to 318.547**, January from 344.186 to 343.676 and July from 301.026 to 296.749; 54/62 days improve. The conservative specialist weight limits the observed January regression. These holdouts were used for research choices and are not untouched tests. February/August validates the component comparisons, not the full historical v6 ensemble. All original validation labels remain unchanged; full fitting excludes negative departure labels under the existing policy.

```sh
python arrival_contest.py train --data runs/data --output runs/arrival-full --permission-ref "PRC2026 registered participant, challenge-only"
python arrival_specialist.py --data runs/data --output runs/arrival-specialist-full --permission-ref "PRC2026 registered participant, challenge-only"
python arrival_contest.py predict --run runs/arrival-full --arrival-specialist runs/arrival-specialist-full --baseline runs/submissions/zestful-fountain_v6.parquet --ranking runs/data/ranking.parquet --template runs/data/submitting.parquet --output runs/submissions/zestful-fountain_v7.parquet --permission-ref "PRC2026 registered participant, challenge-only"
```

Use fresh run/output paths and an unused increasing submission version. Prediction checks baseline, source-data and model digests, feature schemas, expert seed membership and exact template alignment. Source tests cover excluded departure fields, strict completion timing, ties, empty pools, row ordering, finite/nonnegative predictions and the combined inference path. Training and prediction do not upload automatically. An official v7 score is pending.
