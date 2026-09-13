# PRC 2026: retrospective NM residual model

Team: **zestful-fountain**. Original independent GPL-3.0-only implementation.

## Official result

Latest v9: **277.7530 seconds RMSE**, **16 of 129** at **10:23 UTC on 2026-09-13**. All 344,841 pairs scored; leader 245.0207. [Dated snapshot](results/leaderboard-2026-09-13.md). V10 has local selection evidence only; the first-place-through-deadline goal remains incomplete.

Earlier official scores: v4 291.4829, v3 292.4043, v2 297.5883, v1 461.4635 seconds RMSE. Historical ranks and processing times remain in the [8 September](results/leaderboard-2026-09-08.md) and [7 September](results/leaderboard-2026-09-07.md) snapshots. Rankings change; these are dated measurements.

## Data and permitted context

Only organizer-supplied training/ranking tables are used. No external datasets, product/customer records or hidden ranking labels enter this model. The 2025 files contain arrivals and departures; fitting uses departure labels. Airport `BLOCK_TIME_UTC_mvt`, `TAXITIME_SEC_mvt` and movement IDs are never predictors.

The [official schema](https://prc-data-challenge-2026.netlify.app/data.html) retains Network Manager initial, estimated, actual and last-known timestamps in the ranking data. The separate contest model uses those supplied fields explicitly. They are retrospective observations, unsuitable for claiming a predeparture operational forecast. The original `pipeline.py` predeparture and challenge-context modes remain separate. Competition data and trained models are restricted to the challenge; no commercial-product integration is included or implied. A future open-data release must be checked against its published licence.

## Features and estimator

65 features: airport/aircraft/operator/runway/stand categories and interactions; UTC schedule and movement calendars; movement/schedule deltas; trailing arrival/departure counts at airport and runway level over 15/30/60 minutes; headways; initial/estimated/actual NM time deltas, missingness and airport-match indicators. Exact-time events are excluded from trailing counts. Arrivals contribute traffic context through their destination airport only.

A continuous baseline uses valid matching-airport NM actual off-block time, falling back to last-known and first-filed estimates, then 1,000 seconds. For unmatched LIRF records with missing initial NM time, it uses the nonnegative movement/schedule delta. This exception arose from inspection of the Januaryâ€“June 2025 fitting partition: some ground-source records encode very long delays that tree leaves cannot extrapolate. The CatBoost regressor learns the remaining residual. Labels are not rewritten. Positive outliers remain; negative labels are excluded from fitting but retained in validation scoring. Final predictions are floored at zero.

CatBoost RMSE, depth 8, learning rate 0.055, seed 20260907, maximum 2,200 iterations, patience 120. Seasonal validation selected 2,199 iterations; final fitting uses all 2,084,678 nonnegative departure labels. GPU execution uses an NVIDIA RTX 5060 Laptop (8 GB). GPU floating-point reduction is not bitwise deterministic; record model hashes and numerical differences when reproducing. The locked environment uses Python 3.12.

## Validation and rejected changes

These are 2025 holdouts, not leaderboard estimates. The Julyâ€“December split trains on the first six months. The January/July seasonal split trains on the other ten months and is explicitly **not** a chronological forecast. It checks the seasonal mixture represented by the ranking months without reading ranking labels.

| Candidate | 2025 holdout | RMSE seconds |
|---|---|---:|
| Original context, 500 trees | Julyâ€“December | 389.207 |
| Context, 1,000 trees | Julyâ€“December | 382.284 |
| Traffic features with capped labels (rejected) | Julyâ€“December | 434.145 |
| Planned-time residual | Julyâ€“December | 355.680 |
| Supplied NM records, residual | Julyâ€“December | 342.865 |
| NM residual with unmatched LIRF fallback | Julyâ€“December | **267.998** |
| NM residual with same fallback | January and July | **331.284** |

The seasonal test is harder; scores across those two splits are not directly comparable. The local Julyâ€“December result does not imply a 268-second official score. No changes were selected by probing individual ranking targets or exploiting the scoring service. Only one improved submission was made on 7 September before this report. The earlier CPU traffic run was cancelled and has no result.

## Reproduce the submitted method

Obtain authorized dataset access independently. Store all twelve 2025 monthly files, `ranking.parquet` and `submitting.parquet` privately in `runs/data/`. Nothing is downloaded or uploaded implicitly by the training command.

```sh
python -m pip install -r requirements.lock.txt
python -m pytest -q
python -m mypy pipeline.py buckets.py traffic_features.py contest.py carrier_contest.py ensemble_contest.py duration_contest.py specialist_ensemble.py arrival_features.py arrival_specialist.py arrival_contest.py arrival_boost_contest.py nm_neighbor_features.py ordinary_ensemble.py neighbor_boost_contest.py leaderboard.py
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

The arrival specialist uses the original carrier70 plus the same 26 features, depth 5, 1,014 trees, learning rate 0.04 and L2 10. All five seeds 20260907â€“20260911 receive equal weight. Its scope RMSE improves from 3605.043 to 3470.237 on January/July and 3338.878 to 3192.807 on February/August. January alone worsens from 3488.457 to 3752.507; the other three months improve. No seed is selected or discarded.

V7 uses 25% global arrival / 75% v6 outside the unmatched-LIRF scope, and 50% arrival specialist / 50% v6 inside that scope. The complete candidate improves reused January/July RMSE from **321.005 to 318.547**, January from 344.186 to 343.676 and July from 301.026 to 296.749; 54/62 days improve. The conservative specialist weight limits the observed January regression. These holdouts were used for research choices and are not untouched tests. February/August validates the component comparisons, not the full historical v6 ensemble. All original validation labels remain unchanged; full fitting excludes negative departure labels under the existing policy.

```sh
python arrival_contest.py train --data runs/data --output runs/arrival-full --permission-ref "PRC2026 registered participant, challenge-only"
python arrival_specialist.py --data runs/data --output runs/arrival-specialist-full --permission-ref "PRC2026 registered participant, challenge-only"
python arrival_contest.py predict --run runs/arrival-full --arrival-specialist runs/arrival-specialist-full --baseline runs/submissions/zestful-fountain_v6.parquet --ranking runs/data/ranking.parquet --template runs/data/submitting.parquet --output runs/submissions/zestful-fountain_v7.parquet --permission-ref "PRC2026 registered participant, challenge-only"
```

Use fresh run/output paths and an unused increasing submission version. Prediction checks baseline, source-data and model digests, feature schemas, expert seed membership and exact template alignment. Source tests cover excluded departure fields, strict completion timing, ties, empty pools, row ordering, finite/nonnegative predictions and the combined inference path. Training and prediction do not upload automatically. V7 subsequently scored **285.749 seconds RMSE** officially across all 344,841 pairs, improving v6 by 2.6224 seconds. The complete 05:58 UTC snapshot ranks the team 19/96, with first at 245.094. Every value matches independent recomputation exactly. [Official dated result](results/leaderboard-2026-09-10.md).

## Deeper arrival residual candidate v8

`arrival_boost_contest.py` uses the same 123 predictors as the v7 global model with CatBoost depth 9, learning rate 0.05, L2 regularization 7, 254 borders and seed 20260907. Fitting excludes the separately modeled unmatched-LIRF scope and caps fitting residuals at +/-7,200 seconds. Original evaluation labels remain unchanged. The January/July validation run selected 4,999 trees out of a maximum 5,000 with early stopping patience 200. GPU fitting is not bitwise deterministic; saved-model digests identify individual fits.

A fixed 50% blend with v7 was declared before reading the completed validation result. Promotion required at least 0.5 seconds overall improvement, improvement in each month, and at least 40 improved days. The blend scored **315.778** versus v7 **318.547** on the reused January/July holdout: January **341.669** versus 343.676, July **293.251** versus 296.749, and **61 of 62 days** improved. The existing v7 unmatched-LIRF predictions remain exactly unchanged. This is local selection evidence from a reused holdout, not an official score or an untouched generalization test.

Full fitting uses all 2,083,190 eligible ordinary-scope 2025 departures. The standalone feature builder was compared against every cached fitting feature: all **123 columns across 2,085,047 departure rows matched exactly**. Training verifies twelve monthly input files, 2025 movement timestamps, finite labels and immutable output paths. Inference verifies the v7 baseline and ranking/template digests, completed-model status, parameters, model digest, tree count, feature order, finite nonnegative values and exact template alignment.

```sh
python arrival_boost_contest.py train --data runs/data --output runs/arrival-boost-full --trees 4999 --device GPU --permission-ref "PRC2026 registered participant, challenge-only"
python arrival_boost_contest.py predict --run runs/arrival-boost-full --baseline runs/submissions/zestful-fountain_v7.parquet --ranking runs/data/ranking.parquet --template runs/data/submitting.parquet --output runs/submissions/zestful-fountain_v8.parquet --permission-ref "PRC2026 registered participant, challenge-only"
```

Use fresh paths and an unused increasing submission version. Training and prediction do not upload automatically. V8 subsequently scored **278.6349 seconds RMSE** officially, improving v7 by 7.1141 seconds. The complete 09:54 UTC snapshot on 13 September ranks the team 17/127, leader 245.0207. Every prediction independently recomputes exactly and all 383 existing specialist values are preserved. [Dated result](results/leaderboard-2026-09-13.md).

## Airport and completed-NM-neighbor candidate v9

`ordinary_ensemble.py` combines a global LightGBM model with one expert per departure airport. All eleven models use 679 trees, 63 leaves, learning rate 0.035, minimum child samples 80, L2 regularization 10, column fraction 0.9, seed 20260907, two threads, deterministic fitting and column-wise histograms. Full fitting uses all 2,083,190 nonnegative ordinary-scope 2025 departure labels, excluding the existing unmatched-LIRF specialist scope. Fitting residuals are capped at +/-7,200 seconds; original evaluation labels remain unchanged.

Airport experts use the 123 existing arrival-context predictors. The global model adds 30 features from `nm_neighbor_features.py`: for airport, runway and stand groups, counts, means, standard deviations, own-minus-mean differences, latest age and latest value for the supplied NM taxi-time proxy. The proxy is movement time minus NM AOBT, accepted only from 0 to 7,200 seconds with matching movement/NM departure airports. Pools contain DEP rows from the same calendar month and strictly earlier movement times; the queried row and simultaneous movements are excluded. The helper selects only allowed timestamp/group fields before operating, never departure labels or block times. Latest timestamp ties are averaged; missing stands do not form an observed stand group. These are retrospective organizer-supplied predictors, with no live forecasting claim.

The fixed candidate averages the airport and neighbor models equally. Its component comparison against the identically configured pooled 123-feature model improves all four evaluated months:

| Held-out month | Pooled RMSE | Equal airport/neighbor RMSE |
| --- | ---: | ---: |
| January 2025 | 347.313 | 344.175 |
| July 2025 | 301.045 | 296.675 |
| February 2025 | 283.220 | 275.394 |
| August 2025 | 249.154 | 244.701 |

Within each split, models fit the other ten months. The v7 specialist is restored identically for these comparisons. The airport model alone worsens January and February; the equal combination was explored after the individual January/July results, so this is reused research evidence. February/August compares components and does not validate the complete historical v8/v9 ensemble.

Outside the specialist scope, the proposed submission uses 75% v8 plus 25% of the equal candidate. This improves complete reused January/July RMSE from **315.778 to 315.179**, January from 341.669 to 341.197 and July from 293.251 to 292.531; **55/62 days** improve. All v8 specialist values remain exact. The recorded continuation criteria require at least 0.5 seconds overall gain, improvement in both months, at least 40 improved days, all four component-month improvements and an improved official v8 result. All requirements passed on 13 September; the earlier conditional runner timed out before the v8 receipt existed and fitted no models.

The standalone implementation exactly matches every cached feature: **153 columns across 2,085,047 departures**. The promotion wrapper additionally binds the twelve training-file hashes, source snapshot, comparison and decision before fitting. Synthetic tests cover strict timing/month/target exclusion, ties, alignment, configuration and model tampering, and independent full training/prediction round trips.

```sh
python ordinary_ensemble.py train --data runs/data --output runs/ordinary-full --permission-ref "PRC2026 registered participant, challenge-only"
python ordinary_ensemble.py predict --run runs/ordinary-full --baseline runs/submissions/zestful-fountain_v8.parquet --ranking runs/data/ranking.parquet --template runs/data/submitting.parquet --output runs/submissions/zestful-fountain_v9.parquet --permission-ref "PRC2026 registered participant, challenge-only"
```

Use fresh paths and the next unused submission version. Inference checks the v8 baseline, data and template digests, all model hashes, tree counts, feature order, fixed parameters and blend weights before exact-template validation. Neither command uploads. V9 subsequently scored **277.7530 seconds RMSE** officially, improving v8 by 0.8819 seconds. The complete 10:23 UTC snapshot on 2026-09-13 ranks the team 16/129; team best 277.7530, leader 245.0207. Every prediction independently recomputes exactly, preserving all 383 specialist values. [Dated result](results/leaderboard-2026-09-13.md).

## NM-neighbor CatBoost replacement candidate v10

`neighbor_boost_contest.py` adds the same 30 strict-past NM-neighbor predictors to the existing CatBoost's 123 arrival-context features. The 153-feature model retains depth 9, learning rate 0.05, L2 regularization 7, 254 borders, seed 20260907, GPU fitting and two threads. The tree count was fixed at 4,999 before the completed validation result, with no early stopping or blend-weight search. Competition configuration enforces these 4,999 GPU trees; shorter CPU fits require an explicit synthetic data class and are labeled synthetic tests. GPU fitting is not bitwise deterministic; saved-model digests identify each fitted artifact.

The candidate changes only the ordinary component that contributes 37.5% of v9: `v10 = v9 + 0.375 * (CatBoost153 - CatBoost123)`. Each CatBoost prediction retains its original nonnegative floor. The replacement itself is never clipped: a negative or nonfinite result rejects prediction. All LIRF rows with missing `IOBT_flt` preserve the v9 specialist exactly.

The predeclared gate required at least 0.5 seconds overall January/July improvement, improvement in each month, and at least 40 improved days against the actual v9 local baseline. The completed fixed comparison passes:

| Reused 2025 holdout | V9 RMSE | Replacement RMSE |
| --- | ---: | ---: |
| January and July | 315.179420 | 314.319747 |
| January | 341.196848 | 340.168756 |
| July | 292.531371 | 291.825053 |

The gain is 0.859674 seconds, with 52/62 days improved. Fitting uses the other ten months and excludes negative departure labels and the separate unmatched-LIRF scope; fitting residuals alone are capped at +/-7,200 seconds. All 344,419 original evaluation labels, including 80 negative labels and extreme values, remain unchanged. These months have been reused for research decisions; the result is not an untouched generalization test or an official score. The earlier four-month LightGBM comparison motivated these features but does not independently validate this complete CatBoost replacement. V9 remains the verified official best.

Full training uses all twelve authorized 2025 files and the existing 2,083,190 eligible ordinary-scope departures. To reproduce, first retain the exact v8/v9 prediction files and manifests and the original 123-feature full model that produced v8. Replacing that old model with a new fit would not reproduce the component embedded in the retained v9 baseline.

```sh
python neighbor_boost_contest.py train --data runs/data --output runs/neighbor-boost-full --permission-ref "PRC2026 registered participant, challenge-only"
python neighbor_boost_contest.py predict --run runs/neighbor-boost-full --old-run runs/arrival-boost-full --baseline runs/submissions/zestful-fountain_v9.parquet --v8-baseline runs/submissions/zestful-fountain_v8.parquet --ranking runs/data/ranking.parquet --template runs/data/submitting.parquet --output runs/submissions/zestful-fountain_v10.parquet --permission-ref "PRC2026 registered participant, challenge-only"
```

Use fresh paths and an unused increasing version. Inference verifies the v9-to-v8 baseline digest chain, the original CatBoost123 digest embedded in v8, identical twelve-file training provenance, both model schemas and fitted parameters, exact tree counts, data/template digests, scope counts and unchanged v8/v9 specialist values. The output records both model/report digests and both baseline-manifest digests. Synthetic tests exercise a full training/prediction round trip, component subtraction, exact row/specialist preservation, manifest/model tampering and invalid replacement rejection. Neither command uploads; source publication, full fitting and independent all-row verification remain separate requirements before an official submission.
