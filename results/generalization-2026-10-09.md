# Generalization-first PRC validation, 9 October 2026

Status: **all forward/seasonal/airport validation, full CPU fitting, both submissions
and fresh all-row saved-model/API verification complete**. Submission files are
local; no new official score is claimed. Production code is unchanged.

The selected forecast is **50% CatBoost + 50% deep LightGBM with the novel-aircraft
fallback** (`catboost+deep_no_demand_novel_guarded`). Selection uses September,
October and November only. Guarded candidates must improve pooled RMSE by at least
one second without worsening a fold; fixed equal pairs must improve every fold
and gain at least one second over the best eligible single. December and seasonal
diagnostics never alter selection. The MAE CHAMPION is reported separately in
[LEADERBOARD.md](../LEADERBOARD.md); official scores never enter either process.

| Validation | Rows | RMSE seconds | MAE seconds | Signed bias seconds | Worst-decile RMSE seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| September–November forward selection | 531,951 | 403.769790 | 219.245624 | +5.241259 | 1,130.387612 |
| Locked December / January winter proxy | 165,664 | 415.085553 | 235.124140 | +17.793713 | 1,149.571363 |
| February forward seasonal diagnostic | 143,732 | 568.199863 | 251.586915 | -57.344573 | 1,688.190023 |
| June forward seasonal diagnostic | 183,142 | 476.876385 | 218.168032 | +3.266453 | 1,389.561462 |
| July forward seasonal diagnostic | 190,713 | 642.329579 | 242.686678 | -33.915109 | 1,935.471783 |

Additional frozen seasonal proxies: February RMSE 568.199863 seconds; June RMSE
476.876385 seconds; July RMSE 642.329579 seconds. These weaker results remain visible and
do not retune the selected ensemble.

Selected forward fold RMSEs: September 426.618333; October 341.933767;
November 440.764173. Evaluation retains 54 negative labels across selection and
21 in December. Fit excludes negative labels; large positive labels remain.
The selected ensemble's December p90 absolute error is 464.203201 seconds.

| December airport | Rows | RMSE seconds | Signed bias seconds |
| --- | ---: | ---: | ---: |
| EDDF | 17,322 | 294.373452 | +11.761449 |
| EDDM | 12,035 | 418.402848 | -83.599019 |
| EGLL | 20,000 | 396.573106 | -7.794486 |
| EHAM | 19,458 | 275.637707 | +24.960181 |
| LEBL | 13,835 | 252.803632 | +4.791062 |
| LEMD | 17,981 | 246.771926 | +4.796783 |
| LFPG | 19,778 | 407.872040 | +49.976535 |
| LIRF | 12,068 | 964.706076 | -7.127563 |
| LSZH | 10,585 | 281.517613 | +20.975085 |
| LTFM | 22,602 | 362.625369 | +94.832209 |

LIRF is the worst December airport by RMSE. Tail errors remain substantial; mean
MAE alone does not establish stable future performance. Blanket missing-aircraft
fallback improved MAE but worsened RMSE, so its completed results remain visible
while the RMSE stability gate rejects it. The narrower guard preserves missing
categories observed during fitting and falls back for truly unseen aircraft.

Complete-airport November holdouts confirm the deterministic unknown-airport
fallback: LSZH RMSE 424.629765 seconds (10,078 rows); LTFM RMSE 380.874871
seconds (22,189 rows). Every selected model overwrites an unknown airport with
its training global mean, so these predictions are computed exactly from the
excluded-airport training labels. They do not require redundant native fits;
independent CatBoost/LightGBM tests verify that invariant. Their tail and unseen
cohort metrics remain in the unranked diagnostic leaderboard records.

January means the December winter proxy, not a January calendar holdout. All 2025
data was used in past research, so December is not described as historically
untouched. Schedule context uses only published timetable fields with a documented
historical as-of assumption. Actual/NM clocks and observed runway/stand assignments
are excluded. See [GENERALIZATION.md](../GENERALIZATION.md) for methodology,
limitations, source/data hashes and reproduction commands.

The final template has 670,790 departures spanning January, February, June and
July 2026. Final inference preserves template IDs, dtype, order and row count; independent
fresh native inference matches every row exactly.
Reusable GPL source and typed inference are separate from research-only competition
weights; no trained competition model is installed in the customer product.


## Final delivery record

| Field | Value |
| --- | --- |
| BEST_CV_RMSE | 403.7697899305182 seconds |
| JAN_PROXY_RMSE | 415.0855531381534 seconds; December winter proxy, not January calendar holdout |
| FEB_PROXY_RMSE | 568.1998627655648 seconds |
| JUN_PROXY_RMSE | 476.87638530568466 seconds |
| JUL_PROXY_RMSE | 642.3295794174196 seconds |
| WORST_AIRPORT | LIRF; December RMSE 964.7060761070315 seconds |
| BEST_MODEL | catboost+deep_no_demand_novel_guarded |
| ENSEMBLE | 0.5 CatBoost + 0.5 deep no-demand novel-guarded LightGBM |
| LEAKAGE_CHECK | Allowlist and purged forward validation pass; historical schedule as-of availability remains an assumption |
| FINAL_SUBMISSION | runs/generalization-delivery-v4/submissions/final_submission.parquet; 670,790 rows; locally verified, not uploaded |
| REPRODUCIBLE | Frozen selection, input/source/environment hashes, native model reload and exact all-row inference receipts |
| PUBLIC_REPO_READY | GPL source-only archive prepared; data and weights excluded; public review is separate from merge |
| PRODUCT_MODEL_READY | Typed portable source and real bundle API pass; competition weights remain restricted to research |

Final submission SHA-256: `94dc5195f7647faaf38c2aa401b3f128f15c12abc1985331016deece2d1f43d4`.
Ranking submission SHA-256: `147d7b56642767cef34622109a872778e8d69d9e5ce80ab3fcc813eec08cf41c`.
All 2,084,678 nonnegative departure labels are used for final fitting; 369 negative
labels are excluded from fitting only. CPU fit times are 860.328 seconds for CatBoost
and 54.109 seconds for deep LightGBM. Separate inference times on 165,664 rows are
0.344 and 7.266 seconds. Each component reloads with exact parity on those rows.
The final 670,790-row batch takes 14.469 seconds of native model inference;
schedule preparation and serialization are outside that timer.

All 137 PRC tests and 35 maintained strict Python modules pass. The standalone
source archive includes the same complete test suite and pinned dependencies.
The MAE board preserves 314 completed candidates, three unranked partial runs,
six completed unranked diagnostics and two completed unranked refits. Legacy
benchmarks did not measure separate fit/inference times; these remain explicit
nulls rather than estimates. Official feedback is never used for selection.
