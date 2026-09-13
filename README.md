# PhoenixAI PRC 2026 taxi-out research

Independent, original research for team **zestful-fountain**. V12 scored **273.5557 seconds RMSE** on all **344,841 pairs**, improving v11 by **1.4181 seconds**. Complete public pagination at **2026-09-13T16:53:37.305711+00:00** ranks the team **12/131**, up from 14/131 immediately before submission. Leader 245.0207; remaining gap **28.5350 seconds**. First place remains incomplete. [Official result](results/leaderboard-2026-09-13.md#v12-following-context-catboost-replacement). See [METHOD.md](METHOD.md) for validation, availability limits and reproduction.

This directory alone is GPL-3.0-only. The proprietary parent PhoenixAI/RALE repository, product modules, credentials, models and restricted datasets are excluded.

The research evaluates airport demand, flight context and the organizer-supplied Network Manager records for retrospective taxi-out reconstruction. The first model uses CatBoost with original preparation, explicit features, temporal validation and submission controls. The product uses separate observation feeds and deterministic scenario implementations; it does not consume this model or competition data.

## Reproduce

Python 3.12; create an isolated virtual environment, then:

```sh
python -m pip install -r requirements.lock.txt
python -m pytest -q
python -m mypy pipeline.py buckets.py traffic_features.py contest.py carrier_contest.py ensemble_contest.py duration_contest.py specialist_ensemble.py arrival_features.py arrival_specialist.py arrival_contest.py arrival_boost_contest.py nm_neighbor_features.py ordinary_ensemble.py neighbor_boost_contest.py nm_clock_overlay.py clock_overlay_contest.py leaderboard.py
```

Tests train twice on fixed synthetic fixtures, compare predictions, and verify template order, invalid values and feature leakage exclusions. Synthetic scores are test outputs only.

## Neighbor replacement submission v10

`neighbor_boost_contest.py` replaces v9's existing 123-feature CatBoost component with the same fixed 4,999-tree model using 153 features. The frozen 37.5% component replacement improves reused January/July RMSE from 315.179 to 314.320, improving both months and 52/62 days. These local results use reused selection data. V10 subsequently scored **275.1751** officially on all 344,841 pairs, ranking **14/130** at 12:04 UTC on 13 September 2026. [Configuration, guards and reproduction commands](METHOD.md#nm-neighbor-catboost-replacement-candidate-v10).

## Fixed clock-overlay submission v11

`nm_clock_overlay.py` applies one fixed rule to supplied full predictions: for ordinary departures with matching NM origins, both valid clock proxies and signed `AOBT_3_flt - LOBT_flt > 7200` seconds, replace the entire prediction with the nonnegative movement-minus-LOBT proxy. Every other prediction and the existing LIRF/missing-IOBT specialist remain exact. The original GPLv3 module performs no fitting or model inference; `clock_overlay_contest.py` provides a separate file/manifest wrapper for an independently verified v10 baseline.

One fixed local diagnostic improves proposed-v10 reused January/July RMSE from **314.3197465013 to 313.7463286061**, a **0.5734178952-second** gain, with both months improving. It changes 31 of 344,419 rows and retains all original labels, including 80 negatives. The largest beneficial row supplies 45.8% of net squared-error improvement; excluding it only for influence analysis leaves a 0.311181-second gain and improvement in both months. These are reused selection data, with historical holdout material already inspected; this is not an untouched validation result.

Target-free ranking checks found 21 eligible rows and matched their replacement proxies. They do not verify actual overlaid ranking predictions or establish an official gain. This frozen diagnostic had no predeclared promotion criteria; any later release decision follows observation of its results. After observing these results and reconciling the improved official v10 score, the fixed rule was manually selected for a guarded release. Publication and exact verification of all 344,841 actual ranking predictions preceded guarded upload. V11 then scored **274.9738 RMSE** officially, improving v10 by **0.2013 seconds**; rank remained **14/130** at 12:34 UTC on 13 September 2026. [Exact rule, evidence limits and conditional CLI usage](METHOD.md#fixed-nm-clock-overlay-proposal-v11).

## Authorized data access

Use the competition console linked from the provisioning email. Select Other Authentication Methods, then Login with SSO. Create MinIO access keys and save their `accessKey` and `secretKey` in a private file outside this repository. The OpenSky REST `clientId`/`clientSecret` files cannot access competition buckets. Never paste keys in commands, commit them, or put them in browser code.

```sh
python buckets.py --credentials /private/minio.json list-buckets
python buckets.py --credentials /private/minio.json list --bucket ACTUAL_DATA_BUCKET
python buckets.py --credentials /private/minio.json download --bucket ACTUAL_DATA_BUCKET --key ACTUAL_OBJECT_KEY --output runs/data/january.parquet
```

Discover exact keys; do not guess them. Download the 2025 monthly training files, 2026 ranking data and official submitting template. Keep download receipts and hashes private. Reconcile the overview's 11-airport statement with the detailed 10-airport table and the actual files. Confirm data use permission for an industry team and keep any product/commercial permission separate.

Copy `config.example.json` into a private run directory. For authorized real data set `data_class` to `competition` and `data_permission_ref` to the dated permission/provisioning reference. This reference records evidence; it does not create a licence. Default configuration remains synthetic.

```sh
python pipeline.py train --training runs/data/january.parquet runs/data/february.parquet --config runs/config.json --output runs/model-v1
python pipeline.py predict --run runs/model-v1 --ranking runs/data/ranking.parquet --template runs/data/submitting.parquet --output runs/submissions/zestful-fountain_v1.parquet
python pipeline.py validate --template runs/data/submitting.parquet --submission runs/submissions/zestful-fountain_v1.parquet
python buckets.py --credentials /private/minio.json upload --template runs/data/submitting.parquet --submission runs/submissions/zestful-fountain_v1.parquet
python buckets.py --credentials /private/minio.json list --bucket prc-2026-zestful-fountain
```

The training command above illustrates multiple inputs; pass **all twelve** authorized 2025 monthly files for the final fit. October through December is the default temporal holdout, so training inputs must include records on both sides of 1 October. Download the resulting score object separately; an upload receipt is not an official score. Conditional uploads refuse replacement. Keep version numbers increasing.

Immediately before upload, `buckets.py` paginates the team bucket and enforces increasing versions, a conservative maximum of five submissions in the preceding 24 hours, and the one GB capacity limit. Unknown/invalid timestamps or listing failures stop the write. Run only one uploader at a time; these client-side checks do not provide a distributed lock against simultaneous processes.

## Method and evaluation

- Target: `TAXITIME_SEC_mvt`, seconds; departures only. Official metric: RMSE.
- Categories: departure/destination airport, aircraft type, flight rule, market segment, wake category and operator. Missing optional fields become `UNKNOWN`; inspect actual schema before accepting that fallback.
- Calendar: UTC hour, weekday, month, cyclical hour and 30-minute scheduled departure count. Density is batch schedule context, not an observed live queue.
- Excluded: movement/flight IDs as predictors, target, actual movement/off-block times, and runway/stand until their availability at prediction time is established.
- Holdout: chronological split with no repeated flight group across the boundary. No random row split. Airport means use only the fitting partition. Early stopping selects iterations; the final model is refitted on all authorized training rows.
- Report: overall and per-airport RMSE, airport-mean baseline, residual p90, feature importance, input/model hashes and parameters. Residual p90 is not a calibrated confidence interval.
- Submission: exactly `MVT_ID_mvt`, `TAXITIME_SEC_mvt`, exact official IDs/dtype/order, finite nonnegative predictions. Model and template hashes are checked before upload.

The optional `challenge_context` mode adds the supplied runway, stand, airport-context pairs, movement calendar, movement/schedule delta and departure density. This is retrospective ranking-data context, not a predeparture operational forecast. In that original mode, actual off-block proxies (`AOBT_3_flt`, `LOBT_flt`) remain excluded. The separate v2 `contest.py` mode deliberately uses these organizer-supplied retrospective records: see METHOD.md. It must not be represented as a predeparture forecast. The documented task supplies actual movement/runway/stand fields to participants.

For the observed negative taxi labels, `negative_target_policy="drop"` excludes them from fitting only. They remain in chronological holdout scoring and are counted in the report. Final refitting excludes negative labels; positive outliers are retained. The official 2025 training-year check uses movement timestamps because a flight can be scheduled on 31 December 2024 and depart in 2025.

Before claiming competitive performance, evaluate multiple chronological cutoffs including January and July-like seasonal conditions, compare airport/hour medians and boosted models, perform feature ablations, assess unseen airports/operators, and document outliers without using ranking targets. Run a small predefined depth/learning-rate grid using training holdouts only. Keep changes supported by holdout results; do not optimize through excessive leaderboard submissions. Weather, geometry and additional data require documented open availability, licences and as-of alignment before use.

## Participation checklist

1. Obtain authorized MinIO access; inspect the actual schema, airport coverage and target definition.
2. Run baseline and chronological experiments, record all results and retain a reproducible best run.
3. Export **only this independent directory** to a standalone public GPLv3 GitHub repository, excluding `runs`, virtual environments, caches and private config. Do not publish the proprietary parent repository. Complete a secret scan first.
4. Produce and validate a genuine ranking submission; upload to the assigned team bucket and verify the official result file.
5. Retain code commit, environment lock, data hashes and exact configuration for winner reproduction. Prepare the method report and any organizer-requested final materials.
6. Submit well before the published **11 October 2026, 23:59:59 CET** deadline. Ask organizers to reconcile CET/CEST wording; do not rely on a last-hour interpretation.

## Sources and climate context

[Challenge](https://prc-data-challenge-2026.netlify.app/), [eligibility](https://prc-data-challenge-2026.netlify.app/eligibility.html), [data](https://prc-data-challenge-2026.netlify.app/data.html), [ranking](https://prc-data-challenge-2026.netlify.app/ranking.html), [registered team](https://prc-data-challenge-2026.netlify.app/teams/zestful-fountain.html), [OpenSky terms](https://opensky-network.org/about/terms-of-use).

For the wider climate-project and digital monitoring, reporting and verification context, see [Planet2050](https://planet2050.earth/). Taxi-time prediction is not itself a verified fuel reduction, carbon credit or Planet2050-certified result. No Planet2050 data or code is used by this baseline.

## Find our team in the API

The endpoint is paginated and contains repeated teams. Entries can move between pages. [Dated ranking snapshot](results/leaderboard-2026-09-13.md). Run `python leaderboard.py` to follow every `nextCursor` and rank each team by its best score.

## Carrier/specialist submission v3

v3 officially scored **292.4043 seconds RMSE**, rank **12 of 69** at 2026-09-08 04:04 UTC. [Dated result](results/leaderboard-2026-09-08.md). `carrier_contest.py` provides independent training and inference; see METHOD.md. First place remains unachieved.

## Fixed ensemble submission v4

v4 officially scored **291.4829 seconds RMSE**, rank **12 of 69** at 2026-09-08 04:26 UTC. [Result](results/leaderboard-2026-09-08.md). `ensemble_contest.py` adds reproducible LightGBM fitting and the fixed blend; see METHOD.md.

## Duration submission v5

v5 officially scored **290.0659 seconds RMSE**, with all **344,841 pairs** scored. Rank **19 of 89** at **2026-09-09 18:40 UTC**; the leader is at **246.3605**. [Dated result](results/leaderboard-2026-09-09.md). First place remains unachieved; the goal is to reach it and defend it through the published 11 October deadline within the submission rules.

`duration_contest.py` reproduces the 93-feature retrospective model and fixed blend preserving the LIRF specialist. Local reused-holdout RMSE is **321.589**, versus **323.377** for v4; that is distinct from the official score. Full fitting completed on all 2,084,678 nonnegative 2025 departure labels. Independent recomputation exactly matches every submitted value. See [METHOD.md](METHOD.md).

The v6 submission uses `specialist_ensemble.py` to average five predeclared specialist seeds while keeping every other v5 prediction exact. It officially scored **288.3714**, improving v5 by **1.6945 seconds**. The specialist improves in January, February, July and August validation; overall January/July RMSE is 321.005. Reproduction and validation limits are documented in [METHOD.md](METHOD.md#specialist-seed-ensemble-v6-10-september-2026).

## Following-context submission v12

The fixed 183-feature CatBoost candidate passes its predeclared local criteria: reused January/July whole-ensemble RMSE improves from 313.746329 to 312.971010, with both months and 59/62 days improving. This 0.775318-second local gain is not an official score. All 398 specialist and 31 clock-correction predictions remain exact. The existing 37.5% component weight and 4,999-tree settings are unchanged.

`nm_following_features.py` adds thirty features from strictly following, same-UTC-month NM departure-clock observations. The added columns match all 2,085,047 training and 344,841 ranking departures against the fixed experimental references; separate direct synthetic and sampled real-row oracles also pass. These are retrospective supplied observations, with no predeparture availability claim.

`following_boost_contest.py` provides full fitting and guarded prediction for this selected candidate. It checks the v11/v10 prediction chain and the old Cat153 model, then replaces only that component while preserving the specialist and clock scopes. Full fitting, original source publication, exact all-row prediction verification, guarded upload and official result reconciliation are complete for v12. The local gain above remains distinct from the official 1.4181-second improvement. See [the fixed protocol and reproduction commands](METHOD.md#fixed-following-context-catboost-candidate-v12).
