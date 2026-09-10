# PhoenixAI PRC 2026 taxi-out research

Independent, original research for team **zestful-fountain**. Latest official submission v5 scored **290.0659 seconds RMSE** across **344,841 predictions**, rank **19th of 89 scored teams at 18:40 UTC on 9 September 2026**. The leader scored 246.3605 at that snapshot; first place remains unachieved. [Official result](results/leaderboard-2026-09-09.md). See [METHOD.md](METHOD.md) for validation, availability limits and reproduction.

This directory alone is GPL-3.0-only. The proprietary parent PhoenixAI/RALE repository, product modules, credentials, models and restricted datasets are excluded.

The research evaluates airport demand, flight context and the organizer-supplied Network Manager records for retrospective taxi-out reconstruction. The first model uses CatBoost with original preparation, explicit features, temporal validation and submission controls. The product uses separate observation feeds and deterministic scenario implementations; it does not consume this model or competition data.

## Reproduce

Python 3.12; create an isolated virtual environment, then:

```sh
python -m pip install -r requirements.lock.txt
python -m pytest -q
python -m mypy pipeline.py buckets.py traffic_features.py contest.py
```

Tests train twice on fixed synthetic fixtures, compare predictions, and verify template order, invalid values and feature leakage exclusions. Synthetic scores are test outputs only.

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

The endpoint returns **50 submissions per page**, with repeated teams. Our current entry is on the second page. [Readable ranking snapshot](results/leaderboard-2026-09-07.md). Run `python leaderboard.py` to follow every `nextCursor` and rank each team by its best score.

## Latest carrier/specialist submission

v3 officially scored **292.4043 seconds RMSE**, rank **12 of 69** at 2026-09-08 04:04 UTC. [Dated result](results/leaderboard-2026-09-08.md). `carrier_contest.py` provides independent training and inference; see METHOD.md. First place remains unachieved.

## Latest fixed ensemble submission

v4 officially scored **291.4829 seconds RMSE**, rank **12 of 69** at 2026-09-08 04:26 UTC. [Result](results/leaderboard-2026-09-08.md). `ensemble_contest.py` adds reproducible LightGBM fitting and the fixed blend; see METHOD.md.

## Latest duration submission and active continuation

v5 officially scored **290.0659 seconds RMSE**, with all **344,841 pairs** scored. Rank **19 of 89** at **2026-09-09 18:40 UTC**; the leader is at **246.3605**. [Dated result](results/leaderboard-2026-09-09.md). First place remains unachieved; the goal is to reach it and defend it through the published 11 October deadline within the submission rules.

`duration_contest.py` reproduces the 93-feature retrospective model and fixed blend preserving the LIRF specialist. Local reused-holdout RMSE is **321.589**, versus **323.377** for v4; that is distinct from the official score. Full fitting completed on all 2,084,678 nonnegative 2025 departure labels. Independent recomputation exactly matches every submitted value. See [METHOD.md](METHOD.md).

The next candidate, `specialist_ensemble.py`, averages five predeclared specialist seeds while keeping every other v5 prediction exact. It improves the specialist in January, February, July and August validation; overall January/July RMSE is 321.005. This is a local candidate with no official result yet. Reproduction and validation limits are documented in [METHOD.md](METHOD.md#specialist-seed-ensemble-candidate-10-september-2026).
