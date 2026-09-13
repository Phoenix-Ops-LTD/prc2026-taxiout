# Official PRC result: 13 September 2026

V8 scored **278.6349 seconds RMSE** on all **344,841 pairs**, improving v7 by **7.1141 seconds**. The complete 1147-submission snapshot at **2026-09-13T09:54:19.744476+00:00** ranks the team **17/127**, up from 29/127 immediately before upload. The leader scores **245.0207**; the remaining gap is **33.6142 seconds**. First place through the deadline remains incomplete.

The organizer processed `zestful-fountain_v8.parquet` at 2026-09-13T09:54:05.856442Z. The private own-bucket receipt and complete public leaderboard agree.

- Submission SHA-256: `d7c465325f0d479252e42dd8fca054e223ba6cdfce47c6ed6ac081d8e824c4a2`; 4,707,625 bytes.
- All 344,841 values independently recompute exactly, preserving all 383 v7 specialist values. Mean absolute change: 15.430 seconds.
- Full fit: 4,999 GPU trees on 2,083,190 eligible ordinary-scope 2025 departures, 123 features, fitting-only residual cap 7,200. Model SHA-256: `8dbe474bd526c5fa58333aa8f68af2b889504cb540072444bf1289ca05c354d6`.
- Original GPLv3 source: [PR 8](https://github.com/Phoenix-Ops-LTD/prc2026-taxiout/pull/8), source `a23e796`, merge `32332c8ee21f09a5b900312fa2507ff71cadad3d`. Public CI, 35 research tests, thirteen strict mypy modules and parent checks passed for this source.
- Training and verification completed on 10 September, but the initial upload failed with a network connection error. A fresh complete own-bucket listing on 13 September confirmed v8 was absent. The guarded retry succeeded after checking quota, capacity, increasing version, exact schema and digests; conditional creation prevents replacement. It found 0 prior submissions in the rolling 24-hour window.

The selected fixed 50% v7/arrival-CatBoost blend scored 315.778 versus 318.547 on reused January/July validation, improving both months and 61/62 days. Those local scores are separate from the official result. All original evaluation labels remain unchanged. Restricted data, fitted models and credentials remain private.

Sources: [official ranking](https://prc-data-challenge-2026.netlify.app/ranking.html), [complete paginated leaderboard API](https://datacomp.opensky-network.org/api/competitions/bb3693e1-26bc-4a9e-8619-4fe78b4eab0c/leaderboard). See [method and reproduction](../METHOD.md#deeper-arrival-residual-candidate-v8).
