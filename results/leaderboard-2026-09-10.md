# Official PRC result: 10 September 2026

Team **zestful-fountain**, submission **v6**, scored **288.3714 seconds RMSE** across all **344,841 pairs**. The previous best v5 scored 290.0659; improvement is **1.6945 seconds**. The organizer processed v6 at **2026-09-10T05:08:59.596394Z**.

A complete **668-submission** API read at **2026-09-10T05:09:04Z** places the team **22nd of 95 scored teams**. Leader youthful-giraffe scores **245.2901**; the gap is **43.0813 seconds**. Rank is unchanged from the earlier check. First place has not been achieved.

The candidate averages five fixed seeds for the existing unmatched-LIRF specialist. All **344,458 predictions outside that scope remain exactly unchanged** from v5; **383 specialist rows** receive the ensemble. Independent all-row recomputation matches every exported value exactly (maximum absolute difference 0).

- Submission SHA-256: `dbaec23b869f46a3ea7bd3f1a28740be1bdc9e11e97f0ce8db89b7e06ec31f1e`.
- Full fitting: four additional experts, each using all 1,488 nonnegative specialist labels; the fifth expert is retained from the verified v5 baseline.
- Source: [research PR 4](https://github.com/Phoenix-Ops-LTD/prc2026-taxiout/pull/4), source commit `314984b`, merged as `a293cd8`.
- Validation: 28 research tests, nine strict mypy modules, both public research CI runs and secret/artifact scans pass.
- Upload: first submission on 10 September; the guarded uploader found one submission in the preceding 24 hours and verified all bucket pages, version and capacity before writing.

The fixed local specialist comparison improves January, February, July and August. January/July overall RMSE with all other v5 values fixed is 321.005 versus 321.589. These local values are distinct from the official result; see [method and reproduction](../METHOD.md#specialist-seed-ensemble-v6-10-september-2026).

Sources: [official ranking](https://prc-data-challenge-2026.netlify.app/ranking.html), [full paginated leaderboard API](https://datacomp.opensky-network.org/api/competitions/bb3693e1-26bc-4a9e-8619-4fe78b4eab0c/leaderboard). Model artifacts, restricted source data, credentials and full score receipts remain private.
