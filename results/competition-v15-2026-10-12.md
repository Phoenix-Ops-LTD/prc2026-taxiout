# Competition trial v15 - 12 October 2026 UTC

The already fitted, competition-only NM residual CatBoost model passes independent
feature and native-model verification. Recompute all 2,085,047 training feature
rows with departure targets and block times masked; they exactly match the
preserved feature cache. Independently reload both validation models and exactly
reproduce 344,419 seasonal and 1,079,516 chronological validation predictions.
These historical validations are distinct from the purged forward MAE board.

The fixed single model and original NM/schedule fallback reproduce all 344,841
ranking predictions after reload and serialization. No new fitting, weights,
GPU rental or production change occurs. Public source CI passes 163 tests and
43 strict maintained Python modules before upload.

Guarded conditional upload creates `zestful-fountain_v15.parquet`; prior objects
remain intact. The full remote object matches SHA256
`794d5f7f69f4e0cfc638657de431b4926abc95213f7bce3a43465014c87d823e`.
The native full CatBoost model has SHA256
`0bb9158a9f242bdc687970936d2fc1c96f4e9e287454f927e9caaa918a70fd80`.
Model version is `prc2026-nm-residual-standalone/1.0.0`. Source is published at
[d1d664ca3dec44b57479dce6f1f76eecf0712378](https://github.com/Phoenix-Ops-LTD/prc2026-taxiout/tree/d1d664ca3dec44b57479dce6f1f76eecf0712378),
with [passing CI](https://github.com/Phoenix-Ops-LTD/prc2026-taxiout/actions/runs/38258026503).

Organizer status is **Succeeded**, with **344,841 used pairs** and **297.5883
seconds RMSE**, processed at **2026-10-12T13:53:09.530363Z**. This is **24.7400
seconds worse** than the previous submission and best, v14's **272.8483**.
Retain **v14** as the best scored submission. No previous object is replaced.

Complete public pagination at **2026-10-12T13:57:44.745159+00:00** confirms
**rank 78 / 244 scored teams**, across **5,487 submission entries**. The leader
scores 212.8457; our gap to first remains 60.0026 seconds. First place is not
achieved. The observer retries after a transient API timeout and preserves the
original timestamp of the last complete observation during failures.

Exact model/data/source/artifact hashes, official receipts and requested commercial,
LinkedIn and investor drafts are saved in the private score pack
`runs/final-mode-score-pack-20261012-v15.json`. No post is published. The MAE
leaderboard remains 340 completed comparable candidates, with partial runs unranked.

Prepare the distinct 670,790-row final file from the retained best v14 model chain.
First reproduce every historical ranking chain stage against its original bytes,
then independently reload and repeat all final predictions. The final file is
not uploaded during preparation. Competition models remain outside RALE,
NewWorld and customer production.
