# Final submission protocol - 12 October 2026 UTC

The current best scored competition model is the fixed long-following replacement,
model version `prc2026-long-following-replacement/13.0.0`, submitted for ranking as
**v14 / 272.8483 seconds RMSE**. It is an operational-observation competition
model. The separate predeparture forecast and portable research adapter remain
available; competition-trained weights do not enter production.

Use the actual `final_ranking.parquet` and `final_submitting.parquet`, not the
ranking files. The actual final template contains **670,790 departures** in
January, February, June and July 2026. Departure targets and block times are blank.
Shared ranking IDs do not justify reusing ranking predictions: provided clocks
and arrival observations differ between the two datasets.

## Exact native reconstruction

Use the hash-bound GPL modules in `reproducibility/long-following-13.0.0` and
their original native model reports. Each prediction command accepts explicit
`--ranking`, `--template`, fresh `--output` and `--permission-ref` arguments.
Run each module in a fresh Python interpreter to prevent imports from a different
source version. Source files must match `source-hashes.json` byte for byte.
The 21-module following-boost training snapshot also matches the corresponding
files in this 22-module published snapshot; the original snapshot coverage stays
bound to its training report.

| Stage | Prediction module | Native models / dependency |
| --- | --- | --- |
| 5 | `duration_contest.py predict` | duration-full-v17; carrier-seasonal-v9; lirf-specialist-v12; lightgbm-seasonal-v10 |
| 6 | `specialist_ensemble.py predict` | specialist-bagging-full-v31; stage 5 |
| 7 | `arrival_contest.py predict` | arrival-full-v40; arrival-specialist-full-v41 (`--arrival-specialist`); stage 6 |
| 8 | `arrival_boost_contest.py predict` | arrival-catboost-full-v50; stage 7 |
| 9 | `ordinary_ensemble.py predict` | equal-ordinary-full-v64; stage 8 |
| 10 | `neighbor_boost_contest.py predict` | neighbor-boost-full-v68; original arrival-catboost-full-v50; stages 9 and 8 |
| 11 | `clock_overlay_contest.py` | stage 10; fixed original clock rule |
| 12 | `following_boost_contest.py predict` | following-boost-full-v74; original neighbor-boost-full-v68; stages 11 and 10 |
| 13 | `long_following_contest.py predict` | following-long-full-v77; original equal-ordinary-full-v64; stages 12, 11, 10 and 9 |

The final model stage number is historical; the exact same model is ranking v14.
Prediction commands validate native hashes, schemas, component weights, protected
scopes and parent-manifest lineage. No refitting or weight changes occur.

First run all nine stages on the original ranking inputs. Every serialized stage
must match its original ranking artifact exactly. The final ranking digest must be
`145ff5ae9c5fe9d08dc3ef2eef42e787dd411ca051a76ac404ef750b1a918240`,
which also matches the scored v14 remote object. Then run the chain on the final
inputs twice in independent fresh directories. Verify exact equality of all
670,790 predictions, IDs, ordering, dtypes and serialized bytes. Recheck every
source, native-model, report and evaluation input hash after inference. Independently
verify all twelve original training input digests against the fitted-model report.

## Final upload accepted

The verified unique final file was uploaded immediately on **12 October 2026 at
16:38:29 UTC**, replacing the earlier local scheduled wait. The organizer returned
**Succeeded / 670,790 used pairs**, observed at 16:39 UTC. Its result contains no
score: final acceptance is confirmed, while final score and final rank remain
unknown. The complete remote object matches the verified SHA256 below.

The scheduled upload guard is stopped, and its stop marker remains present.
Read-only final-result observation and public rank monitoring continue. Their
local processes require this computer to remain running, but the accepted final
file is already stored remotely and no longer depends on a scheduled upload.

Immediately before writing, recheck authoritative own-bucket score receipts: the
retained best must still be v14 / 272.8483. A changed best requires a newly verified
final artifact and replacement of this guard. Recheck input/model/source hashes,
template alignment, bucket capacity, quota and absence of an existing final object.
Create only `zestful-fountain_final.parquet`, conditionally with `IfNoneMatch: *`.
Read back the complete object and verify SHA256 before recording upload success.
An upload is not an official final score; final evaluation may remain pending.
The private `runs/latest-final-submission.json` receipt records acceptance,
artifact/manifest/result hashes and full row count. Detailed source/native/data
bindings remain in the release manifest and original native verification report.

The local stop marker is `runs/final-deadline-guard.stop`. A possible completed
write is never retried blindly: reconcile the existing remote key and its complete
bytes first. Existing ranking objects are never replaced. Public rank observation
continues independently and never changes the local forward MAE leaderboard.

Current validation: all nine historical ranking stages reproduce original bytes;
both complete final native passes and all nine intermediate stage repeats match
exactly. Final prediction SHA256 is
`b1adbc9b2a62f72feef1590cd2ba4a3e6d74e1ffc5300acb20e90c7b1f6356ee`.
The read-only authoritative bucket/quotas/training-hash preflight also passes.
All 168 research tests and
43 strict maintained Python modules pass public CI. Runtime data, native models,
credentials, detailed receipts and artifact archives remain outside Git.
