# Best scored model: final accepted - 12 October 2026 UTC

The retained competition ranking submission is **v14 / 272.8483 seconds RMSE**.
It improves v12's previous best by **0.7074 seconds**. V15 scores 297.5883 and
duplicates historical v2; it is not retained. First place remains unachieved.
Local screening rejects a fixed equal mixture before fitting or upload because
it worsens both historical validation months and all 62 days. The forecast MAE
board remains 340 completed comparable candidates, with its CHAMPION unchanged.

Prepare the separate final artifact using model version
`prc2026-long-following-replacement/13.0.0`, the exact model submitted as v14.
The 670,790-row template covers January, February, June and July 2026. Departure
targets and block times are blank. No ranking predictions are reused on the final
data; each complete model chain is recomputed from the provided final inputs.

All nine historical ranking stages reproduce their original parquet bytes,
ending in the scored v14 SHA256
`145ff5ae9c5fe9d08dc3ef2eef42e787dd411ca051a76ac404ef750b1a918240`.
Two independent native runs on the final inputs reproduce **all nine stages
exactly**, including all 670,790 final values, IDs, order, dtypes and serialized
parquet bytes. All **106** frozen model/report/source/input bindings remain
unchanged. Independently verify the twelve original training input hashes against
the full fitted-model report. The reconstruction and repeat take 3,805.218 seconds
on CPU, with no fitting or GPU rental.

| Binding | SHA256 |
| --- | --- |
| Final prediction artifact | `b1adbc9b2a62f72feef1590cd2ba4a3e6d74e1ffc5300acb20e90c7b1f6356ee` |
| Final ranking input | `4b0692a792846079516a2749e034b0abda55df72c213eabdcda52a4d1f6bc69f` |
| Final template | `b3445d80e29bf599a5583ca9675f63ddf57b51fdf1ca3f498facb38ea84f3924` |

The separate final artifact was conditionally uploaded as
`zestful-fountain_final.parquet` on **12 October 2026 at 16:38:29 UTC**.
The complete 8,641,691-byte remote object matches the verified prediction SHA256.
The organizer returns **Succeeded / 670,790 used pairs**, observed at 16:39 UTC.
Its receipt contains no score: final acceptance is confirmed, while final score
and final rank remain unknown. The original forecast final is retained separately.

The fresh authoritative preflight confirms v14 remains the own-bucket best,
its remote bytes match the reconstructed ranking artifact, all training hashes,
106 source/native/report/input bindings and template checks pass, and the unique
final key is absent before creation. Three ranking predictions precede this upload
in the rolling 24-hour window; bucket usage before upload is 70,619,903 bytes.
Existing ranking objects remain unchanged. The earlier local scheduled wait is
superseded and its guard stopped; no automatic final replacement is permitted.
A read-only final-result observer checks for a published score every minute.
Public rank observation continues separately. The accepted file is already stored
remotely and does not depend on the local computer staying on for upload.

Detailed receipts, manifest and the exact source/model/data bindings remain in
ignored `runs/final-release-20261012-v14`; the dashboard reads the safe aggregate
acceptance state from `runs/latest-final-submission.json`.

All original model source is openly published with its exact 22-module hash
snapshot. The complete source ZIP includes that snapshot and passes **168 tests**
after independent extraction; public CI also passes **168 tests / 43 strict
maintained modules**. Desktop/mobile and stale-heartbeat browser checks pass.
Data, native weights, credentials and runtime receipts remain outside Git.
Public and private review PRs remain draft and unmerged. Production, NewWorld,
customer aviation and deployment settings are untouched.

See [reconstruction protocol](../FINAL_DELIVERY.md),
[v14 result](competition-v14-2026-10-12.md),
[v15 duplicate correction](competition-v15-2026-10-12.md), and
[rejected bounded screens](competition-screens-2026-10-12.md).
