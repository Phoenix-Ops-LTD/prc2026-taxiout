# Best scored model: final artifact ready - 12 October 2026 UTC

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

The named local artifact is
`runs/final-ready-20261012-v14/zestful-fountain_final.parquet`; it remains private
and **not uploaded**. The original forecast final is retained separately. A full
read-only upload preflight passes: v14 is still the authoritative own-bucket best,
its complete remote bytes match the reconstructed ranking artifact, all training
hashes and final template checks pass, the unique final key is absent, three
ranking predictions fall in the previous 24 hours, and bucket usage is 70,619,903
bytes. These checks repeat immediately before any future write.

The active isolated local guard schedules the unique conditional final upload for
**14 October 2026, 08:00 UTC**, two hours before the organizer's **10:00 UTC**
deadline. It requires the best still to be v14; a changed best requires new final
preparation. Existing ranking and final objects are never replaced. Verify complete
remote bytes before recording upload success. Official final evaluation remains
unknown until the platform produces its result. The host and guard process must
remain running. The read-only dashboard shows guard status, schedule and a stale
heartbeat warning, alongside rank, scored-team count and submission-entry count.

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
