# Verified forecast ranking submission - 12 October 2026 UTC

The owner explicitly requests submission of the existing CV-selected forecast.
Publish the already validated ranking predictions as `zestful-fountain_v13.parquet`.
This is the forecast/1.3.0 CatBoost + deep LightGBM novel-input ensemble, fixed at
50%/50%; it is distinct from the earlier unsubmitted retrospective v13 proposal.
No fitting, feature change or weight adjustment occurs for this submission.

Fresh saved-model inference reproduces all 344,841 ranking predictions exactly.
The uploaded file SHA256 is
`147d7b56642767cef34622109a872778e8d69d9e5ce80ab3fcc813eec08cf41c`.
The guarded uploader checks every team-bucket page, the next increasing version,
zero previous submissions in the rolling 24 hours, the one GB capacity limit,
exact template schema/IDs and finite nonnegative values. Conditional creation
prevents replacement. A subsequent full remote read matches the local bytes.
Public source and its passing CI are available in draft public PR 17.

The own-bucket organizer receipt reports **Succeeded**, **344,841 used pairs** and
**572.3617 seconds RMSE**, observed at 2026-10-12T12:42:32.117979+00:00.
This is worse than the previous official best of 273.5557 seconds. The challenge
ranks each team by its best submitted RMSE, so the prior best remains in force.
First place is not achieved. Complete public observation before this upload was
rank 78 of 243 scored teams, with a leader score of 212.8457 seconds.

This official observation is excluded from local ranking and model selection.
The MAE leaderboard and frozen CV-selected forecast remain unchanged. The local
dashboard shows the latest upload and its result separately from the team's best
official score. Private receipt, bucket keys and submission artifacts remain in
ignored runtime files. The separately verified final-evaluation file remains
local; this action submits ranking predictions only.

Validation: fresh all-row native parity and full uploaded-byte parity pass;
155 research tests and 41 strict maintained Python modules pass. No production
services, repositories or customer aviation code are changed.

Sources: [official ranking/submission rules](https://prc-data-challenge-2026.netlify.app/ranking.html),
[complete paginated ranking API](https://datacomp.opensky-network.org/api/competitions/bb3693e1-26bc-4a9e-8619-4fe78b4eab0c/leaderboard).
