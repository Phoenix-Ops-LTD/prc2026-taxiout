# Official-score retention mode - 12 October 2026 UTC

The owner makes the deadline and best official competition score the submission
retention objective. Deadline remains 14 October 2026 at 10:00 UTC. Local MAE/CV
records remain honest validation evidence; they cannot override a better scored
submission. Competition ranking already retains each team's minimum scored RMSE.
All trial uploads use fresh increasing version keys with conditional creation;
previous objects and their manifests are never replaced.

The existing verified forecast is accepted as v13: 344,841 pairs, 572.3617 seconds
RMSE. V12 remains the official winner at 273.5557 seconds. Complete public
observation at 12:50:43 UTC records rank 78 / 243 teams and 5,482 submission
entries. Improvement over the immediately preceding submission is -298.8060
seconds. The exact model weights, native/input/source/artifact hashes and all
three requested public-text drafts are retained privately in
`runs/final-mode-score-pack-20261012-v13.json`. No post is published.

One bounded CPU challenger tests a fixed 7,200-second upper cap on fitting targets
only. It retains the novel-aircraft guard, original feature allowlist and purged
September/October/November folds. Raw validation labels, including extreme and
negative values, remain unchanged. A fixed equal pair with the frozen CatBoost
component is also evaluated; there is no weight search or fitting to official
feedback. Native all-row reload parity and source/input/model hashes are required.
Only a completed single/pair with every fold RMSE within 5% of the incumbent can
qualify for an official trial. This is a trial-validity bound, not a claim of
improved official performance. The minimum pooled CV RMSE of that fixed set
chooses the trial artifact. Retention follows the best accepted official score.

A cycle may take no more than one hour and requires two hours remaining before
the deadline at each stage. Existing CPU capacity is used; no new GPU cost is
incurred. Any qualified candidate must receive a full 2025 refit, exact native
ranking and separate final-template prediction verification, passing public
source CI, guarded upload and a confirmed official score. Later diagnostics and
hidden labels are not used to tune it. Further repetitions require a concrete
new hypothesis with plausible gain against the best official submission; a poor
official result does not justify sweeping weights or known ranking months.

Official rank/history observation and local experiment monitoring remain active.
Official feedback may govern scored submission retention; it does not enter the
local MAE leaderboard, training targets or model-fitting objectives. The local
final-evaluation artifact is separate and is not uploaded by a ranking trial.

Validation before the trial: 162 research tests and 42 strict maintained Python
modules pass. The CPU monitor carries separate measured fit/inference clocks and
fitting-cap parameters. Partially completed runs remain unranked. Production,
customer aviation, private data rights and original artifacts are unchanged.
