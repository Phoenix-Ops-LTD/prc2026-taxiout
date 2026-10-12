# Exact preserved competition source

This directory contains the independently authored GPL-3.0-only source bytes
bound to the original LightGBM long-following component replacement, model
version `prc2026-long-following-replacement/13.0.0`. The old local v13 candidate
was never uploaded; the later forecast v13 is a different model. A new release
must use the next unused remote version.

Run `long_following_contest.py predict` from this directory against the preserved
new/old native models, exact v12/v11/v10/v9 baseline chain, original ranking file
and submitting template. The CLI checks every source/model/report/input/baseline
hash, unchanged protected rows, exact template schema, all serialized values and
finite nonnegative predictions. `source-hashes.json` records all 22 module hashes.
The local gitattributes rule preserves those exact module bytes. Private data,
models and credentials are excluded.

This uses organizer-provided operational observations for competition-only
retrospective prediction. It is not a predeparture production model and is not
installed in RALE or NewWorld. Complete validation and increasing-version bucket
checks are required before uploading. Ranking and final-evaluation templates must
remain separate. See the repository METHOD.md for the fixed .125 component
replacement formula and limitations of reused seasonal validation.
