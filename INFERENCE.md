# Portable taxi-out inference source

`taxiout.py` exposes a generic typed request/context API with no hardcoded competition
file paths and no production dependency. The independent GPLv3 implementation is
exported separately as `taxiout_inference_source.zip`. It excludes data and weights.
Competition weights remain research-only; use a separately licensed model bundle
for commercial deployment.

```python
from pathlib import Path
from taxiout import ScheduleContext, ScheduleMovement, TaxiOutInput, TaxiOutPredictor

taxiOut = TaxiOutPredictor(Path("/your/model-bundle"))
request = TaxiOutInput(
    airport="EGLL", aircraft="A320", destination="LFPG",
    runway="09L", stand="A1", flightNumber="BA123",
    scheduledTimeUtc="2026-06-01T12:00:00Z", movementId="query-flight",
)
context = ScheduleContext(
    knownAtUtc="2026-06-01T11:00:00Z",
    movements=[ScheduleMovement(
        movementId="other-flight", airport="EGLL", phase="DEP",
        scheduledTimeUtc="2026-06-01T12:05:00Z",
    )],
)
result = taxiOut.predict(request, context)
print(result.model_dump())
```

The result contains `predictionSeconds`, `confidence`, `range`, `modelVersion` and
`reasonCodes`. Confidence is null until calibration is established. The range is an
empirical December residual range, not a coverage guarantee. Runway/stand are
accepted and explicitly unused: the training source does not prove predeparture
assignment availability. Unseen aircraft/operators map to fit-only UNKNOWN;
unseen airports use the global fit mean.

Each context movement names its departure or arrival airport. Clocks must be UTC;
snapshot time must precede departure. Duplicate movement IDs are rejected. Demand
excludes the queried movement by identity. When matching scheduled departures exist,
provide the query identity to avoid counting itself. The caller supplies actual
known schedule provenance; missing historical snapshots cannot be reconstructed.

Install pinned requirements in an isolated Python 3.12 environment. This source
bundle includes the feature/model implementation, standalone utilities, licence,
data boundary and this guide. The complete research archive additionally includes
training, validation and tests. See GENERALIZATION.md.

# Saved prediction policies

`TaxiOutPredictor` loads recorded guard metadata. The novel-category policy
preserves known aircraft and missing categories observed during fitting. Truly
novel inputs use the fitting set's airport/hour, airport or global baseline.
The earlier blanket missing-category guard remains supported for its recorded
experiments, even where forward RMSE rejects it. Responses carry
`UNSEEN_AIRCRAFT_BASELINE_FALLBACK` when applied. Use this typed interface for guarded
bundles; the original unguarded research CLI does not interpret guard metadata.
