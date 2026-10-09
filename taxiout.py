# SPDX-License-Identifier: GPL-3.0-only
"""Generic application-facing taxiOut.predict contract, with no competition file paths."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from forecast_model import PredictInput, PredictOutput, TaxiOut
from forecast_guard import GuardedForecast
from forecast_novel_guard import NovelGuardedForecast, load_model
from pipeline import ID, PHASE, TIME


def utc(value: str) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None or timestamp.utcoffset() != pd.Timedelta(0):
        raise ValueError("Explicit UTC required")
    return value


class ScheduleMovement(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    movementId: str = Field(min_length=1)
    airport: str = Field(min_length=1)
    phase: Literal["DEP", "ARR"]
    scheduledTimeUtc: str

    _utc = field_validator("scheduledTimeUtc")(utc)


class ScheduleContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    knownAtUtc: str
    movements: list[ScheduleMovement] = Field(default_factory=list)

    _utc = field_validator("knownAtUtc")(utc)


class TaxiOutInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    airport: str = Field(min_length=1)
    aircraft: str = "UNKNOWN"
    destination: str = "UNKNOWN"
    runway: str | None = None
    stand: str | None = None
    flightNumber: str | None = None
    scheduledTimeUtc: str
    movementId: str | None = None

    _utc = field_validator("scheduledTimeUtc")(utc)


class TaxiOutPredictor:
    """Use a separately licensed model bundle and the caller's actual known timetable.

    Airport in each schedule movement is the airport where its departure or arrival
    occurs. Restricted competition bundles remain explicitly labelled research-only.
    """
    def __init__(self, model_bundle: Path) -> None:
        self.adapter = TaxiOut(model_bundle)
        self.adapter.models = {n: load_model(model_bundle / n) for n in self.adapter.manifest["weights"]}

    def predict(self, request: TaxiOutInput, context: ScheduleContext) -> PredictOutput:
        identifiers = [m.movementId for m in context.movements]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Schedule movement IDs must be unique")
        if request.movementId is None and any(m.phase == "DEP" and m.airport == request.airport
            and pd.Timestamp(m.scheduledTimeUtc) == pd.Timestamp(request.scheduledTimeUtc) for m in context.movements):
            raise ValueError("movementId required when context could include the queried departure")
        records = [{ID: "context-" + str(index), PHASE: m.phase, TIME: m.scheduledTimeUtc,
                    "ADEP_mvt": m.airport if m.phase == "DEP" else "UNKNOWN",
                    "ADES_mvt": m.airport if m.phase == "ARR" else "UNKNOWN"}
                   for index, m in enumerate(context.movements) if m.movementId != request.movementId]
        schedule = pd.DataFrame(records, columns=[ID, PHASE, TIME, "ADEP_mvt", "ADES_mvt"])
        data = request.model_dump(exclude={"movementId"})
        data["scheduleKnownAtUtc"] = context.knownAtUtc
        result = self.adapter.predict(PredictInput.model_validate(data), schedule)
        if any(isinstance(m, GuardedForecast) and ((request.aircraft == "UNKNOWN"
            and (not isinstance(m, NovelGuardedForecast) or not m.missing_aircraft_seen_in_fit))
            or request.aircraft not in m.vocabulary["AIRCRAFT_TYPE_mvt"]) for m in self.adapter.models.values()):
            result.reasonCodes.append("UNSEEN_AIRCRAFT_BASELINE_FALLBACK")
        return result
