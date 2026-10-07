"""Scoring lookup from official tables stored as data, never as code.

Table file: CSV with columns  exercise,sex,age_min,age_max,reps,points
Each row says: for this exercise/sex/age band, `reps` valid repetitions earns
`points`. The scorer returns the points of the highest listed `reps` that does
not exceed the participant's count (0 below the lowest listed row, the top
row's points above the highest).

No table is bundled. data/scoring/README.md says where to get the official
MINDEF/HPB numbers; do not invent or approximate them.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import SCORING_DIR

COLUMNS = ["exercise", "sex", "age_min", "age_max", "reps", "points"]


class ScoreTable:
    def __init__(self, df: pd.DataFrame):
        missing = set(COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"scoring table missing columns: {sorted(missing)}")
        self.df = df[COLUMNS].copy()
        self.df["sex"] = self.df["sex"].str.lower().str[0]
        self.df["exercise"] = self.df["exercise"].str.lower()

    @classmethod
    def load(cls, path: str | Path | None = None) -> "ScoreTable":
        return cls(pd.read_csv(path or SCORING_DIR / "ippt_tables.csv"))

    def points(self, exercise: str, sex: str, age: int, reps: int) -> float:
        rows = self.df[(self.df.exercise == exercise.lower()) & (self.df.sex == sex.lower()[0])
                       & (self.df.age_min <= age) & (age <= self.df.age_max)]
        if rows.empty:
            raise LookupError(f"no scoring rows for {exercise}, sex={sex}, age={age}")
        rows = rows.sort_values("reps")
        reached = rows[rows.reps <= reps]
        return float(reached.points.iloc[-1]) if not reached.empty else 0.0
