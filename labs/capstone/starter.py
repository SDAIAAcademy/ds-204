"""Read one synthetic capstone case and print what has to be decided before any model.

    uv run python capstone/starter.py bayader        # or: awtad, nasaim

It trains nothing. Copy it into your own repository as the first lines of your work.
"""
import json
import sys
from pathlib import Path

import pandas as pd

LABS = Path(__file__).resolve().parents[1]


def load(problem_id: str):
    """The table and its data dictionary for one synthetic case of the shortlist."""
    shortlist = json.loads((LABS / "capstone" / "shortlist.json").read_text(encoding="utf-8"))["problems"]
    problem = next(p for p in shortlist if p["id"] == problem_id and p["kind"] == "synthetic")
    table = pd.read_csv(LABS / problem["file"], parse_dates=[problem["date"]])
    dictionary = pd.read_csv(LABS / problem["dictionary"])
    return problem, table, dictionary


if __name__ == "__main__":
    problem, table, dictionary = load(sys.argv[1] if len(sys.argv) > 1 else "bayader")
    print(problem["title"])
    print("Decision:", problem["decision"])
    print(f"{len(table):,} rows, {table[problem['unit']].nunique():,} values of {problem['unit']}, "
          f"{problem['date']} from {table[problem['date']].min().date()} to {table[problem['date']].max().date()}")
    print(f"Target {problem['target']}: {table[problem['target']].mean():.1%}")
    print("\nWhen is each column written?")
    print(dictionary.to_string(index=False))
    print("\nBefore any model, write down:")
    for question in ("Who acts on the score, and how many can they act on?",
                     "On which day is the score made, and which columns exist on that day?",
                     "Which split rehearses the use: who is new, what is later, what is known?",
                     "Which rows stay closed until the end?",
                     "What would a constant answer and a simple rule score?"):
        print(" -", question)
    print("\nThink about:", problem["think_about"])
