"""Step 7b: Clean the real dataset down to what we need for semantic search."""

import ast
import json
from pathlib import Path

import pandas as pd

RAW_PATH = Path(__file__).resolve().parent.parent / "data" / "books_real.csv"
CLEAN_PATH = Path(__file__).resolve().parent.parent / "data" / "books_real_clean.json"


def parse_stringified_list(value: str) -> list[str]:
    """Convert a string like "['Fiction', 'Fantasy']" into an actual list.

    ast.literal_eval safely evaluates Python literal syntax (lists, dicts,
    numbers) without the security risk of eval(). It's the standard tool
    for exactly this situation.
    """
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return parsed
        return [str(parsed)]
    except (ValueError, SyntaxError):
        return [value]


def main() -> None:
    df = pd.read_csv(RAW_PATH)
    print(f"Loaded {len(df)} raw rows.")

    # Keep only the columns semantic search actually needs.
    df = df[["title", "authors", "description", "genres", "average_rating"]]

    # Drop rows with no description — nothing to embed without one.
    before = len(df)
    df = df.dropna(subset=["description"])
    print(f"Dropped {before - len(df)} rows with missing descriptions.")

    # Parse the stringified lists back into real lists, then join for display.
    df["authors"] = df["authors"].apply(parse_stringified_list)
    df["genres"] = df["genres"].apply(parse_stringified_list)

    df["authors_display"] = df["authors"].apply(lambda a: ", ".join(a))
    # Metadata filters need ONE simple value, so we take the first genre only.
    df["primary_genre"] = df["genres"].apply(
        lambda g: g[0] if g else "Unknown"
    )

    # Build the final clean records as plain dicts.
    books = []
    for i, row in df.iterrows():
        books.append({
            "id": str(i),
            "title": row["title"],
            "author": row["authors_display"],
            "genre": row["primary_genre"],
            "description": row["description"],
            "rating": float(row["average_rating"]),
        })

    with open(CLEAN_PATH, "w", encoding="utf-8") as f:
        json.dump(books, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(books)} cleaned books to {CLEAN_PATH}")
    print(f"\nSample record:\n{json.dumps(books[0], indent=2)}")


if __name__ == "__main__":
    main()