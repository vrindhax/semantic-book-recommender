"""Step 7a: Inspect the real dataset before indexing it."""

import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "books_real.csv"


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}\n")

    print("First 3 rows:")
    print(df.head(3))

    print("\nMissing values per column:")
    print(df.isnull().sum())


if __name__ == "__main__":
    main()