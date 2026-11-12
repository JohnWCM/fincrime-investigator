
from collections import Counter
from pathlib import Path
import pandas as pd

DATA = Path("data/raw/HI-Small_Trans.csv")

required = [
    "From Bank",
    "Account",
    "To Bank",
    "Account.1",
]

counts = Counter()

for df in pd.read_csv(
    DATA,
    usecols=required,
    dtype=str,
    chunksize=200_000,
):
    source = (
        df["From Bank"].str.strip()
        + "::"
        + df["Account"].str.strip()
    )

    destination = (
        df["To Bank"].str.strip()
        + "::"
        + df["Account.1"].str.strip()
    )

    # Count each transaction once per account.
    # Self-transfers should not count twice.
    counts.update(source.dropna().tolist())

    non_self = destination[source != destination]
    counts.update(non_self.dropna().tolist())

print("\nTOP 30 MOST ACTIVE ACCOUNTS\n")

for rank, (account, n) in enumerate(
    counts.most_common(30), 1
):
    print(f"{rank:2}. {account:30} {n:,} transactions")
