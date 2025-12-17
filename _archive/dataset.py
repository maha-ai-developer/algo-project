import os
import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise SystemExit(f"[ERROR] CSV not found: {path}")
    df = pd.read_csv(path)
    # Expect at least: date/time, open, high, low, close, volume
    # Normalize
    df = df.rename(
        columns={
            c: c.lower().strip()
            for c in df.columns
        }
    )
    if "date" in df.columns and "time" in df.columns:
        df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"])
    elif "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    else:
        df["datetime"] = pd.to_datetime(df["date"])
    df = df.sort_values("datetime")
    df = df.set_index("datetime")
    return df
