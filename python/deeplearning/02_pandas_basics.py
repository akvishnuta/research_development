"""
Pandas — Tabular / labeled data wrangling.

Covers: Series, DataFrame creation, I/O, groupby, merge/join,
datetime handling, pipe-able transformations, and memory-saving dtypes.
"""

import pandas as pd
import numpy as np

# ── 1. Series ──────────────────────────────────────────────────────────────

s = pd.Series([10, 20, 30, 40], index=["a", "b", "c", "d"], name="values")
print("── Series ──")
print(s)
print(f"mean={s.mean():.1f}, std={s.std():.1f}")
print(f"b = {s['b']}, b..d =\n{s.loc['b':'d']}\n")

# ── 2. DataFrame Creation ──────────────────────────────────────────────────

df = pd.DataFrame(
    {
        "city": ["NYC", "LA", "Chicago", "Houston", "Phoenix"],
        "pop_m": [8.4, 3.8, 2.7, 2.3, 1.6],
        "area_km2": [783, 1302, 589, 1659, 1343],
        "founded": [1624, 1781, 1837, 1836, 1868],
    }
)
# Add a calculated column
df["density"] = df["pop_m"] * 1e6 / df["area_km2"]

print("── DataFrame ──")
print(df)
print(f"shape = {df.shape}\n")

# ── 3. Read / Write CSV (in-memory with StringIO) ──────────────────────────

from io import StringIO

csv_data = """name,age,score
Alice,30,95
Bob,25,87
Charlie,35,92
"""
df_csv = pd.read_csv(StringIO(csv_data))
print("── Read CSV ──")
print(df_csv)

# Write back
out = StringIO()
df_csv.to_csv(out, index=False)
print(f"CSV round-trip OK.\n")

# ── 4. Info / Describe / Memory ────────────────────────────────────────────

print("── Info ──")
df.info()
print(f"\nDescribe:\n{df.describe()}\n")

# ── 5. GroupBy ─────────────────────────────────────────────────────────────

sales = pd.DataFrame(
    {
        "region": ["East", "West", "East", "West", "East"],
        "product": ["A", "A", "B", "B", "A"],
        "revenue": [100, 150, 80, 120, 110],
    }
)
print("── GroupBy ──")
print(sales.groupby("region")["revenue"].agg(["sum", "mean", "count"]))
print()
print(sales.groupby(["region", "product"])["revenue"].sum(), "\n")

# ── 6. Merge / Join ────────────────────────────────────────────────────────

left = pd.DataFrame({"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"]})
right = pd.DataFrame({"id": [1, 2, 4], "score": [95, 87, 70]})

print("── Merge ──")
print(pd.merge(left, right, on="id", how="inner"))
print(pd.merge(left, right, on="id", how="left"), "\n")

# ── 7. Datetime ────────────────────────────────────────────────────────────

dates = pd.date_range("2024-01-01", periods=5, freq="D")
ts = pd.Series(np.random.randn(5), index=dates)
print("── Datetime ──")
print(ts)
print(f"Q1 mean: {ts.loc['2024-01-01':'2024-03-31'].mean():.3f}\n")

# ── 8. Pipe-able Transformation ────────────────────────────────────────────

def winsorize(df: pd.DataFrame, col: str, limits=(0.05, 0.05)) -> pd.DataFrame:
    """Clip outliers at given quantiles."""
    lo, hi = df[col].quantile([limits[0], 1 - limits[1]])
    df[col] = df[col].clip(lo, hi)
    return df


def zscore_scale(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df[f"{col}_z"] = (df[col] - df[col].mean()) / df[col].std()
    return df


np.random.seed(42)
raw = pd.DataFrame({"value": np.random.randn(100) * 50 + 100})
# Introduce outliers
raw.loc[0, "value"] = 9999
raw.loc[1, "value"] = -9999

clean = raw.pipe(winsorize, "value").pipe(zscore_scale, "value")
print("── Pipe ──")
print(f"Raw range: [{raw['value'].min():.0f}, {raw['value'].max():.0f}]")
print(f"Clean range: [{clean['value'].min():.0f}, {clean['value'].max():.0f}]")
print(f"Z-score range: [{clean['value_z'].min():.2f}, {clean['value_z'].max():.2f}]\n")

# ── 9. Categorical dtype (memory efficient) ────────────────────────────────

df_cat = pd.DataFrame({"color": ["red"] * 100 + ["green"] * 200 + ["blue"] * 300})
df_cat["color"] = df_cat["color"].astype("category")
print("── Categoricals ──")
print(f"memory: {df_cat.memory_usage(deep=True)['color']} bytes  vs  "
      f"{df_cat['color'].astype('object').memory_usage(deep=True)} bytes (object)")

print("\n✅ Pandas basics complete.")
