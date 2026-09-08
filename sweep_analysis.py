from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

BASE = "https://media.githubusercontent.com/media/JeffreyRStevens/ncaavolleyballr/refs/heads/main/data-csv"
YEARS = [2023, 2024, 2025]


def main() -> None:
    out = Path("sweep_output")
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for year in YEARS:
        report[str(year)] = {}
        for level in ["teammatch", "playermatch"]:
            url = f"{BASE}/wvb_{level}_div1_{year}.csv"
            df = pd.read_csv(url)
            report[str(year)][level] = {
                "url": url,
                "rows": int(len(df)),
                "columns": list(df.columns),
                "dtypes": {c: str(t) for c, t in df.dtypes.items()},
                "head": df.head(3).where(pd.notna(df.head(3)), None).to_dict(orient="records"),
            }
    (out / "schema_inspection.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
