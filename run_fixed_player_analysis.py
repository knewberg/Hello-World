from collections import defaultdict, deque
from pathlib import Path
import shutil

import pandas as pd
import sweep_analysis as sa


def build_player_snapshots_fixed(player_all: pd.DataFrame, team_pre: pd.DataFrame) -> pd.DataFrame:
    """Freeze player state for every team-match date, not only dates present in player data."""
    targets = (
        team_pre[["year", "Team", "Date", "pre_team_sets"]]
        .drop_duplicates(["year", "Team", "Date"])
        .sort_values(["year", "Team", "Date"])
    )
    player_groups = {
        (int(year), str(team)): g.sort_values(["Date", "ContestID"], kind="stable")
        for (year, team), g in player_all.groupby(["year", "Team"], sort=False)
    }
    snapshots = []
    for (year, team), tg in targets.groupby(["year", "Team"], sort=False):
        g = player_groups.get((int(year), str(team)))
        by_date = {}
        if g is not None and not g.empty:
            by_date = {dt: day for dt, day in g.groupby("Date", sort=True)}
        cum = defaultdict(lambda: defaultdict(float))
        recent = deque(maxlen=5)
        for row in tg.itertuples(index=False):
            dt = row.Date
            snapshots.append({
                "year": int(year),
                "Team": team,
                "Date": dt,
                **sa.player_snapshot(cum, recent, float(row.pre_team_sets)),
            })
            day = by_date.get(dt)
            if day is None:
                continue
            for contest_id, contest in day.groupby("ContestID", sort=False):
                players = sa.aggregate_player_rows(contest)
                team_match_sets = max((vals.get("S", 0.0) for vals in players.values()), default=0.0)
                recent.append({"team_sets": float(team_match_sets), "players": players})
                for player, vals in players.items():
                    for c, v in vals.items():
                        cum[player][c] += float(v)
    return pd.DataFrame(snapshots)


sa.build_player_snapshots = build_player_snapshots_fixed
sa.OUT_DIR = Path(__file__).resolve().parent / "sweep_output_fixed"
if sa.OUT_DIR.exists():
    shutil.rmtree(sa.OUT_DIR)
sa.OUT_DIR.mkdir(exist_ok=True)
sa.main()

# Additional linkage diagnostics for the corrected output.
p24 = pd.read_csv(sa.OUT_DIR / "sweep_predictions_2024.csv")
p25 = pd.read_csv(sa.OUT_DIR / "sweep_predictions_2025.csv")
rows = []
for year, p in [(2024, p24), (2025, p25)]:
    both = (p["A_pl_effective_depth"] > 0) & (p["B_pl_effective_depth"] > 0)
    rows.append({
        "year": year,
        "n": len(p),
        "both_team_player_history_rate": float(both.mean()),
        "teamA_player_history_rate": float((p["A_pl_effective_depth"] > 0).mean()),
        "teamB_player_history_rate": float((p["B_pl_effective_depth"] > 0).mean()),
    })
pd.DataFrame(rows).to_csv(sa.OUT_DIR / "player_history_coverage.csv", index=False)
print(pd.DataFrame(rows).to_string(index=False))
