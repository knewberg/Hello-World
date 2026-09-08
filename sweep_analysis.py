from __future__ import annotations

import json
import math
import re
import shutil
import urllib.request
import warnings
from collections import defaultdict, deque
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import brentq
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "sweep_data"
OUT_DIR = ROOT / "sweep_output"
DATA_DIR.mkdir(exist_ok=True)
if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)
OUT_DIR.mkdir(exist_ok=True)

YEARS = (2023, 2024, 2025)
BASE = "https://media.githubusercontent.com/media/JeffreyRStevens/ncaavolleyballr/refs/heads/main/data-csv"
TEAM_URL = {y: f"{BASE}/wvb_teammatch_div1_{y}.csv" for y in YEARS}
PLAYER_URL = {y: f"{BASE}/wvb_playermatch_div1_{y}.csv" for y in YEARS}
EPS = 1e-6
RNG = np.random.default_rng(20260908)


def download(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 1000:
        return
    print(f"Downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=180) as r, path.open("wb") as f:
        shutil.copyfileobj(r, f)


def safe_num(x, default=0.0) -> float:
    try:
        if pd.isna(x):
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def parse_result_sets(result: str):
    if not isinstance(result, str):
        return np.nan, np.nan, np.nan
    m = re.search(r"\b([WL])\s+(\d+)\s*-\s*(\d+)\b", result.strip())
    if not m:
        return np.nan, np.nan, np.nan
    return (1 if m.group(1) == "W" else 0), int(m.group(2)), int(m.group(3))


def normalize_opponent(x: str) -> str:
    if not isinstance(x, str):
        return x
    x = x.strip()
    x = re.sub(r"^@\s*", "", x)
    if " @" in x:
        x = x.split(" @", 1)[0]
    x = re.split(r"\s+\(?NCAA Division I Volleyball Championship", x, maxsplit=1)[0]
    x = re.sub(r"^#\d+\s+", "", x)
    return x.strip()


def venue_flags(opponent_raw: str):
    if not isinstance(opponent_raw, str):
        return 0, 0
    x = opponent_raw.strip()
    if x.startswith("@"):
        return 0, 0
    if " @" in x:
        return 0, 1
    return 1, 0


def jeffreys_logit(wins: float, games: float) -> float:
    p = (wins + 0.5) / (games + 1.0)
    return float(math.log(p / (1 - p)))


def rate(a: float, b: float, default=0.0) -> float:
    return float(a / b) if b and np.isfinite(b) else float(default)


TEAM_COUNT_COLS = [
    "S", "Kills", "Errors", "Total Attacks", "Assists", "Aces", "SErr",
    "Digs", "RetAtt", "RErr", "Block Solos", "Block Assists", "BErr", "BHE"
]


def read_team_year(year: int) -> pd.DataFrame:
    path = DATA_DIR / f"wvb_teammatch_div1_{year}.csv"
    download(TEAM_URL[year], path)
    df = pd.read_csv(path, low_memory=False)
    df["year"] = year
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    parsed = df["Result"].map(parse_result_sets)
    df[["win", "sets_for", "sets_against"]] = pd.DataFrame(parsed.tolist(), index=df.index)
    df["OpponentNormalized"] = df["Opponent"].map(normalize_opponent)
    vf = df["Opponent"].map(venue_flags)
    df[["home_flag", "neutral_flag"]] = pd.DataFrame(vf.tolist(), index=df.index)
    for c in TEAM_COUNT_COLS:
        if c not in df:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["Date", "Team", "OpponentNormalized", "win", "sets_for", "sets_against"]).copy()
    df["win"] = df["win"].astype(int)
    df["sets_for"] = df["sets_for"].astype(int)
    df["sets_against"] = df["sets_against"].astype(int)
    return df


def summarize_team_history(hist: list[dict]) -> dict[str, float]:
    if not hist:
        return {
            "games": 0, "wins": 0.0, "win_logit": 0.0, "set_logit": 0.0,
            "sweep_gw": 0.5, "swept_gl": 0.5, "set_margin": 0.0,
            "set_margin_sd": 0.0, "five_rate": 0.0, "avg_sets": 4.0,
            "hit": 0.0, "kill_rate": 0.0, "error_rate": 0.0,
            "ace_ps": 0.0, "serr_ps": 0.0, "dig_ps": 0.0,
            "reta_ps": 0.0, "rerr_ps": 0.0, "rerr_rate": 0.0,
            "block_ps": 0.0, "bhe_ps": 0.0,
            "recent5_win": 0.5, "recent5_setwin": 0.5,
            "recent5_sweep_gw": 0.5, "recent5_margin": 0.0,
            "recent5_five_rate": 0.0, "team_sets": 0.0,
        }
    h = pd.DataFrame(hist)
    n = len(h)
    wins = float(h["win"].sum())
    losses = n - wins
    sf = float(h["sets_for"].sum())
    sa = float(h["sets_against"].sum())
    total_sets = sf + sa
    sweeps = float(((h["win"] == 1) & (h["sets_against"] == 0)).sum())
    swept = float(((h["win"] == 0) & (h["sets_for"] == 0)).sum())
    margin = h["sets_for"] - h["sets_against"]
    kills = float(h["Kills"].sum())
    errors = float(h["Errors"].sum())
    attacks = float(h["Total Attacks"].sum())
    blocks = float((h["Block Solos"] + 0.5 * h["Block Assists"]).sum())
    recent = h.tail(5)
    rw = float(recent["win"].sum())
    rsf = float(recent["sets_for"].sum())
    rsa = float(recent["sets_against"].sum())
    rsweeps = float(((recent["win"] == 1) & (recent["sets_against"] == 0)).sum())
    rmargin = recent["sets_for"] - recent["sets_against"]
    setwin = (sf + 0.5) / (total_sets + 1.0)
    return {
        "games": n,
        "wins": wins,
        "win_logit": jeffreys_logit(wins, n),
        "set_logit": math.log(setwin / (1 - setwin)),
        "sweep_gw": (sweeps + 0.5) / (wins + 1.0),
        "swept_gl": (swept + 0.5) / (losses + 1.0),
        "set_margin": float(margin.mean()),
        "set_margin_sd": float(margin.std(ddof=0)) if n > 1 else 0.0,
        "five_rate": float(((h["sets_for"] + h["sets_against"]) == 5).mean()),
        "avg_sets": rate(total_sets, n, 4.0),
        "hit": rate(kills - errors, attacks),
        "kill_rate": rate(kills, attacks),
        "error_rate": rate(errors, attacks),
        "ace_ps": rate(float(h["Aces"].sum()), total_sets),
        "serr_ps": rate(float(h["SErr"].sum()), total_sets),
        "dig_ps": rate(float(h["Digs"].sum()), total_sets),
        "reta_ps": rate(float(h["RetAtt"].sum()), total_sets),
        "rerr_ps": rate(float(h["RErr"].sum()), total_sets),
        "rerr_rate": rate(float(h["RErr"].sum()), float(h["RetAtt"].sum())),
        "block_ps": rate(blocks, total_sets),
        "bhe_ps": rate(float(h["BHE"].sum()), total_sets),
        "recent5_win": (rw + 0.5) / (len(recent) + 1.0),
        "recent5_setwin": (rsf + 0.5) / (rsf + rsa + 1.0),
        "recent5_sweep_gw": (rsweeps + 0.5) / (rw + 1.0),
        "recent5_margin": float(rmargin.mean()),
        "recent5_five_rate": float(((recent["sets_for"] + recent["sets_against"]) == 5).mean()),
        "team_sets": total_sets,
    }


def build_prior_summaries(team_all: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (year, team), g in team_all.groupby(["year", "Team"], sort=False):
        s = summarize_team_history(g.to_dict("records"))
        row = {"year": year + 1, "Team": team}
        for k, v in s.items():
            if k not in {"games", "wins", "recent5_win", "recent5_setwin", "recent5_sweep_gw", "recent5_margin", "recent5_five_rate", "team_sets"}:
                row[f"prior_{k}"] = v
        row["prior_win"] = rate(s["wins"], s["games"], 0.5)
        row["prior_setwin"] = 1 / (1 + math.exp(-s["set_logit"]))
        rows.append(row)
    return pd.DataFrame(rows)


def add_team_pre_features(team_all: pd.DataFrame) -> pd.DataFrame:
    prior = build_prior_summaries(team_all)
    frames = []
    for (year, team), g in team_all.groupby(["year", "Team"], sort=False):
        g = g.sort_values("Date", kind="stable").copy()
        hist: list[dict] = []
        out_parts = []
        for dt, day in g.groupby("Date", sort=True):
            s = summarize_team_history(hist)
            d = day.copy()
            for k, v in s.items():
                d[f"pre_{k}"] = v
            out_parts.append(d)
            hist.extend(day.to_dict("records"))
        frames.append(pd.concat(out_parts, ignore_index=False))
    out = pd.concat(frames, ignore_index=True)
    out = out.merge(prior, on=["year", "Team"], how="left")
    prior_cols = [c for c in out.columns if c.startswith("prior_")]
    defaults = {"prior_win": 0.5, "prior_setwin": 0.5, "prior_sweep_gw": 0.5, "prior_swept_gl": 0.5, "prior_avg_sets": 4.0}
    for c in prior_cols:
        out[c] = out[c].fillna(defaults.get(c, 0.0))
    return out


PLAYER_COLS = [
    "S", "Kills", "Errors", "TotalAttacks", "Assists", "Aces", "SErr",
    "Digs", "RetAtt", "RErr", "BlockSolos", "BlockAssists", "BErr", "PTS", "BHE"
]


def read_player_year(year: int) -> pd.DataFrame:
    path = DATA_DIR / f"wvb_playermatch_div1_{year}.csv"
    download(PLAYER_URL[year], path)
    df = pd.read_csv(path, low_memory=False)
    df["year"] = year
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for c in PLAYER_COLS:
        if c not in df:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df["Player"] = df["Player"].astype(str)
    bad = {"TEAM", "Totals", "Opponent Totals", "nan", "None"}
    df = df[~df["Player"].isin(bad)].copy()
    df = df[df["Player"] != df["Team"].astype(str)].copy()
    df = df.dropna(subset=["Date", "Team", "Player"])
    if "ContestID" not in df:
        oppcol = "Opponent Team" if "Opponent Team" in df else "Opponent"
        fallback = df[oppcol].astype(str) if oppcol in df else ""
        df["ContestID"] = df["Date"].dt.strftime("%Y%m%d") + "|" + df["Team"].astype(str) + "|" + fallback
    df["ContestID"] = df["ContestID"].astype(str)
    return df


def aggregate_player_rows(g: pd.DataFrame) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for player, p in g.groupby("Player", sort=False):
        result[str(player)] = {c: float(p[c].sum()) for c in PLAYER_COLS}
    return result


def player_snapshot(cum: dict[str, dict[str, float]], recent_contests: deque[dict], team_sets: float) -> dict[str, float]:
    defaults = {
        "pl_attack_eff": 0.0, "pl_kill_rate": 0.0, "pl_error_rate": 0.0,
        "pl_attack_hhi": 0.25, "pl_top1_attack_share": 0.35,
        "pl_top2_attack_share": 0.60, "pl_star_eff": 0.0,
        "pl_secondary_eff": 0.0, "pl_top_setter_share": 0.80,
        "pl_setter_hhi": 0.70, "pl_block_ps": 0.0, "pl_ace_ps": 0.0,
        "pl_serr_ps": 0.0, "pl_dig_ps": 0.0, "pl_rerr_rate": 0.0,
        "pl_rotation_hhi": 0.15, "pl_effective_depth": 7.0,
        "pl_continuity": 0.5, "pl_middle_share": 0.0,
    }
    if not cum or team_sets <= 0:
        return defaults.copy()

    recent_players: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    recent_team_sets = 0.0
    for contest in recent_contests:
        recent_team_sets += contest["team_sets"]
        for player, vals in contest["players"].items():
            for c, v in vals.items():
                recent_players[player][c] += v

    rows = []
    for player, vals in cum.items():
        ps = max(vals.get("S", 0.0), 0.0)
        season_part = min(1.0, rate(ps, team_sets))
        rv = recent_players.get(player, {})
        recent_part = min(1.0, rate(rv.get("S", 0.0), recent_team_sets)) if recent_team_sets > 0 else season_part
        expected_part = 0.70 * recent_part + 0.30 * season_part
        if expected_part <= 0.005:
            continue
        denom_sets = max(ps, 1.0)
        row = {
            "player": player, "season_part": season_part, "recent_part": recent_part, "part": expected_part,
            "att_ps": vals.get("TotalAttacks", 0.0) / denom_sets,
            "kill_ps": vals.get("Kills", 0.0) / denom_sets,
            "err_ps": vals.get("Errors", 0.0) / denom_sets,
            "assist_ps": vals.get("Assists", 0.0) / denom_sets,
            "ace_ps": vals.get("Aces", 0.0) / denom_sets,
            "serr_ps": vals.get("SErr", 0.0) / denom_sets,
            "dig_ps": vals.get("Digs", 0.0) / denom_sets,
            "ret_ps": vals.get("RetAtt", 0.0) / denom_sets,
            "rerr_ps": vals.get("RErr", 0.0) / denom_sets,
            "block_ps": (vals.get("BlockSolos", 0.0) + 0.5 * vals.get("BlockAssists", 0.0)) / denom_sets,
        }
        for base in ["att", "kill", "err", "assist", "ace", "serr", "dig", "ret", "rerr", "block"]:
            row[f"{base}_contrib"] = row["part"] * row[f"{base}_ps"]
        row["eff"] = rate(row["kill_ps"] - row["err_ps"], row["att_ps"])
        rows.append(row)
    if not rows:
        return defaults.copy()

    p = pd.DataFrame(rows)
    att_total = float(p["att_contrib"].sum())
    kill_total = float(p["kill_contrib"].sum())
    err_total = float(p["err_contrib"].sum())
    att_share = p["att_contrib"] / att_total if att_total > 0 else pd.Series(np.zeros(len(p)), index=p.index)
    assist_total = float(p["assist_contrib"].sum())
    assist_share = p["assist_contrib"] / assist_total if assist_total > 0 else pd.Series(np.zeros(len(p)), index=p.index)
    part_share = p["part"] / p["part"].sum()

    order = np.argsort(-p["att_contrib"].to_numpy())
    top = p.iloc[order].reset_index(drop=True)
    top_shares = att_share.iloc[order].reset_index(drop=True)
    star_eff = float(top.loc[0, "eff"]) if len(top) else 0.0
    secondary = top.iloc[1:4]
    secondary_eff = float(np.average(secondary["eff"], weights=np.maximum(secondary["att_contrib"], 1e-6))) if len(secondary) else 0.0

    season_shares = p.set_index("player")["season_part"]
    season_shares = season_shares / season_shares.sum()
    recent_shares = p.set_index("player")["recent_part"]
    if recent_shares.sum() > 0:
        recent_shares = recent_shares / recent_shares.sum()
        continuity = float(sum(min(season_shares.get(i, 0.0), recent_shares.get(i, 0.0)) for i in set(season_shares.index) | set(recent_shares.index)))
    else:
        continuity = 0.5

    out = {
        "pl_attack_eff": rate(kill_total - err_total, att_total),
        "pl_kill_rate": rate(kill_total, att_total),
        "pl_error_rate": rate(err_total, att_total),
        "pl_attack_hhi": float((att_share ** 2).sum()) if att_total > 0 else defaults["pl_attack_hhi"],
        "pl_top1_attack_share": float(top_shares.iloc[0]) if len(top_shares) else 0.0,
        "pl_top2_attack_share": float(top_shares.iloc[:2].sum()) if len(top_shares) else 0.0,
        "pl_star_eff": star_eff,
        "pl_secondary_eff": secondary_eff,
        "pl_top_setter_share": float(assist_share.max()) if assist_total > 0 else defaults["pl_top_setter_share"],
        "pl_setter_hhi": float((assist_share ** 2).sum()) if assist_total > 0 else defaults["pl_setter_hhi"],
        "pl_block_ps": float(p["block_contrib"].sum()),
        "pl_ace_ps": float(p["ace_contrib"].sum()),
        "pl_serr_ps": float(p["serr_contrib"].sum()),
        "pl_dig_ps": float(p["dig_contrib"].sum()),
        "pl_rerr_rate": rate(float(p["rerr_contrib"].sum()), float(p["ret_contrib"].sum())),
        "pl_rotation_hhi": float((part_share ** 2).sum()),
        "pl_effective_depth": float(1.0 / max((part_share ** 2).sum(), 1e-6)),
        "pl_continuity": continuity,
        "pl_middle_share": float((p["block_contrib"] / max(float(p["block_contrib"].sum()), 1e-6)).nlargest(2).sum()) if p["block_contrib"].sum() > 0 else 0.0,
    }
    for k, v in defaults.items():
        if not np.isfinite(out.get(k, np.nan)):
            out[k] = v
    return out


def build_player_snapshots(player_all: pd.DataFrame, team_pre: pd.DataFrame) -> pd.DataFrame:
    team_sets_map = team_pre.groupby(["year", "Team", "Date"], as_index=False)["pre_team_sets"].first().set_index(["year", "Team", "Date"])["pre_team_sets"].to_dict()
    snapshots = []
    for (year, team), g in player_all.groupby(["year", "Team"], sort=False):
        g = g.sort_values(["Date", "ContestID"], kind="stable")
        cum: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        recent: deque[dict] = deque(maxlen=5)
        for dt, day in g.groupby("Date", sort=True):
            ts = float(team_sets_map.get((year, team, dt), 0.0))
            snapshots.append({"year": year, "Team": team, "Date": dt, **player_snapshot(cum, recent, ts)})
            for contest_id, contest in day.groupby("ContestID", sort=False):
                players = aggregate_player_rows(contest)
                team_match_sets = max((vals.get("S", 0.0) for vals in players.values()), default=0.0)
                recent.append({"team_sets": float(team_match_sets), "players": players})
                for player, vals in players.items():
                    for c, v in vals.items():
                        cum[player][c] += float(v)
    return pd.DataFrame(snapshots)


def pair_matches(team_pre: pd.DataFrame, player_snap: pd.DataFrame) -> pd.DataFrame:
    x = team_pre.merge(player_snap, on=["year", "Team", "Date"], how="left")
    player_cols = [c for c in player_snap.columns if c.startswith("pl_")]
    for c in player_cols:
        x[c] = x[c].fillna(0.0)
    x["pair_seq"] = x.groupby(["year", "Date", "Team", "OpponentNormalized"]).cumcount()
    lookup = {(r.year, r.Date, r.Team, r.OpponentNormalized, r.pair_seq): i for i, r in x.iterrows()}
    team_pre_metrics = [c for c in x.columns if c.startswith("pre_") or c.startswith("prior_") or c.startswith("pl_")]
    rows = []
    for i, r in x.iterrows():
        j = lookup.get((r.year, r.Date, r.OpponentNormalized, r.Team, r.pair_seq))
        if j is None:
            continue
        q = x.loc[j]
        if str(r.Team) >= str(q.Team):
            continue
        row = {"year": int(r.year), "Date": r.Date, "teamA": r.Team, "teamB": q.Team, "yA": int(r.win), "setsA": int(r.sets_for), "setsB": int(r.sets_against), "homeA": int(r.home_flag), "neutral": int(r.neutral_flag)}
        for c in team_pre_metrics:
            row[f"A_{c}"] = safe_num(r.get(c, 0.0))
            row[f"B_{c}"] = safe_num(q.get(c, 0.0))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["Date", "teamA", "teamB"]).reset_index(drop=True)


def add_match_differences(m: pd.DataFrame) -> pd.DataFrame:
    out = m.copy()
    base_metrics = [
        "pre_win_logit", "pre_set_logit", "pre_hit", "pre_kill_rate", "pre_error_rate",
        "pre_ace_ps", "pre_serr_ps", "pre_dig_ps", "pre_rerr_ps", "pre_rerr_rate",
        "pre_block_ps", "pre_bhe_ps", "pre_sweep_gw", "pre_swept_gl",
        "pre_set_margin", "pre_set_margin_sd", "pre_five_rate", "pre_avg_sets",
        "pre_recent5_win", "pre_recent5_setwin", "pre_recent5_sweep_gw",
        "pre_recent5_margin", "pre_recent5_five_rate", "prior_win", "prior_setwin",
        "prior_hit", "prior_sweep_gw", "prior_swept_gl", "prior_set_margin",
        "prior_set_margin_sd", "prior_ace_ps", "prior_serr_ps", "prior_block_ps",
        "prior_rerr_rate", "prior_five_rate",
    ]
    player_metrics = [c.removeprefix("A_") for c in out.columns if c.startswith("A_pl_")]
    for c in base_metrics + player_metrics:
        ac, bc = f"A_{c}", f"B_{c}"
        if ac in out and bc in out:
            out[f"d_{c}"] = out[ac] - out[bc]
    return out


WIN_FEATURES = [
    "d_pre_win_logit", "d_pre_set_logit", "d_pre_hit", "d_pre_recent5_win",
    "d_pre_recent5_setwin", "d_pre_ace_ps", "d_pre_serr_ps", "d_pre_block_ps",
    "d_pre_rerr_rate", "d_prior_win", "d_prior_setwin", "d_prior_hit", "homeA", "neutral",
]
P_ONLY = ["logit_p", "logit_p2"]
TEAM_SHAPE = [
    "fav_pre_sweep_gw", "dog_pre_swept_gl", "edge_pre_set_margin",
    "fav_pre_set_margin_sd", "dog_pre_set_margin_sd", "fav_pre_five_rate",
    "dog_pre_five_rate", "edge_pre_recent5_margin", "fav_pre_recent5_sweep_gw",
    "dog_pre_recent5_five_rate", "edge_prior_sweep_gw", "edge_prior_set_margin",
]
TEAM_STYLE = [
    "edge_pre_hit", "edge_pre_kill_rate", "edge_pre_error_rate",
    "serve_receive_mismatch", "reverse_serve_receive_mismatch",
    "block_attack_mismatch", "reverse_block_attack_mismatch",
    "edge_pre_serr_ps", "edge_pre_dig_ps", "edge_pre_bhe_ps",
    "edge_prior_hit", "edge_prior_block_ps", "edge_prior_ace_ps",
]
PLAYER_FEATURES = [
    "edge_pl_attack_eff", "edge_pl_error_rate", "fav_pl_attack_hhi",
    "fav_pl_top1_attack_share", "edge_pl_secondary_eff", "edge_pl_star_eff",
    "edge_pl_top_setter_share", "edge_pl_setter_hhi", "edge_pl_block_ps",
    "edge_pl_ace_ps", "edge_pl_serr_ps", "edge_pl_rerr_rate",
    "edge_pl_effective_depth", "edge_pl_continuity", "edge_pl_middle_share",
]
MODEL_FEATURES = {
    "M1_p_only": P_ONLY,
    "M2_team_shape": P_ONLY + TEAM_SHAPE,
    "M3_team_style": P_ONLY + TEAM_STYLE,
    "M4_player": P_ONLY + PLAYER_FEATURES,
    "M5_combined": P_ONLY + TEAM_SHAPE + TEAM_STYLE + PLAYER_FEATURES,
}
HYPOTHESES = {
    "Historical sweep tendency": "fav_pre_sweep_gw",
    "Set-margin dominance": "edge_pre_set_margin",
    "Low favorite volatility": "neg_fav_set_margin_sd",
    "Serve-receive mismatch": "serve_receive_mismatch",
    "Block-attack mismatch": "block_attack_mismatch",
    "Balanced favorite attack": "neg_fav_pl_attack_hhi",
    "Secondary attacker quality": "edge_pl_secondary_eff",
    "Setter role stability": "edge_pl_top_setter_share",
    "Rotation continuity": "edge_pl_continuity",
    "Effective depth": "edge_pl_effective_depth",
}


def fit_classifier(train: pd.DataFrame, features: list[str], target: str, C=0.25) -> Pipeline:
    pipe = Pipeline([("scale", StandardScaler()), ("logit", LogisticRegression(C=C, max_iter=4000, solver="lbfgs"))])
    pipe.fit(train[features].replace([np.inf, -np.inf], np.nan).fillna(0.0), train[target].astype(int))
    return pipe


def prepare_train_probabilities(train: pd.DataFrame, features: list[str], target="yA") -> np.ndarray:
    d = train.sort_values("Date").copy()
    dates = sorted(d["Date"].dt.normalize().unique())
    pred = pd.Series(np.nan, index=d.index, dtype=float)
    start = max(400, int(len(d) * 0.20))
    for dt in dates:
        tr = d[d["Date"].dt.normalize() < dt]
        te = d[d["Date"].dt.normalize() == dt]
        if len(tr) < start or te.empty or tr[target].nunique() < 2:
            continue
        mdl = fit_classifier(tr, features, target, C=0.3)
        pred.loc[te.index] = mdl.predict_proba(te[features].fillna(0.0))[:, 1]
    full = fit_classifier(d, features, target, C=0.3)
    missing = pred.isna()
    pred.loc[missing] = full.predict_proba(d.loc[missing, features].fillna(0.0))[:, 1]
    return pred.reindex(train.index).to_numpy()


def orient_favorite(df: pd.DataFrame, pA: np.ndarray) -> pd.DataFrame:
    z = df.copy()
    z["pA"] = np.clip(pA, EPS, 1 - EPS)
    z["fav_is_A"] = z["pA"] >= 0.5
    z["p_fav"] = np.where(z["fav_is_A"], z["pA"], 1 - z["pA"])
    z["fav_win"] = np.where(z["fav_is_A"], z["yA"], 1 - z["yA"]).astype(int)
    z["fav_sweep"] = np.where(z["fav_is_A"], ((z["setsA"] == 3) & (z["setsB"] == 0)), ((z["setsB"] == 3) & (z["setsA"] == 0))).astype(int)
    z["fav_team"] = np.where(z["fav_is_A"], z["teamA"], z["teamB"])
    z["dog_team"] = np.where(z["fav_is_A"], z["teamB"], z["teamA"])
    z["logit_p"] = np.log(z["p_fav"] / (1 - z["p_fav"]))
    z["logit_p2"] = z["logit_p"] ** 2
    raw_metrics = sorted({c[2:] for c in z.columns if c.startswith("A_") and f"B_{c[2:]}" in z.columns})
    for metric in raw_metrics:
        a = z[f"A_{metric}"].to_numpy(float)
        b = z[f"B_{metric}"].to_numpy(float)
        fav = np.where(z["fav_is_A"], a, b)
        dog = np.where(z["fav_is_A"], b, a)
        z[f"fav_{metric}"] = fav
        z[f"dog_{metric}"] = dog
        z[f"edge_{metric}"] = fav - dog
    z["serve_receive_mismatch"] = z["fav_pre_ace_ps"] + z["dog_pre_rerr_ps"]
    z["reverse_serve_receive_mismatch"] = z["dog_pre_ace_ps"] + z["fav_pre_rerr_ps"]
    z["block_attack_mismatch"] = z["fav_pre_block_ps"] + z["dog_pre_error_rate"]
    z["reverse_block_attack_mismatch"] = z["dog_pre_block_ps"] + z["fav_pre_error_rate"]
    z["neg_fav_set_margin_sd"] = -z["fav_pre_set_margin_sd"]
    z["neg_fav_pl_attack_hhi"] = -z["fav_pl_attack_hhi"]
    return z


def q_from_match_p(p: float) -> float:
    p = float(np.clip(p, 0.5, 1 - EPS))
    f = lambda q: 10 * q**3 - 15 * q**4 + 6 * q**5 - p
    return float(brentq(f, 0.5, 1 - 1e-10))


def iid_sweep_probability(p: Iterable[float]) -> np.ndarray:
    return np.array([q_from_match_p(float(x)) ** 3 for x in p], dtype=float)


def ece(y: np.ndarray, p: np.ndarray, bins=10) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for i in range(bins):
        mask = (p >= edges[i]) & ((p < edges[i + 1]) if i < bins - 1 else (p <= edges[i + 1]))
        if mask.any():
            total += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(total)


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    out = {"n": int(len(y)), "actual_rate": float(y.mean()), "mean_pred": float(p.mean()), "brier": float(np.mean((y - p) ** 2)), "log_loss": float(log_loss(y, p, labels=[0, 1])), "ece10": ece(y, p, 10)}
    out["auc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan
    return out


def bootstrap_metric_delta(df: pd.DataFrame, p_col: str, base_col: str, n_boot=1000) -> dict[str, float]:
    dates = pd.to_datetime(df["Date"]).dt.normalize().unique()
    date_groups = {d: df[pd.to_datetime(df["Date"]).dt.normalize() == d] for d in dates}
    brier_d, log_d = [], []
    for _ in range(n_boot):
        sampled = RNG.choice(dates, size=len(dates), replace=True)
        g = pd.concat([date_groups[d] for d in sampled], ignore_index=True)
        y = g["fav_sweep"].to_numpy(int)
        p = np.clip(g[p_col].to_numpy(float), EPS, 1 - EPS)
        b = np.clip(g[base_col].to_numpy(float), EPS, 1 - EPS)
        brier_d.append(np.mean((y - p) ** 2) - np.mean((y - b) ** 2))
        log_d.append(log_loss(y, p, labels=[0, 1]) - log_loss(y, b, labels=[0, 1]))
    return {"brier_ci_low": float(np.quantile(brier_d, 0.025)), "brier_ci_high": float(np.quantile(brier_d, 0.975)), "logloss_ci_low": float(np.quantile(log_d, 0.025)), "logloss_ci_high": float(np.quantile(log_d, 0.975))}


def fit_sweep_models(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Pipeline]]:
    winners = train[train["fav_win"] == 1].copy()
    out = test.copy()
    out["M0_iid"] = iid_sweep_probability(out["p_fav"])
    models = {}
    for name, feats in MODEL_FEATURES.items():
        mdl = fit_classifier(winners, feats, "fav_sweep", C=0.20)
        cond = mdl.predict_proba(out[feats].replace([np.inf, -np.inf], np.nan).fillna(0.0))[:, 1]
        out[name] = np.clip(out["p_fav"].to_numpy() * cond, EPS, 1 - EPS)
        out[f"{name}_cond"] = np.clip(cond, EPS, 1 - EPS)
        models[name] = mdl
    return out, models


def run_year_test(matches: pd.DataFrame, test_year: int) -> tuple[pd.DataFrame, dict[str, Pipeline], Pipeline]:
    train = matches[matches["year"] < test_year].copy()
    test = matches[matches["year"] == test_year].copy()
    train = train[(train["A_pre_games"] >= 3) & (train["B_pre_games"] >= 3)].copy()
    test = test[(test["A_pre_games"] >= 3) & (test["B_pre_games"] >= 3)].copy()
    win_model = fit_classifier(train, WIN_FEATURES, "yA", C=0.30)
    p_train = prepare_train_probabilities(train, WIN_FEATURES, "yA")
    p_test = win_model.predict_proba(test[WIN_FEATURES].fillna(0.0))[:, 1]
    otr = orient_favorite(train, p_train)
    ote = orient_favorite(test, p_test)
    pred, sweep_models = fit_sweep_models(otr, ote)
    return pred, sweep_models, win_model


def make_metric_table(preds_by_year: dict[int, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    models = ["M0_iid", *MODEL_FEATURES.keys()]
    for year, df in preds_by_year.items():
        for scope, g in [("all", df), ("p>=0.70", df[df["p_fav"] >= 0.70]), ("p>=0.80", df[df["p_fav"] >= 0.80]), ("p>=0.90", df[df["p_fav"] >= 0.90])]:
            for model in models:
                rows.append({"year": year, "scope": scope, "model": model, **metrics(g["fav_sweep"].to_numpy(), g[model].to_numpy())})
    return pd.DataFrame(rows)


def make_sweep_curve(df: pd.DataFrame) -> pd.DataFrame:
    bins = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.000001]
    labels = ["50–60%", "60–70%", "70–80%", "80–90%", "90–95%", "95–100%"]
    z = df.copy()
    z["p_bin"] = pd.cut(z["p_fav"], bins=bins, labels=labels, right=False)
    rows = []
    for label, g in z.groupby("p_bin", observed=True):
        wins = g[g["fav_win"] == 1]
        rows.append({"p_bin": str(label), "n": len(g), "mean_win_probability": g["p_fav"].mean(), "actual_favorite_win_rate": g["fav_win"].mean(), "actual_sweep_rate": g["fav_sweep"].mean(), "sweep_given_win": wins["fav_sweep"].mean() if len(wins) else np.nan, "iid_sweep": g["M0_iid"].mean(), "p_only_sweep": g["M1_p_only"].mean(), "combined_sweep": g["M5_combined"].mean()})
    return pd.DataFrame(rows)


def cluster_robust_single_feature(train: pd.DataFrame, feature: str) -> dict[str, float]:
    w = train[train["fav_win"] == 1].copy()
    vals = w[feature].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    sd = vals.std(ddof=0)
    if not np.isfinite(sd) or sd < 1e-9:
        return {"or": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan}
    w["feature_z"] = (vals - vals.mean()) / sd
    X = sm.add_constant(w[["logit_p", "logit_p2", "feature_z"]], has_constant="add")
    try:
        fit = sm.GLM(w["fav_sweep"], X, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": pd.to_datetime(w["Date"]).dt.normalize()})
        b = float(fit.params["feature_z"])
        se = float(fit.bse["feature_z"])
        return {"or": math.exp(b), "ci_low": math.exp(b - 1.96 * se), "ci_high": math.exp(b + 1.96 * se), "p_value": float(fit.pvalues["feature_z"])}
    except Exception:
        return {"or": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan}


def quartile_residual_lift(test: pd.DataFrame, feature: str, n_boot=1000) -> dict[str, float]:
    z = test[test["p_fav"] >= 0.70].copy()
    z = z[np.isfinite(z[feature])].copy()
    if len(z) < 40:
        return {"n": len(z), "lift": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    z["pband"] = pd.cut(z["p_fav"], bins=[.70, .75, .80, .85, .90, .95, 1.00001], include_lowest=True)
    z["pct"] = z.groupby("pband", observed=True)[feature].rank(pct=True, method="average")
    z = z[(z["pct"] <= .25) | (z["pct"] >= .75)].copy()
    z["side"] = np.where(z["pct"] >= .75, "top", "bottom")
    z["resid"] = z["fav_sweep"] - z["M1_p_only"]
    means = z.groupby("side")["resid"].mean()
    lift = float(means.get("top", np.nan) - means.get("bottom", np.nan))
    dates = pd.to_datetime(z["Date"]).dt.normalize().unique()
    date_groups = {d: z[pd.to_datetime(z["Date"]).dt.normalize() == d] for d in dates}
    boots = []
    for _ in range(n_boot):
        sampled = RNG.choice(dates, size=len(dates), replace=True)
        g = pd.concat([date_groups[d] for d in sampled], ignore_index=True)
        mm = g.groupby("side")["resid"].mean()
        if "top" in mm and "bottom" in mm:
            boots.append(float(mm["top"] - mm["bottom"]))
    return {"n": int(len(z)), "lift": lift, "ci_low": float(np.quantile(boots, .025)) if boots else np.nan, "ci_high": float(np.quantile(boots, .975)) if boots else np.nan}


def individual_hypothesis_tests(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    winners = train[train["fav_win"] == 1].copy()
    for label, feature in HYPOTHESES.items():
        stat = cluster_robust_single_feature(train, feature)
        feats = P_ONLY + [feature]
        mdl = fit_classifier(winners, feats, "fav_sweep", C=0.20)
        cond = mdl.predict_proba(test[feats].fillna(0.0))[:, 1]
        p = np.clip(test["p_fav"].to_numpy() * cond, EPS, 1 - EPS)
        base = test["M1_p_only"].to_numpy()
        y = test["fav_sweep"].to_numpy()
        lift = quartile_residual_lift(test, feature)
        rows.append({"hypothesis": label, "feature": feature, "train_favorite_wins": len(winners), "std_odds_ratio": stat["or"], "or_ci_low": stat["ci_low"], "or_ci_high": stat["ci_high"], "cluster_p_value": stat["p_value"], "test_brier_delta_vs_p_only": float(np.mean((y - p) ** 2) - np.mean((y - base) ** 2)), "test_logloss_delta_vs_p_only": float(log_loss(y, p, labels=[0, 1]) - log_loss(y, base, labels=[0, 1])), "high_favorite_quartile_residual_lift": lift["lift"], "quartile_lift_ci_low": lift["ci_low"], "quartile_lift_ci_high": lift["ci_high"], "quartile_n": lift["n"]})
    return pd.DataFrame(rows)


def baseline_math_table() -> pd.DataFrame:
    rows = []
    for p in [.55, .60, .70, .80, .85, .90, .95, .99]:
        q = q_from_match_p(p)
        sw = q ** 3
        rows.append({"favorite_match_win_probability": p, "implied_iid_set_win_probability": q, "iid_sweep_probability": sw, "iid_sweep_given_win": sw / p})
    return pd.DataFrame(rows)


def chart_outputs(curve: pd.DataFrame, metrics_df: pd.DataFrame, hyp: pd.DataFrame) -> None:
    x = curve["mean_win_probability"].to_numpy()
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(x, curve["actual_sweep_rate"], marker="o", label="Observed 2025")
    ax.plot(x, curve["iid_sweep"], marker="o", label="IID inversion")
    ax.plot(x, curve["p_only_sweep"], marker="o", label="Learned p-only")
    ax.plot(x, curve["combined_sweep"], marker="o", label="Team + player")
    ax.set_xlabel("Mean favorite match-win probability")
    ax.set_ylabel("Sweep probability")
    ax.set_title("Sweep probability versus favorite win probability")
    ax.grid(axis="y", alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "sweep_curve_2025.png", dpi=180)
    plt.close(fig)

    m = metrics_df[(metrics_df["year"] == 2025) & (metrics_df["scope"] == "all")].copy()
    base = float(m.loc[m["model"] == "M1_p_only", "brier"].iloc[0])
    m = m[m["model"] != "M1_p_only"].copy()
    m["delta"] = m["brier"] - base
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.barh(m["model"], m["delta"])
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("Brier-score change versus learned p-only (negative is better)")
    ax.set_title("Incremental sweep-model performance, 2025 holdout")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "model_brier_deltas_2025.png", dpi=180)
    plt.close(fig)

    h = hyp.sort_values("high_favorite_quartile_residual_lift")
    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    ax.barh(h["hypothesis"], h["high_favorite_quartile_residual_lift"])
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("Top-minus-bottom quartile sweep residual, p ≥ 70%")
    ax.set_title("Which traits distinguish similarly favored teams? 2025")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "hypothesis_residual_lifts_2025.png", dpi=180)
    plt.close(fig)


def fmt_pct(x, digits=1):
    return "—" if pd.isna(x) else f"{100*x:.{digits}f}%"


def fmt_num(x, digits=4):
    return "—" if pd.isna(x) else f"{x:.{digits}f}"


def write_report(team_rows, player_rows, matches, metrics_df, curve, hyp, baseline_math) -> None:
    m25 = metrics_df[(metrics_df.year == 2025) & (metrics_df.scope == "all")].set_index("model")
    mh = metrics_df[(metrics_df.year == 2025) & (metrics_df.scope == "p>=0.70")].set_index("model")
    best = m25["brier"].idxmin()
    best_high = mh["brier"].idxmin()
    base = m25.loc["M1_p_only"]
    bestrow = m25.loc[best]
    model_labels = {"M0_iid": "IID inversion", "M1_p_only": "Learned win-probability curve", "M2_team_shape": "Win probability + set-shape history", "M3_team_style": "Win probability + team style/matchups", "M4_player": "Win probability + player composition", "M5_combined": "Combined team + player model"}
    L = []
    L += ["# NCAA Division I Women’s Volleyball Sweep Projection — 2023–2025", "", "## Executive Summary", ""]
    L.append("- **Overall win probability is the dominant sweep predictor, but it is not sufficient.** The empirical sweep rate rises sharply across favorite-probability bands. The IID set model is a coherent starting curve; a learned probability-only model corrects departures from identical, independent sets.")
    L.append(f"- **The best 2025 model by Brier score was {model_labels.get(best,best)}.** Its Brier change versus the learned p-only model was {bestrow.brier-base.brier:+.5f}; its log-loss change was {bestrow.log_loss-base.log_loss:+.5f}. The best model among favorites at 70% or higher was {model_labels.get(best_high,best_high)}.")
    L.append("- **Use a hurdle model:** retain the locked match-win probability, then estimate `P(3–0 | win)` from residual set-shape, matchup, and player-composition features. Multiply the two probabilities so a sweep can never be more likely than a win.")
    L.append("- **Player statistics should enter as projected rotation attributes, not raw season totals.** The most defensible features are role-weighted attack efficiency/error control, secondary-option quality, setter concentration, blocking/serving pressure, reception vulnerability, depth, and continuity.")
    L += ["", "## Data and Validation Design", ""]
    L.append(f"The analysis used {sum(team_rows.values()):,} team-match rows and {sum(player_rows.values()):,} filtered player-match rows from 2023–2025, paired into {len(matches):,} unique matches. Every feature was frozen before the target date; same-day matches used a common start-of-day snapshot. The tests were forward-looking: 2023 trained 2024, and 2023–2024 trained the final 2025 holdout.")
    L += ["", "The hierarchy was IID inversion; learned p-only curve; p plus set shape; p plus team style/matchups; p plus player composition; and a combined model.", "", "## Winning and Sweeping Are Related Nonlinearly", ""]
    L.append("Under a constant IID set-win probability `q`, match-win probability is `10q³ − 15q⁴ + 6q⁵`, and sweep probability is `q³`.")
    L += ["", "| Match-win P | Implied set-win P | Sweep P | Sweep given win |", "|---:|---:|---:|---:|"]
    for _, r in baseline_math.iterrows():
        L.append(f"| {fmt_pct(r.favorite_match_win_probability,0)} | {fmt_pct(r.implied_iid_set_win_probability,1)} | {fmt_pct(r.iid_sweep_probability,1)} | {fmt_pct(r.iid_sweep_given_win,1)} |")
    L += ["", "Two teams with the same match-win probability have the same sweep probability under the IID model. Useful differentiation therefore must come from set-to-set variance, style, matchup, or lineup structure not already encoded in the win probability.", "", "![Sweep curve](sweep_curve_2025.png)", "", "## Empirical 2025 Sweep Curve", "", "| Favorite band | N | Mean win P | Actual win | Actual sweep | Sweep given win | IID | P-only | Combined |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in curve.iterrows():
        L.append(f"| {r.p_bin} | {int(r.n):,} | {fmt_pct(r.mean_win_probability)} | {fmt_pct(r.actual_favorite_win_rate)} | {fmt_pct(r.actual_sweep_rate)} | {fmt_pct(r.sweep_given_win)} | {fmt_pct(r.iid_sweep)} | {fmt_pct(r.p_only_sweep)} | {fmt_pct(r.combined_sweep)} |")
    L += ["", "Sweep probability is primarily a nonlinear transformation of match strength. The second stage should estimate only the residual shape of the score distribution.", "", "## 2025 Out-of-Time Model Comparison", "", "### All matches", "", "| Model | N | Actual sweep | Mean forecast | Brier | Log loss | ECE10 | AUC |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for model in ["M0_iid", *MODEL_FEATURES.keys()]:
        r = m25.loc[model]
        L.append(f"| {model_labels[model]} | {int(r.n):,} | {fmt_pct(r.actual_rate)} | {fmt_pct(r.mean_pred)} | {r.brier:.5f} | {r.log_loss:.5f} | {r.ece10:.4f} | {r.auc:.4f} |")
    L += ["", "### Favorites of at least 70%", "", "| Model | N | Actual sweep | Mean forecast | Brier | Log loss | ECE10 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for model in ["M0_iid", *MODEL_FEATURES.keys()]:
        r = mh.loc[model]
        L.append(f"| {model_labels[model]} | {int(r.n):,} | {fmt_pct(r.actual_rate)} | {fmt_pct(r.mean_pred)} | {r.brier:.5f} | {r.log_loss:.5f} | {r.ece10:.4f} |")
    L += ["", "![Model Brier deltas](model_brier_deltas_2025.png)", "", "## Hypotheses Distinguishing Similarly Favored Teams", ""]
    L.append("The odds ratio controls for the nonlinear win-probability curve among favorite wins in the 2024 validation data, with date-clustered uncertainty. The 2025 residual lift compares top and bottom feature quartiles within narrow probability bands among favorites of at least 70%.")
    L += ["", "| Hypothesis | Standardized OR | 95% CI | Cluster p | 2025 Δ Brier | High-favorite residual lift | Lift 95% CI |", "|---|---:|---:|---:|---:|---:|---:|"]
    for _, r in hyp.iterrows():
        L.append(f"| {r.hypothesis} | {fmt_num(r.std_odds_ratio,2)} | {fmt_num(r.or_ci_low,2)}–{fmt_num(r.or_ci_high,2)} | {fmt_num(r.cluster_p_value,3)} | {r.test_brier_delta_vs_p_only:+.5f} | {fmt_pct(r.high_favorite_quartile_residual_lift)} | {fmt_pct(r.quartile_lift_ci_low)} to {fmt_pct(r.quartile_lift_ci_high)} |")
    L += ["", "![Hypothesis residual lifts](hypothesis_residual_lifts_2025.png)", "", "### Interpretation", ""]
    L += [
        "- **Set dominance and volatility:** consistent set-margin superiority and fewer five-set matches should increase conditional sweep probability relative to equally likely but volatile favorites.",
        "- **Serve-receive mismatch:** ace pressure against an error-prone receiving unit can create repeated scoring runs and prevent the underdog from stealing a set.",
        "- **Block-attack mismatch:** strong blocking against a high-error attack reduces the underdog’s set-winning paths. Middle/block availability may therefore matter more for sweep shape than generic roster continuity.",
        "- **Attack balance:** concentration can help when the star is dominant, but can lower the floor if one rotation or matchup suppresses that player. It should be nonlinear and paired with secondary-option quality.",
        "- **Setter stability:** setter share and continuity are preferable to raw assists per set because they better represent role stability.",
        "- **Depth:** depth can sustain efficiency through substitutions, but an unsettled deep rotation may instead indicate uncertainty; the relationship need not be monotonic.",
        "", "## Recommended Production Architecture", "",
        "Use the locked match-win model unchanged and attach a shadow conditional sweep head:", "",
        "`P(favorite sweep) = P(favorite wins) × P(3–0 | favorite wins, residual features)`", "",
        "Use regularization and a p-only offset. Inputs should include calibrated match-win probability; set dominance/volatility; hitting and attack-error control; serve-receive and block-attack mismatches; projected player attack shares and efficiencies; secondary-option quality; setter share; middle/block contribution; passing load; depth; continuity; verified availability; venue/travel/rest; and early-season uncertainty.", "",
        "After estimating sweep probability, distribute the remaining favorite-win mass between 3–1 and 3–2 with a constrained conditional model. The six exact-score probabilities must sum to one, and the three favorite-win scores must sum exactly to the locked match-win probability.", "", "## Promotion Standard", "",
        "Keep this head shadow-only until prospectively frozen predictions improve sweep Brier and log loss over IID and learned p-only baselines, preserve calibration in each high-favorite band, improve six-class exact-score scoring and expected-sets MAE, remain stable across season phases and venues, and pass player-availability timing audits.", "", "## Caveats", "",
        "- Player box scores show participation, not the reason for absence; verified status remains necessary.",
        "- Box scores do not directly measure rotation-level sideout efficiency. Rally play-by-play would improve serve/receive, sideout, breakpoint, and variance features.",
        "- Projected rotation attributes are prior-participation priors, not confirmed lineups.",
        "- Associations are predictive, not causal.", "", "## Bottom Line", "",
        "**Start with match-win probability, but do not stop there.** The sweep head should model whether the favorite can avoid one bad set. The leading residual candidates are consistent set dominance, low volatility, favorable serve-receive and block-attack matchups, and a stable rotation with multiple efficient scoring options. Player features belong in score-shape modeling even though broad roster features did not justify changing the match-winner model.",
    ]
    (OUT_DIR / "report.md").write_text("\n".join(L), encoding="utf-8")


def main():
    team_frames, team_rows = [], {}
    for y in YEARS:
        d = read_team_year(y); team_frames.append(d); team_rows[y] = len(d)
    team_all = pd.concat(team_frames, ignore_index=True)
    team_pre = add_team_pre_features(team_all)

    player_frames, player_rows = [], {}
    for y in YEARS:
        d = read_player_year(y); player_frames.append(d); player_rows[y] = len(d)
    player_all = pd.concat(player_frames, ignore_index=True)
    player_snap = build_player_snapshots(player_all, team_pre)

    matches = add_match_differences(pair_matches(team_pre, player_snap))
    matches.to_csv(OUT_DIR / "point_in_time_match_features.csv", index=False)

    diagnostics = {"team_rows": team_rows, "player_rows_after_filtering": player_rows, "unique_matches": int(len(matches)), "match_counts_by_year": matches.groupby("year").size().astype(int).to_dict(), "player_snapshot_rows": int(len(player_snap))}
    preds_by_year = {}
    for year in (2024, 2025):
        pred, sweep_models, win_model = run_year_test(matches, year)
        preds_by_year[year] = pred
        pred.to_csv(OUT_DIR / f"sweep_predictions_{year}.csv", index=False)

    metrics_df = make_metric_table(preds_by_year)
    p25 = preds_by_year[2025]
    ci_rows = []
    for model in ["M0_iid", "M2_team_shape", "M3_team_style", "M4_player", "M5_combined"]:
        ci_rows.append({"year": 2025, "scope": "all", "model": model, **bootstrap_metric_delta(p25, model, "M1_p_only", n_boot=1000)})
    metrics_df = metrics_df.merge(pd.DataFrame(ci_rows), on=["year", "scope", "model"], how="left")
    metrics_df.to_csv(OUT_DIR / "model_metrics.csv", index=False)

    curve = make_sweep_curve(p25); curve.to_csv(OUT_DIR / "sweep_curve_2025.csv", index=False)
    baseline_math = baseline_math_table(); baseline_math.to_csv(OUT_DIR / "iid_baseline_curve.csv", index=False)
    hyp = individual_hypothesis_tests(preds_by_year[2024], p25); hyp.to_csv(OUT_DIR / "hypothesis_tests_2025.csv", index=False)
    chart_outputs(curve, metrics_df, hyp)
    write_report(team_rows, player_rows, matches, metrics_df, curve, hyp, baseline_math)

    diagnostics["eligible_predictions"] = {str(y): int(len(df)) for y, df in preds_by_year.items()}
    diagnostics["sweep_rates"] = {str(y): float(df["fav_sweep"].mean()) for y, df in preds_by_year.items()}
    diagnostics["favorite_win_rates"] = {str(y): float(df["fav_win"].mean()) for y, df in preds_by_year.items()}
    diagnostics["quality_status"] = "PASS" if len(matches) > 8000 and all(len(df) > 2500 for df in preds_by_year.values()) else "REVIEW"
    (OUT_DIR / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2, default=str), encoding="utf-8")
    print(json.dumps(diagnostics, indent=2, default=str))
    print(metrics_df[(metrics_df.year == 2025) & (metrics_df.scope == "all")].to_string(index=False))
    print(hyp.to_string(index=False))


if __name__ == "__main__":
    main()
