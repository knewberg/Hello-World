from datetime import date
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sweep_head import (
    CalibrationObservation,
    CalibrationState,
    CURRENT_2026_STATE,
    RegularizedSweepCalibrator,
    SweepHeadV12,
)


def current_2026_observations():
    values = [
        ("2026-09-06", 0.8133, 0),
        ("2026-09-06", 0.6436, 1),
        ("2026-09-07", 0.9278557008613764, 1),
        ("2026-09-06", 0.8709, 0),
        ("2026-09-06", 0.7113, 0),
        ("2026-09-07", 0.5543985584445036, 1),
        ("2026-09-06", 0.7063, 0),
        ("2026-09-07", 0.899358973628886, 0),
        ("2026-09-07", 0.6793682547670311, 1),
        ("2026-09-06", 0.8205, 0),
        ("2026-09-07", 0.5104596622665691, 0),
        ("2026-09-06", 0.7706, 0),
    ]
    return [CalibrationObservation(date.fromisoformat(d), p, True, bool(s), False) for d, p, s in values]


def test_current_bridge_is_reproduced():
    state = RegularizedSweepCalibrator(5.0).fit_completed(current_2026_observations())
    assert abs(state.favorite_sweep_delta - CURRENT_2026_STATE.favorite_sweep_delta) < 1e-12
    assert state.favorite_win_observations == 12
    assert state.upset_win_observations == 0


def test_stronger_regularization_shrinks_bridge():
    obs = current_2026_observations()
    weak = RegularizedSweepCalibrator(1.0).fit_completed(obs)
    strong = RegularizedSweepCalibrator(20.0).fit_completed(obs)
    assert abs(strong.favorite_sweep_delta) < abs(weak.favorite_sweep_delta)


def test_same_day_results_are_not_used():
    obs = current_2026_observations()
    cal = RegularizedSweepCalibrator(5.0)
    assert cal.fit_before_date(obs, "2026-09-07").favorite_win_observations == 7
    assert cal.fit_before_date(obs, "2026-09-08").favorite_win_observations == 12


def test_no_upsets_means_no_underdog_update():
    state = RegularizedSweepCalibrator(5.0).fit_completed(current_2026_observations())
    assert state.underdog_sweep_delta == 0.0
    assert state.upset_win_observations == 0


def test_probability_identities_hold():
    model = SweepHeadV12(CURRENT_2026_STATE)
    for p_a in [0.01, 0.10, 0.35, 0.50, 0.70, 0.90, 0.99]:
        d = model.team_a_distribution(p_a)
        assert abs(sum(d.values()) - 1.0) < 1e-12
        assert abs(d["A_3_0"] + d["A_3_1"] + d["A_3_2"] - p_a) < 1e-12


def test_calibration_changes_score_shape_not_win_mass():
    base = SweepHeadV12(CalibrationState())
    current = SweepHeadV12(CURRENT_2026_STATE)
    p_a = 0.82
    b = base.team_a_distribution(p_a)
    c = current.team_a_distribution(p_a)
    assert abs((b["A_3_0"] + b["A_3_1"] + b["A_3_2"]) - p_a) < 1e-12
    assert abs((c["A_3_0"] + c["A_3_1"] + c["A_3_2"]) - p_a) < 1e-12
    assert c["A_3_0"] < b["A_3_0"]
