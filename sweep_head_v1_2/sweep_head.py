"""Rapid regularized set/sweep calibration for NCAA women's volleyball.

This module is downstream of the locked match-win model. It never changes
P(win). It maps that immutable probability into a six-state exact-score
distribution and permits only L2-regularized intercept updates to conditional
sweep heads during a season.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import math
from typing import Iterable, Mapping, Optional, Sequence

EPS = 1e-12


@dataclass(frozen=True)
class Head:
    intercept: float
    slope_logit_p: float


END_2025 = {
    "favorite_sweep": Head(-0.7598785573618496, 0.5239503243870701),
    "favorite_31": Head(0.33970175356622606, 0.23559725156812755),
    "underdog_sweep": Head(-0.6694051065523932, -0.2782583542069120),
    "underdog_31": Head(0.38998871025084936, -0.29360832156569616),
}


@dataclass(frozen=True)
class CalibrationObservation:
    match_date: date
    p_favorite: float
    favorite_won: bool
    favorite_swept: bool
    underdog_swept: bool = False


@dataclass(frozen=True)
class CalibrationFit:
    delta: float
    observations: int
    events: int
    penalty_lambda: float
    converged: bool
    iterations: int


@dataclass(frozen=True)
class CalibrationState:
    favorite_sweep_delta: float = 0.0
    underdog_sweep_delta: float = 0.0
    favorite_win_observations: int = 0
    upset_win_observations: int = 0
    penalty_lambda: float = 5.0
    fitted_through_date: Optional[date] = None
    formula_version: str = "sweep-head-v1.2-regularized-intercept"


CURRENT_2026_STATE = CalibrationState(
    favorite_sweep_delta=-0.20340621161358413,
    underdog_sweep_delta=0.0,
    favorite_win_observations=12,
    upset_win_observations=0,
    penalty_lambda=5.0,
    fitted_through_date=date(2026, 9, 7),
)


def _clip_probability(p: float) -> float:
    return min(max(float(p), EPS), 1.0 - EPS)


def _logit(p: float) -> float:
    p = _clip_probability(p)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _parse_date(value) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _head_probability(head: Head, p_favorite: float, intercept_delta: float = 0.0) -> float:
    return _sigmoid(head.intercept + intercept_delta + head.slope_logit_p * _logit(p_favorite))


class RegularizedSweepCalibrator:
    """Fit L2-regularized intercept bridges with strict date safety.

    For fixed historical head ``a + b*logit(p)``, estimate only ``delta`` by
    maximizing current-season conditional-sweep log likelihood minus
    ``0.5 * lambda * delta**2``. This is a one-parameter convex problem.
    """

    def __init__(self, penalty_lambda: float = 5.0, max_iterations: int = 50, tolerance: float = 1e-12):
        if penalty_lambda <= 0:
            raise ValueError("penalty_lambda must be > 0")
        self.penalty_lambda = float(penalty_lambda)
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)

    def _fit_delta(self, head: Head, p_values: Sequence[float], y_values: Sequence[int]) -> CalibrationFit:
        n = len(p_values)
        if n != len(y_values):
            raise ValueError("p_values and y_values must have the same length")
        if n == 0:
            return CalibrationFit(0.0, 0, 0, self.penalty_lambda, True, 0)

        delta = 0.0
        lam = self.penalty_lambda
        events = int(sum(int(y) for y in y_values))
        converged = False

        for iteration in range(1, self.max_iterations + 1):
            grad = -lam * delta
            information = lam
            for p, y in zip(p_values, y_values):
                eta = head.intercept + head.slope_logit_p * _logit(p) + delta
                pred = _sigmoid(eta)
                grad += int(y) - pred
                information += pred * (1.0 - pred)
            new_delta = delta + grad / max(information, EPS)
            if abs(new_delta - delta) <= self.tolerance:
                delta = new_delta
                converged = True
                break
            delta = new_delta

        return CalibrationFit(delta, n, events, lam, converged, iteration)

    def fit_before_date(self, observations: Iterable[CalibrationObservation], target_date) -> CalibrationState:
        """Fit only on dates strictly before target_date; same-day matches share state."""
        target = _parse_date(target_date)
        prior = [obs for obs in observations if _parse_date(obs.match_date) < target]
        return self.fit_completed(prior)

    def fit_completed(self, observations: Iterable[CalibrationObservation]) -> CalibrationState:
        observations = list(observations)
        fav_wins = [obs for obs in observations if obs.favorite_won]
        upsets = [obs for obs in observations if not obs.favorite_won]

        fav_fit = self._fit_delta(
            END_2025["favorite_sweep"],
            [obs.p_favorite for obs in fav_wins],
            [int(obs.favorite_swept) for obs in fav_wins],
        )
        dog_fit = self._fit_delta(
            END_2025["underdog_sweep"],
            [obs.p_favorite for obs in upsets],
            [int(obs.underdog_swept) for obs in upsets],
        )
        fitted_through = max((_parse_date(obs.match_date) for obs in observations), default=None)
        return CalibrationState(
            favorite_sweep_delta=fav_fit.delta,
            underdog_sweep_delta=dog_fit.delta,
            favorite_win_observations=fav_fit.observations,
            upset_win_observations=dog_fit.observations,
            penalty_lambda=self.penalty_lambda,
            fitted_through_date=fitted_through,
        )


class SweepHeadV12:
    """Constrained exact-score distribution using a versioned calibration state."""

    def __init__(self, calibration_state: CalibrationState = CURRENT_2026_STATE):
        self.state = calibration_state

    def favorite_oriented_distribution(self, p_favorite: float) -> dict[str, float]:
        pf = _clip_probability(p_favorite)
        if pf < 0.5:
            raise ValueError("p_favorite must be >= 0.5")

        fs = _head_probability(END_2025["favorite_sweep"], pf, self.state.favorite_sweep_delta)
        f31 = _head_probability(END_2025["favorite_31"], pf)
        us = _head_probability(END_2025["underdog_sweep"], pf, self.state.underdog_sweep_delta)
        u31 = _head_probability(END_2025["underdog_31"], pf)

        f30 = pf * fs
        f_remaining = pf - f30
        f31p = f_remaining * f31
        f32 = f_remaining - f31p

        pu = 1.0 - pf
        u30 = pu * us
        u_remaining = pu - u30
        u31p = u_remaining * u31
        u32 = u_remaining - u31p

        result = {
            "F_3_0": f30, "F_3_1": f31p, "F_3_2": f32,
            "U_3_2": u32, "U_3_1": u31p, "U_3_0": u30,
        }
        self._validate_favorite(result, pf)
        return result

    def team_a_distribution(self, team_a_win_probability: float) -> dict[str, float]:
        p_a = _clip_probability(team_a_win_probability)
        a_is_favorite = p_a >= 0.5
        pf = p_a if a_is_favorite else 1.0 - p_a
        f = self.favorite_oriented_distribution(pf)

        if a_is_favorite:
            result = {
                "A_3_0": f["F_3_0"], "A_3_1": f["F_3_1"], "A_3_2": f["F_3_2"],
                "B_3_2": f["U_3_2"], "B_3_1": f["U_3_1"], "B_3_0": f["U_3_0"],
            }
        else:
            result = {
                "A_3_0": f["U_3_0"], "A_3_1": f["U_3_1"], "A_3_2": f["U_3_2"],
                "B_3_2": f["F_3_2"], "B_3_1": f["F_3_1"], "B_3_0": f["F_3_0"],
            }
        self._validate_team_a(result, p_a)
        return result

    def sweep_chances(self, team_a_win_probability: float) -> dict[str, float]:
        d = self.team_a_distribution(team_a_win_probability)
        return {
            "team_a_sweep": d["A_3_0"],
            "team_b_sweep": d["B_3_0"],
            "any_sweep": d["A_3_0"] + d["B_3_0"],
        }

    @staticmethod
    def _validate_favorite(d: Mapping[str, float], p_favorite: float) -> None:
        if abs(sum(d.values()) - 1.0) > 1e-9:
            raise RuntimeError("Six-score probabilities do not sum to 1")
        if abs(d["F_3_0"] + d["F_3_1"] + d["F_3_2"] - p_favorite) > 1e-9:
            raise RuntimeError("Favorite score mass does not equal locked win probability")

    @staticmethod
    def _validate_team_a(d: Mapping[str, float], p_a: float) -> None:
        if abs(sum(d.values()) - 1.0) > 1e-9:
            raise RuntimeError("Six-score probabilities do not sum to 1")
        a_mass = d["A_3_0"] + d["A_3_1"] + d["A_3_2"]
        if abs(a_mass - p_a) > 1e-9:
            raise RuntimeError("Team A score mass does not equal locked win probability")
