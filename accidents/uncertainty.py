"""How much of each margin is real, and how much is 24 months of luck.

    python -m accidents.uncertainty
    python -m accidents.uncertainty --check

The repository reports two margins over the seasonal-naive baseline: +23.4% on the
published window and -15.5% three years earlier. Both are single numbers computed
on 24 months of seven series, and a single number says nothing about whether it
would survive a different 24 months. This puts an interval around each.

**Why the month is the resampling unit.** The seven series are measured on the same
months and move together: a month with more traffic has more of everything. Drawing
individual (series, month) cells would break that correlation and report an interval
narrower than the truth. Drawing whole months keeps the seven series that share a
month together, and keeps the comparison paired, so every method is scored on exactly
the same resample and the margin between them is what varies.

**Why two bootstraps and not one.** Resampling months independently assumes the months
are independent, and they are not: accident counts in March tell you something about
April. That assumption makes the interval too narrow. The moving-block bootstrap draws
runs of consecutive months instead, which keeps short-range dependence inside each
block, and is the standard correction. Neither is the right answer on its own. Both are
reported, and where they disagree the wider one is the one to believe.

Nothing here refits a model or moves a published figure. It reuses the same fits, on
the same split, and only varies which test months are counted.

Writes:
    results/uncertainty_run.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .data import ROOT, load_monthly, series_list, split
from .model import SeasonalNaive, SeasonalTrend, SeriesMean, fit_all
from .pipeline import TOLERANCE, choose_window, comparable, differences
from .robustness import METHODS, PANDEMIC_FREE, PUBLISHED

RESULTS = ROOT / "results" / "uncertainty_run.json"

#: Enough that the percentile endpoints are stable to the third decimal.
DRAWS = 10_000

#: Fixed, and recorded in the artifact, because an interval that moves between runs
#: is not a measurement.
SEED = 20260928

#: Consecutive months per block. Three keeps a quarter together, which is the
#: shortest run over which these series visibly carry their own momentum.
BLOCK = 3

#: Percentile endpoints.
LOW, HIGH = 2.5, 97.5


def window_predictions(
    df: pd.DataFrame, validation_years: tuple[int, int], train_end_year: int
) -> tuple[list[str], np.ndarray, dict[str, np.ndarray]]:
    """Refit the published protocol and keep the test predictions, not just the scores.

    Returns the series keys, an (n_series, n_months) array of actuals, and the same
    shape per method. `robustness.run_window` throws these arrays away once it has
    scored them, and the bootstrap needs them.
    """
    train, test = split(df, train_end_year)
    test = test[test["year"] <= train_end_year + 2].reset_index(drop=True)
    window, _ = choose_window(df[df["year"] <= train_end_year], validation_years)

    factories = {
        "seasonal_trend": lambda: SeasonalTrend(window=window),
        "seasonal_naive": SeasonalNaive,
        "series_mean": SeriesMean,
    }
    fitted = {name: fit_all(train, factory) for name, factory in factories.items()}

    keys, actual_rows = [], []
    pred_rows: dict[str, list[np.ndarray]] = {name: [] for name in METHODS}
    months: list[tuple[int, int]] | None = None

    for s in series_list(df):
        mask = (test["category"] == s.category) & (test["kind"] == s.kind)
        part = test[mask].sort_values(["year", "month"])
        here = list(zip(part["year"].tolist(), part["month"].tolist(), strict=True))
        if months is None:
            months = here
        elif here != months:
            # Every series must cover the same months or a column of the resampled
            # matrix would mix one series' March with another's April.
            raise SystemExit("series do not share a common test calendar")
        keys.append(s.key)
        actual_rows.append(part["value"].to_numpy(dtype=float))
        for name in METHODS:
            pred_rows[name].append(
                fitted[name][s.key].predict(
                    part["year"].to_numpy(), part["month"].to_numpy()
                ).astype(float)
            )

    return (
        keys,
        np.vstack(actual_rows),
        {name: np.vstack(rows) for name, rows in pred_rows.items()},
    )


def macro_scaled_mae(actual: np.ndarray, predicted: np.ndarray, take: np.ndarray):
    """Scaled MAE per series over the selected months, averaged across series.

    `take` is either one resample, shape (months,), or a whole bootstrap of them,
    shape (draws, months). Every draw is evaluated in the same pass: ten thousand
    of them through a Python loop took most of two minutes and made the check too
    slow to keep in CI, where the same arithmetic broadcast across an axis takes
    seconds. The returned value is a float for one resample and an array of them
    for many.

    The denominator follows the resample rather than staying at the full-sample
    mean. Scaled MAE is a ratio of two statistics of one sample, and holding half
    of it still would report an interval narrower than the quantity varies.
    """
    a = actual[:, take]
    p = predicted[:, take]
    err = np.abs(p - a).mean(axis=-1)
    denom = a.mean(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_series = np.where(denom != 0, err / denom, np.nan)
    out = np.nanmean(per_series, axis=0)
    return float(out) if out.ndim == 0 else out


def margin(actual: np.ndarray, preds: dict[str, np.ndarray], take: np.ndarray):
    """(baseline - model) / baseline, as a share, on the selected months."""
    base = macro_scaled_mae(actual, preds["seasonal_naive"], take)
    model = macro_scaled_mae(actual, preds["seasonal_trend"], take)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(np.asarray(base) != 0, (np.asarray(base) - model) / base, np.nan)
    return float(out) if out.ndim == 0 else out


def iid_draws(n: int, rng: np.random.Generator, draws: int) -> np.ndarray:
    """Months drawn independently, which assumes they carry no memory."""
    return rng.integers(0, n, size=(draws, n))


def block_draws(n: int, rng: np.random.Generator, draws: int, block: int) -> np.ndarray:
    """Runs of consecutive months, which lets each run carry its own momentum."""
    per_draw = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(draws, per_draw))
    offsets = np.arange(block)
    return (starts[:, :, None] + offsets[None, None, :]).reshape(draws, -1)[:, :n]


def interval(samples: np.ndarray) -> dict:
    return {
        "low": float(np.percentile(samples, LOW)),
        "high": float(np.percentile(samples, HIGH)),
        "share_above_zero": float(np.mean(samples > 0)),
    }


def measure(df: pd.DataFrame, validation_years: tuple[int, int], train_end_year: int,
            label: str) -> dict:
    keys, actual, preds = window_predictions(df, validation_years, train_end_year)
    n_months = actual.shape[1]
    every = np.arange(n_months)
    point = margin(actual, preds, every)

    rng = np.random.default_rng(SEED)
    out = {
        "label": label,
        "validation_years": list(validation_years),
        "train_end_year": train_end_year,
        "series": len(keys),
        "test_months": n_months,
        "margin": point,
    }
    for name, idx in (("iid_months", iid_draws(n_months, rng, DRAWS)),
                      ("moving_block", block_draws(n_months, rng, DRAWS, BLOCK))):
        out[name] = interval(np.asarray(margin(actual, preds, idx)))
    out["moving_block"]["block_months"] = BLOCK
    return out


def verdict(window: dict) -> str:
    """What the two intervals together allow the repository to claim."""
    wider = max(window["iid_months"], window["moving_block"],
                key=lambda i: i["high"] - i["low"])
    crosses = wider["low"] <= 0 <= wider["high"]
    direction = "above" if window["margin"] > 0 else "below"
    if crosses:
        return (f"The wider interval spans zero, so this margin is not separable from "
                f"nothing on {window['test_months']} months.")
    return (f"The wider interval stays {direction} zero, so the sign of this margin "
            f"survives resampling.")


def report(window: dict) -> None:
    print(f"\n  {window['label']}  "
          f"(train to {window['train_end_year']}, {window['test_months']} test months)")
    print(f"    margin                     {window['margin']:+.1%}")
    for name in ("iid_months", "moving_block"):
        i = window[name]
        print(f"    {name:<24}   [{i['low']:+.1%}, {i['high']:+.1%}]   "
              f"above zero in {i['share_above_zero']:.1%} of draws")
    print(f"    {verdict(window)}")


def main(check: bool = False) -> int:
    df = load_monthly()
    windows = {
        "published": measure(df, *PUBLISHED, "published, 2021-2022"),
        "pandemic_free": measure(df, *PANDEMIC_FREE, "pandemic free, 2018-2019"),
    }
    payload = {
        "environment": {
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "python": platform.python_version(),
        },
        "question": ("Both margins are computed on 24 months. How much of each one is "
                     "the model, and how much is which 24 months were held out?"),
        "method": (f"Paired bootstrap over test months, {DRAWS} draws, seed {SEED}. "
                   f"Every method is scored on the same resample. Reported twice: "
                   f"months drawn independently, and drawn in blocks of {BLOCK} "
                   f"consecutive months to allow for month-to-month dependence."),
        "draws": DRAWS,
        "seed": SEED,
        "windows": windows,
    }
    payload["finding"] = " ".join(verdict(w) for w in windows.values())

    if check:
        if not RESULTS.is_file():
            print(f"{RESULTS} is missing. Run: python -m accidents.uncertainty",
                  file=sys.stderr)
            return 1
        stored = json.loads(RESULTS.read_text(encoding="utf-8"))
        diffs = differences(comparable(stored), comparable(payload))
        if diffs:
            print(f"{RESULTS.name} does not match this run:", file=sys.stderr)
            for path, a, b in diffs[:20]:
                print(f"  {path}: stored {a}, recomputed {b}", file=sys.stderr)
            return 1
        print(f"\n  {RESULTS.name} matches this run to within {TOLERANCE:g} relative\n")
        return 0

    for window in windows.values():
        report(window)
    print()
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"  wrote {RESULTS.relative_to(ROOT).as_posix()}\n")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="recompute and compare against the committed artifact")
    raise SystemExit(main(**vars(parser.parse_args())))
