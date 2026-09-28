"""Tests for the bootstrap intervals around the two margins.

The intervals decide what the README is allowed to claim, so the things guarded
here are the ones that would make them misleading rather than merely wrong:

  * the bootstrap computing a different margin from the one already published,
    which would put an interval around a number nobody reported;
  * the resample breaking the pairing, so the methods are scored on different
    months and the margin between them stops meaning anything;
  * the block bootstrap drawing scattered months instead of runs, which would
    make it a second copy of the independent one rather than a check on it;
  * the finding itself, that the published margin clears zero and the earlier
    one does not, silently reversing.
"""

import json

import numpy as np
import pytest

from accidents.data import load_monthly
from accidents.robustness import PANDEMIC_FREE, PUBLISHED, run_window
from accidents.uncertainty import (
    BLOCK,
    DRAWS,
    RESULTS,
    SEED,
    block_draws,
    iid_draws,
    macro_scaled_mae,
    margin,
    measure,
    window_predictions,
)

TOLERANCE = 1e-9


@pytest.fixture(scope="module")
def df():
    return load_monthly()


@pytest.fixture(scope="module")
def published(df):
    keys, actual, preds = window_predictions(df, *PUBLISHED)
    return keys, actual, preds, np.arange(actual.shape[1])


def test_the_bootstrap_scores_the_same_margin_as_the_published_run(df, published):
    """On every month, the two code paths must agree exactly.

    `robustness.run_window` scores the window and throws the predictions away;
    this module keeps them and rescores. If the two ever disagree, the interval
    belongs to a different quantity from the point estimate it is drawn around,
    and reporting them together would be dishonest rather than imprecise.
    """
    _, actual, preds, every = published
    assert margin(actual, preds, every) == pytest.approx(
        run_window(df, *PUBLISHED, label="published").margin(), abs=TOLERANCE
    )


def test_the_earlier_window_agrees_too(df):
    _, actual, preds = window_predictions(df, *PANDEMIC_FREE)
    every = np.arange(actual.shape[1])
    assert margin(actual, preds, every) == pytest.approx(
        run_window(df, *PANDEMIC_FREE, label="pandemic free").margin(), abs=TOLERANCE
    )


def test_every_series_covers_the_same_months(published):
    _, actual, preds, _ = published
    assert actual.shape == (7, 24)
    for name, matrix in preds.items():
        assert matrix.shape == actual.shape, name


def test_a_resample_scores_every_method_on_the_same_months(published):
    """The pairing is the whole point: an unpaired bootstrap measures the wrong thing.

    Scoring the model on one resample and the baseline on another would add the
    variance of two independent samples to a difference that in truth shares one,
    and the interval would come out far too wide.
    """
    _, actual, preds, _ = published
    rng = np.random.default_rng(1)
    take = rng.integers(0, actual.shape[1], size=actual.shape[1])
    # Recomputing each method on `take` and differencing must equal the margin
    # helper, which is the only thing that guarantees one resample was used.
    base = macro_scaled_mae(actual, preds["seasonal_naive"], take)
    model = macro_scaled_mae(actual, preds["seasonal_trend"], take)
    assert margin(actual, preds, take) == pytest.approx(
        (base - model) / base, abs=TOLERANCE
    )


def test_scaled_mae_rescales_its_own_denominator(published):
    """The denominator has to follow the resample, not stay at the full-sample mean.

    Scaled MAE is a ratio of two statistics of the same sample. Freezing the
    denominator would hold half the ratio still and report an interval narrower
    than the quantity actually varies.
    """
    _, actual, preds, _ = published
    half = np.arange(12)
    on_half = macro_scaled_mae(actual, preds["seasonal_trend"], half)
    frozen = float(
        np.nanmean(
            np.abs(preds["seasonal_trend"][:, half] - actual[:, half]).mean(axis=1)
            / actual.mean(axis=1)
        )
    )
    assert on_half != pytest.approx(frozen, abs=1e-6)


def test_block_draws_are_runs_of_consecutive_months():
    rng = np.random.default_rng(7)
    idx = block_draws(24, rng, 200, BLOCK)
    assert idx.shape == (200, 24)
    # Within each block the months must step by one.
    blocks = idx[:, : (24 // BLOCK) * BLOCK].reshape(200, -1, BLOCK)
    steps = np.diff(blocks, axis=2)
    assert (steps == 1).all()
    assert idx.max() < 24


def test_independent_draws_stay_in_range():
    rng = np.random.default_rng(7)
    idx = iid_draws(24, rng, 200)
    assert idx.shape == (200, 24)
    assert idx.min() >= 0 and idx.max() < 24


def test_the_interval_is_the_same_on_every_run(df):
    """A fixed seed, or the artifact would move on its own between runs."""
    first = measure(df, *PANDEMIC_FREE, "pandemic free")
    second = measure(df, *PANDEMIC_FREE, "pandemic free")
    for key in ("iid_months", "moving_block"):
        assert first[key]["low"] == second[key]["low"]
        assert first[key]["high"] == second[key]["high"]


def test_the_finding_still_holds():
    """The published margin clears zero; the earlier one does not.

    This is the sentence the README and the CV both rest on. If a change to the
    model, the data or the draw count ever reverses it, that is a finding and not
    a test failure to paper over: read the new numbers before touching this.
    """
    stored = json.loads(RESULTS.read_text(encoding="utf-8"))
    assert stored["draws"] == DRAWS and stored["seed"] == SEED

    pub = stored["windows"]["published"]
    assert pub["margin"] > 0
    for kind in ("iid_months", "moving_block"):
        assert pub[kind]["low"] > 0, f"published margin no longer clears zero on {kind}"

    free = stored["windows"]["pandemic_free"]
    assert free["margin"] < 0
    for kind in ("iid_months", "moving_block"):
        assert free[kind]["low"] < 0 < free[kind]["high"], (
            f"the earlier margin is now separable from zero on {kind}; "
            "the README claim needs rewriting, not this assertion"
        )
