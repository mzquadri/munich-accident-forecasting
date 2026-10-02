# Munich road accident forecasting

Monthly forecasts for Munich's published road accident counts, served behind a
small HTTP API. Built for the Digital Product School coding challenge, then
rebuilt after the original model turned out not to work.

## Why this exists in its current form

The first version of this project fitted one linear regression to every row of
the city's export at once. That file holds seven different monthly series, three
accident categories crossed with an accident type, and their averages range from
21 to 3,512 accidents per month, a factor of 169. Fitting a single line through all seven, with
only a year and a month as input, produced a model with an R² of 0.0009 and a
mean absolute error of 824 against a target averaging 780. The API it served
accepted no way to say which series you wanted, and described its output as a
number of deaths, which is not a quantity the dataset contains.

That model is reproduced exactly in `docs/provenance.md` before anything was
changed, so the criticism is measured rather than asserted. What follows is the
rebuild.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/01_series-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="docs/figures/01_series.png">
  <img src="docs/figures/01_series.png" alt="The seven series, on a log scale, spanning a factor of 169">
</picture>

## The data

Munich publishes monthly road accident counts through its open data portal. The
snapshot used here is the May 2024 export, included in the repository as the
challenge input.

| | |
| --- | --- |
| Series | 7, being 3 categories crossed with an accident type |
| Observations | 1,932 monthly counts |
| Period | 2000 to 2022 |
| Categories | Alkoholunfälle, Fluchtunfälle, Verkehrsunfälle |
| Types | insgesamt, Verletzte und Getötete, mit Personenschäden |

Three details in the file matter and each is handled in `accidents/data.py`. `MONAT`
holds YYYYMM rather than a month number. Rows with `MONAT` set to `Summe` are
annual subtotals and would swamp the monthly series. Rows exist for 2023 and 2024
with no values, because those months were not published, which is not the same
thing as zero accidents.

## The model

Each series is fitted separately, because they do not share a scale. Within a
series the model is a linear trend in time plus a monthly seasonal offset, fitted
on the most recent six years. Predictions are clipped at zero.

The six-year window is a choice, so it was made on 2019 and 2020 and the test
years were left alone. `docs/evaluation.md` records every window that was tried.

## Results

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/02_model_vs_baselines-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="docs/figures/02_model_vs_baselines.png">
  <img src="docs/figures/02_model_vs_baselines.png" alt="The model against both baselines, scaled MAE per series and averaged">
</picture>

Held out: 2021 and 2022, 168 monthly observations, scored once. Errors are
averaged across the seven series after scaling each by its own level, so the
small series are not drowned out by the large ones.

| Method | Scaled MAE | MAPE |
| --- | --- | --- |
| Seasonal trend | **0.153** | **20.5%** |
| Seasonal naive | 0.200 | 25.4% |
| Series mean | 0.296 | 42.2% |

The seasonal naive baseline, which repeats the same calendar month of the
previous year, is the comparison that matters for seasonal counts. The model
beats it on all seven series.

One caveat belongs next to that result rather than at the bottom of the page. The
baseline carries 2020 forward, and 2020 was a pandemic year, so it forecasts low
on all seven series. Part of the margin is that timing rather than the model.

### How much of that margin is the pandemic

Most of it.

Stating the caveat was not the same as measuring it, so the identical protocol
was run three years earlier, over a stretch where nothing unusual happened.
Window selected on 2016 and 2017, refitted on everything to 2017, scored once on
2018 and 2019. The baseline there repeats 2017, an ordinary year. Same code, same
metrics, same number of held-out rows.

| Held out | Seasonal trend | Seasonal naive | Margin over the baseline | Series won |
| --- | --- | --- | --- | --- |
| 2021-2022, as published | **0.153** | 0.200 | +23.4% | 7 of 7 |
| 2018-2019, ordinary years | 0.166 | **0.143** | -15.5% | 3 of 7 |

On an ordinary period the advantage is gone. The point estimate has the baseline
ahead by 15.5%, and the model is in front on three of the seven series rather than
all seven.

The published figure is not wrong and it is not a property of the method either.
It is what happens when a trend fitted through a collapse is scored against a
baseline that repeats the collapse. Read it as a measurement of 2021 and 2022.

That the comparison runs at all is the point of keeping the protocol in one place:
the published window is reproduced by the same function that produces this one, and
its macro figures match `results/reference_run.json` to 1e-9, which is what makes
the second row trustworthy.

### Whether either margin survives 24 months

Both rows above rest on 24 months of seven series, and a single number says nothing
about whether it would hold on a different 24 months. A paired bootstrap answers
that: every method is scored on the same resampled months, so what varies is the
margin between them rather than the noise in each.

It is run twice. Drawing months independently assumes they carry no memory, which
for monthly accident counts is false and makes the interval too narrow. Drawing
runs of three consecutive months keeps short-range dependence inside each run.
Neither is right alone; where they disagree, believe the wider one.

| Held out | Margin | Months drawn independently | Drawn in blocks of three |
| --- | --- | --- | --- |
| 2021-2022, as published | +23.4% | [+10.8%, +35.0%] | [+7.2%, +33.3%] |
| 2018-2019, ordinary years | -15.5% | [-39.7%, +4.3%] | [-36.8%, +6.9%] |

The published margin clears zero on both. The earlier one does not: its interval
spans zero either way, so on 24 months the baseline's apparent 15.5% lead cannot be
told apart from no difference at all.

That sharpens the finding rather than weakening it. What this repository can defend
is not that the baseline wins on an ordinary period. It is that the model's
advantage is measurable on the published window and absent on an ordinary one, and
that the gap between those two conclusions is the pandemic sitting in the baseline's
memory.

Reproduce with `python -m accidents.robustness` and `python -m accidents.uncertainty`.
Both runs are pinned in `results/` and checked in CI alongside everything else.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/03_forecast_vs_actual-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="docs/figures/03_forecast_vs_actual.png">
  <img src="docs/figures/03_forecast_vs_actual.png" alt="Held-out forecasts against what actually happened, four of the seven series">
</picture>

## Reproducing

Python 3.11.

```bash
pip install -r requirements.txt
python -m accidents.pipeline
```

That reads the CSV, selects the trend window on the validation years, refits,
scores against both baselines, and writes `models/forecaster.joblib` and
`results/reference_run.json`. It runs in a few seconds.

Figures are regenerated from that run:

```bash
pip install -r requirements-dev.txt
python scripts/figures/generate_figures.py
```

## The API

```bash
uvicorn app:app --reload
```

| Endpoint | Purpose |
| --- | --- |
| `GET /` | Liveness, and whether the model artifact loaded |
| `GET /series` | The seven series that can be forecast, and the last year trained on |
| `POST /predict` | A forecast for one series, year and month |

```bash
curl -s -X POST localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"category": "Alkoholunfälle", "kind": "insgesamt", "year": 2021, "month": 6}'
```

A request naming a series that does not exist returns 404 with the list of valid
ones, since a caller has no way to guess German category names. If the artifact
is missing the service reports that at `/` and returns 503 rather than pretending.

## Limitations

The test period is two years covering a pandemic and its rebound, which is an
unusual stretch to be judged on.

The model sees only its own history. There are no covariates for weather, traffic
volume, policy, road works or changes in reporting, so it cannot anticipate a
change in any of them.

This is a coding challenge exercise, not a public safety tool. Nothing here has
been validated for operational use, and the counts it forecasts are aggregates
for a whole city.

It runs on a CPU and there is nothing here for a GPU to do. The model is fitted on
1,764 monthly observations and the seven fitted series together serialise to a few
kilobytes. Fitting all seven takes single-digit milliseconds and one forecast
takes a small fraction of one, measured on the author's machine and not pinned
anywhere, because those are properties of the machine. Moving arithmetic that
size to an accelerator would cost more in transfer than it replaced.

## Attribution

Data: Landeshauptstadt München open data portal, included as the challenge input
and not redistributed under any claim of ownership. Check the portal's current
terms before reuse.

The challenge framing is the Digital Product School's. The implementation and
evaluation are the author's own. `docs/provenance.md` separates the original
submission from the rebuild, and the original notebooks are preserved unmodified
in `notebooks/`.

## Licence

MIT for the code, see [LICENSE](LICENSE). The dataset is covered by its own terms.
