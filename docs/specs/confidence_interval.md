# Spec / Research: Prediction (Confidence) Intervals in MMF

Status: **Research / Draft**. No implementation yet.
Branch: `docs/confidence-interval-spec`

## 0. TL;DR

- Today MMF only outputs **point forecasts**. Several wrapped models already compute
  quantiles or samples internally (Chronos, TimesFM, Moirai, most statsforecast models,
  Prophet), but MMF throws them away.
- Some MMF models have **no native interval at all**. These are the intermittent and
  baseline statsforecast models (Croston\*, ADIDA, IMAPA, TSB, WindowAverage, MFLES) and
  the NeuralForecast and MLForecast models as configured today (point losses).
- The gap can be filled **for every model with one model-agnostic mechanism**: split
  conformal prediction on backtest residuals. MMF already writes the needed residuals,
  per series, per backtest window and per horizon step, to the `evaluation_output` table
  (`forecast` and `actual` arrays). Calibration therefore costs no extra training.
- Backtest length is the binding constraint. A per-series, per-horizon-step conformal
  quantile at level `1-α` needs at least `⌈1/α⌉-1` windows to be finite. That is
  **9 windows for 90%** and **19 windows for 95%**, and it is still noisy at those
  sizes (±9 and ±5 coverage points respectively). With the README defaults
  (`backtest_length=30`, `stride=10`, `prediction_length=10`), MMF produces only
  **3 windows**. Pure per-series calibration is therefore not viable by default. MMF's
  advantage is **borrowing strength across many series**: per-series quantiles shrunk
  toward a group pool (ABC×XYZ by default), which stays stable with only a few windows.
  This was validated on M5 (§7.6.4).
- **Native intervals, where a model has them, beat every conformal variant on interval
  score** at similar coverage in the M5 experiment (AutoETS: MSIS 9.9 vs ≥ 13.9 at
  95%). They are not calibrated, though: they over-cover by up to 12 points at 80%.
  CQR on the native bands corrects that to nominal coverage, and improves the interval
  score by 4% on average, even with 3 windows. That includes Chronos-Bolt, whose
  clamped 95% band only covers 80%. Residual conformal calibration is the fallback for models
  without native intervals.
- What similar products ship is very consistent: a configurable coverage level
  (default 0.95 in BigQuery and Snowflake), or a list of quantiles (default
  P10/P50/P90 in Amazon, TimeGPT and Azure), returned as `lower`/`upper` columns next
  to the point forecast.

Recommended direction (detailed in §7):
1. Pass native quantiles through where they exist (`method: auto`), as the first choice.
2. Use model-agnostic, MMF-level conformal calibration from the backtest output for every
   model without native intervals. Default: `coverage.target: series`, with shrinkage
   toward ABC×XYZ groups.
3. Evaluate intervals with coverage, Winkler/MSIS and pinball loss in
   `evaluation_output`.

---

## 1. Terminology

| Term | Meaning | What users usually want |
|---|---|---|
| **Confidence interval** | Uncertainty about a *parameter* (e.g. the mean of the forecast distribution). | Rarely, in forecasting. |
| **Prediction interval (PI)** | A range expected to contain the *future observation* `y_{T+h}` with probability `1-α`. | **Yes.** This is what "forecast confidence interval" means in practice. |
| **Quantile forecast** | `q_τ(y_{T+h})` for a set of levels τ (e.g. 0.1, 0.5, 0.9). A central PI at `1-α` is the pair `(q_{α/2}, q_{1-α/2})`. | Yes, for inventory and safety stock (P90, P95). |
| **Marginal vs conditional coverage** | Marginal: coverage holds on average over series and time. Conditional: it holds per series or per context. | Users assume conditional. Most methods only guarantee marginal coverage. |

The rest of this document says "interval" or "PI". The user-facing feature can still be
called "confidence intervals", as BigQuery (`confidence_level`) and Snowflake do.

---

## 2. Current state in MMF (code audit)

The pipeline only carries point forecasts:

- `ForecastingRegressor.calculate_metrics` (`mmf_sa/models/abstract_model.py:165`)
  stores `forecast` (array of length `prediction_length`), `actual`, and one scalar
  `metric_value` per (series, window).
- `ForecastingRegressor.backtest` (`mmf_sa/models/abstract_model.py:84`) rolls the
  forecast origin by `stride` from `max_date - backtest_length` up to the end. The
  number of windows is

  ```
  W = floor((backtest_length - prediction_length) / stride) + 1
  ```

  The README example (30 / 10 / 10) gives **W = 3**.
- Local models are **refit at each window** (`StatsFcForecaster.predict`). Global
  models are trained once before the backtest and then predicted with rolling context
  (`Forecaster.backtest_global_model`, `mmf_sa/Forecaster.py:399`; retraining is
  commented out). Foundation models are zero-shot.
- `scoring_output` holds only `target` arrays.

What each pipeline already computes and discards:

| Pipeline | What is computed but dropped | Where |
|---|---|---|
| Chronos-Bolt | 9 quantiles (0.1…0.9) | `ChronosPipeline.py:503,566,819`: only `quantiles.index(0.5)` is kept |
| Chronos-2 | 21 quantiles (0.01, 0.05, 0.1…0.9, 0.95, 0.99) | same code path |
| TimesFM 2.5 | `quantile_forecast` of shape `(batch, h, 10)` (mean + q10…q90) | `TimesFMPipeline.py:265,565`: `forecasts, _ = model.forecast(...)` |
| Moirai / Moirai-MoE | `num_samples` sample paths (configured at **10**) | `MoiraiPipeline.py:238,333`: `np.median(samples)` |
| statsforecast | Native `lo-*/hi-*` are available for ARIMA/ETS/CES/Theta/TBATS/Naive/SNaive if `level=` is passed | `StatsFcForecastingPipeline.py:92,130`: `predict(h)` is called without `level` |
| SKTimeProphet | `predict_interval` (Prophet's own simulation) | `SKTimeForecastingPipeline.py:69`: only `predict` is called |
| NeuralForecast | Nothing (all configs use point losses: `smape`) | `NeuralForecastPipeline.py:465` |
| MLForecast | Nothing (point LightGBM) | `MLForecastPipeline.py:448` |

Also relevant: local models apply `fillna(0)` and `clip(0)` to the target
(`Forecaster.py:343`, `StatsFcForecastingPipeline.py:37`). The lower bound of an
interval should be clipped to 0 the same way for non-negative targets. This is made
configurable via `clip_lower` (§7.3.1).

---

## 3. Native interval support per model family

Verified against library source and docs (statsforecast `main`, chronos-forecasting
`main`, TimesFM commit pinned in `abstract_model.py`, neuralforecast `main`). Earlier
library versions may differ.

### 3.1 statsforecast (local)

In `statsforecast/models.py`, each model's `predict(h, level=...)` either computes
intervals analytically or raises
`"You must pass prediction_intervals to compute them."`.

| MMF model | Native PI | How |
|---|---|---|
| AutoARIMA, AutoETS, AutoCES, AutoTheta, AutoTBATS | ✅ | Analytical, from the state-space or ARIMA error variance, assuming Gaussian errors |
| Naive, SeasonalNaive | ✅ | Analytical (random-walk variance, `σ√h`) |
| WindowAverage, SeasonalWindowAverage | ❌ | Conformal only |
| CrostonClassic / Optimized / SBA, ADIDA, IMAPA, TSB | ❌ | Conformal only |
| AutoMFLES (and MSTL) | ❌ | Conformal only |

`ConformalIntervals(n_windows=2, h=1, method="conformal_distribution")` can be attached
per model or globally (`StatsForecast(prediction_intervals=...)`). Implementation
details matter for §6:

- It does `n_windows` **non-overlapping** refits at the end of each series. That gives
  `n_windows` absolute errors per horizon step, as a matrix of shape `(n_windows, h)`.
- The quantile is the exact finite-sample order statistic `ceil((n+1)(1-α))`, **clipped
  to n**. With the default `n_windows=2`, every level up to 66% therefore ends up as
  "the max of 2 errors". There is no real coverage guarantee at 80–95%.
- The series needs at least `2h+1` points. If the series is short, the library silently
  reduces `n_windows`.
- Methods: `conformal_error` gives `ŷ ± q_{1-α}(|e|)`. `conformal_distribution` builds
  pseudo-paths `ŷ ± |e|` and takes empirical quantiles.

Doing this inside MMF would **duplicate** MMF's own backtest refits, because local
models already refit per window. That costs `n_windows` extra fits per series.

### 3.2 MLForecast (global: LightGBM)

- `MLForecast.fit(..., prediction_intervals=PredictionIntervals(n_windows, h, method))`,
  then `predict(h, level=[80, 95])`. This is the same conformal scheme as above,
  calibrated per horizon step and per series.
  https://nixtlaverse.nixtla.io/mlforecast/docs/how-to-guides/prediction_intervals.html
- Alternative: train one LightGBM per quantile with `objective="quantile", alpha=τ`.
  This costs one model per quantile and suffers from quantile crossing. The docs warn
  that "most prediction intervals are too narrow since models do not account for all
  sources of uncertainty".

### 3.3 NeuralForecast (global)

Two independent paths
(https://raw.githubusercontent.com/Nixtla/neuralforecast/main/neuralforecast/core.py):

- **Probabilistic loss** (re-training required):
  - `MQLoss(level=[80, 90])`: fixed quantiles chosen at construction.
  - `DistributionLoss("Normal" | "StudentT" | "Poisson" | "NegativeBinomial" | ...)`:
    parametric, with any `level` at predict time.
  - `IQLoss()`: implicit quantile, with any level at predict time.

  The Auto\* models would need `loss=` changed and HPO re-run. The tuning objective
  (`valid_loss`) would also change.
- **Conformal**: `nf.fit(df, prediction_intervals=PredictionIntervals(n_windows, method))`
  with point losses, then `predict(level=...)`. It needs at least 2 windows.

Count-aware distributions (Poisson, NegBin, Tweedie) are the main reason to prefer the
loss route for intermittent retail data (DeepAR literature, §4.4).

### 3.4 sktime (local: Prophet)

- `predict_interval(fh, coverage=0.9)` and `predict_quantiles(fh, alpha=[...])` on any
  estimator that has the tag `capability:pred_int=True`. Prophet has it.
- Model-agnostic wrappers: `ConformalIntervals`, `NaiveVariance`, `BaggingForecaster`,
  `SquaringResiduals`. These give any sktime forecaster intervals.
  https://www.sktime.net/en/stable/api_reference/forecasting.html
- Prophet's own intervals come from simulating future trend changepoints plus
  observation noise. `interval_width` defaults to 0.8 and `uncertainty_samples` to
  1000. They are known to be poorly calibrated: they ignore seasonality uncertainty
  unless MCMC is used, and they widen linearly.
  (Taylor & Letham, 2018; https://facebook.github.io/prophet/docs/uncertainty_intervals.html)

### 3.5 Foundation models

| Model | Output | Levels available | Notes |
|---|---|---|---|
| Chronos (T5) | Sample paths (token sampling) | Any (empirical) | Not wrapped in MMF today |
| Chronos-Bolt | Direct multi-step **quantiles** | 0.1…0.9 (9 levels, from `config.json`) | Levels outside [0.1, 0.9] are **extrapolated** with a warning, so 95% PIs are unreliable |
| Chronos-2 | Direct quantiles | 21 levels incl. 0.01, 0.05, 0.95, 0.99 | `predict_quantiles(..., quantile_levels=...)`, `predict_df` |
| TimesFM 2.5 | Point + `quantile_forecast[..., 10]` = mean, q10…q90 | 0.1…0.9 | Optional 30M continuous quantile head (`use_continuous_quantile_head=True`, `fix_quantile_crossing=True` in `ForecastConfig`). MMF's `ForecastConfig` sets neither today. |
| Moirai / Moirai-MoE | Sample paths from a mixture distribution | Any (empirical) | `num_samples=10` in `models_conf.yaml` is far too few for tail quantiles (q05/q95 would be the min/max of 10 draws). Needs ≥100–200. |

Zero-shot quantiles from foundation models are **not guaranteed to be calibrated** on a
given dataset. Benchmarks (Chronos paper, GIFT-Eval) report WQL/CRPS, which mixes
calibration and sharpness, rather than coverage. Applying a conformal correction on top
of them is common practice (CQR, §4.2).

### 3.6 Summary matrix

| Family | Native PI today (lib) | Wired in MMF | Cheapest path to PIs |
|---|---|---|---|
| statsforecast ARIMA/ETS/CES/Theta/TBATS/Naive | ✅ analytical | ❌ | pass `level=` |
| statsforecast intermittent + window baselines + MFLES | ❌ | ❌ | conformal |
| Prophet (sktime) | ✅ simulation | ❌ | `predict_interval`, ideally + conformal |
| MLForecast LGBM | ❌ (conformal option) | ❌ | conformal |
| NeuralForecast (point loss) | ❌ (conformal option) | ❌ | conformal, or retrain with `MQLoss`/`DistributionLoss` |
| Chronos-Bolt / Chronos-2 / TimesFM | ✅ quantiles | ❌ (median only) | pass through, + conformal recalibration |
| Moirai | ✅ samples | ❌ (median only) | raise `num_samples`, take quantiles |

---

## 4. Literature review: how to fill the gap

### 4.1 Classical approaches

- **Analytical / model-based PIs** (`ŷ ± z_{1-α/2} σ_h`). These are exact under the
  model's assumptions. The literature consistently finds them **too narrow** in
  practice, because they ignore parameter, model-selection and structural-change
  uncertainty (Chatfield, 1993; Makridakis et al., 2020 for M4). Most methods in M4
  under-covered the nominal 95%.
- **Empirical PIs from past errors.** Williams & Goodman (1971) build intervals from
  the empirical distribution of out-of-sample errors at each horizon. They are the
  direct ancestor of time-series split-conformal and are what Nixtla's
  `conformal_error` implements. Lee & Scholtes (2014) revisited this for supply chains.
- **Bootstrapped residual paths.** Simulate future paths by resampling in-sample
  one-step residuals through the model's recursion (Hyndman & Athanasopoulos, FPP3
  §5.5). This handles non-Gaussian errors, but it assumes the residuals are i.i.d. and
  that the fitted model is correct. Its cost is `B` simulations per series.
- **Quantile regression / pinball-loss training.** Learn `q_τ` directly. Examples are
  LightGBM with the quantile objective, `MQLoss`, and MQ-RNN/MQ-CNN (Wen et al., 2017).
  Many top M5-Uncertainty entries used LightGBM quantile models (Makridakis et al.,
  2022).
- **Parametric likelihood (DeepAR-style).** Predict the parameters of a Gaussian,
  Student-t or Negative-Binomial distribution, then sample (Salinas et al., 2020). This
  is the natural fit for count and intermittent demand.

### 4.2 Conformal prediction (distribution-free)

- **Split / inductive conformal** (Vovk et al., 2005; Lei et al., 2018). Compute
  non-conformity scores `s_i = |y_i − ŷ_i|` on a calibration set of size `n`. Then
  `PI = ŷ ± s_(k)` with `k = ⌈(n+1)(1−α)⌉`. If the scores are exchangeable, marginal
  coverage is at least `1−α` in finite samples. The coverage obtained for a fixed
  calibration set is Beta-distributed, which is what §6.1 uses (Angelopoulos & Bates,
  2023).
- **Normalised / locally adaptive scores** (Papadopoulos et al., 2008; Lei et al.,
  2018). Use `s_i = |y_i − ŷ_i| / σ̂_i`, where `σ̂_i` is a scale estimate, and set
  `PI = ŷ ± σ̂ · q`. This lets scores be **pooled across series of different
  magnitudes**, and it is the key enabler for MMF (§7). A good `σ̂` for MMF is the
  MASE scale: the in-sample MAE of the seasonal naive forecast.
- **Conformalised Quantile Regression, CQR** (Romano et al., 2019). Start from a
  model's own quantiles `(q̂_lo, q̂_hi)`, score `s = max(q̂_lo − y, y − q̂_hi)`, and
  widen or shrink by the conformal quantile of `s`. This is the right way to
  **recalibrate foundation-model quantiles** (Chronos, TimesFM, Moirai) while keeping
  their adaptivity.
- **Multi-step forecasts.** Stankeviciute et al. (2021, CF-RNN) calibrate each horizon
  step separately and apply a Bonferroni correction (`α/h`) for *joint* trajectory
  coverage. Per-step calibration without Bonferroni is what Nixtla does. It gives
  per-step marginal coverage, which is usually what planners want.

### 4.3 Conformal under non-exchangeability (time series)

Time-series residuals are autocorrelated, heteroscedastic and drift over time, which
breaks the exchangeability assumption. Remedies:

| Method | Idea | Relevance to MMF |
|---|---|---|
| **Weighted conformal / nexCP** (Barber et al., 2023, *Ann. Stat.*) | Weight calibration scores by recency. The coverage gap is bounded by the total-variation distance between score distributions. | Simple and batch-friendly. Exponential decay over backtest windows. |
| **ACI** (Gibbs & Candès, 2021) | Online update `α_{t+1} = α_t + γ(α − err_t)`. Long-run coverage is guaranteed under arbitrary shift. | Needs a sequential stream of realised errors. Fits a *monitoring* loop better than a batch run. |
| **AgACI** (Zaffran et al., 2022) | Aggregates ACI over several γ values, which removes the need to tune γ. | Same as ACI. |
| **EnbPI** (Xu & Xie, 2021) | Bootstrap ensemble with leave-one-out residuals. No refits on a hold-out set. | Costly (B models). Implemented in MAPIE. |
| **SPCI** (Xu & Xie, 2023) | Quantile regression on past residuals, which captures residual autocorrelation. | Research-grade. |
| **Conformal PID** (Angelopoulos et al., 2023) | A control-theory generalisation of ACI with a scorecaster. | Research-grade. |

Libraries: MAPIE (`TimeSeriesRegressor` with EnbPI and ACI), Nixtla
statsforecast/mlforecast/neuralforecast (`ConformalIntervals`/`PredictionIntervals`),
sktime `ConformalIntervals`, and statsforecast `ConformalSeasonalPool` (an adaptive
sample-based conformal model).

### 4.4 Intermittent and low-volume series

This is critical for MMF, given that its Croston/TSB/ADIDA defaults target retail.

- Symmetric `ŷ ± q` intervals around a Croston *rate* forecast are meaningless. The
  lower bound is negative, and actual demand is mostly 0.
- Prefer **signed, asymmetric scores**: separate quantiles of `y − ŷ` for the lower and
  upper sides. Clip the lower bound at 0. Report quantiles (P50/P90/P95) rather than
  symmetric intervals.
- Coverage for count data is ill-defined when the bounds land on integer values.
  Kolassa (2016) recommends evaluating predictive count distributions with proper
  scoring rules (ranked probability score, pinball loss) rather than coverage alone.
  Snyder, Ord & Beaumont (2012) give parametric count models (Poisson / NegBin
  state-space) as the classical alternative.
- In practice, pool scores within an intermittency class: ADI/CV² (Syntetos–Boylan)
  smooth / erratic / intermittent / lumpy. MMF does **not** compute this today. The
  optional profiling skill
  (`skills/databricks-skills/many-model-forecasting/2-profile-and-classify-series.md`)
  only writes a two-valued `forecastability_class` (`high_confidence` / `low_signal`),
  derived from sparsity, entropy and SNR. The classes are cheap to compute from the
  target (§7.6.4 does it in a few lines).

### 4.5 Hierarchies

Intervals do not add up across a hierarchy. MMF has a reconciliation module
(`mmf_sa/reconciliation.py`). Coherent probabilistic forecasts need probabilistic
reconciliation:

- Panagiotelis et al. (2023): bootstrap or Gaussian approaches with MinT.
- Nixtla `hierarchicalforecast`: `Normality`, `Bootstrap`, `PERMBU` intervals.

This is out of scope for a first version. It should be listed as a follow-up.

---

## 5. What similar products do

| Product | User-facing API | Method | Defaults |
|---|---|---|---|
| **BigQuery ML ARIMA_PLUS** | `ML.FORECAST(..., STRUCT(h AS horizon, 0.95 AS confidence_level))` → `prediction_interval_lower_bound/upper_bound` | Analytical (ARIMA std error) | 0.95. https://cloud.google.com/bigquery/docs/reference/standard-sql/bigqueryml-syntax-forecast |
| **Snowflake ML FORECAST** | `!FORECAST(..., CONFIG_OBJECT => {'prediction_interval': 0.95})` → `LOWER_BOUND/UPPER_BOUND` | Gradient boosting, method not documented | 0.95. https://docs.snowflake.com/en/sql-reference/classes/forecast/methods/forecast |
| **Azure AutoML forecasting** | `fitted_model.forecast_quantiles(X, quantiles=[0.025, 0.5, 0.975])` | Model-dependent (residual-based) | https://learn.microsoft.com/en-us/azure/machine-learning/how-to-automl-forecasting-faq |
| **Amazon Forecast / SageMaker DeepAR** | `ForecastTypes=["0.1","0.5","0.9"]` (any `mean` + quantiles) | DeepAR likelihood sampling. Metric: weighted quantile loss (wQL). | P10/P50/P90. *Amazon Forecast stopped accepting new customers in 2024*; the pattern lives on in SageMaker/AutoGluon. |
| **AutoGluon-TimeSeries** | `predictor.predict(...)` returns `mean` + `quantile_levels` columns | Native per model (Chronos, DeepAR, ETS...). Ensembles are optimised on WQL. | Quantiles 0.1…0.9 |
| **Nixtla TimeGPT** | `forecast(df, h, level=[80, 95])` or `quantiles=[...]` | **Conformal on historical rolling backtests** | User-chosen. https://nixtla.github.io/nixtlar/articles/quantiles.html |
| **Databricks AutoML forecasting** (Prophet/ARIMA) | Output includes `yhat_lower`, `yhat_upper` | Prophet simulation / ARIMA analytical | Prophet's 0.8. *Not re-verified for the latest AutoML version.* |
| **Prophet** | `interval_width`, `uncertainty_samples` → `yhat_lower/upper` | Trend-changepoint simulation + noise | 0.8 |
| **GluonTS** (evaluation) | `Evaluator(quantiles=...)` → `Coverage[q]`, `wQuantileLoss[q]`, `MSIS`, `mean_wQuantileLoss` | n/a | |

Takeaways for MMF:
1. **One knob**, in two forms: `confidence_level` (a scalar such as 0.95) or `quantiles`
   (a list). Both are common, and they are interconvertible.
2. Output is **two extra columns** next to the point forecast, as in Snowflake and
   BigQuery, or quantile columns, as in AutoGluon and Amazon.
3. Products that target heterogeneous model zoos (TimeGPT, Nixtla) use **conformal on
   rolling backtests**, not per-model analytics. That is MMF's situation.
4. Evaluation uses WQL/MSIS/coverage, alongside point metrics.

---

## 6. How long should the backtest be?

### 6.1 Theory: finite-sample conformal

With `n` calibration scores and target coverage `1−α`:

- A **finite** upper bound requires `n ≥ ⌈1/α⌉ − 1`: **4** for 80%, **9** for 90%,
  **19** for 95%, and **99** for 99%. Below that, the exact method returns an infinite
  bound. Nixtla clips to the max score, which under-covers silently.
- For a given calibration draw, the realised coverage is
  `Beta(k, n+1−k)`, `k=⌈(n+1)(1−α)⌉` (Angelopoulos & Bates, 2023, §3.2). Its standard
  deviation is roughly `√(α(1−α)/(n+2))`. Computed exactly:

| n (scores per pool) | sd of coverage @80% | @90% | @95% |
|---:|---:|---:|---:|
| 4 | 0.163 | ∞ (undefined) | ∞ |
| 9 | 0.121 | 0.090 | ∞ |
| 19 | 0.087 | 0.065 | 0.048 |
| 49 | 0.056 | 0.042 | 0.027 |
| 99 | 0.040 | 0.030 | 0.022 |
| 199 | 0.028 | 0.021 | 0.015 |
| 499 | 0.018 | 0.013 | 0.010 |
| 999 | 0.013 | 0.009 | 0.007 |

Rule of thumb: about **100 scores per pool** for ±3 coverage points at 90%, and about
**500** for ±1.3 points.

### 6.2 What a "score per pool" is in MMF

Calibration is per horizon step `j ∈ {1..h}`, because error grows with `j`. So:

| Pooling strategy | Scores per pool | MMF default (W=3) | Comment |
|---|---|---|---|
| Per series × per step | `W` | 3, **unusable** | Needs W≥19 for 90%. That is ≈19 refits per series for local models. |
| Per series, pooled over steps (with `√j` or fitted scaling) | `W · h` | 30 | Borderline. Assumes a known error-growth shape. |
| **Per segment × per step, normalised scores** | `W · N_segment` | 3 × (hundreds–thousands) | **Stable.** Coverage is marginal over the segment's series. |
| Global × per step, normalised | `W · N` | very large | Stable, but ignores heterogeneity |

Here `N_segment` is the number of series in a homogeneous group, for example an
ABC×XYZ or Syntetos–Boylan class computed by MMF, a store, or a category.

### 6.3 Practical guidance (from the literature and the products)

- **Number of windows W.**
  - Pooled calibration: **W ≥ 5–10** distinct forecast origins is enough for pool
    size. More windows mainly add *diversity of origin*.
  - Per-series calibration: W ≥ 20 (90%) to 40 (95%).
  - Nixtla's default `n_windows=2` is a code minimum, not a recommendation. Their
    tutorials use 5–10.
- **Span.** The windows should cover **at least one full seasonal cycle**, so that
  calibration sees peak and off-peak errors. That means ≥ 7 days for daily data with
  weekly seasonality, ≥ 52 weeks for weekly retail, and ≥ 12 months for monthly data.
  Otherwise the intervals are calibrated to a single regime.
- **Stride.** `stride < prediction_length` gives overlapping windows. This is fine for
  per-step scores (each window still gives one score per step), but the scores become
  correlated, so the *effective* n is smaller. For local models, compute cost is
  `O(W)` refits per series. `stride=1` maximises W at the highest cost.
- **Recency vs size.** A longer backtest gives more scores, but older ones may come
  from a different regime (Barber et al., 2023). Mitigations: exponential recency
  weights over windows (weighted conformal), or cap the backtest at about 1–2 seasonal
  cycles.
- **Train/calibrate/score mismatch.** In evaluation, global models are trained on less
  data than in final scoring (`Forecaster.backtest_global_model`). Calibration scores
  are therefore slightly pessimistic, giving conservative intervals. That is
  acceptable and standard.

Recommended settings for users who want better intervals. These are **opt-in**: the
README defaults are *not* changed (see §6.4 and §9). Values are to be validated
empirically.

| freq | prediction_length | backtest_length | stride | W |
|---|---:|---:|---:|---:|
| D | 28 | 28 + 7·9 = 91 | 7 | 10 |
| W | 13 | 13 + 52 = 65 | 4 | 14 |
| M | 6 | 6 + 18 = 24 | 2 | 10 |
| H | 24 | 24 + 24·13 = 336 | 24 | 14 |

### 6.4 Workaround: good intervals with the default W = 3

`W` is the **number of backtest windows** (forecast origins) per series:

```
W = floor((backtest_length − prediction_length) / stride) + 1
```

With the README defaults (`backtest_length=30`, `prediction_length=10`, `stride=10`),
the origins are at `T−29`, `T−19` and `T−9`, so W = 3. Each window gives one error per
horizon step, so every series has only 3 errors per step. Raising W multiplies the
refit cost of local models by W/3, so the defaults stay as they are. The workarounds
below need no change to `backtest_length` or `stride`:

1. **Pool across series** (`coverage.target: marginal | group`, §7.6). This is the
   main lever: a pool holds `3 · N_group` scores per step, e.g. 3 000 for a group of
   1 000 series. It costs nothing extra.
2. **Pool across horizon steps within a series** (`pool_horizon`, §7.6.3 Option 3).
   This gives n = W·h = 30 scores per series (enough for a finite 90% bound) and
   restores error growth with a group-level step profile.
3. **Shrink per-series quantiles toward the group** (`shrinkage_kappa`). Short
   histories automatically lean on the group.
4. **Denser calibration windows only where they are cheap** (optional, P1). Global and
   foundation models are *not* refit per window (§2), so an optional
   `calibration_stride` (e.g. 1) can add windows for those families at
   prediction-only cost. Local models keep the default stride.
5. **Fail loudly instead of under-covering.** When a pool has fewer than
   `⌈2/α⌉ − 1` scores for a requested level (two-sided, signed), widen to the next
   pooling level, or emit `null` bounds with a warning. Never silently clip to the
   max score, as statsforecast and AutoGluon do.

---

## 7. Proposed design (sketch for discussion)

### 7.1 Layer A: native pass-through (first choice where available)

- statsforecast: call `predict(h, level=levels)` for models whose native PI is
  available (§3.1).
- Chronos / TimesFM: keep the quantile tensor, mapping levels to the nearest trained
  quantiles. For TimesFM, enable `use_continuous_quantile_head`. Chronos-Bolt only
  has quantiles 0.1–0.9 and silently clamps anything wider. Its "95%" band is
  therefore its 80% band: on M5 it covered 0.80. Wider levels must go through CQR,
  and MMF should log a warning (§7.6.4).
- Moirai: take `np.quantile(samples, τ)` and raise the default `num_samples` (≥100).
- Prophet: `predict_interval`.

This gives "model-native" bands. They are **not guaranteed calibrated**, and they are
**missing for half of the zoo**. On M5 they were nonetheless the sharpest intervals at
near-nominal coverage (§7.6.4). They do over-cover at 80%, however (AutoETS 0.87,
SeasonalNaive 0.92), which CQR fixes (§7.2). MMF should still report their empirical
coverage (§8).

### 7.2 Layer B: MMF conformal calibration (universal fallback)

It runs after evaluation, entirely from `evaluation_output` (it already has `forecast`
and `actual` arrays), so there is **no extra training**:

1. Explode the arrays to `(series, model, window_start, step j, ŷ, y)`.
2. Compute a scale `σ̂_i` per series. The default is the MASE denominator (in-sample
   MAE of the seasonal naive) on the training part. Fall back to the mean of `|y|`.
3. Compute signed normalised scores `e = (y − ŷ)/σ̂_i`.
4. For each `(model, pool, step j)`, store the conformal quantiles of `e` for the
   lower (`α/2`) and upper (`1−α/2`) sides, optionally recency-weighted. What a `pool`
   is depends on `coverage.target` (`marginal`, `group` or `series`). See §7.6.
5. Persist the result to a new table `calibration_output` (`run_id`, `model`,
   `coverage_target`, `pool_key`, `step`, `level`, `q_lo`, `q_hi`, `n_scores`,
   `shrinkage_weight`).

Scoring then produces `lo = ŷ + σ̂_i·q_lo` and `hi = ŷ + σ̂_i·q_hi`. After that, the
lower bound is optionally clipped according to `clip_lower` (§7.3.1).

For models with native quantiles, Layer B runs **CQR** on the native bands instead
(§4.2). This keeps their adaptivity while fixing their calibration. It requires the
backtest to also store the native `lo`/`hi` arrays per window, which adds no extra
training because the same fit produces them. The default, `cqr_score: asymmetric`
with an adaptive tail split, uses one score per tail. Whatever share of `α` the lower
tail does not use (it rarely misses once clipped at 0) goes to the upper tail. The
correction is pooled per `group_col` group (ABC×XYZ by default, `cqr_pool: group`).
On M5, with W=3 or more, it reached nominal coverage within 2 points on the four
models with native bands (AutoETS, SeasonalNaive, Chronos-Bolt, Chronos-2), and
improved MSIS by 4% on average (§7.6.4). Per-series shrinkage is slightly sharper
for the statsforecast models. It fails, however, when the native band is far off
(Chronos-Bolt at 95%), so it is not the default for CQR.

### 7.3 Configuration (proposal)

```yaml
prediction_intervals:
  enabled: true
  levels: [80, 95]            # or quantiles: [0.1, 0.5, 0.9]
  method: auto                # auto (native + CQR if available, else conformal) | conformal | native | cqr
  cqr_score: asymmetric       # asymmetric | scaled | symmetric, see §7.6.4
  cqr_tail_split: adaptive    # adaptive (unused lower-tail alpha goes to the upper tail) | equal (alpha/2 each; over-covers when clipped)
  cqr_pool: group             # group (per coverage.group_col) | marginal | series (shrinkage as in coverage), see §7.6.4
  coverage:                   # see §7.6
    target: series            # marginal | group | series
    group_col: auto           # auto = ABC×XYZ computed by MMF | sb_class | null | <column> | [col1, col2]
    level_group_col: null     # e.g. "abc_class": drives levels_by_group, independent of group_col
    min_scores_per_group: 100 # smaller groups are merged into a global "other" pool
    levels_by_group: null     # e.g. {A: [95], B: [90], C: [80]}; overrides `levels`
    shrinkage_kappa: 10       # series target only: q_i = λ·q_series + (1-λ)·q_group, λ = W/(W+κ)
    pool_horizon: auto        # series target only: pool steps within a series (auto = if W < 19)
  score: signed_normalized    # abs | signed | signed_normalized
  recency_decay: null         # e.g. 0.9 per window (weighted conformal)
  clip_lower: auto            # auto | none | <number>, see §7.3.1
  output_format: quantile_map # quantile_map | lo_hi | both, see §7.4
```

#### 7.3.1 Lower-bound clipping (`clip_lower`)

Residual-based intervals are symmetric or nearly so around `ŷ`. For small `ŷ`
(low-volume or intermittent series), `ŷ − margin` easily falls below the physically
possible minimum. Example: a Croston forecast of 0.3 with a margin of 2 gives
`[-1.7, 2.3]`.

| Value | Behaviour | When to use |
|---|---|---|
| `auto` (default) | Clip at 0 **only if** the training target has `min(y) ≥ 0` across all series. The decision is made once per run and logged to MLflow as the tag `pi_clip_lower`. | Most users. It does the right thing for demand/sales and leaves signed targets alone. |
| `none` | Never clip. | Targets that can be negative: profit/loss, temperature, returns, net flows, differenced series. |
| `<number>` (e.g. `0`) | Always clip: `lo = max(lo, clip_lower)`. | A known physical floor (0 for counts, a contractual minimum, etc.). |

Notes:
- **Coverage is preserved.** If `y ≥ c` always holds, then `y ∈ [lo, hi]` exactly when
  `y ∈ [max(lo, c), hi]`. Clipping only narrows the interval; the conformal guarantee
  is unchanged. If the floor is wrong (`y < c` can happen), clipping *breaks* coverage.
  That is why `auto` checks the data rather than assuming non-negativity.
- **Order of operations.** Add the conformal margin first, then clip. The point
  forecast `ŷ` is never modified by this setting.
- **Clipping is a safety net, not a fix.** A frequently clipped lower bound means a
  symmetric score fits the error distribution poorly. `score: signed_normalized` learns
  separate lower and upper quantiles, so the lower side is usually already tight. MMF
  should report the share of clipped bounds per model in `evaluation_output` as a
  diagnostic.
- **Interaction with existing target clipping.** Local pipelines currently force
  `clip(0)` on the target and on point forecasts (`Forecaster.py:343`,
  `StatsFcForecastingPipeline.py:37,105,141`). For signed targets, `clip_lower: none`
  only makes sense once that hard-coded clipping is made configurable too. This is
  tracked as an open question (§9).
- An analogous `clip_upper` (e.g. `1.0` for rates and percentages) can be added with
  the same semantics, without `auto`.

### 7.4 Output schema changes

`scoring_output` gets new columns, controlled by `output_format`:

| `output_format` | Columns added | Notes |
|---|---|---|
| `quantile_map` (**default**) | `<target>_quantiles: MAP<STRING, ARRAY<DOUBLE>>`, keyed by quantile level (`"0.025"`, `"0.1"`, `"0.9"`, `"0.975"`, ...), one array of length `prediction_length` per key | Schema is stable whatever levels are configured, and it scales to many quantiles (e.g. M5-style 9 quantiles). A central PI at level `L` lives under keys `(1−L/100)/2` and `1−(1−L/100)/2`. |
| `lo_hi` | `<target>_lo_<L>`, `<target>_hi_<L>: ARRAY<DOUBLE>` for each level `L` in `levels` (e.g. `y_lo_80`, `y_hi_80`, `y_lo_95`, `y_hi_95`) | Matches the Snowflake/BigQuery style (`LOWER_BOUND/UPPER_BOUND`). Easiest for SQL, Genie and dashboards (`examples/fresh_retail_net`). The schema changes when levels change. |
| `both` | Both of the above | Convenience for dashboards while keeping the map as the canonical form. |

Key naming in the map: quantile levels are formatted with no trailing zeros
(`"0.1"`, not `"0.10"`). The median `"0.5"` is included only when `method` produces it
(native/CQR). With conformal, the point forecast stays in `<target>`.

`evaluation_output` also gets a `forecast_quantiles: MAP<STRING, ARRAY<DOUBLE>>`
column, with the same keys as above, holding the native quantiles of each backtest
window (null for models without native bands). It is only written when intervals are
enabled. The interval metrics (§8, §11.4) go to their own table, not to
`evaluation_output`.

### 7.5 Phasing

1. **P0**: Layer B with all three `coverage.target` values (default `series` with
   ABC×XYZ shrinkage), MMF-computed grouping columns (ABC, XYZ, Syntetos–Boylan),
   interval metrics, and native pass-through plus CQR for **statsforecast and
   Chronos** (`level=` / `predict_quantiles` in backtest and forecast). Layer B is
   model-agnostic and has no training cost. See §11 for the plan.
2. **P1**: Layer A for TimesFM, Moirai and Prophet, through the same CQR path. This is
   adapter work only: each model returns its quantiles in the `forecast_quantiles`
   format.
3. **P2**: NeuralForecast probabilistic losses (`MQLoss`, `DistributionLoss`, NegBin),
   and LightGBM quantile objective.
4. **P3**: probabilistic hierarchical reconciliation, and online ACI in a monitoring
   job.

### 7.6 Coverage target: marginal vs group vs per-series

This section answers open question 1: **for which population must "90% coverage" hold?**

- **Marginal coverage:** 90% of all (series, date) points fall in their interval.
  Some series can be at 70% and others at 99%.
- **Group-conditional coverage:** 90% holds within each group (e.g. ABC class, store,
  intermittency class).
- **Per-series coverage:** 90% holds for each series individually.

The stronger the target, the more calibration data it needs (§6) and the wider the
intervals become for short or noisy series.

#### 7.6.1 What other products and libraries actually do

| Product / library | Where calibration scores come from | Pool | Effective coverage target |
|---|---|---|---|
| **statsforecast** `ConformalIntervals` | `n_windows` refits at the end of **each series** | Per series × per step: `n_windows` scores (default 2) | **Per series** in intent, but with n=2 the bound is the max of 2 errors, so there is no real guarantee |
| **mlforecast** `PredictionIntervals` (default) | CV windows; scores reshaped `(n_windows, n_series, h)` and quantiled over windows only | Per series × per step | **Per series** (same caveat) |
| **mlforecast** transfer methods (`ScaleAlignment`, `ErrorScaling`, `Recalibration`; on `main`, newer than the MMF pin) | Scores "**pooled over every calibration series**", normalised by a per-series scale (MAD or std of Δy) | All series × per step, signed | **Marginal**, scale-adapted per series. This is the same idea as §7.2. |
| **neuralforecast** `PredictionIntervals` | Same Nixtla scheme as mlforecast | Per series × per step | Per series |
| **Nixtla TimeGPT** | Rolling historical forecasts per series (server-side) | Per series | Per series (details not public) |
| **AutoGluon-TimeSeries** (Croston/ADIDA/IMAPA "conformalized" models) | Up to **5** windows per series, absolute errors "**pooled across the prediction horizon**" | Per series, all steps (n ≤ 5·h), constant width | **Per series**, at the cost of ignoring error growth with horizon |
| AutoGluon / Amazon Forecast / SageMaker DeepAR / Chronos / TimesFM | Model likelihood or quantile loss, trained globally | Implicit (global model) | **Marginal** over the training distribution, with no guarantee |
| BigQuery ARIMA_PLUS, Snowflake, Prophet, Databricks AutoML | Per-series model variance or simulation | n/a (parametric) | Per series in form, with no calibration guarantee |
| Azure AutoML | Std-dev of validation-fold residuals (`ForecastingPipelineWrapper.stddev`) | Not documented | Not documented |

Sources: statsforecast `python/statsforecast/models.py` (`_conformity_scores`),
mlforecast `mlforecast/conformal_prediction.py` (`_add_conformal_*_intervals`,
`_add_transfer_intervals`), AutoGluon
`timeseries/src/autogluon/timeseries/models/local/statsforecast.py`
(`AbstractConformalizedStatsForecastModel`, `max_num_conformalization_windows = 5`),
Azure `ForecastingPipelineWrapper` API reference.

Observations:
1. **Every conformal library defaults to per-series calibration**, but with 2–5 windows.
   They accept a noisy and under-covering result rather than pool. Nobody exposes the
   coverage target as an explicit choice.
2. Nixtla's newest code adds **cross-series pooling with per-series scaling** for
   series with little or no history ("transfer"). This is the first sign of a library
   moving to MMF's situation.
3. AutoGluon shows the other cheap trick: **pool across horizon steps within a series**
   (n = W·h). This gives enough scores, but the interval width no longer grows with
   the horizon.

#### 7.6.2 What safety-stock users need

- Inventory policies set service levels **per SKU** (cycle service level, so that
  safety stock is `z_α·σ_L` per item; Silver, Pyke & Thomas, 2017). In practice the
  targets are set **per class**: ABC/XYZ classes get different service levels, such as
  A=95%, B=90%, C=80% (Teunter, Babai & Syntetos, 2010).
- Planners notice per-SKU failures. A marginal 90% that hides 60% coverage on all
  fast movers is a real business problem. A class-level guarantee is what the
  service-level *policy* is written against.
- Empirical studies of safety stocks show that variance from forecast errors
  underestimates the true uncertainty (Prak, Teunter & Syntetos, 2017). Empirical
  quantiles per SKU work when there is enough history (Trapero, Cardós & Kourentzes,
  2019). This argues for empirical (conformal) calibration over parametric `z·σ`, at
  a level of pooling that the data can support.

Conclusion: the realistic requirement is **group-conditional coverage at the
service-level class**, with per-series adaptivity through scale normalisation. Per-series
guarantees are desirable but only achievable with long backtests.

#### 7.6.3 Three options (all configurable via `coverage.target`)

**Option 1: `marginal`** (global pool, normalised scores)

- Pool: all series of a model × step `j`. The scores `e = (y−ŷ)/σ̂_i` are already
  scale-adapted, so a large series gets a proportionally wide interval.
- Guarantee: `≥ 1−α` on average over all (series, date) points. The split-conformal
  guarantee holds under exchangeability across series and windows (Lei et al., 2018).
- Data need: `W · N ≥ ~100` scores per step, which is trivially met even with the
  default W=3.
- Risk: heterogeneous populations. Intermittent series are typically under-covered
  and smooth series over-covered, because one pooled quantile fits neither.
- When to use: quick start, homogeneous series, dashboards, when no grouping column
  exists.

**Option 2: `group`** (Mondrian / group-conditional conformal) — choose when a
per-group *guarantee* is required

- Pool: series sharing a `group_col` value × step `j`. These are Mondrian conformal
  predictors (Vovk et al., 2003): running split conformal separately inside each
  category gives the `1−α` guarantee **within every group**.
- `group_col` candidates:

  | Grouping | Source | Captures | M5 result (§7.6.4) |
  |---|---|---|---|
  | **ABC×XYZ** (`auto`): ABC = 80/15/5 volume Pareto, XYZ = terciles of CV | MMF, from the target | Importance × variability | **Best of all groupings** in 12/12 configs (MSIS), best ABC balance |
  | Syntetos–Boylan (`sb_class`): ADI 1.32 / CV² 0.49 cut-offs | MMF, from the target | Error-distribution shape | Best balance across its own classes, otherwise ≈ marginal |
  | ABC alone | MMF (volume; revenue if a price column exists) | Importance | Second best |
  | Error tier (terciles of seasonal-naive MAE / mean) | MMF | Difficulty | ≈ marginal |
  | `forecastability_class` (profiling) | Profiling table, if present | Signal vs noise (2 groups) | Not tested |
  | Business column (category, store, hierarchy level) | User | Domain segment | ≈ marginal (no gain) |

  A list (e.g. `[sb_class, category]`) builds groups from the combinations. Small
  groups are merged as below.
- `levels_by_group` maps the service-level policy directly, e.g. A: 95%, C: 80%. The
  service-level column (`level_group_col`, typically ABC) is separate from the
  calibration grouping (`group_col`). Business importance decides *which* level a group
  gets; error shape decides *how* the quantile is pooled.
- Data need: `W · N_group ≥ min_scores_per_group` (default 100; ≥ ~40 is enough for
  a finite 95% bound). Smaller groups are merged into a pooled "other" group, and the
  merge is logged.
- If `group_col` is null, this is identical to `marginal`.
- Extension (later): Gibbs, Cherian & Candès (2025) give conformal methods with
  coverage guaranteed over *overlapping* groups, e.g. both "store" and "category" at
  once.

**Option 3: `series`** (per-series, with shrinkage to the group) — **recommended default**

- Pool: the series' own scores. Two tricks keep it usable when W is small:
  1. **Horizon pooling** (`pool_horizon`). Pool all steps of the series
     (n = W·h, as AutoGluon does), then restore error growth with a **step profile
     learned at group level**: `q_i,j = q_i · r_j`, where `r_j` is the ratio of the
     step-`j` quantile to the all-steps quantile in the series' group. The series sets
     the level and the group sets the shape.
  2. **Shrinkage** (partial pooling, credibility-style):
     `q_i = λ_i · q_series,i + (1−λ_i) · q_group`, with `λ_i = W_i/(W_i+κ)` and
     `κ = shrinkage_kappa` (default 10, tuned on M5, §7.6.4). Short series fall back toward the group
     quantile, and series with a long history get their own.
     **Count windows, not points.** The weight must use the number of windows `W_i`
     (independent forecast origins), not `W_i·h`. Errors within a window are strongly
     correlated, so `W·h` overstates the evidence. On M5, `λ = W·h/(W·h+κ)` (≈ 0.8 at
     W=3) under-covered badly (0.74 at 80%, 0.89 at 95%), while `λ = W/(W+κ)` was well
     calibrated (§7.6.4). With κ=20 that is λ ≈ 0.13 at W=3 and 0.375 at W=12; with the
     default κ=10 it is 0.23 and 0.55.
- Guarantee: **no finite-sample guarantee** for the shrunk quantile. Coverage must be
  checked empirically per series (§8). Exact per-series conformal is recovered when
  `λ_i → 1`, which needs W ≥ 19 for 90% (§6.1).
- Data need: works with W=3 thanks to shrinkage. It becomes mostly series-specific
  once `W ≳ κ`.
- When to use: the default. It had the best interval score of all the universal
  methods on M5 at near-nominal coverage, for W=3 and W=12.

Summary:

| | `marginal` | `group` | `series` (default) |
|---|---|---|---|
| Guarantee | Average over all points | Within each group | Empirical only (exact when W large) |
| Works with default W=3 | ✅ | ✅ if groups have ≥ ~35 series | ✅ via shrinkage (mostly group-driven at W=3) |
| M5 MSIS, AutoETS 95%, W=3 (§7.6.4) | 17.4 | 16.1 (ABC×XYZ) | **13.9** (ABC×XYZ, κ=10) |
| Adaptive to series scale | ✅ (normalised scores) | ✅ | ✅✅ |
| Adaptive to series *shape* (e.g. erratic vs smooth) | ❌ | ✅ (if the grouping captures it) | ✅ |
| Matches safety-stock practice | Weak | **ABC/XYZ service levels** | Per-SKU service level |
| Extra compute | none | none | none (all from `evaluation_output`) |

All three are computed from the same exploded backtest scores, so MMF can report
coverage for all three views in `evaluation_output` whichever one drives the
intervals. Users then see the trade-off on their own data before switching.

#### 7.6.4 Experiment: choosing the options on M5

Notebook: `docs/specs/experiments/pi_grouping_m5.py`. Raw results (232 rows, from the
confirmation run): `docs/specs/experiments/pi_grouping_m5_results.json`, plus
`pi_grouping_m5_results_seed42.json` for the second seed. It ran on serverless in the
`fe-sandbox-air-eu-test` workspace (runs `878662839893805`, `324298550294595` and
`458608515774101`), taking about 20 min each.

**Setup**

- **Data:** 4,966 random M5 item×store series (seed 7). Leading zeros were trimmed,
  and only series with at least a year of history before calibration were kept. The
  mix is 77% intermittent, 13% lumpy, 8% smooth and 2% erratic (Syntetos–Boylan), and
  32/32/36% A/B/C.
- **Models:** SeasonalNaive(7), CrostonSBA and AutoETS(ZNZ, 7), refit for every window
  (statsforecast `cross_validation`, as MMF local models do). h = 28.
- **Calibration:** windows strictly before the test period, with stride 14. W=3 is
  the last 3 windows (MMF-default-like); W=12 is all of them.
- **Test:** 4 non-overlapping 28-day windows (2016-02-01 to 2016-05-22), never seen
  during calibration.
- **Scoring:** signed scores normalised by the in-sample seasonal-naive MAE, exact
  finite-sample conformal ranks, and the lower bound clipped at 0.
- **Grouping features** use only data from before the first calibration window.
- **Metrics:** coverage, MSIS (interval score / seasonal-naive MAE; lower is better),
  and the worst coverage gap across the four Syntetos–Boylan classes and across A/B/C.

**Results**, AutoETS with W=3 (CrostonSBA and SeasonalNaive show the same ranking):

| Method | Coverage at 80% | MSIS at 80% | Coverage at 95% | MSIS at 95% | Worst class gap at 95% (SB / ABC) |
|---|---|---|---|---|---|
| marginal | 0.773 | 7.50 | 0.942 | 17.4 | 0.036 / 0.055 |
| group: Syntetos–Boylan | 0.773 | 7.48 | 0.943 | 17.2 | 0.014 / 0.047 |
| group: ABC | 0.773 | 7.43 | 0.942 | 16.5 | 0.017 / 0.011 |
| **group: ABC×XYZ** | 0.775 | 7.40 | 0.942 | 16.1 | **0.012 / 0.011** |
| group: category / store / error tier / SB×category | ≈ marginal | ≈ marginal | ≈ marginal | ≈ marginal | no better |
| series, λ = W/(W+κ), κ=20, shrunk to SB groups | 0.775 | 7.22 | 0.944 | 15.5 | 0.015 / 0.044 |
| series, λ = W/(W+κ), κ=20, shrunk to ABC×XYZ | 0.777 | 7.15 | 0.942 | 14.7 | 0.012 / 0.011 |
| **series, λ = W/(W+κ), κ=10, shrunk to ABC×XYZ** (recommended default) | 0.777 | **7.00** | 0.941 | **13.9** | **0.013 / 0.013** |
| series, λ = W·h/(W·h+κ) | 0.737 | 6.70 | 0.890 | 13.4 | 0.068 / 0.092 |
| Nixtla style: per series × step, absolute | 0.700 | 7.05 | **0.700** | 19.6 | 0.258 / 0.274 |
| AutoGluon style: per series, steps pooled, absolute | 0.714 | 6.70 | 0.900 | 12.9 | 0.055 / 0.063 |
| AutoETS native (statsforecast `level=`) | 0.872 | **5.93** | 0.944 | **9.9** | 0.015 / 0.015 |

With W=12 at 95%, the recommended default scores 11.7, against 16.2 for ABC×XYZ groups.
Nixtla-style recovers at 80% (0.806 coverage, 6.43) but still under-covers at 95%
(0.894).

**Confirmation run** (run `324298550294595`; same seed and series). The extended
notebook reproduced all 148 rows of the first run exactly, then added series shrinkage
toward ABC×XYZ and a κ sweep:

- **Shrinkage target:** shrinking toward ABC×XYZ beats shrinking toward Syntetos–Boylan
  in **all 48** model × level × W × κ combinations. It also has a much smaller ABC
  coverage gap (≈ 0.01 vs ≈ 0.04 at 95%).
- **κ trade-off** (ABC×XYZ shrinkage, averaged over the 12 model × level × W
  combinations):

  | κ | Mean coverage gap vs nominal | MSIS vs κ=20 |
  |---|---:|---:|
  | 5 | −1.8 pt | −6.2% |
  | **10** | **−1.3 pt** | **−3.8%** |
  | 20 | −1.1 pt | 0 |
  | 40 | −1.1 pt | +4.2% |

  κ=10 buys about 4% sharper intervals for 0.2 points of coverage. κ=5 starts to
  under-cover visibly at 95% (0.930–0.943). → **default `shrinkage_kappa: 10`**. Use 20
  when coverage matters more than width.

**Second seed** (run `458608515774101`, seed 42). This run used 4,955 different
series with the same mix: 75% intermittent, 13% lumpy, 9% smooth, 3% erratic, and
32/32/35% A/B/C. Every conclusion holds:

| Check | Seed 7 | Seed 42 |
|---|---|---|
| ABC×XYZ is the best grouping | 12/12 | 12/12 |
| Series shrinkage beats the best grouping (MSIS) | 12/12 | 12/12 |
| ABC×XYZ shrinkage beats SB shrinkage (MSIS) | 48/48 | 47/48 ¹ |
| κ=5 / 10 / 20 / 40: mean coverage gap | −1.8 / −1.3 / −1.1 / −1.1 pt | −1.6 / −1.1 / −1.0 / −1.0 pt |
| κ=5 / 10 / 40: MSIS vs κ=20 | −6.2 / −3.8 / +4.2% | −8.3 / −4.9 / +5.3% |
| AutoETS 95%, W=3, MSIS: marginal / ABC×XYZ / default / native | 17.4 / 16.1 / 13.9 / 9.9 | 17.7 / 16.6 / 13.9 / 10.4 |
| AutoETS 95%, W=3, Nixtla-style coverage | 0.700 | 0.707 |

¹ The one loss is CrostonSBA at 80% with W=3 and κ=5: 6.882 vs 6.878, a tie.

Across the 232 matching rows, coverage differs between the seeds by 0.2 points on
average, and by at most 1.0 point. The 1–7% MSIS gaps between groupings are therefore
larger than the seed-to-seed noise.

**CQR on native bands** (run `307705752027548`, seed 7, 31 min). Raw results:
`pi_grouping_m5_results_cqr.json`. The native AutoETS and SeasonalNaive bands are
refit on the same calibration windows, then corrected by a conformal quantile of their
own misses (Romano et al., 2019). Both calibration and test clip the native lower bound
at 0. Three scores were tested, each pooled over all series, per ABC×XYZ group, or per
series shrunk toward ABC×XYZ (κ=10):

- `sym`: one score `max(lo − y, y − hi)/σ` at `1−α`.
- `asym`: one score per tail, `(lo − y)/σ` and `(y − hi)/σ`, each at `1−α/2`.
- `scaled`: `|y − ŷ| / native half-width` at `1−α`, giving `ŷ ± q·half-width`. This
  is normalised conformal with the native width as the local scale.

Averaged over the 8 model × level × W combinations:

| Variant | MSIS vs native | Worst case | Mean coverage gap vs nominal |
|---|---:|---:|---:|
| native (no correction) | 0 | 0 | +5.1 pt (SN 80%: +11.6, AutoETS 80%: +7.2) |
| `sym`, any pool | −0.6 to −2.8% | 0% | +4.2 to +5.3 pt |
| `asym`, series shrink κ=10 | **−7.3%** | −0.5% | +3.6 pt |
| `asym`, ABC×XYZ | −5.7% | +0.5% | +3.5 pt |
| **`scaled`, ABC×XYZ** | +0.6% | +8.7% | **−1.1 pt** |
| `scaled`, marginal | +1.0% | +5.7% | −1.1 pt |

Selected rows, at 95%, W=3:

| | AutoETS coverage | AutoETS MSIS | SeasonalNaive coverage | SeasonalNaive MSIS |
|---|---:|---:|---:|---:|
| native | 0.944 | 9.94 | 0.972 | 11.34 |
| `asym`, series κ=10 | 0.968 | 9.10 | 0.963 | 10.23 |
| `scaled`, ABC×XYZ | 0.943 | 9.65 | 0.945 | 10.92 |
| residual conformal default (`series`, ABC×XYZ, κ=10) | 0.941 | 13.92 | 0.946 | 16.33 |

- **`sym` is close to a no-op.** On intermittent series, `y = 0` with a clipped
  lower bound of 0 gives a score of 0 on most points, so the quantile barely moves.
- **`asym` is the sharpest.** It beats native on MSIS in all 8 combinations, by 7% on
  average and by up to 10% for SeasonalNaive. It over-covers by about 3.5 points,
  however. Once the lower bound is clipped at 0, the lower tail almost never misses,
  so the realised coverage tends toward `1−α/2`. The one-tailed variant below fixes
  this.
- **`scaled` hits nominal coverage.** It turns native AutoETS's 0.87 at 80% into
  0.78–0.79, and SeasonalNaive's 0.92 into 0.78. At 95%, its MSIS is 2–4% better than
  native. At 80%, the narrower and correctly sized interval scores worse on MSIS (AutoETS
  6.25 vs 5.93), because native over-coverage is cheap there.
- **W=3 is enough.** Every variant gives nearly the same numbers at W=3 and W=12
  (`scaled`, ABC×XYZ: +0.9% vs +0.3% MSIS). The native band already provides the
  per-series, per-step shape, and CQR only estimates one correction per pool.
- **Pooling matters little.** Shrinkage per series helps `asym` slightly. For `scaled`,
  the ABC×XYZ pool is the best choice, and per-series shrinkage adds nothing.
- CQR on native bands keeps a large lead over residual conformal: AutoETS 95% at W=3
  scores 9.1–9.7, against 13.9 for the residual conformal default.

**One-tailed asymmetric CQR, on two seeds** (runs `154621151140624` with seed 7 and
`650019621875992` with seed 42; raw results in `pi_grouping_m5_results_cqr_1t_seed7.json`
and `..._seed42.json`). The lower correction is computed as in `asym`. The lower tail
then gets its realised miss rate `m_lo` on the calibration windows, and the upper tail
is calibrated at `1 − α + m_lo` (capped at `1 − α/2`). The seed-7 run reproduced all
304 earlier rows exactly. Averaged over the 8 model × level × W combinations:

| Variant | MSIS vs native, seed 7 / 42 | Mean coverage gap, seed 7 / 42 | Worst coverage gap |
|---|---:|---:|---:|
| `asym` (equal split), series κ=10 | −7.3% / −8.1% | +3.6 / +3.7 pt | 6.5 pt |
| **`asym_1t`, series κ=10** | **−2.4% / −3.4%** | **−0.4 / −0.4 pt** | **1.0 pt** |
| `asym_1t`, ABC×XYZ | −0.9% / −0.9% | −0.4 / −0.4 pt | 1.2 pt |
| `scaled`, ABC×XYZ | +0.6% / −0.5% | −1.1 / −1.1 pt | 2.1 pt |

- **`asym_1t` with per-series shrinkage is the best calibrated variant.** It is never
  more than 1 point from nominal. Its MSIS beats `scaled` (ABC×XYZ) by about 3% on
  average on both seeds, and in 7 and 6 of the 8 combinations. Fewer series are
  under-covered by more than 10 points: 9.9% against about 11.8%.
- At 95% it beats native on MSIS in every combination (−1.4% to −8.8%). At 80% it
  costs MSIS for AutoETS (+5 to +7%), as every nominal variant does: native AutoETS
  over-covers there (0.87), which MSIS barely penalises.
- Per-series shrinkage helps `asym_1t` more than it helps `scaled`, on these two
  models. The Chronos run below shows it does not generalise.
- Seeds 7 and 42 agree on the ranking and on the coverage figures to within 0.1 point.

**CQR on Chronos quantiles** (run `252308978398800`,
seed 7, same 4,966 series, 84 min; raw results in `pi_grouping_m5_results_chronos.json`).
Chronos-Bolt-small and Chronos-2 were run zero-shot on CPU, on the driver, with a
512-day context. Their 2.5/10/50/90/97.5% quantiles serve as the native band, and the
median as the point forecast. Every method above was applied to them unchanged. The
SeasonalNaive rows reproduce the earlier seed-7 run exactly.

Native bands:

| Model | Coverage 80% | MSIS 80% | Coverage 95% | MSIS 95% | Series under-covered by >10 pt at 95% |
|---|---:|---:|---:|---:|---:|
| AutoETS (for reference) | 0.872 | 5.93 | 0.944 | 9.94 | 4.5% |
| Chronos-2 | 0.783 | 5.90 | 0.966 | 9.95 | 2.4% |
| Chronos-Bolt-small | 0.797 | 6.04 | **0.800** | **15.05** | **61.8%** |

Chronos-2 is as sharp as AutoETS and close to calibrated. Chronos-Bolt's 95% band is
just its 80% band, because the model clamps to its trained 0.1–0.9 quantiles.

`asym_1t` CQR, W=3, by pool:

| | Chronos-2 95% | Chronos-Bolt 80% | Chronos-Bolt 95% |
|---|---|---|---|
| native | 0.966 / 9.95 | 0.797 / 6.04 | 0.800 / 15.05 |
| **ABC×XYZ group** | 0.960 / 9.99 | 0.815 / 6.03 | **0.943 / 10.65** |
| series, κ=10 | 0.958 / 10.06 | 0.775 / 5.98 | **0.877** / 10.25 |

(coverage / MSIS)

Across the four models with native bands (AutoETS, SeasonalNaive, Chronos-Bolt and
Chronos-2), × 2 levels × W ∈ {3, 12}:

| CQR pool (`asym_1t`) | MSIS vs native | Mean coverage gap | Worst coverage gap | Series under-covered by >10 pt |
|---|---:|---:|---:|---:|
| **ABC×XYZ group** | **−4.1%** | **+0.3 pt** | **1.8 pt** | **9.9%** |
| marginal | −3.3% | −0.4 pt | 2.5 pt | 12.8% |
| series, κ=10 | −5.3% | −1.7 pt | **9.3 pt** | 15.5% |
| `scaled`, ABC×XYZ | +3.5% | −1.2 pt | 2.6 pt | 11.1% |
| residual conformal (`series`, ABC×XYZ, κ=10) | +15.6% | −1.5 pt | 2.9 pt | 17.3% |

- **CQR fixes Chronos-Bolt's broken 95% band.** With the ABC×XYZ pool, coverage goes
  from 0.80 to 0.94–0.95, MSIS from 15.0 to 10.7 (−29%), and the share of badly
  under-covered series from 62% to 8%. Even W=3 is enough.
- **Per-series shrinkage fails when the needed correction is large.** For
  Chronos-Bolt at 95% it stays at 0.86–0.88, even at W=12. The per-series 97.5%
  quantile is estimated from 3 to 12 windows of strongly correlated steps, so it is
  biased low. For the statsforecast models the correction is small, so the bias did
  not show.
- **CQR does no harm on an already good band.** For Chronos-2 the ABC×XYZ pool changes
  MSIS by +0.2% and coverage by at most 1.8 points. At 80% it halves the share of
  badly under-covered series, from 24% to 12%.
- `cqr_score: scaled` with per-series shrinkage blows up on Chronos (MSIS up to 370 at
  95%, W=12). Some native half-widths are near zero, and the 1e-3·σ floor is too low.
  Do not offer that combination.
- Residual conformal on the Chronos median is again far behind native + CQR (+21%
  MSIS for Chronos-2).

→ **`method: auto` = native + CQR** for models with native bands, with
`cqr_score: asymmetric`, `cqr_tail_split: adaptive`, and **`cqr_pool: group`**
(ABC×XYZ by default) as the defaults. `cqr_pool: series` is about 1–2% sharper on
well-specified bands (statsforecast). It under-covers when the native band is badly
miscalibrated, though, and that is exactly the case CQR is for.
`cqr_tail_split: equal` gives sharper but over-covering bands. `method: native` keeps
the uncorrected bands.

**Findings**

1. **Grouping column:** ABC×XYZ ranked first among all groupings in every one of the 12
   model × level × W combinations. Syntetos–Boylan gives the best balance across its
   own classes. Business columns (category, store) bring nothing over marginal. The
   differences are small, 1–7% MSIS. → **`group_col: auto` = ABC×XYZ.**
2. **Coverage target:** per-series shrinkage counting *windows* is the best universal
   method. It is near-nominal, and it beats the best grouping on MSIS in all 12 model ×
   level × W combinations. Shrinking toward ABC×XYZ with κ=10 is the best variant
   (confirmation run). → **`coverage.target: series`, `group_col: auto` (ABC×XYZ),
   `shrinkage_kappa: 10`** by default. `group` remains available when a per-group
   guarantee is required.
3. **Normalisation is not enough.** At equal coverage, pooled methods have a much worse
   interval score than native intervals (AutoETS at 95%: 16–17 vs 9.9; the
   recommended default narrows the gap to 13.9). The seasonal-naive scale does
   not size each series' interval well, and misses on spiky series are large and
   heavily penalised. This is why per-series adaptivity matters more than the choice of
   group.
4. **Native intervals first, then CQR.** Where available (AutoETS, SeasonalNaive),
   native intervals beat every residual conformal variant on MSIS, but over-cover by
   up to 12 points at 80% (Chronos-Bolt under-covers by 15 points at 95%).
   One-tailed asymmetric CQR, pooled per ABC×XYZ group, brings all four models
   within 2 points of nominal coverage, and improves MSIS by 4% on average, even
   with W=3.
   → `method: auto` = native + CQR.
5. **The product baselines fail as §6 predicts.** With W=3, Nixtla-style calibration
   collapses to the maximum of 3 errors (0.70 coverage at 95%). AutoGluon-style
   horizon pooling is sharp but under-covers by 5–9 points.
6. All pooled methods under-cover by about 2–3 points at 80%. This is likely
   calibration-to-test drift (Aug 2015–Jan 2016 vs Feb–May 2016) and correlated
   scores. Recency weighting is worth testing.

**Caveats and follow-ups**

- One dataset (mostly intermittent), one test period, local models only. No global or
  foundation models were tested. Two seeds (two different sets of series) agree.
- κ was tuned on the same test period it is reported on. The second seed picks the
  same κ, but a second dataset or test period would be a stronger check.
- CQR was tested on statsforecast bands (two seeds) and Chronos bands (one seed).
  TimesFM, Moirai and global models (NeuralForecast, LightGBM) are untested. So is
  recency weighting.

---

## 8. How to evaluate intervals

| Metric | Formula / idea | Use |
|---|---|---|
| **Empirical coverage** | mean of `1[lo ≤ y ≤ hi]`, per level and per step | Calibration check (should be close to nominal) |
| **Mean interval width** (often scaled) | mean of `(hi − lo)` / scale | Sharpness: narrower is better at equal coverage |
| **Winkler / interval score** | `(hi−lo) + (2/α)(lo−y)·1[y<lo] + (2/α)(y−hi)·1[y>hi]` | A single proper score for one interval (Gneiting & Raftery, 2007) |
| **MSIS** | Interval score / in-sample seasonal naive MAE | M4 official interval metric (Makridakis et al., 2020) |
| **Pinball (quantile) loss** | `max(τ(y−q), (τ−1)(y−q))` | Per quantile |
| **Weighted quantile loss (wQL)** | `2 Σ pinball / Σ abs(y)` | Amazon Forecast, GluonTS, AutoGluon |
| **WSPL** | Scaled pinball, weighted over 9 quantiles | M5-Uncertainty official metric |
| **CRPS** | Integral of pinball over all τ (approx. mean wQL over many τ) | Full-distribution score (Chronos, GIFT-Eval) |

Following Gneiting et al. (2007), the goal is to **maximise sharpness subject to
calibration**. Report coverage *and* a proper score. Coverage alone can be gamed:
infinitely wide intervals always cover.

---

## 9. Open questions

1. ~~Marginal vs per-series coverage? Which grouping column?~~ **Decided: configurable,
   chosen on M5** (§7.6.4). `coverage.target: series` (shrinkage with `λ = W/(W+κ)`)
   by default, `group_col: auto` = ABC×XYZ, `shrinkage_kappa: 10`, and `method: auto`
   (native + one-tailed asymmetric CQR pooled per ABC×XYZ group first). ABC×XYZ
   shrinkage and κ were confirmed in a second run and on a second seed. CQR was
   tested on Chronos-Bolt and Chronos-2. *Remaining:* a second dataset
   (`fresh_retail_net`), TimesFM and global models, and recency weighting.
2. ~~Output schema?~~ **Decided:** quantile map by default, with lo/hi columns
   available (`output_format: quantile_map | lo_hi | both`, §7.4).
   *Remaining:* whether the `fresh_retail_net` Genie views should use `both`.
3. ~~Change README defaults to W ≥ 10?~~ **Decided: no.** Defaults stay at W = 3. Use
   the cross-series and horizon pooling and shrinkage workarounds in §6.4. The longer
   backtest settings in §6.3 are documented as opt-in.

4. ~~Where does calibration live?~~ **Decided: a separate step.**
   `run_calibration` runs inside `run_forecast` between evaluation and scoring. It
   reads `evaluation_output` and writes `calibration_output`; scoring joins on that
   table. It can also be run on its own, to recalibrate without retraining (§11.1).
5. ~~Joint trajectory coverage or per-step?~~ **Decided: per-step**, as every product
   does. Joint coverage is out of scope.
6. ~~Interaction with model selection?~~ **Decided:** calibrate the selected model as
   it is, and report its coverage. The optimism from selecting on the same windows is
   small compared with the 1–2 point differences measured. A held-out last window can
   be added later.
7. ~~Make the hard-coded `clip(0)` on local models configurable?~~ **Decided: not in
   P0.** Document that `clip_lower: none` has no effect on statsforecast models,
   because their target and forecasts are clipped at 0
   (`StatsFcForecastingPipeline.py:37,105,141`). Making it configurable changes point
   forecasts and is a separate change.
8. ~~Where does the backtest store native bands?~~ **Decided:** a new
   `forecast_quantiles` column in `evaluation_output` (§7.4). It is only filled when
   intervals are enabled. Existing tables are migrated with `mergeSchema`.
9. ~~How is the configuration passed?~~ **Decided: both ways.** A
   `prediction_intervals` block in the YAML config is the default, and a
   `run_forecast(prediction_intervals={...})` argument overrides it, like the other
   parameters.
10. ~~Where do interval metrics go?~~ **Decided: a separate table**,
    `<evaluation_output>_interval_metrics` (`run_forecast(interval_metrics_output=...)`).
    The first choice was new `metric_name` rows in `evaluation_output`. It was reversed
    during implementation: the post-evaluation notebook, Skill 5's model selection, the
    Results Explorer app and users' own queries average `metric_value` per series and
    model without filtering `metric_name`, and reconciliation explodes the `forecast`
    arrays. The extra rows would have silently corrupted model selection (§11.4).
11. ~~Where does ABC×XYZ come from?~~ **Decided:** it is computed inside
    `run_calibration` from the training data (the last 365 days before the first
    backtest window). It is stored in `calibration_output`, and `group_col: <column>`
    overrides it with a user column.

*Deferred:* a second dataset (`fresh_retail_net`), recency weighting, `both` for the
Genie views (Q2), the Moirai sample count, and TimesFM and global models (P1/P2).

---

## 10. References

Conformal prediction & time series
- Vovk, V., Gammerman, A., Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer.
- Papadopoulos, H., Gammerman, A., Vovk, V. (2008). Normalized nonconformity measures for regression conformal prediction. *AIA*.
- Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R., Wasserman, L. (2018). Distribution-free predictive inference for regression. *JASA* 113(523).
- Romano, Y., Patterson, E., Candès, E. (2019). Conformalized quantile regression. *NeurIPS*.
- Stankeviciute, K., Alaa, A., van der Schaar, M. (2021). Conformal time-series forecasting. *NeurIPS*.
- Gibbs, I., Candès, E. (2021). Adaptive conformal inference under distribution shift. *NeurIPS*.
- Xu, C., Xie, Y. (2021). Conformal prediction interval for dynamic time-series (EnbPI). *ICML*.
- Zaffran, M., Féron, O., Goude, Y., Josse, J., Dieuleveut, A. (2022). Adaptive conformal predictions for time series. *ICML*.
- Barber, R. F., Candès, E., Ramdas, A., Tibshirani, R. (2023). Conformal prediction beyond exchangeability. *Annals of Statistics* 51(2).
- Xu, C., Xie, Y. (2023). Sequential predictive conformal inference for time series (SPCI). *ICML*.
- Angelopoulos, A., Candès, E., Tibshirani, R. (2023). Conformal PID control for time series prediction. *NeurIPS*.
- Angelopoulos, A., Bates, S. (2023). Conformal prediction: a gentle introduction. *Foundations and Trends in ML* (arXiv:2107.07511).
- Vovk, V., Lindsay, D., Nouretdinov, I., Gammerman, A. (2003). Mondrian confidence machine. Technical report, Royal Holloway, University of London.
- Gibbs, I., Cherian, J., Candès, E. (2025). Conformal prediction with conditional guarantees. *JRSS-B* (arXiv:2305.12616).

Classical forecasting intervals & evaluation
- Williams, W. H., Goodman, M. L. (1971). A simple method for the construction of empirical confidence limits for economic forecasts. *JASA* 66(336).
- Chatfield, C. (1993). Calculating interval forecasts. *JBES* 11(2).
- Tashman, L. (2000). Out-of-sample tests of forecasting accuracy: an analysis and review. *IJF* 16(4).
- Gneiting, T., Raftery, A. (2007). Strictly proper scoring rules, prediction, and estimation. *JASA* 102(477).
- Gneiting, T., Balabdaoui, F., Raftery, A. (2007). Probabilistic forecasts, calibration and sharpness. *JRSS-B* 69(2).
- Lee, Y. S., Scholtes, S. (2014). Empirical prediction intervals revisited. *IJF* 30(2).
- Bergmeir, C., Hyndman, R., Koo, B. (2018). A note on the validity of cross-validation for evaluating autoregressive time series prediction. *CSDA* 120.
- Hyndman, R., Athanasopoulos, G. *Forecasting: Principles and Practice*, 3rd ed., §5.5 & §5.9. https://otexts.com/fpp3/
- Makridakis, S., Spiliotis, E., Assimakopoulos, V. (2020). The M4 Competition: 100,000 time series and 61 forecasting methods. *IJF* 36(1).
- Makridakis, S. et al. (2022). The M5 uncertainty competition: results, findings and conclusions. *IJF* 38(4).
- Petropoulos, F. et al. (2022). Forecasting: theory and practice. *IJF* 38(3).

Intermittent demand
- Syntetos, A., Boylan, J. (2005). The accuracy of intermittent demand estimates. *IJF* 21(2).
- Snyder, R., Ord, J. K., Beaumont, A. (2012). Forecasting the intermittent demand for slow-moving inventories: a modelling approach. *IJF* 28(2).
- Kolassa, S. (2016). Evaluating predictive count data distributions in retail sales forecasting. *IJF* 32(3).

Inventory / safety stock
- Silver, E., Pyke, D., Thomas, D. (2017). *Inventory and Production Management in Supply Chains*, 4th ed. CRC Press.
- Teunter, R., Babai, M. Z., Syntetos, A. (2010). ABC classification: service levels and inventory costs. *Production and Operations Management* 19(3).
- Prak, D., Teunter, R., Syntetos, A. (2017). On the calculation of safety stocks when demand is forecasted. *EJOR* 256(2).
- Trapero, J. R., Cardós, M., Kourentzes, N. (2019). Empirical safety stock estimation based on kernel and GARCH models. *Omega* 84.

Models
- Taylor, S., Letham, B. (2018). Forecasting at scale (Prophet). *The American Statistician* 72(1).
- Wen, R. et al. (2017). A multi-horizon quantile recurrent forecaster (MQ-RNN). *NeurIPS TS workshop*.
- Salinas, D., Flunkert, V., Gasthaus, J., Januschowski, T. (2020). DeepAR. *IJF* 36(3).
- Ansari, A. F. et al. (2024). Chronos: learning the language of time series. *TMLR*.
- Das, A., Kong, W., Sen, R., Zhou, Y. (2024). A decoder-only foundation model for time-series forecasting (TimesFM). *ICML*.
- Woo, G. et al. (2024). Unified training of universal time series forecasting transformers (Moirai). *ICML*.
- Panagiotelis, A., Gamakumara, P., Athanasopoulos, G., Hyndman, R. (2023). Probabilistic forecast reconciliation. *EJOR* 306(2).

Library / product docs (accessed 2026-10)
- statsforecast `ConformalIntervals`: https://nixtlaverse.nixtla.io/statsforecast/docs/tutorials/conformalprediction.html, source `python/statsforecast/{models,utils}.py`
- mlforecast prediction intervals: https://nixtlaverse.nixtla.io/mlforecast/docs/how-to-guides/prediction_intervals.html
- neuralforecast core (`PredictionIntervals`, `MQLoss`, `DistributionLoss`, `IQLoss`): https://github.com/Nixtla/neuralforecast
- sktime forecasting API: https://www.sktime.net/en/stable/api_reference/forecasting.html
- MAPIE time series: https://mapie.readthedocs.io
- Chronos: https://github.com/amazon-science/chronos-forecasting, model configs on Hugging Face (`amazon/chronos-bolt-*`, `amazon/chronos-2`)
- TimesFM 2.5: https://github.com/google-research/timesfm
- BigQuery ML.FORECAST: https://cloud.google.com/bigquery/docs/reference/standard-sql/bigqueryml-syntax-forecast
- Snowflake FORECAST: https://docs.snowflake.com/en/sql-reference/classes/forecast/methods/forecast
- Azure AutoML forecasting: https://learn.microsoft.com/en-us/azure/machine-learning/how-to-automl-forecasting-faq
- Nixtla TimeGPT intervals/quantiles: https://nixtla.github.io/nixtlar/articles/prediction-intervals.html
- Prophet uncertainty intervals: https://facebook.github.io/prophet/docs/uncertainty_intervals.html
- mlforecast conformal & transfer intervals (source): https://github.com/Nixtla/mlforecast/blob/main/mlforecast/conformal_prediction.py
- AutoGluon conformalized local models (source): https://github.com/autogluon/autogluon/blob/master/timeseries/src/autogluon/timeseries/models/local/statsforecast.py
- Azure `ForecastingPipelineWrapper` (validation-fold `stddev`): https://learn.microsoft.com/en-us/python/api/azureml-training-tabular/azureml.training.tabular.models.forecasting_pipeline_wrapper.forecastingpipelinewrapper

---

## 11. Implementation plan (P0)

This section follows the decisions in §9. P0 covers residual conformal for every model,
plus native + CQR for statsforecast and Chronos.

### 11.1 Pipeline flow

```
run_forecast
 ├─ evaluate_models    backtest as today; when intervals are enabled, models with native
 │                     bands also write forecast_quantiles per window -> evaluation_output
 ├─ run_calibration    NEW: reads evaluation_output (this run_id) and the training data
 │                     1. per-series features: σ (seasonal-naive MAE), ABC, XYZ, SB
 │                     2. scores: residual (signed, normalised) or CQR (asym, adaptive split)
 │                     3. pooled quantiles + shrinkage per (model, pool, step, level)
 │                     4. out-of-sample interval metrics (§11.4) -> <evaluation_output>_interval_metrics
 │                     -> calibration_output
 └─ score_models       forecast as today, plus native bands where available; joins
                       calibration_output -> interval columns in scoring_output (§7.4)
```

`run_calibration` is also public, so that a past run can be recalibrated with other
settings (levels, `coverage.target`, κ) without retraining: `run_calibration(spark,
evaluation_output=..., run_id=..., prediction_intervals={...})`. Because scoring needs
the calibration, a recalibration only applies to scoring runs that happen after it.

`prediction_intervals.enabled` defaults to **`false`** in the shipped
`forecasting_conf_*.yaml`. Existing pipelines and table schemas are therefore unchanged
until a user opts in. The `true` shown in §7.3 is the recommended setting.

### 11.2 Files to change

| File | Change |
|---|---|
| `mmf_sa/__init__.py` | `run_forecast(prediction_intervals=None)`, merged over the YAML block. Calls `run_calibration` between evaluation and scoring. Exports `run_calibration`. |
| `mmf_sa/forecasting_conf_{hourly,daily,weekly,monthly}.yaml` | Default `prediction_intervals` block (§7.3) with `enabled: false`. |
| `mmf_sa/models/models_conf.yaml` | Per-model `native_intervals: true\|false` capability flag. Initially true for the statsforecast models listed in §3.1 and for the Chronos models. |
| `mmf_sa/models/abstract_model.py` | `backtest` / `calculate_metrics` carry an optional `forecast_quantiles` dict per window. `predict` accepts the quantile levels to produce. |
| `mmf_sa/models/statsforecast/StatsFcForecastingPipeline.py` | When native intervals are requested, pass `level=` to `predict`, and map `-lo-/-hi-` to quantile keys. Point-forecast clipping is unchanged (§9 Q7). |
| `mmf_sa/models/chronosforecast/ChronosPipeline.py` | Use `predict_quantiles` in the distributed, single (serverless) and backtest paths, keeping the median as the point forecast. Log a warning when a requested level is outside a Bolt model's trained 0.1–0.9 range (§7.1). |
| `mmf_sa/Forecaster.py` | Add `forecast_quantiles` to the evaluation UDF schemas and writers. `score_*` writes the native bands and applies `calibration_output`. A `run_calibration` method orchestrates the Spark side. |
| `mmf_sa/calibration.py` (new) | Pure pandas/numpy: finite-sample ranks, residual scores, CQR scores (`asymmetric` + adaptive split, `scaled`, `symmetric`), group merging, shrinkage, ABC/XYZ/SB features, `clip_lower`, `output_format`, interval metrics. Ported from `docs/specs/experiments/pi_grouping_m5.py`. |
| `tests/unit/test_calibration.py` (new) | Rank, shrinkage and CQR unit tests, plus a golden test against the notebook's numbers on a small fixed M5 sample. |
| `README.md`, `mmf_sa/README.md`, `skills/.../4-execute-mmf-forecast.md`, `5-post-process-and-evaluate.md` | Document the parameters, the tables, and the `clip(0)` limitation. Update the outdated "GPU models cannot use serverless" note. |

### 11.3 Tables

`evaluation_output` (new column, nullable):
`forecast_quantiles MAP<STRING, ARRAY<DOUBLE>>`, with keys as in §7.4.

`calibration_output` (new, appended per `run_id`). As implemented in step 4, it has one
row per model × series × horizon step × level, whatever the coverage target. The
per-series features are repeated on every row, so scoring joins on the series id alone:

| Column | Type | Notes |
|---|---|---|
| `run_id`, `run_date`, `use_case`, `model` | STRING / TIMESTAMP | |
| `method` | STRING | `conformal`, `cqr` or `native` (what `method: auto` resolved to) |
| `score` | STRING | `signed_normalized` / `signed` (conformal), `asymmetric` / `symmetric` / `scaled` (CQR), `native` |
| `coverage_target` | STRING | `marginal` / `group` / `series` (conformal), the `cqr_pool` (CQR), `native` |
| `<group_id>`, `pool_key` | STRING | series id; `pool_key` = its (merged) group label, `__other__` or `__all__` |
| `step` | INT | 1..`prediction_length` |
| `level` | DOUBLE | central level in percent |
| `q_lo`, `q_hi` | DOUBLE | in units of σ (conformal `signed_normalized`, CQR `asymmetric` / `symmetric`), of 1 (`signed`), or of the native half-width (`scaled`) |
| `n_windows`, `shrinkage_weight`, `infinite_bound` | INT / DOUBLE / BOOLEAN | diagnostics |
| `clip_lower` | DOUBLE | resolved `clip_lower`, null for no clipping |
| `sigma`, `abc_class`, `xyz_class`, `sb_class` | DOUBLE / STRING | per-series scale and classes |

`scoring_output`: interval columns per `output_format` (§7.4).

### 11.4 Interval metrics

Coverage and MSIS have to be measured on windows that were not used to calibrate.
`run_calibration` therefore computes them leave-one-window-out: for each backtest
window, it recalibrates on the other windows with the same settings and scores the
held-out one. With W=3 that means calibrating on 2 windows, so these metrics are
slightly pessimistic for the shipped bands. They are written to
`<evaluation_output>_interval_metrics` (columns `<group_id>`,
`backtest_window_start_date`, `metric_name`, `metric_value`, `model`, `run_id`,
`run_date`, `use_case`), with `metric_name` ∈ {`coverage_<L>`, `msis_<L>`, `width_<L>`},
one row per series × window × model. The native bands, when
present, are also reported as `native_coverage_<L>` / `native_msis_<L>` /
`native_width_<L>`. A weighted quantile loss (`wql`) is deferred.

### 11.5 P0 tasks (one PR each)

1. **Config and plumbing.** Add the `prediction_intervals` block, the `run_forecast`
   argument, the `native_intervals` flags and a no-op `run_calibration`. With
   `enabled: false` everything behaves exactly as today (regression test on the
   existing integration test).
2. **`calibration.py` core and unit tests.** Port the experiment math. The golden
   test reproduces the notebook's coverage and MSIS on a fixed 30-series M5 sample
   within 1e-9.
3. **Native bands in the backtest.** Write `forecast_quantiles` for statsforecast and
   Chronos (local UDF and foundation paths), with the `mergeSchema` migration.
4. **`run_calibration` on Spark.** Features, scores, pooling, `calibration_output`,
   and the leave-one-window-out metrics. The first version pools on the driver.
   4b. **Distributed calibration** (§11.6). Same results without collecting the
   backtest to the driver.
5. **Scoring.** Native bands at forecast time, apply the calibration, `clip_lower`,
   `output_format`. The calibration is joined to the forecasts and applied per
   partition (§11.6).
6. **Docs and example.** README, skills, and an example notebook on M5 daily that
   shows coverage per ABC×XYZ class.
7. **Validation.** An end-to-end MMF run on the 4,966-series M5 sample (seed 7) must
   reproduce the experiment's coverage within 0.5 point, and its MSIS within 2%, for
   AutoETS, SeasonalNaive, CrostonSBA and Chronos-Bolt. MMF's backtest windows
   differ from the notebook's (`backtest_length` / `stride`), so the windows are
   set to match. A second run on about 1M synthetic series, with the driver memory
   capped, checks that nothing scales with the number of series on the driver.

**Task 7 outcome** (notebooks and results in `tests/validation/` on the feature branch,
run on serverless with the default environment):

- The 0.5-point / 2% criterion above assumed MMF could replay the experiment, which
  calibrates once and scores 4 later windows with refits. MMF calibrates and forecasts in
  one run, so the M5 check scores the 28 days after the experiment's test cutoff instead,
  with the experiment's last 3 calibration windows as MMF's backtest.
- **Held-out coverage on 4,966 M5 series** (MMF / experiment):

  | Model | 80% | 95% |
  |---|---:|---:|
  | AutoETS (CQR) | 0.794 / 0.788 | 0.943 / 0.944 |
  | SeasonalNaive (CQR) | 0.808 / 0.799 | 0.947 / 0.946 |
  | CrostonSBA (conformal) | 0.790 / 0.783 | 0.948 / 0.944 |

  Coverage is within 1 point of the experiment. MSIS is 7–37% higher, which is not
  comparable: 1 held-out window instead of 4, and σ measured on a different year.
- **Spark equals the numpy reference** on MMF's own backtest: the maximum difference of
  the corrections, coverage and MSIS is 0 for every model and level.
- **Scale** (synthetic backtests, h = 7, 3 windows, 2 models):

  | Series | Calibration | Scoring per model |
  |---:|---:|---:|
  | 100k | 4.1 min | ~35 s |
  | 1M | 13.1 min | 2–3 min |

  All row counts were as expected, with no null intervals. Getting there required batching
  the per-series work into hash buckets of ~2,000 series per UDF call, computing the
  features in SQL, and materializing intermediate results to temporary tables where
  `persist()` is unavailable (serverless clients before environment version 5). The first
  version took 61 min at 100k.
- The local-model backtest itself, one statsforecast fit per series and window, remains the
  bottleneck of an end-to-end run at that scale. It is outside the prediction-interval code.

### 11.6 Scaling to millions of series

For 1M series, W = 3, H = 28 and 5 quantiles, the backtest of one model is about
600M doubles (≈ 5 GB), and its scoring rows about 170M doubles. Neither may be
collected to the driver. What is left on the driver must not grow with the number of
series: pool tables (groups × steps × levels), group-label maps and scalars.

**Scoring (done in step 5).** The corrections are pivoted in Spark to one row per
series (`q_lo` / `q_hi` arrays ordered by step, σ, method, score, per level). They are
left-joined to the forecasts and applied per partition with `mapInPandas`, by the same
pure code as the reference `apply_calibration`. Only the fallback for series without
calibration (the per-step median correction, an exact `percentile` per level and
step) is collected.

**Calibration (task 4b).** The math splits into pooled statistics and a per-series
combination, and each has a distributed form:

1. **Long scores.** Per model, `posexplode` the `forecast` / `actual` arrays (and the
   native quantiles) into rows (series, window, step, y, ŷ, lo, hi). The window index
   is the rank of the window among the series' last W windows. Join the per-series σ
   and pool label, and compute the scores: `(y − ŷ)/σ` for conformal, and `(lo − y)/σ`
   / `(y − hi)/σ` (or the symmetric or scaled score) for CQR.
2. **Group merging.** Count series per label in Spark, and map labels below
   `min_scores_per_group` to `__other__`. The map has one entry per label.
3. **Pool quantiles, exactly.** The conformal quantile is an order statistic: rank
   `k = ⌈(n+1)q⌉` (or `⌊(n+1)q⌋` for the lower tail), clipped to [1, n]. It is computed
   in SQL from the pool's count, with the same float rounding guard as
   `conformal_rank`. `row_number()` over (pool, step) ordered by score, filtered to
   `rn = k`, gives exactly the numpy value: ties do not change the k-th value. The
   steps-pooled quantile `q_all` per pool is the same with the partition on the pool
   only. The result is small (pools × steps).
4. **Per-series combination.** `applyInPandas` by series, over its W×H scores. It
   computes the per-series quantiles with `calibration.tail_quantiles` /
   `upper_quantile`, combines them with the broadcast pool table (shrinkage with
   λ = W/(W+κ), and the step profile), and emits the series' `calibration_output` rows.
   Targets without a per-series part (marginal, group, the CQR group pool) just join
   the pool table.
5. **CQR adaptive split.** `m_lo` = the mean, over all scores, of `s_lo > q_lo(pool,
   step)`: one join and one `avg`. Then the upper-tail quantiles are computed at
   `1 − α + m_lo` (capped at `1 − α/2`), with step 3 again.
6. **Leave-one-window-out metrics.** Duplicate the long scores with a `fold` column
   (fold j drops window j), add `fold` to every partition key in steps 3–5, and score
   window j per series in the step-4 UDF. That is W times the data in one pass, rather
   than W passes.
7. **Classes without the driver.** ABC uses the cumulative volume share (a window
   ordered by volume descending, then by series id). XYZ uses the rank `r` of CV
   (ordered by CV, then series id): X when `r ≤ 1 + (N−1)/3`, Y when
   `r ≤ 1 + 2(N−1)/3`, otherwise Z. This matches `pd.qcut` on `rank(method="first")`
   exactly.

The numpy functions stay as the reference. The golden and randomized tests compare
the Spark path against `calibrate_model` to 1e-12, on the M5 fixture and on random
cubes with ties, small groups and infinite ranks. They run locally on PySpark ≥ 3.5
with Python ≤ 3.11, because OSS PySpark 3.4 cannot run its Python workers on
Python 3.12, nor return `array<timestamp>` from `applyInPandas`.

Cost: per model and level, the sort in step 3 covers N×W×H rows per fold, so about
250M rows for 1M series at W = 3 (× 3 folds). That is minutes on a mid-size cluster,
and much less than the backtest itself, which refits every local model W times.
