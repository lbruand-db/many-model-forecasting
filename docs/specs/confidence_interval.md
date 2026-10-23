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
  **3 windows**. Per-series calibration is therefore not viable by default. MMF's
  advantage is **pooling scale-normalised residuals across many series** (by segment
  and horizon step), which makes calibration stable even with only a few windows.
- What similar products ship is very consistent: a configurable coverage level
  (default 0.95 in BigQuery and Snowflake), or a list of quantiles (default
  P10/P50/P90 in Amazon, TimeGPT and Azure), returned as `lower`/`upper` columns next
  to the point forecast.

Recommended direction (detailed in §7):
1. Pass native quantiles through where they exist, but keep this optional.
2. Use model-agnostic, MMF-level conformal calibration from the backtest output as the
   default and universal mechanism.
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
interval should be clipped to 0 the same way for non-negative targets.

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
  smooth / erratic / intermittent / lumpy. MMF already computes this in the optional
  profiling skill (`skills/databricks-skills/many-model-forecasting/2-profile-and-classify-series.md`).

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
intermittency class from profiling, a store, or a category.

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

Concrete suggested defaults (to validate empirically):

| freq | prediction_length | backtest_length | stride | W |
|---|---:|---:|---:|---:|
| D | 28 | 28 + 7·9 = 91 | 7 | 10 |
| W | 13 | 13 + 52 = 65 | 4 | 14 |
| M | 6 | 6 + 18 = 24 | 2 | 10 |
| H | 24 | 24 + 24·13 = 336 | 24 | 14 |

---

## 7. Proposed design (sketch for discussion)

### 7.1 Layer A: native pass-through (optional, per family)

- statsforecast: call `predict(h, level=levels)` for models whose native PI is
  available (§3.1).
- Chronos / TimesFM: keep the quantile tensor, mapping levels to the nearest trained
  quantiles. For TimesFM, enable `use_continuous_quantile_head`.
- Moirai: take `np.quantile(samples, τ)` and raise the default `num_samples` (≥100).
- Prophet: `predict_interval`.

This gives "model-native" bands. They are **not guaranteed calibrated**, and they are
**missing for half of the zoo**.

### 7.2 Layer B: MMF conformal calibration (default, universal)

It runs after evaluation, entirely from `evaluation_output` (it already has `forecast`
and `actual` arrays), so there is **no extra training**:

1. Explode the arrays to `(series, model, window_start, step j, ŷ, y)`.
2. Compute a scale `σ̂_i` per series. The default is the MASE denominator (in-sample
   MAE of the seasonal naive) on the training part. Fall back to the mean of `|y|`.
3. Compute signed normalised scores `e = (y − ŷ)/σ̂_i`.
4. For each `(model, pool, step j)`, store the conformal quantiles of `e` for the
   lower (`α/2`) and upper (`1−α/2`) sides, optionally recency-weighted. Here `pool`
   is one of `series`, `segment` (from a user column such as an intermittency class
   or category) or `global`. Automatic fallback: use `series` if `W ≥ ⌈2/α⌉ − 1`,
   else `segment`, else `global`.
5. Persist the result to a new table `calibration_output` (`run_id`, `model`, `pool`,
   `step`, `level`, `q_lo`, `q_hi`, `n_scores`).

Scoring then produces `lo = clip(ŷ + σ̂_i·q_lo, 0)` and `hi = ŷ + σ̂_i·q_hi`.

For models with native quantiles, Layer B can instead run **CQR** on the native bands
(§4.2), which keeps their adaptivity while fixing calibration.

### 7.3 Configuration (proposal)

```yaml
prediction_intervals:
  enabled: true
  levels: [80, 95]            # or quantiles: [0.1, 0.5, 0.9]
  method: conformal           # conformal | native | cqr (native+conformal)
  pooling: auto               # series | segment | global | auto
  segment_col: null           # e.g. "intermittency_class"
  score: signed_normalized    # abs | signed | signed_normalized
  recency_decay: null         # e.g. 0.9 per window (weighted conformal)
  clip_lower: 0
```

### 7.4 Output schema changes

- `scoring_output`: add `<target>_lo_<lv>` / `<target>_hi_<lv>` array columns (one per
  level), or a `quantiles` map of `level → array`. Pick one and document it. The first
  form matches Snowflake/BigQuery; the second scales better.
- `evaluation_output`: add per-window interval metrics (§8) when intervals are enabled.

### 7.5 Phasing

1. **P0**: Layer B with `global` and `segment` pooling, plus interval metrics. This is
   model-agnostic and has no training cost.
2. **P1**: Layer A for foundation models and statsforecast, plus CQR.
3. **P2**: NeuralForecast probabilistic losses (`MQLoss`, `DistributionLoss`, NegBin),
   and LightGBM quantile objective.
4. **P3**: probabilistic hierarchical reconciliation, and online ACI in a monitoring
   job.

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

1. Is the target marginal coverage over series (cheap, stable) or per-series coverage
   (needs a long backtest)? What do MMF users actually need for safety stock?
2. Which output schema: wide `lo/hi` columns or a quantile map? This affects the
   Genie/dashboard views in `examples/fresh_retail_net`.
3. Should we change the README defaults so that W ≥ 10, given the cost for local
   models on big data?
4. Should calibration live in the evaluation run, in scoring, or as a separate step
   (`run_calibration`)? Scoring currently does not read `evaluation_output`.
5. Do we want *joint* trajectory coverage (Bonferroni / CF-RNN) or per-step coverage?
   Per-step is the proposed default.
6. Interaction with the model selection in skill 5 (best model per series): calibrate
   the *selected* model per series. Selection on the same backtest windows biases the
   calibration slightly toward optimism.

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
