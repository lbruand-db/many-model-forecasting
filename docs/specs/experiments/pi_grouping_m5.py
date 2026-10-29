# Databricks notebook source
# MAGIC %md
# MAGIC # Prediction-interval grouping experiment on M5
# MAGIC
# MAGIC Companion to `docs/specs/confidence_interval.md` §7.6. Compares the coverage
# MAGIC targets / grouping columns for conformal calibration of backtest residuals:
# MAGIC
# MAGIC * calibration on `W` rolling windows strictly before a held-out test period
# MAGIC   (`W=3` mimics MMF defaults, `W=12` a long backtest),
# MAGIC * evaluation on `TEST_WINDOWS` non-overlapping 28-day windows,
# MAGIC * signed, scale-normalised scores (`(y - yhat) / sigma_i`, sigma = in-sample
# MAGIC   seasonal-naive MAE), finite-sample conformal ranks, lower bound clipped at 0.
# MAGIC
# MAGIC Results are returned with `dbutils.notebook.exit(json)`; nothing is written to tables.

# COMMAND ----------

# MAGIC %pip install statsforecast==2.0.0 datasetsforecast==0.0.8 --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import json
import math
import time

import numpy as np
import pandas as pd
from datasetsforecast.m5 import M5
from statsforecast import StatsForecast
from statsforecast.models import AutoETS, CrostonSBA, SeasonalNaive

dbutils.widgets.text("n_series", "2000")
dbutils.widgets.text("seed", "7")
N_SERIES = int(dbutils.widgets.get("n_series"))
SEED = int(dbutils.widgets.get("seed"))

H = 28                 # forecast horizon (M5)
CAL_STEP = 14          # stride between calibration windows
W_LONG = 12            # long calibration; W=3 uses the last 3 of these windows
W_LIST = [3, W_LONG]
TEST_WINDOWS = 4
TEST_STEP = 28
LEVELS = [80, 95]
FEATURE_DAYS = 365     # history used for sigma and grouping features
MIN_SCORES_PER_GROUP = 100
KAPPA = 20
KAPPAS = [5, 10, 20, 40]               # shrinkage sweep for the series option (no extra fits)
SHRINK_GROUP_COLS = ["sb", "abc_xyz"]  # groups the series option shrinks toward
t0 = time.time()

# COMMAND ----------

# MAGIC %md ## Data

# COMMAND ----------

# M5.load() builds the full long table with exogenous features and runs out of memory
# on a serverless notebook, so read the raw wide CSV and sample series before melting.
M5.download(directory="/tmp/m5")
m5_path = "/tmp/m5/m5/datasets"
cal = pd.read_csv(f"{m5_path}/calendar.csv", usecols=["date"], parse_dates=["date"])
cal["d"] = [f"d_{i + 1}" for i in range(len(cal))]   # same mapping as datasetsforecast
wide = pd.read_csv(f"{m5_path}/sales_train_evaluation.csv")
wide["unique_id"] = wide["item_id"] + "_" + wide["store_id"]   # CATEGORY_DEPT_ITEM_STATE_STORE
day_cols = [c for c in wide.columns if c.startswith("d_")]
rng = np.random.default_rng(SEED)
# Oversample, as some series are dropped by the history-length filter below.
pick = rng.choice(len(wide), size=min(len(wide), int(1.5 * N_SERIES)), replace=False)
wide = wide.iloc[np.sort(pick)]
Y = wide.melt(id_vars=["unique_id"], value_vars=day_cols, var_name="d", value_name="y")
Y = Y.merge(cal, on="d").rename(columns={"date": "ds"}).drop(columns="d")
Y["y"] = Y["y"].astype(float)
Y = Y.sort_values(["unique_id", "ds"]).reset_index(drop=True)
del wide

# Drop leading zeros (pre-launch period of each item).
Y = Y[Y.groupby("unique_id")["y"].cumsum() > 0]

T = Y["ds"].max()
test_cutoff = T - pd.Timedelta(days=TEST_WINDOWS * TEST_STEP)           # last day of calibration data
first_cal_cutoff = test_cutoff - pd.Timedelta(days=H + (W_LONG - 1) * CAL_STEP)
feature_cutoff = first_cal_cutoff                                         # features use data <= this date
min_start = feature_cutoff - pd.Timedelta(days=FEATURE_DAYS + 28)

starts = Y.groupby("unique_id")["ds"].min()
eligible = starts[starts <= min_start].index.to_numpy()
ids = np.sort(rng.choice(eligible, size=min(N_SERIES, len(eligible)), replace=False))
Y = Y[Y["unique_id"].isin(ids)].reset_index(drop=True)
print(f"{len(ids)} series, T={T.date()}, test_cutoff={test_cutoff.date()}, "
      f"first_cal_cutoff={first_cal_cutoff.date()}")

# COMMAND ----------

# MAGIC %md ## Per-series scale and grouping features (history <= feature_cutoff only)

# COMMAND ----------

hist = Y[(Y["ds"] <= feature_cutoff) & (Y["ds"] > feature_cutoff - pd.Timedelta(days=FEATURE_DAYS))]


def series_features(g):
    y = g["y"].to_numpy(dtype=float)
    snaive_mae = np.mean(np.abs(y[7:] - y[:-7]))
    nz = y[y > 0]
    adi = len(y) / max(len(nz), 1)
    cv2 = (nz.std() / nz.mean()) ** 2 if len(nz) > 1 else 0.0
    return pd.Series({
        "sigma": snaive_mae if snaive_mae > 0 else np.mean(np.abs(y)),
        "volume": y.sum(),
        "cv": y.std() / y.mean() if y.mean() > 0 else np.inf,
        "adi": adi,
        "cv2": cv2,
        "rel_noise": snaive_mae / y.mean() if y.mean() > 0 else np.inf,
    })


F = hist.groupby("unique_id").apply(series_features)
F = F[F["sigma"] > 0]
ids = F.index.to_numpy()
Y = Y[Y["unique_id"].isin(ids)].reset_index(drop=True)

# Syntetos-Boylan classes.
F["sb"] = np.select(
    [(F.adi < 1.32) & (F.cv2 < 0.49), (F.adi < 1.32), (F.cv2 < 0.49)],
    ["smooth", "erratic", "intermittent"],
    "lumpy",
)
# ABC on volume (80/15/5).
share = F["volume"].sort_values(ascending=False).cumsum() / F["volume"].sum()
F["abc"] = pd.cut(share, [-1, 0.8, 0.95, 2], labels=["A", "B", "C"]).astype(str).reindex(F.index)
# XYZ = terciles of CV (X = least volatile).
F["xyz"] = pd.qcut(F["cv"].rank(method="first"), 3, labels=["X", "Y", "Z"]).astype(str)
F["abc_xyz"] = F["abc"] + F["xyz"]
# Error tier = terciles of seasonal-naive MAE relative to mean demand.
F["err_tier"] = pd.qcut(F["rel_noise"].rank(method="first"), 3, labels=["low", "mid", "high"]).astype(str)
# Business columns from the M5 id: CATEGORY_DEPT_ITEM_STATE_STORE.
F["category"] = [i.split("_")[0] for i in F.index]
F["store"] = ["_".join(i.split("_")[-2:]) for i in F.index]
F["sb_x_category"] = F["sb"] + "|" + F["category"]
print(F["sb"].value_counts().to_dict(), F["abc"].value_counts().to_dict())

GROUP_COLS = ["sb", "abc", "abc_xyz", "err_tier", "category", "store", "sb_x_category"]
N = len(ids)
idx_of = {u: i for i, u in enumerate(ids)}
sigma = F.loc[ids, "sigma"].to_numpy()

# COMMAND ----------

# MAGIC %md ## Backtests (statsforecast cross-validation, refit per window like MMF local models)

# COMMAND ----------

MODELS = ["SeasonalNaive", "CrostonSBA", "AutoETS"]
NATIVE = ["SeasonalNaive", "AutoETS"]
N_CHUNKS = 64
NATIVE_COLS = [f"{m}_{side}_{lv}" for m in NATIVE for side in ("lo", "hi") for lv in LEVELS]
OUT_COLS = ["unique_id", "ds", "cutoff", "kind", "y"] + MODELS + NATIVE_COLS


def make_models(names):
    lib = {
        "SeasonalNaive": lambda: SeasonalNaive(season_length=7),
        "CrostonSBA": lambda: CrostonSBA(),
        "AutoETS": lambda: AutoETS(season_length=7, model="ZNZ"),
    }
    return [lib[n]() for n in names]


def run_cv(pdf):
    """Backtests for one chunk of series; runs on a Spark worker with n_jobs=1.

    statsforecast's multiprocessing pool (n_jobs=-1) aborts the serverless notebook
    kernel, so parallelism comes from Spark instead, as in MMF's local-model UDFs.
    """
    pdf = pdf.drop(columns="chunk")
    pdf["ds"] = pd.to_datetime(pdf["ds"])
    sf = StatsForecast(models=make_models(MODELS), freq="D", n_jobs=1)
    cal = sf.cross_validation(df=pdf[pdf["ds"] <= test_cutoff], h=H, n_windows=W_LONG, step_size=CAL_STEP)
    test = sf.cross_validation(df=pdf, h=H, n_windows=TEST_WINDOWS, step_size=TEST_STEP)
    sf_nat = StatsForecast(models=make_models(NATIVE), freq="D", n_jobs=1)
    nat = sf_nat.cross_validation(df=pdf, h=H, n_windows=TEST_WINDOWS, step_size=TEST_STEP, level=LEVELS)
    cal["kind"], test["kind"], nat["kind"] = "cal", "test", "native"
    out = pd.concat([cal, test, nat], ignore_index=True)
    out.columns = [c.replace("-", "_") for c in out.columns]
    for c in ("ds", "cutoff"):
        out[c] = pd.to_datetime(out[c]).dt.strftime("%Y-%m-%d")
    return out.reindex(columns=OUT_COLS)


def to_cube(cv, col, cutoffs):
    """(series, window, step) array from a cross_validation frame."""
    cv = cv[cv["cutoff"].isin(cutoffs)].copy()
    cv["i"] = cv["unique_id"].map(idx_of)
    cv["w"] = cv["cutoff"].map({c: k for k, c in enumerate(sorted(cutoffs))})
    cv["j"] = (cv["ds"] - cv["cutoff"]).dt.days - 1
    out = np.full((N, len(cutoffs), H), np.nan)
    out[cv["i"].to_numpy(), cv["w"].to_numpy(), cv["j"].to_numpy()] = cv[col].to_numpy()
    return out


from pyspark.sql import types as st

schema = st.StructType(
    [st.StructField(c, st.StringType()) for c in ["unique_id", "ds", "cutoff", "kind"]]
    + [st.StructField(c, st.DoubleType()) for c in OUT_COLS[4:]]
)
Y_in = Y.assign(ds=Y["ds"].dt.strftime("%Y-%m-%d"), chunk=Y["unique_id"].map(idx_of) % N_CHUNKS)
cv_all = (
    spark.createDataFrame(Y_in[["unique_id", "ds", "y", "chunk"]])
    .groupBy("chunk")
    .applyInPandas(run_cv, schema)
    .toPandas()
)
cv_all = cv_all.rename(columns={c: c.replace("_lo_", "-lo-").replace("_hi_", "-hi-") for c in NATIVE_COLS})
for c in ("ds", "cutoff"):
    cv_all[c] = pd.to_datetime(cv_all[c])
cal_cv = cv_all[cv_all["kind"] == "cal"]
test_cv = cv_all[cv_all["kind"] == "test"]
native_cv = cv_all[cv_all["kind"] == "native"]
print(f"backtests done in {time.time() - t0:.0f}s ({len(cv_all)} rows)")

cal_cutoffs = sorted(cal_cv["cutoff"].unique())
test_cutoffs = sorted(test_cv["cutoff"].unique())
assert len(cal_cutoffs) == W_LONG and len(test_cutoffs) == TEST_WINDOWS
assert max(cal_cutoffs) + pd.Timedelta(days=H) <= min(test_cutoffs) + pd.Timedelta(days=0)

Y_cal = to_cube(cal_cv, "y", cal_cutoffs)
Y_test = to_cube(test_cv, "y", test_cutoffs)
assert not np.isnan(Y_cal).any() and not np.isnan(Y_test).any()

# COMMAND ----------

# MAGIC %md ## Conformal helpers

# COMMAND ----------


def conf_rank(n, q, upper):
    """1-based finite-sample conformal rank; returns (rank clipped to [1, n], was_infinite)."""
    pos = round((n + 1) * q, 8)
    k = math.ceil(pos) if upper else math.floor(pos)
    return min(max(k, 1), n), (k < 1 or k > n)


def tail_quantiles(scores, level):
    """Signed scores of shape (n, ...) -> (q_lo, q_hi, inf) along axis 0 for central level."""
    a = (1 - level / 100) / 2
    n = scores.shape[0]
    s = np.sort(scores, axis=0)
    k_lo, inf_lo = conf_rank(n, a, upper=False)
    k_hi, inf_hi = conf_rank(n, 1 - a, upper=True)
    return s[k_lo - 1], s[k_hi - 1], (inf_lo or inf_hi)


def abs_quantile(scores, level):
    n = scores.shape[0]
    s = np.sort(scores, axis=0)
    k, inf = conf_rank(n, level / 100, upper=True)
    return s[k - 1], inf


def group_quantiles(E, labels, level):
    """E: (N, W, H) normalised signed scores. Returns per-series (N, H) q_lo, q_hi, inf flags."""
    W = E.shape[1]
    labels = pd.Series(labels)
    sizes = labels.map(labels.value_counts())
    labels = labels.where(sizes * W >= MIN_SCORES_PER_GROUP, "__other__").to_numpy()
    q_lo = np.empty((N, H))
    q_hi = np.empty((N, H))
    inf = np.zeros(N, dtype=bool)
    for g in np.unique(labels):
        m = labels == g
        lo, hi, f = tail_quantiles(E[m].reshape(-1, H), level)
        q_lo[m], q_hi[m], inf[m] = lo, hi, f
    return q_lo, q_hi, inf


# COMMAND ----------

# MAGIC %md ## Build intervals for every method and evaluate on the test windows

# COMMAND ----------


def evaluate(name, model, W, level, lo, hi, yhat, inf_share=0.0):
    """lo/hi/yhat: (N, K, H) arrays on the original scale."""
    lo = np.maximum(lo, 0.0)
    a = 1 - level / 100
    y = Y_test
    cover = (y >= lo) & (y <= hi)
    s = sigma[:, None, None]
    width = (hi - lo) / s
    iscore = ((hi - lo) + (2 / a) * (lo - y) * (y < lo) + (2 / a) * (y - hi) * (y > hi)) / s
    per_series = cover.mean(axis=(1, 2))
    nominal = level / 100
    row = {
        "method": name, "model": model, "W": W, "level": level,
        "coverage": float(cover.mean()),
        "msis": float(iscore.mean()),
        "scaled_width": float(width.mean()),
        "series_cov_mad": float(np.mean(np.abs(per_series - nominal))),
        "share_series_undercovered_10pt": float(np.mean(per_series < nominal - 0.10)),
        "share_series_overcovered_10pt": float(np.mean(per_series > min(nominal + 0.10, 0.999))),
        "share_lo_clipped": float(np.mean(lo == 0.0)),
        "share_inf_bounds": float(inf_share),
    }
    for col in ["sb", "abc"]:
        lab = F.loc[ids, col].to_numpy()
        for g in np.unique(lab):
            row[f"cov_{col}_{g}"] = float(cover[lab == g].mean())
    return row


results = []
for model in MODELS:
    yhat_cal_all = np.maximum(to_cube(cal_cv, model, cal_cutoffs), 0.0)
    yhat_test = np.maximum(to_cube(test_cv, model, test_cutoffs), 0.0)
    n_nan = int(np.isnan(yhat_cal_all).sum() + np.isnan(yhat_test).sum())
    if n_nan:
        print(f"WARNING {model}: {n_nan} NaN forecasts replaced by 0")
        yhat_cal_all = np.nan_to_num(yhat_cal_all)
        yhat_test = np.nan_to_num(yhat_test)
    for W in W_LIST:
        Yc, Fc = Y_cal[:, -W:, :], yhat_cal_all[:, -W:, :]
        E = (Yc - Fc) / sigma[:, None, None]      # signed normalised scores (N, W, H)
        A = np.abs(Yc - Fc)                       # absolute raw scores (N, W, H)
        for level in LEVELS:
            def from_q(q_lo, q_hi):
                lo = yhat_test + sigma[:, None, None] * q_lo[:, None, :]
                hi = yhat_test + sigma[:, None, None] * q_hi[:, None, :]
                return lo, hi

            # Option 1: marginal (one global pool per step).
            q_lo, q_hi, inf = group_quantiles(E, np.zeros(N, dtype=int), level)
            results.append(evaluate("marginal", model, W, level, *from_q(q_lo, q_hi), yhat_test, inf.mean()))

            # Option 2: group-conditional for each candidate grouping column.
            for col in GROUP_COLS:
                q_lo, q_hi, inf = group_quantiles(E, F.loc[ids, col].to_numpy(), level)
                results.append(evaluate(f"group:{col}", model, W, level, *from_q(q_lo, q_hi), yhat_test, inf.mean()))

            # Option 3: per-series, horizon-pooled, step profile + shrinkage toward a group.
            s_lo, s_hi, s_inf = tail_quantiles(E.reshape(N, -1).T, level)  # per series, n = W*H
            for gcol in SHRINK_GROUP_COLS:
                labels = F.loc[ids, gcol].to_numpy()
                g_lo, g_hi, _ = group_quantiles(E, labels, level)
                r_lo = np.ones((N, H))
                r_hi = np.ones((N, H))
                for g in np.unique(labels):
                    m = labels == g
                    all_lo, all_hi, _ = tail_quantiles(E[m].reshape(-1), level)
                    if abs(all_lo) > 1e-9:
                        r_lo[m] = g_lo[m] / all_lo
                    if abs(all_hi) > 1e-9:
                        r_hi[m] = g_hi[m] / all_hi
                variants = [("nW", W, k) for k in KAPPAS]
                if gcol == "sb":
                    variants.append(("nWH", W * H, KAPPA))
                for n_eff_name, n_eff, kappa in variants:
                    lam = n_eff / (n_eff + kappa)
                    q_lo = lam * s_lo[:, None] * r_lo + (1 - lam) * g_lo
                    q_hi = lam * s_hi[:, None] * r_hi + (1 - lam) * g_hi
                    # Names of the first run (SB group, kappa=20) are kept for comparability.
                    if gcol == "sb" and kappa == KAPPA:
                        name = f"series_shrink_{n_eff_name}"
                    else:
                        name = f"series_shrink_{n_eff_name}:{gcol}:k{kappa}"
                    results.append(evaluate(name, model, W, level,
                                            *from_q(q_lo, q_hi), yhat_test, float(s_inf)))

            # Baseline: Nixtla-style per series x per step, absolute raw scores (conformal_error).
            q, inf = abs_quantile(A.transpose(1, 0, 2), level)          # (N, H)
            results.append(evaluate("baseline:nixtla_series_step", model, W, level,
                                    yhat_test - q[:, None, :], yhat_test + q[:, None, :], yhat_test, float(inf)))

            # Baseline: AutoGluon-style per series, absolute scores pooled over steps.
            q, inf = abs_quantile(A.reshape(N, -1).T, level)            # (N,)
            results.append(evaluate("baseline:autogluon_series_pooled", model, W, level,
                                    yhat_test - q[:, None, None], yhat_test + q[:, None, None], yhat_test, float(inf)))

    if model in NATIVE:
        for level in LEVELS:
            lo = to_cube(native_cv, f"{model}-lo-{level}", test_cutoffs)
            hi = to_cube(native_cv, f"{model}-hi-{level}", test_cutoffs)
            results.append(evaluate("baseline:native", model, 0, level, lo, hi, yhat_test))
    print(f"{model} evaluated at {time.time() - t0:.0f}s")

R = pd.DataFrame(results)
display(R.sort_values(["model", "level", "W", "msis"]))

# COMMAND ----------

summary = {
    "n_series": int(N),
    "sb_counts": F["sb"].value_counts().to_dict(),
    "abc_counts": F["abc"].value_counts().to_dict(),
    "group_counts": {c: int(F[c].nunique()) for c in GROUP_COLS},
    "runtime_s": round(time.time() - t0, 1),
    "results": R.to_dict(orient="records"),
}
dbutils.notebook.exit(json.dumps(summary, default=float))
