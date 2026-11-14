"""
Stage 4 (ASOS): persistence baseline vs Random Forest (UrbanAtmos).

Input:   ASOS_03_Daegu_daily_features_after_report.csv   (output of A_ASOS_03_make_features_after_report.py)
Output:  ASOS_04_model_results.csv        metrics per target, model and split (val, test)
         ASOS_04_predictions.csv          every val/test prediction of both models (for error analysis)
         ASOS_04_feature_importance.png   top features of each Random Forest
         ASOS_04_modeling_report.txt      text report of every step

Task: on day D (data up to 23:00 of D) predict day D+1:
  - y_tropical_night      classification (0/1)
  - y_heatwave_advisory   classification (0/1)
  - y_tmax_feels          regression (degC)

Models:
  - persistence (baseline): "tomorrow is like today", fixed rules, nothing is learned
  - Random Forest: many decision trees, each built on a random sample of train days
    and random subsets of features; the forest averages the trees

Steps:
  1. load, pick the feature columns, split by the `split` column (no shuffling)
  2. persistence baseline (no training)
  3. Random Forest settings: expanding-window check by year inside train only
  4. val: fit on train, choose the decision threshold (best F1), compare with persistence
  5. test: refit on train + val with the frozen settings and threshold, evaluate ONCE
  6. explain: permutation importance on val, errors on test per month
  7. report and save (always overwrites the output files)

Requirements:  pip install pandas numpy scikit-learn matplotlib   (scikit-learn >= 1.4 for NaN in Random Forest)
Run:           python A_ASOS_04_modeling.py
"""

import matplotlib

matplotlib.use("Agg")    # save figures to files, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, recall_score)

# ---------- SETTINGS ----------
INPUT_CSV = "ASOS_03_Daegu_daily_features_after_report.csv"
RESULTS_CSV = "ASOS_04_model_results.csv"
PREDICTIONS_CSV = "ASOS_04_predictions.csv"
IMPORTANCE_PNG = "ASOS_04_feature_importance.png"
REPORT_TXT = "ASOS_04_modeling_report.txt"

CLASS_TARGETS = ["y_tropical_night", "y_heatwave_advisory"]
REG_TARGET = "y_tmax_feels"
TARGETS = CLASS_TARGETS + [REG_TARGET]

# Columns that are never features: targets, tomorrow's air temperature (leakage) and the split label
NOT_FEATURES = ["check_tmax_air_next_day", "split"]

# Persistence rules use the same KMA thresholds as the targets (Stage 3)
FEELS_ADVISORY = 33.0              # daily max feels-like for the advisory
TROPICAL_NIGHT_THRESHOLD = 25.0    # night minimum air temperature

# Random Forest
N_TREES = 500
RANDOM_STATE = 42                  # same random draws on every run -> same results
N_JOBS = -1                        # use all CPU cores

# Settings tried for each forest (kept small: only ~1368 training days)
CLASS_GRID = [{"min_samples_leaf": leaf, "class_weight": weight}
              for leaf in [5, 10, 20] for weight in [None, "balanced"]]
REG_GRID = [{"min_samples_leaf": leaf, "max_features": feats}
            for leaf in [5, 10, 20] for feats in ["sqrt", 1.0]]
CLASS_FIXED = {"max_features": "sqrt"}   # each split looks at about 8 of the 59 features

# Expanding-window check inside train: fit on 2014..Y-1, score on year Y
FIRST_CHECK_YEAR = 2017

# Thresholds tried on val for turning probabilities into 0/1 (best F1 wins, ties -> lower threshold)
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.01), 2)

# Explanation
PERM_REPEATS = 10                  # shuffles per feature in permutation importance
TOP_FEATURES = 10                  # features shown in the figure and in the report

# Set to False while experimenting, so the test set stays unseen
EVALUATE_TEST = True

report: list[str] = []


def log(text: str = "") -> None:
    """Prints a line to the console and keeps it for the report file."""
    print(text)
    report.append(text)


def log_table(df: pd.DataFrame) -> None:
    """Logs a small DataFrame with an indent."""
    for line in df.to_string().splitlines():
        log("       " + line)


# ---------- STEP 1 ----------
def load_data(path: str) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(path, parse_dates=["date"], index_col="date").sort_index()
    y_cols = [c for c in df.columns if c.startswith("y_")]
    x_cols = [c for c in df.columns if c not in y_cols + NOT_FEATURES]

    missing = [t for t in TARGETS if t not in df.columns]
    if missing:
        raise KeyError(f"Targets not found in the input file: {missing}")

    log("1. LOADING AND SPLIT")
    log(f"   Rows: {len(df)}, feature columns: {len(x_cols)}, targets: {', '.join(TARGETS)}")
    log(f"   Range: {df.index.min().date()} -> {df.index.max().date()}")
    log("   Rows per split: " + ", ".join(f"{s}: {(df['split'] == s).sum()}" for s in ["train", "val", "test"]))
    log("   Positive share per split (%):")
    log_table((df.groupby("split")[CLASS_TARGETS].mean() * 100).round(1).reindex(["train", "val", "test"]))
    return df, x_cols


# ---------- STEP 2 ----------
def persistence(df: pd.DataFrame, target: str) -> pd.Series:
    """Baseline 'tomorrow is like today'. Fixed rules, nothing is learned from data."""
    if target == REG_TARGET:
        return df["feels_like_max"]                                          # tomorrow = today
    if target == "y_heatwave_advisory":
        return (df["feels_like_max"] >= FEELS_ADVISORY).astype(int)          # today felt >= 33
    if target == "y_tropical_night":
        return (df["night_min_air_prev"] >= TROPICAL_NIGHT_THRESHOLD).astype(int)   # last full night >= 25
    raise ValueError(f"No persistence rule for {target}")


# ---------- METRICS ----------
def class_metrics(y: pd.Series, pred: pd.Series, prob: pd.Series | None = None) -> dict:
    """Precision / recall / F1 at the 0/1 decision; PR-AUC and Brier from probabilities.
    Persistence gives no probability, so its 0/1 answer is used for Brier and PR-AUC is left empty."""
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    return {
        "positive_share_%": round(100 * y.mean(), 1),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "PR_AUC": average_precision_score(y, prob) if prob is not None else np.nan,
        "Brier": brier_score_loss(y, prob if prob is not None else pred),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
    }


def reg_metrics(y: pd.Series, pred: pd.Series) -> dict:
    """MAE = average miss in degC; RMSE = same, but large misses count more; bias = average (pred - real)."""
    return {
        "MAE": mean_absolute_error(y, pred),
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "bias": float((pred - y).mean()),
    }


# ---------- STEP 3 ----------
def make_forest(target: str, params: dict):
    """A Random Forest for one target. Leaves of a classifier store the share of yes-days,
    leaves of a regressor store the average temperature of tomorrow."""
    common = {"n_estimators": N_TREES, "random_state": RANDOM_STATE, "n_jobs": N_JOBS}
    if target == REG_TARGET:
        return RandomForestRegressor(**common, **params)
    return RandomForestClassifier(**common, **CLASS_FIXED, **params)


def forest_output(model, X: pd.DataFrame) -> np.ndarray:
    """Probability of 'yes' for a classifier, predicted degC for a regressor."""
    if isinstance(model, RandomForestClassifier):
        return model.predict_proba(X)[:, 1]
    return model.predict(X)


def check_score(target: str, y: pd.Series, out: np.ndarray) -> float:
    """Score used to compare settings: PR-AUC for 0/1 targets (no threshold needed yet),
    minus MAE for the regression target, so that higher is always better."""
    if target == REG_TARGET:
        return -mean_absolute_error(y, out)
    return average_precision_score(y, out) if y.sum() > 0 else np.nan


def choose_settings(train: pd.DataFrame, x_cols: list[str], target: str) -> dict:
    """Expanding-window check by year, inside train only:
    fit 2014..2016 -> score 2017, fit 2014..2017 -> score 2018, ... fit 2014..2021 -> score 2022.
    The settings with the best mean score win. Val and test are not used here."""
    grid = REG_GRID if target == REG_TARGET else CLASS_GRID
    years = train.index.year
    check_years = [y for y in sorted(set(years)) if y >= FIRST_CHECK_YEAR]

    rows = []
    for params in grid:
        scores = []
        for year in check_years:
            fit, check = train[years < year], train[years == year]
            model = make_forest(target, params).fit(fit[x_cols], fit[target])
            scores.append(check_score(target, check[target], forest_output(model, check[x_cols])))
        rows.append({**params, "mean_score": np.nanmean(scores)})

    table = pd.DataFrame(rows)
    best = table.loc[table["mean_score"].idxmax()]
    score_name = "-MAE" if target == REG_TARGET else "PR-AUC"
    log(f"   {target}: mean {score_name} over check years {check_years[0]}-{check_years[-1]}")
    log_table(table.round(3).astype(str).set_index(list(grid[0].keys())))
    chosen = {k: best[k] for k in grid[0]}
    # pandas turns None into NaN and ints into floats; restore the original types
    chosen = {k: (None if pd.isna(v) else int(v) if k == "min_samples_leaf" else v) for k, v in chosen.items()}
    if "max_features" in chosen and chosen["max_features"] != "sqrt":
        chosen["max_features"] = float(chosen["max_features"])
    log(f"   -> chosen: {chosen}")
    return chosen


# ---------- STEP 4 ----------
def choose_threshold(y: pd.Series, prob: np.ndarray) -> float:
    """Threshold with the best F1 on val. On a tie the lower threshold wins (more warnings = safer)."""
    f1 = [f1_score(y, (prob >= t).astype(int), zero_division=0) for t in THRESHOLDS]
    return float(THRESHOLDS[int(np.argmax(f1))])


def evaluate(target: str, y: pd.Series, base: pd.Series, out: np.ndarray, threshold: float | None) -> list[dict]:
    """Metrics of persistence and Random Forest on the same days."""
    if target == REG_TARGET:
        return [{"model": "persistence", **reg_metrics(y, base)},
                {"model": "random_forest", **reg_metrics(y, pd.Series(out, index=y.index))}]
    pred = pd.Series((out >= threshold).astype(int), index=y.index)
    return [{"model": "persistence", **class_metrics(y, base)},
            {"model": "random_forest", "threshold": threshold, **class_metrics(y, pred, pd.Series(out, index=y.index))}]


def prediction_rows(part: pd.DataFrame, target: str, out: np.ndarray, threshold: float | None) -> pd.DataFrame:
    """One row per day: real value, persistence and Random Forest answers (for error analysis)."""
    rows = pd.DataFrame({
        "split": part["split"],
        "target": target,
        "real": part[target],
        "persistence": persistence(part, target),
        "random_forest_output": out,          # probability (0/1 targets) or degC (regression)
    })
    rows["random_forest"] = (out >= threshold).astype(int) if threshold is not None else out
    return rows


def log_comparison(rows: list[dict], target: str) -> None:
    if target == REG_TARGET:
        cols = ["model", "MAE", "RMSE", "bias"]
    else:
        cols = ["model", "positive_share_%", "precision", "recall", "F1", "PR_AUC", "Brier", "TP", "FP", "FN", "TN"]
    log_table(pd.DataFrame(rows)[cols].set_index("model").round(3))
    if target == REG_TARGET:
        gain = 1 - rows[1]["MAE"] / rows[0]["MAE"]
        log(f"       Random Forest MAE is {100 * gain:.1f}% {'smaller' if gain >= 0 else 'larger'} than persistence")


# ---------- STEP 6 ----------
def importance(model, X: pd.DataFrame, y: pd.Series, target: str) -> pd.Series:
    """Permutation importance: shuffle one feature column at a time and measure how much
    the score drops. A big drop means the forest relied on that feature."""
    scoring = "neg_mean_absolute_error" if target == REG_TARGET else "average_precision"
    result = permutation_importance(model, X, y, scoring=scoring, n_repeats=PERM_REPEATS,
                                    random_state=RANDOM_STATE, n_jobs=N_JOBS)
    return pd.Series(result.importances_mean, index=X.columns).sort_values(ascending=False)


def errors_per_month(y: pd.Series, pred: pd.Series) -> pd.DataFrame:
    """Missed events (FN) and false alarms (FP) per month."""
    month = y.index.month
    return pd.DataFrame({
        "real_yes": y.groupby(month).sum().astype(int),
        "missed_FN": ((pred == 0) & (y == 1)).groupby(month).sum(),
        "false_alarm_FP": ((pred == 1) & (y == 0)).groupby(month).sum(),
    }).rename_axis("month")


def plot_importance(importances: dict[str, pd.Series]) -> None:
    fig, axes = plt.subplots(1, len(importances), figsize=(6 * len(importances), 5))
    for ax, (target, imp) in zip(axes, importances.items()):
        top = imp.head(TOP_FEATURES)[::-1]
        ax.barh(top.index, top.values, color="#4C72B0")
        unit = "MAE increase (degC)" if target == REG_TARGET else "PR-AUC drop"
        ax.set_title(target)
        ax.set_xlabel(f"{unit} when shuffled")
    fig.suptitle("Random Forest: permutation importance on val (model fitted on train)")
    fig.tight_layout()
    fig.savefig(IMPORTANCE_PNG, dpi=120)
    plt.close(fig)


# ---------- MAIN ----------
def main() -> None:
    df, x_cols = load_data(INPUT_CSV)
    train, val, test = (df[df["split"] == s] for s in ["train", "val", "test"])

    log("\n2. PERSISTENCE BASELINE (no training)")
    log(f"   {REG_TARGET}: tomorrow = today's feels_like_max")
    log(f"   y_heatwave_advisory: 1 if today's feels_like_max >= {FEELS_ADVISORY:g}")
    log(f"   y_tropical_night: 1 if night_min_air_prev >= {TROPICAL_NIGHT_THRESHOLD:g}")

    log(f"\n3. RANDOM FOREST SETTINGS ({N_TREES} trees, expanding window by year inside train)")
    settings = {t: choose_settings(train, x_cols, t) for t in TARGETS}

    results, predictions, importances, thresholds, val_models = [], [], {}, {}, {}

    log("\n4. VAL: forest fitted on train, threshold chosen on val, compared with persistence")
    for target in TARGETS:
        model = make_forest(target, settings[target]).fit(train[x_cols], train[target])
        out = forest_output(model, val[x_cols])
        if target in CLASS_TARGETS:
            thresholds[target] = choose_threshold(val[target], out)
        rows = evaluate(target, val[target], persistence(val, target), out, thresholds.get(target))
        log(f"   {target}" + (f" (threshold {thresholds[target]:.2f})" if target in thresholds else ""))
        log_comparison(rows, target)
        results += [{"target": target, "split": "val", **r} for r in rows]
        predictions.append(prediction_rows(val, target, out, thresholds.get(target)))
        val_models[target] = model

    if EVALUATE_TEST:
        log("\n5. TEST (used once): forest refitted on train + val with the frozen settings and threshold")
        train_val = pd.concat([train, val])
        for target in TARGETS:
            model = make_forest(target, settings[target]).fit(train_val[x_cols], train_val[target])
            out = forest_output(model, test[x_cols])
            rows = evaluate(target, test[target], persistence(test, target), out, thresholds.get(target))
            log(f"   {target}")
            log_comparison(rows, target)
            results += [{"target": target, "split": "test", **r} for r in rows]
            pred_rows = prediction_rows(test, target, out, thresholds.get(target))
            predictions.append(pred_rows)
            if target in CLASS_TARGETS:
                log("       Random Forest errors on test per month:")
                log_table(errors_per_month(test[target], pred_rows["random_forest"]))
    else:
        log("\n5. TEST skipped (EVALUATE_TEST = False)")

    log(f"\n6. FEATURE IMPORTANCE (permutation on val, forest fitted on train, top {TOP_FEATURES})")
    for target in TARGETS:
        importances[target] = importance(val_models[target], val[x_cols], val[target], target)
        log(f"   {target}")
        for name, value in importances[target].head(TOP_FEATURES).items():
            log(f"       {name:<32} {value:.4f}")
    plot_importance(importances)

    log("\n7. NOTES")
    log("   - Heat-wave labels are runs of days, so the label of D+1 also depends on D+2 (Stage 3).")
    log("   - Positive shares grow from train to test (hotter recent summers); compare models on the same split.")
    log("   - A Random Forest cannot predict values above those seen in training (leaves store averages).")
    log("   - Correlated features (e.g. feels_like_mean / feels_like_max / lags) share their importance.")
    log("   - The test set is small (2 summers), so the test numbers are noisy.")

    results = pd.DataFrame(results).round(4)
    counts = ["TP", "FP", "FN", "TN"]
    results[counts] = results[counts].astype("Int64")    # whole numbers, empty for the regression rows
    results.to_csv(RESULTS_CSV, index=False)
    pd.concat(predictions).to_csv(PREDICTIONS_CSV)
    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    log(f"\n   Saved: {RESULTS_CSV}, {PREDICTIONS_CSV}, {IMPORTANCE_PNG} and {REPORT_TXT}")


if __name__ == "__main__":
    main()
