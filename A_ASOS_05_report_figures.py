"""
Stage 5 (ASOS): figures for the final report (UrbanAtmos).

Input:   ASOS_03_Daegu_daily_features_after_report.csv   (Stage 3 after report)
         ASOS_04_model_results.csv, ASOS_04_predictions.csv  (Stage 4)
Output:  report_figures/*.png  (used in final_report.md)

Figures:
  1. fig_feels_like.png        how humidity raises the feels-like temperature (KMA formula)
  2. fig_hot_days_per_year.png tropical nights and advisory days per year, with the train / val / test split
  3. fig_threshold_val.png     how the Random Forest decision threshold was chosen on val
  4. fig_season_strip.png      every test day as a coloured cell: caught / missed / false alarm / quiet
  5. fig_waffle_test.png       the confusion matrix drawn as 304 squares (one per test day)
  6. fig_tmax_summer.png       real vs predicted max feels-like temperature, summer 2025
  7. fig_scoreboard.png        persistence vs Random Forest on every target, val and test

Nothing is trained here; the script only reads the saved Stage 4 results.

Requirements:  pip install pandas numpy matplotlib
Run:           python A_ASOS_05_report_figures.py   (after A_ASOS_04_modeling.py)
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")    # save figures to files, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle

from A_ASOS_03_make_features import feels_like_kma   # same formula as Stage 3

# ---------- SETTINGS ----------
FEATURES_CSV = "ASOS_03_Daegu_daily_features_after_report.csv"
RESULTS_CSV = "ASOS_04_model_results.csv"
PREDICTIONS_CSV = "ASOS_04_predictions.csv"
OUT_DIR = Path("report_figures")

CLASS_TARGETS = {"y_tropical_night": "Tropical night", "y_heatwave_advisory": "Heatwave advisory"}
REG_TARGET = "y_tmax_feels"
REGRESSION_WINDOW = ("2025-07-01", "2025-08-31")   # dates of the predicted day (D+1)

# Colours (checked for colour-blind safety)
SURFACE, GRID, AXIS = "#fcfcfb", "#e1e0d9", "#c3c2b7"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
MODEL = {"persistence": "#eb6834", "random_forest": "#6250d6"}           # orange / violet
MODEL_NAME = {"persistence": "Persistence", "random_forest": "Random Forest"}
OUTCOME = {                                                                 # one colour per confusion-matrix box
    "caught": ("#2a78d6", "Caught (true positive)"),
    "missed": ("#e34948", "Missed (false negative)"),
    "false_alarm": ("#eda100", "False alarm (false positive)"),
    "quiet": (GRID, "Correctly quiet (true negative)"),
}
METRIC = {"F1": "#6250d6", "precision": "#1baf7a", "recall": "#eda100"}
RAMP = ["#86b6ef", "#2a78d6", "#104281"]                                     # light -> dark blue

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "semibold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "legend.frameon": False,
    "legend.labelcolor": INK_2,
    "lines.linewidth": 2,
    "lines.solid_capstyle": "round",
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})


def save(fig, name: str) -> None:
    fig.savefig(OUT_DIR / name)
    plt.close(fig)
    print(f"   Saved: {OUT_DIR / name}")


def outcome(real: pd.Series, pred: pd.Series) -> pd.Series:
    """Confusion-matrix box of each day."""
    return pd.Series(np.select(
        [(real == 1) & (pred == 1), (real == 1) & (pred == 0), (real == 0) & (pred == 1)],
        ["caught", "missed", "false_alarm"], "quiet"), index=real.index)


# ---------- FIGURE 1 ----------
def fig_feels_like() -> None:
    """Feels-like temperature against humidity for three air temperatures."""
    rh = np.linspace(20, 100, 161)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for ta, colour in zip([30, 33, 35], RAMP):
        fl = feels_like_kma(pd.Series(np.full_like(rh, ta)), pd.Series(rh))
        ax.plot(rh, fl, color=colour, label=f"air {ta} °C")
        ax.annotate(f"air {ta} °C", (rh[-1], fl.iloc[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", color=INK_2)
    for level, text in [(33, "advisory level 33 °C"), (35, "alarm level 35 °C")]:
        ax.axhline(level, color=MUTED, linewidth=1)
        ax.annotate(text, (21, level), xytext=(0, 4), textcoords="offset points", color=INK_2, fontsize=9)
    ax.set_xlim(20, 100)
    ax.set_xlabel("Relative humidity (%)")
    ax.set_ylabel("Feels-like temperature (°C)")
    ax.set_title("Same air temperature, different feels-like: humidity decides", loc="left")
    ax.legend(loc="lower right")
    save(fig, "fig_feels_like.png")


# ---------- FIGURE 2 ----------
def fig_hot_days(df: pd.DataFrame) -> None:
    """Positive days per year (season May-September), with the split shaded."""
    per_year = df.groupby(df.index.year)[list(CLASS_TARGETS)].sum().astype(int)
    years = per_year.index.to_numpy()
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.2), sharex=True)
    for ax, (col, name) in zip(axes, CLASS_TARGETS.items()):
        ax.bar(years, per_year[col], width=0.55, color=RAMP[1], zorder=3)
        for x, v in zip(years, per_year[col]):
            ax.annotate(str(v), (x, v), xytext=(0, 3), textcoords="offset points", ha="center",
                        fontsize=8, color=INK_2)
        ax.set_title(f"{name} days per summer (May-September)", loc="left")
        ax.set_ylim(0, per_year[col].max() * 1.25)
        ax.grid(axis="x", visible=False)
        for (start, end), label in [((2013.5, 2022.5), "train"), ((2022.5, 2024.5), "val"), ((2024.5, 2026.5), "test")]:
            if label != "train":
                ax.axvspan(start, end, color="#f0efec", zorder=0)
            ax.text((start + end) / 2, per_year[col].max() * 1.2, label, ha="center", va="top",
                    color=MUTED, fontsize=9)
    axes[1].set_xticks(years)
    save(fig, "fig_hot_days_per_year.png")


# ---------- FIGURE 3 ----------
def fig_threshold(pred: pd.DataFrame, results: pd.DataFrame) -> None:
    """Precision, recall and F1 of the Random Forest on val for every threshold."""
    thresholds = np.round(np.arange(0.05, 0.96, 0.01), 2)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, (target, name) in zip(axes, CLASS_TARGETS.items()):
        part = pred[(pred["target"] == target) & (pred["split"] == "val")]
        real, prob = part["real"], part["random_forest_output"]
        curves = {"precision": [], "recall": [], "F1": []}
        for t in thresholds:
            p = (prob >= t).astype(int)
            tp, fp, fn = ((p == 1) & (real == 1)).sum(), ((p == 1) & (real == 0)).sum(), ((p == 0) & (real == 1)).sum()
            prec = tp / (tp + fp) if tp + fp else np.nan
            rec = tp / (tp + fn)
            curves["precision"].append(prec)
            curves["recall"].append(rec)
            curves["F1"].append(2 * prec * rec / (prec + rec) if prec and rec else 0)
        for name_m in ["precision", "recall", "F1"]:
            ax.plot(thresholds, curves[name_m], color=METRIC[name_m], label=name_m,
                    linewidth=2.6 if name_m == "F1" else 2)

        row = results[(results["target"] == target) & (results["split"] == "val")].set_index("model")
        chosen = row.loc["random_forest", "threshold"]
        best_f1 = row.loc["random_forest", "F1"]
        base_f1 = row.loc["persistence", "F1"]
        ax.axvline(chosen, color=INK_2, linewidth=1)
        ax.scatter(chosen, best_f1, s=70, color=METRIC["F1"], edgecolor=SURFACE, linewidth=2, zorder=4)
        ax.axhline(base_f1, color=MUTED, linewidth=1)
        ax.annotate(f"persistence F1 {base_f1:.2f}", (0.06, base_f1), xytext=(0, -12), textcoords="offset points",
                    color=INK_2, fontsize=9, bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1, "alpha": 0.85})
        ax.set_title(f"{name} (val 2023-2024)\nchosen threshold {chosen:.2f} -> best F1 {best_f1:.2f}", loc="left")
        ax.set_xlabel("Threshold (say 'yes' if probability ≥ threshold)")
        ax.set_xlim(0.05, 0.95)
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("Score (0 = worst, 1 = perfect)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    save(fig, "fig_threshold_val.png")


# ---------- FIGURE 4 ----------
def fig_season_strip(pred: pd.DataFrame) -> None:
    """Every test day as one cell; rows: real / persistence / Random Forest, per summer."""
    test = pred[pred["split"] == "test"].copy()
    test["day"] = test.index + pd.Timedelta(days=1)          # the day that is predicted (D+1)
    years = sorted(test["day"].dt.year.unique())
    rows = ["real", "persistence", "random_forest"]

    fig, axes = plt.subplots(len(CLASS_TARGETS) * len(years), 1, figsize=(12, 6.8))
    axes = iter(axes)
    for target, name in CLASS_TARGETS.items():
        for year in years:
            ax = next(axes)
            part = test[(test["target"] == target) & (test["day"].dt.year == year)]
            x0 = pd.Timestamp(f"{year}-05-01")
            for r, row_name in enumerate(rows):
                y = len(rows) - 1 - r
                if row_name == "real":
                    colours = np.where(part["real"] == 1, INK_2, GRID)
                    counts = f"{int(part['real'].sum())} real event days"
                else:
                    box = outcome(part["real"], part[row_name])
                    colours = [OUTCOME[b][0] for b in box]
                    n = box.value_counts()
                    counts = (f"caught {n.get('caught', 0)} · missed {n.get('missed', 0)} · "
                              f"false alarms {n.get('false_alarm', 0)}")
                for day, colour in zip(part["day"], colours):
                    ax.add_patch(Rectangle(((day - x0).days, y + 0.08), 0.86, 0.84, color=colour, linewidth=0))
                ax.text(-2, y + 0.5, "Real" if row_name == "real" else MODEL_NAME[row_name],
                        ha="right", va="center", color=INK, fontsize=9)
                ax.text(155, y + 0.5, counts, ha="left", va="center", color=INK_2, fontsize=8.5)
            ax.set_xlim(0, 153)
            ax.set_ylim(0, len(rows))
            ax.set_yticks([])
            months = pd.date_range(f"{year}-05-01", f"{year}-09-01", freq="MS")
            ax.set_xticks([(m - x0).days for m in months])
            ax.set_xticklabels([m.strftime("%b") for m in months])
            ax.grid(False)
            ax.spines[["left", "bottom"]].set_visible(False)
            ax.tick_params(length=0)
            ax.set_title(f"{name} · summer {year}", loc="left", fontsize=10)
    handles = [Patch(color=INK_2, label="Event happened")] + [Patch(color=c, label=l) for c, l in OUTCOME.values()]
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Test summers day by day: what really happened and how each model answered",
                 x=0.01, ha="left", fontsize=12, fontweight="semibold", color=INK)
    fig.tight_layout(rect=(0.05, 0.02, 0.86, 0.97))
    save(fig, "fig_season_strip.png")


# ---------- FIGURE 5 ----------
def fig_waffle(pred: pd.DataFrame) -> None:
    """Confusion matrix as 304 squares (one per test day), sorted by box."""
    test = pred[pred["split"] == "test"]
    order = ["caught", "missed", "false_alarm", "quiet"]
    n_cols = 19                                           # 19 x 16 = 304 days
    fig, axes = plt.subplots(len(CLASS_TARGETS), 2, figsize=(9.5, 8.6))
    for i, (target, name) in enumerate(CLASS_TARGETS.items()):
        part = test[test["target"] == target]
        for j, model in enumerate(["persistence", "random_forest"]):
            ax = axes[i, j]
            box = outcome(part["real"], part[model])
            counts = box.value_counts().reindex(order, fill_value=0)
            cells = np.repeat(order, counts.to_numpy())
            for k, b in enumerate(cells):
                col, row = k % n_cols, k // n_cols
                ax.add_patch(Rectangle((col, -row), 0.86, 0.86, color=OUTCOME[b][0], linewidth=0))
            n_rows = int(np.ceil(len(cells) / n_cols))
            ax.set_xlim(-0.2, n_cols)
            ax.set_ylim(-n_rows - 0.2, 1)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title(f"{name}: {MODEL_NAME[model]}", loc="left")
            ax.text(0, -n_rows - 0.4,
                    f"caught {counts['caught']} · missed {counts['missed']} · "
                    f"false alarms {counts['false_alarm']} · quiet {counts['quiet']}",
                    va="top", color=INK_2, fontsize=9)
    handles = [Patch(color=c, label=l) for c, l in OUTCOME.values()]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("The confusion matrix as 304 test days (one square = one day)",
                 x=0.02, ha="left", fontsize=12, fontweight="semibold", color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    save(fig, "fig_waffle_test.png")


# ---------- FIGURE 6 ----------
def fig_tmax(pred: pd.DataFrame) -> dict:
    """Real vs predicted max feels-like temperature for summer 2025, plus the size of the misses."""
    part = pred[(pred["target"] == REG_TARGET) & (pred["split"] == "test")].copy()
    part["day"] = part.index + pd.Timedelta(days=1)
    window = part[(part["day"] >= REGRESSION_WINDOW[0]) & (part["day"] <= REGRESSION_WINDOW[1])]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.3), gridspec_kw={"width_ratios": [2.3, 1]})
    ax1.plot(window["day"], window["real"], color=INK, label="Real")
    for model in ["persistence", "random_forest"]:
        ax1.plot(window["day"], window[model], color=MODEL[model], label=MODEL_NAME[model], linewidth=1.8)
    ax1.axhline(33, color=MUTED, linewidth=1)
    ax1.annotate("advisory level 33 °C", (window["day"].iloc[-1], 33), xytext=(0, -14), textcoords="offset points",
                 ha="right", color=INK_2, fontsize=9,
                 bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1, "alpha": 0.85})
    ax1.set_title("Max feels-like temperature, July-August 2025 (test)", loc="left")
    ax1.set_ylabel("°C")
    ax1.legend(loc="lower right", ncol=3)
    ax1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %b"))

    # misses on all test days, in buckets
    bins, labels = [0, 1, 2, 3, np.inf], ["< 1 °C", "1-2 °C", "2-3 °C", "≥ 3 °C"]
    shares = {}
    x = np.arange(len(labels))
    for k, model in enumerate(["persistence", "random_forest"]):
        miss = (part[model] - part["real"]).abs()
        share = pd.cut(miss, bins, right=False, labels=labels).value_counts(normalize=True).reindex(labels) * 100
        shares[model] = share.round(1)
        bars = ax2.bar(x + (k - 0.5) * 0.36, share, width=0.34, color=MODEL[model], label=MODEL_NAME[model], zorder=3)
        for bar, v in zip(bars, share):
            ax2.annotate(f"{v:.0f}%", (bar.get_x() + bar.get_width() / 2, v), xytext=(0, 3),
                         textcoords="offset points", ha="center", fontsize=8, color=INK_2)
    ax2.set_xticks(x, labels)
    ax2.set_title("Size of the miss, all 304 test days", loc="left")
    ax2.set_ylabel("Share of days (%)")
    ax2.grid(axis="x", visible=False)
    ax2.legend(loc="upper right")
    fig.tight_layout()
    save(fig, "fig_tmax_summer.png")
    return shares


# ---------- FIGURE 7 ----------
def fig_scoreboard(results: pd.DataFrame) -> None:
    """Dumbbell chart: persistence vs Random Forest. F1 (higher is better) and MAE (lower is better)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.2), gridspec_kw={"width_ratios": [1.6, 1]})
    rows = [(t, s) for t in CLASS_TARGETS for s in ["val", "test"]]
    for ax, metric, items, label_of in [
        (ax1, "F1", rows, lambda t, s: f"{CLASS_TARGETS[t]} · {s}"),
        (ax2, "MAE", [(REG_TARGET, "val"), (REG_TARGET, "test")], lambda t, s: f"Max feels-like · {s}"),
    ]:
        for y, (target, split) in enumerate(reversed(items)):
            r = results[(results["target"] == target) & (results["split"] == split)].set_index("model")[metric]
            ax.plot([r["persistence"], r["random_forest"]], [y, y], color=AXIS, linewidth=2, zorder=1)
            for model in ["persistence", "random_forest"]:
                ax.scatter(r[model], y, s=70, color=MODEL[model], edgecolor=SURFACE, linewidth=2, zorder=3,
                           label=MODEL_NAME[model] if y == 0 and metric == "F1" else None)
                right = r[model] >= r.max()            # the larger value is labelled on the right
                ax.annotate(f"{r[model]:.3f}" if metric == "F1" else f"{r[model]:.2f}", (r[model], y),
                            xytext=(9 if right else -9, 0), textcoords="offset points",
                            ha="left" if right else "right", va="center", fontsize=9, color=INK_2)
        ax.set_yticks(range(len(items)), [label_of(t, s) for t, s in reversed(items)])
        ax.set_ylim(-0.6, len(items) - 0.4)
        ax.grid(axis="y", visible=False)
        ax.margins(x=0.25)
    ax1.set_title("F1 for the yes/no targets (higher is better)", loc="left")
    ax2.set_title("MAE in °C (lower is better)", loc="left")
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    save(fig, "fig_scoreboard.png")


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(FEATURES_CSV, parse_dates=["date"], index_col="date")
    results = pd.read_csv(RESULTS_CSV)
    pred = pd.read_csv(PREDICTIONS_CSV, parse_dates=["date"], index_col="date")

    print("Making report figures")
    fig_feels_like()
    fig_hot_days(df)
    fig_threshold(pred, results)
    fig_season_strip(pred)
    fig_waffle(pred)
    shares = fig_tmax(pred)
    fig_scoreboard(results)

    print("   Share of test days by size of the miss (%), y_tmax_feels:")
    print(pd.DataFrame(shares).rename(columns=MODEL_NAME).to_string())


if __name__ == "__main__":
    main()
