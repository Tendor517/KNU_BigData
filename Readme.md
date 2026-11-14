## How to use the files

I grouped every file with the dataset it works on. Every important file has the dataset name at the beginning (for ASOS files I also added `A_` in front).

### Files corresponding to every dataset

| file | ASOS | ERA5 |
|---|---|---|
| python | `A_ASOS_01_fetch...` -> downloads ASOS data, <br>`A_ASOS_02_clean...` -> cleans ASOS data, <br>`A_ASOS_02_after_report...` -> removes columns that are not helpful (too many NaN = not good), <br>`A_ASOS_03_make_f...` -> adds features and converts hourly data into daily data, <br>`A_ASOS_03_make_features_after_report...` -> drops the two alarm targets (too rare to model), <br>`A_ASOS_04_modeling.py` -> persistence baseline vs Random Forest, <br>`A_ASOS_05_report_figures.py` -> figures for the final report | `01_fetch_daegu...` -> downloads ERA5 data, <br>`02_clean_open...` -> cleans ERA5 data (no NaN, because the model took care of them), <br>`03_make_features` -> adds features to the data, does not convert them into daily format |
| outputs | `A_ASOS_01...` -> `ASOS_01_Daegu...csv` `OUR DATA`, <br>`A_ASOS_02...` -> `ASOS_02_Daegu...csv` `cleaned data` + report, <br>`A_ASOS_02_after_report.py` -> `ASOS_022_...` `removes 2 columns` (too many NaN), <br>`A_ASOS_03...` -> `ASOS_03_Daegu_daily...` `features turned into daily data` + report, <br>`A_ASOS_03_..._after_report.py` -> `ASOS_03_Daegu_daily_features_after_report.csv` `ready to model` + report, <br>`A_ASOS_04_modeling.py` -> `ASOS_04_model_results.csv`, `ASOS_04_predictions.csv`, `ASOS_04_feature_importance.png` + `ASOS_04_modeling_report.txt`, <br>`A_ASOS_05_report_figures.py` -> `report_figures/*.png` (used in `final_report.md`) | `01_fetch_daegu...` -> `ERA5_01...`, <br>`02_clean...` -> `ERA5_02...csv` data + `Daegu_clean_report.txt`, <br>`03_make_fea...` -> `ERA5_03_Daegu...` |
| other | `asos_raw` is the raw data used to build `ASOS_01_Daegu...csv`, which is our base dataset for ASOS and is used by all other steps | `Parameter_description.txt` is a file that describes the parameters in the ERA5 dataset |

`FOR MODELING DO NOT USE THE ERA5 DATASET !!!!` It is not ready for modeling yet. The only file needed for modeling is `ASOS_03_Daegu_daily_features_after_report.csv`.

## Stage 0: problem to solve

We want to predict whether a heatwave or a tropical night is going to occur.

### Conditions

| type | Advisory | Alarm | Major alert |
|---|---|---|---|
| heatwave | When the daily maximum perceived temperature is expected to remain at 33℃ or higher for two or more consecutive days:<br>`Tmax >= 33℃ for days >= 2` | When the daily maximum perceived temperature is expected to remain at 35℃ or higher for two or more consecutive days:<br>`Tmax >= 35℃ for days >= 2` | In areas where the daily maximum wind chill temperature is observed to be 35℃ or higher for three or more days:<br>`Tmax >= 35℃ for days >= 3` |
| tropical night | Tropical nights are defined as days when the nighttime minimum temperature is 25℃ or higher:<br>`Tmin >= 25℃` | none | none |

## Stage 1: Dataset

To satisfy the conditions we downloaded 2 datasets. First we used the ERA5 dataset from Open-Meteo, which is created by models based on satellite data. We combined two datasets: the Open-Meteo weather dataset and the Open-Meteo air quality dataset. Later in the project we decided to switch to the ASOS dataset from the KMA (Korea Meteorological Administration), which contains the official weather measurements in South Korea.

### Comparison

| compare | ASOS (KMA dataset) | ERA5 (Open-Meteo dataset) |
|---|---|---|
| data | measured by weather stations | retrieved from satellite-based models |
| number of variables available | | |
| number of variables picked | | |
| how many years available | | |
| years picked | from 2014 until now, because in 2013 the station's location was changed | from 2022 until now, because the air pollution dataset is available from 2022 |
| continuous variables | | |
| discrete variables | | |

## Stage 2: Cleaning data

This is not a very important step for modeling, but briefly:
- there were strange values in the dataset that we needed to replace, such as `-9.0` and `-99.0` (KMA codes for missing data)
- we put all of the data on a full hourly time axis
- we filled gaps shorter than 3 hours with linear interpolation:

```math
y = y_0 + \frac{y_1 - y_0}{x_1 - x_0} \cdot (x - x_0)
```

After the report:
- we had to remove two columns because they had too many missing values (NaN):
  - `cloud_cover_tenths`: NaN in 100% of 2014, 57% of 2015, ~22% of 2016-2017, 13% of 2018
  - `visibility_10m`: NaN in 100% of 2014, 35% of 2015

## Stage 3: daily features dataset (`ASOS_03_Daegu_daily_features.csv`)

Created by `A_ASOS_03_make_features.py` from `ASOS_022_Daegu_after_report_clean.csv`.
A text report with counts per split and per year is saved next to it
(`ASOS_03_Daegu_daily_features_report.txt`).

### What the dataset is

- One row is one summer day **D** in Daegu (KMA station 143), May-September 2014-2026 (about 1976 rows).
- The features describe day D, using data only up to 23:00 of D.
- The targets (`y_*`) describe day **D+1**. The task is "predict tomorrow".
- The index is `date`, the date of D.

### Column groups

| group | columns | notes |
|---|---|---|
| Daily features | `temperature_*`, `feels_like_*`, `relative_humidity_*`, `dew_point_*` (min / max / mean), pressures, wind, solar, sunshine, ground temperature, precipitation, `temperature_range`, `feels_like_max_hour` | 26 columns |
| Memory features | `*_lag1/2/3/7`, `*_roll3/7`, `station_pressure_diff1/3`, `hot33_streak`, `hot35_streak`, `night_min_air_prev` | past only, nothing from the future |
| Season | `doy_sin`, `doy_cos` | day of the year as a circle |
| Targets | `y_tmax_feels` (regression); `y_tropical_night`, `y_heatwave_advisory`, `y_heatwave_alarm`, `y_heatwave_alarm_extended` (binary 0/1) | day D+1 |
| Not a feature | `check_tmax_air_next_day`, `split` | see the rules below |

### Target definitions

- **Feels-like temperature**: KMA summer formula (effective from 2022-06-02) with Stull's wet-bulb temperature,
  computed hourly from air temperature and relative humidity, then aggregated to days.
- **Tropical night** (`y_tropical_night`): night minimum **air** temperature (18:01-09:00) >= 25 degC.
- **Heat wave advisory** (`y_heatwave_advisory`): daily max feels-like >= 33 degC for 2 or more consecutive days.
- **Alarm** (`y_heatwave_alarm`): daily max feels-like >= 35 degC for 2 or more consecutive days.
- **Extended alarm** (`y_heatwave_alarm_extended`): daily max feels-like >= 35 degC for 3 or more consecutive days.
- **Regression target** (`y_tmax_feels`): daily maximum feels-like temperature of day D+1.

### Rules for using the data

1. **Features are every column except `y_*`, `check_tmax_air_next_day` and `split`.**
   `check_tmax_air_next_day` is tomorrow's air temperature, so it would be leakage as a feature.
   It exists only for sanity checks against official KMA values.
2. **Use the `split` column**: train 2014-2022, val 2023-2024, test 2025-2026.
   Never shuffle rows, because neighbouring days are strongly correlated.
3. **Tune on val and use test only once**, at the very end.
4. **No scaling is needed for Random Forest.** The file is stored unscaled
   (a scaler would only be needed for models such as logistic regression; fit it on train only).
5. **Missing values**: only a few remain (solar radiation, sunshine, precipitation).
   Random Forest in scikit-learn >= 1.4 handles NaN directly, so no imputer is needed.
6. **Which targets to train**: `y_tropical_night`, `y_heatwave_advisory` and `y_tmax_feels`.
   The two alarm targets have almost no positive days (train: 0.4% and 0.0%; val and test: about 3 days each),
   so they are dropped after the report (see "Stage 3 after report" below).

### Minimal loading example and train / val / test split

The split is chronological and stored in the `split` column (it is decided by the date D of the row,
and the cut points are the settings `TRAIN_END` and `VAL_END` in `A_ASOS_03_make_features.py`).

| split | years (date D) | rows | role |
|---|---|---|---|
| `train` | 2014-2022 | 1368 | fit models, imputers and scalers |
| `val` | 2023-2024 | 304 | choose hyperparameters and the decision threshold |
| `test` | 2025-2026 | 304 | final evaluation, used once |

Share of positive days (%) in each split:

| target | train | val | test |
|---|---|---|---|
| `y_tropical_night` | 12.6 | 16.4 | 19.7 |
| `y_heatwave_advisory` | 9.3 | 15.1 | 21.4 |

Rules:
- Rows are never shuffled, and no row of a later split is used to fit anything on an earlier one.
- The split falls between seasons (the data contain only May-September), so no summer is cut in two.
- The positive share grows from train to test (more hot days in recent years),
  so compare models on the same split and always report the positive share next to the metric.
- For the final model, after the choices are made on val, you may refit on train + val
  and then evaluate once on test.

```python
import pandas as pd

df = pd.read_csv("ASOS_03_Daegu_daily_features_after_report.csv", parse_dates=["date"], index_col="date")
y_cols = [c for c in df.columns if c.startswith("y_")]
X_cols = [c for c in df.columns if c not in y_cols + ["split", "check_tmax_air_next_day"]]

target = "y_heatwave_advisory"
train, val, test = (df[df["split"] == s] for s in ["train", "val", "test"])
X_train, y_train = train[X_cols], train[target]
X_val, y_val = val[X_cols], val[target]
X_test, y_test = test[X_cols], test[target]
```

### Modelling advice

- **Class imbalance**: positives are rare (about 9-21% depending on target and split).
  `class_weight="balanced"` can help; in Stage 4 it is one of the tuned options, not forced.
- **Metrics**: do not use accuracy. Use PR-AUC, recall and precision (or F1) at a threshold chosen on val,
  and the Brier score for probabilities. Report the positive share per split next to every result.
- **Baseline to beat**: persistence (tomorrow equals today, e.g. `hot33_streak > 0`).
  Climatology (by day of year) is an easier baseline and is not used in Stage 4.
- **Tuning inside train**: expanding-window time-series cross-validation by year, not random K-fold.
- **Regression** (`y_tmax_feels`): report MAE and RMSE against the persistence baseline
  (`feels_like_max` of today).

### Known limitations

- Heat wave labels are runs of days, so the label of D+1 uses information from D+2
  (and D+3 for the extended alarm). This is part of the definition and should be mentioned in the report.
- Hourly data give daily maxima and minima slightly below the official KMA values (KMA uses 10-minute data),
  so days close to a threshold can be classified differently.
- Feels-like temperature is computed by us, not downloaded from KMA.
- 2026 ends on 29 September. The test set is small (304 days), so yearly positive counts are small and results are noisy.

## Stage 3 after report (`ASOS_03_Daegu_daily_features_after_report.csv`)

Created by `A_ASOS_03_make_features_after_report.py` from `ASOS_03_Daegu_daily_features.csv`.
It drops `y_heatwave_alarm` and `y_heatwave_alarm_extended` (6 and 0 positive days in train, too rare to learn).
Result: 1976 rows, 59 feature columns, targets `y_tmax_feels`, `y_tropical_night`, `y_heatwave_advisory`,
plus `check_tmax_air_next_day` and `split`. This is the input of Stage 4.

## Stage 4: Modeling

Script: `A_ASOS_04_modeling.py`. Input: `ASOS_03_Daegu_daily_features_after_report.csv` only (not ERA5).
It follows the rules from Stage 3: chronological `split`, no shuffling, test used once.

### How to run

```bash
pip install pandas numpy scikit-learn matplotlib   # scikit-learn >= 1.4 (Random Forest with NaN)
python A_ASOS_04_modeling.py                        # about 1 minute
```

`EVALUATE_TEST = True` in the settings. Set it to `False` while experimenting, so the test set stays unseen.

| output | content |
|---|---|
| `ASOS_04_model_results.csv` | metrics per target, model and split (val, test) |
| `ASOS_04_predictions.csv` | every val / test day: real value, persistence answer, Random Forest probability (or degC) and answer |
| `ASOS_04_feature_importance.png` | top 10 features of each Random Forest (permutation importance on val) |
| `ASOS_04_modeling_report.txt` | the full log of every step |

### Goal

Predict tomorrow (day D+1) from the data of day D (59 features, everything except `y_*`, `check_tmax_air_next_day` and `split`):

| task | target | type |
|---|---|---|
| tropical night | `y_tropical_night` | classification (0/1) |
| heatwave advisory | `y_heatwave_advisory` | classification (0/1) |
| max feels-like temperature | `y_tmax_feels` | regression (degC) |

### Models

**Persistence (baseline)**: "tomorrow is like today". Fixed rules with the KMA thresholds, nothing is learned:

| target | rule |
|---|---|
| `y_tmax_feels` | tomorrow = today's `feels_like_max` |
| `y_heatwave_advisory` | 1 if today's `feels_like_max >= 33` (same as `hot33_streak > 0`) |
| `y_tropical_night` | 1 if `night_min_air_prev >= 25` (the last complete night) |

**Random Forest**: 500 decision trees. Each tree is built on a random sample of training days (drawn with replacement)
and, at every split, looks at a random subset of features. Each tree asks yes/no questions such as
"`night_min_air_prev` >= 24.85?". A leaf stores the share of yes-days (classification) or the average
`y_tmax_feels` (regression) of the training days that end there. The forest averages all trees.
It needs no scaling and handles the few NaN values directly.

### Step by step (as implemented)

1. **Load and split.** Train 2014-2022 (1368 days), val 2023-2024 (304), test 2025-2026 (304).
2. **Persistence.** Apply the rules above to val and test. No training.
3. **Random Forest settings.** For each target, a small grid is checked with an expanding window by year
   inside train only (fit 2014-2016 -> score 2017, ..., fit 2014-2021 -> score 2022). Score: PR-AUC for 0/1 targets,
   MAE for `y_tmax_feels`. The best mean score wins.
   - classification grid: `min_samples_leaf` in {5, 10, 20} x `class_weight` in {None, "balanced"}, `max_features="sqrt"`
   - regression grid: `min_samples_leaf` in {5, 10, 20} x `max_features` in {"sqrt", 1.0}
4. **Val.** Fit on train with the chosen settings, predict val. For 0/1 targets choose the threshold
   (0.05-0.95, step 0.01) with the best F1 on val (tie -> lower threshold, i.e. more warnings).
   Compare with persistence on the same days.
5. **Test once.** Refit on train + val with the frozen settings and threshold, then evaluate on test a single time.
6. **Explain.** Permutation importance on val (forest fitted on train) and test errors (missed days, false alarms) per month.

### Choices made (inside train / on val)

| target | chosen settings | threshold (val) |
|---|---|---|
| `y_tropical_night` | `min_samples_leaf=20`, `class_weight=None` | 0.55 |
| `y_heatwave_advisory` | `min_samples_leaf=5`, `class_weight=None` | 0.31 |
| `y_tmax_feels` | `min_samples_leaf=5`, `max_features=1.0` | - |

### Metrics

- Do not use accuracy (always answering "no" would already give about 84% on val).
- Classification: precision, recall, **F1** (main metric, at the threshold chosen on val), PR-AUC and Brier score.
  Persistence gives no probability, so its PR-AUC is empty and its Brier score uses the 0/1 answer.
- Regression: **MAE** (main metric), RMSE and bias (mean of prediction - real), compared with persistence.
- The positive share of the split is written next to every result.

### Results

Classification (PR-AUC: - = not defined for a 0/1 rule):

| target | model | split | positive share (%) | precision | recall | F1 | PR-AUC | Brier |
|---|---|---|---|---|---|---|---|---|
| `y_tropical_night` | persistence | val | 16.4 | 0.620 | 0.620 | 0.620 | - | 0.125 |
| `y_tropical_night` | Random Forest | val | 16.4 | 0.711 | 0.640 | **0.674** | 0.685 | 0.085 |
| `y_tropical_night` | persistence | test | 19.7 | 0.600 | 0.600 | 0.600 | - | 0.158 |
| `y_tropical_night` | Random Forest | test | 19.7 | 0.688 | 0.550 | **0.611** | 0.755 | 0.084 |
| `y_heatwave_advisory` | persistence | val | 15.1 | 0.745 | 0.826 | 0.784 | - | 0.069 |
| `y_heatwave_advisory` | Random Forest | val | 15.1 | 0.778 | 0.913 | **0.840** | 0.909 | 0.043 |
| `y_heatwave_advisory` | persistence | test | 21.4 | 0.783 | 0.831 | 0.806 | - | 0.086 |
| `y_heatwave_advisory` | Random Forest | test | 21.4 | 0.767 | 0.862 | **0.812** | 0.815 | 0.070 |

Regression (`y_tmax_feels`, degC):

| model | split | MAE | RMSE | bias |
|---|---|---|---|---|
| persistence | val | 1.633 | 2.158 | -0.05 |
| Random Forest | val | **1.454** | 1.850 | -0.19 |
| persistence | test | 1.638 | 2.210 | -0.03 |
| Random Forest | test | **1.482** | 1.961 | -0.15 |

### Reading the results

- **`y_tmax_feels`**: Random Forest is better on both splits: 11.0% smaller MAE on val and 9.6% on test
  (1.48 vs 1.64 degC). The negative bias means it predicts slightly too low on average (0.15-0.19 degC),
  consistent with recent summers being hotter than the training years and with a forest not predicting
  values above those seen in training.
- **`y_heatwave_advisory`**: clearly better on val (F1 0.840 vs 0.784), only slightly better on test (0.812 vs 0.806).
  It catches more advisory days (recall 0.862 vs 0.831), with a few more false alarms.
  Persistence is a strong baseline because hot spells last several days.
- **`y_tropical_night`**: better on val (F1 0.674 vs 0.620), about equal on test (0.611 vs 0.600).
  On test the forest gives fewer false alarms (15 vs 24) but misses more nights (27 vs 24), mostly in August (16 of the 28 August tropical nights).
  The threshold 0.55 chosen on val looks too high for the hotter test summers (19.7% positives vs 16.4% on val).
- Its probabilities are useful: Brier score lower than persistence on every split (e.g. 0.084 vs 0.158 for tropical nights on test), and test PR-AUC 0.755 / 0.815
  (a random guess would score about the positive share, 0.20 / 0.21).
- The test results were not used to change anything.

### Most important features (permutation importance on val)

- `y_tropical_night`: `night_min_air_prev`, `temperature_min`, `feels_like_min`, `doy_cos`, `dew_point_min`.
- `y_heatwave_advisory`: `feels_like_mean`, `temperature_max`, `ground_temperature_mean`, `temperature_range`.
- `y_tmax_feels`: `feels_like_mean` dominates (MAE +1.9 degC when shuffled), then `feels_like_max`.
- Correlated features (e.g. `feels_like_mean`, `feels_like_max` and their lags) share importance,
  so a low value does not always mean a feature is useless.

### Limitations

- Heat wave labels are runs of days, so the label of D+1 also uses information from D+2 (part of the definition).
- Hourly data give daily maxima and minima slightly below the official KMA values (KMA uses 10-minute data).
- Feels-like temperature is computed by us, not downloaded from KMA.
- The test set is small (2 summers, 304 days), so test results are noisy.
- Positive shares grow from train to test (hotter recent summers); a threshold chosen on val may not fit later years.
- ERA5 features are not used.

## Stage 5: Final report

The final report is [`final_report.md`](final_report.md). It explains every stage (data, cleaning, features, modeling,
conclusions) for a non-expert audience, with the formulas and algorithms explained in plain language, and it is the
reference for the presentation (one section per stage, each with a key message and suggested slides).

Its figures are made by `A_ASOS_05_report_figures.py` from the Stage 3 and Stage 4 outputs (nothing is trained there):

```bash
python A_ASOS_05_report_figures.py      # after A_ASOS_04_modeling.py; writes report_figures/*.png
```

| figure | content |
|---|---|
| `fig_feels_like.png` | feels-like temperature against humidity (KMA formula) |
| `fig_hot_days_per_year.png` | tropical nights and advisory days per summer, with the train / val / test split |
| `fig_threshold_val.png` | precision, recall and F1 of the Random Forest on val for every threshold |
| `fig_scoreboard.png` | persistence vs Random Forest: F1 and MAE on val and test |
| `fig_waffle_test.png` | the test confusion matrices drawn as 304 squares, one per day |
| `fig_season_strip.png` | every test day as a cell: caught / missed / false alarm / correctly quiet |
| `fig_tmax_summer.png` | real vs predicted max feels-like temperature (summer 2025) and the size of the misses |
