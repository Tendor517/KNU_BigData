"""
STAGE 3: after report from Stage 3

Input: ASOS_03_Daegu_daily_features.csv
Output: ASOS_03_Daegu_daily_features_after_report.csv

Task: drops 2 columns, 2 targets, because those are to rare to distinguishe:
* y_heatwave_alarm_extended
* y_heatwave_alarm

Requirements:  pip install pandas numpy
Run:           python 03_make_features.py
"""
import pandas as pd

# ---------- SETTINGS ----------
INPUT_CSV = "ASOS_03_Daegu_daily_features.csv"
OUTPUT_CSV = "ASOS_03_Daegu_daily_features_after_report.csv"
REPORT_TXT = "ASOS_03_Daegu_daily_features_after_report.txt"

DROP_COLUMNS = ["y_heatwave_alarm_extended", "y_heatwave_alarm"]
TIME_COLUMN = "date"      

report: list[str] = []

def log(text: str = "") -> None:
    """Prints a line to the console and keeps it for the report file."""
    print(text)
    report.append(text)

def main():
    df = pd.read_csv(INPUT_CSV, parse_dates=["date"], index_col="date").sort_index()
    log(f"Loaded: {len(df)} rows, {len(df.columns)} columns")

    absent = [c for c in DROP_COLUMNS if c not in df.columns]
    if absent:
        raise KeyError(f"Columns to drop not found: {absent}")
    
    df = df.drop(columns=DROP_COLUMNS)
    log(f"Dropped: {', '.join(DROP_COLUMNS)}")
    log(f"Rows: {len(df)}, columns: {len(df.columns)}")
    
    df.to_csv(OUTPUT_CSV)
    with open(REPORT_TXT, "w", encoding="utf-8") as f:
            f.write("\n".join(report))
    log(f"\n   Saved: {OUTPUT_CSV} and {REPORT_TXT}")


if __name__ == "__main__":
    main()
