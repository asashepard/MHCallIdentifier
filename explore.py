# explore.py
import pandas as pd
import numpy as np
from features import engineer_features

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "16"  # Limit to 16 cores for multiprocessing

# helpers
def read_and_engineer(csv_path: str):
    """Load raw (or pre-filtered) Austin CAD CSV and return engineered DataFrame."""
    df = pd.read_csv(
        csv_path,
        parse_dates=[
            "Response Datetime",
            "First Unit Arrived Datetime",
            "Call Closed Datetime",
        ],
    )

    # Run feature engineering (fit=True because this is an exploratory pass)
    engineered_df, _ = engineer_features(df, artefacts=None, fit=True)
    return engineered_df

DROP_COLS = ["Response Datetime", "First Unit Arrived Datetime", "Call Closed Datetime", "Incident Number"]

def get_feature_names(df: pd.DataFrame, target: str = "Mental Health Flag"):
    """Return every column except the target."""
    return [c for c in df.columns if c != target and c not in DROP_COLS]


def write_feature_diffs(
    df: pd.DataFrame,
    features: list[str],
    target: str = "Mental Health Flag",
    output_file: str = "feature_differences.txt",
):
    """
    For each feature, compare distributions between target==1 vs 0 and
    append results to a text file (one long report).
    """
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Austin PD Mental-Health vs Non-Mental-Health Feature Differences\n")
        f.write("=" * 70 + "\n\n")

        for feat in features:
            # skip high-cardinality text columns if desired
            if feat.startswith("TXT_"):
                continue

            f.write(f"Feature: {feat}\n")
            ser = df[feat]

            if pd.api.types.is_numeric_dtype(ser):
                mean1 = ser[df[target] == 1].mean()
                mean0 = ser[df[target] == 0].mean()
                f.write(f"  Numeric: mean(MH=1) = {mean1:.3f}, mean(MH=0) = {mean0:.3f}\n")
                f.write(f"  Δ mean           = {mean1 - mean0:.3f}\n\n")
            else:
                f.write("  Categorical frequency (% rows)\n")
                freq1 = (
                    ser[df[target] == 1]
                    .value_counts(normalize=True)
                    .rename("mh1")
                )
                freq0 = (
                    ser[df[target] == 0]
                    .value_counts(normalize=True)
                    .rename("mh0")
                )
                all_vals = freq1.index.union(freq0.index)
                for val in all_vals:
                    f1 = freq1.get(val, 0.0)
                    f0 = freq0.get(val, 0.0)
                    f.write(
                        f"    {val}: MH=1 {f1:.2%} | MH=0 {f0:.2%} | Δ {f1 - f0:.2%}\n"
                    )
                f.write("\n")


# main
if __name__ == "__main__":
    CSV_PATH = "data/austin_dispatched_filtered.csv"

    engineered = read_and_engineer(CSV_PATH)
    feats = get_feature_names(engineered, target="Mental Health Flag")
    write_feature_diffs(engineered, feats)

    print("Feature-difference report written to feature_differences.txt")
