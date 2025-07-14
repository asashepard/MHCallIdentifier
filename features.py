# features.py
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from pandas.tseries.holiday import USFederalHolidayCalendar
from typing import Tuple, Dict, Any

# Feature-engineering utilities for Austin PD CAD mental-health project
# Returns a *new* DataFrame enriched with domain-specific signals plus a dictionary of fitted artefacts

# Helpers
def _secs(td: pd.Series) -> pd.Series:
    """Convert a Timedelta series to seconds (float)."""
    return td.dt.total_seconds()


def _season(month: int) -> str:
    """Map calendar month to a simple season bucket."""
    return {
        12: "cold", 1: "cold", 2: "cold",      # Dec-Feb
        3: "mild", 4: "mild", 5: "mild",      # Mar-May
        6: "hot", 7: "hot", 8: "hot",        # Jun-Aug
        9: "mild", 10: "mild", 11: "cold"     # Sep-Nov
    }[month]

# Main entry-point
def engineer_features(
    df: pd.DataFrame,
    artefacts: Dict[str, Any] | None = None,
    fit: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    # initialise
    if artefacts is None:
        artefacts = {}

    df = df.copy()

    # 1. Response- & arrival-based timing metrics
    df["ResponseToArrival"] = _secs(df["First Unit Arrived Datetime"] - df["Response Datetime"])
    df["ArrivalToClose"]   = _secs(df["Call Closed Datetime"]       - df["First Unit Arrived Datetime"])
    df["ResponseLagRatio"] = df["ResponseToArrival"] / (df["ArrivalToClose"] + 1)

    # Median-gap normalisation per (Council District × Priority Level)
    key = df["Council District"].astype(str) + "_" + df["Priority Level"].astype(str)
    if fit:
        artefacts["rt_medians"] = df.groupby(key)["ResponseToArrival"].median().to_dict()
    med_lookup = artefacts["rt_medians"]
    df["RespToArr_DiffMed"] = df["ResponseToArrival"] - key.map(med_lookup).fillna(0)

    # 2. Call-complexity proxy
    n_units_z   = (df["Number of Units Arrived"] - df["Number of Units Arrived"].mean()) / (df["Number of Units Arrived"].std() + 1)
    resp_time_z = (df["Response Time"]            - df["Response Time"].mean())            / (df["Response Time"].std() + 1)
    cat_change  = (df["Initial Problem Category"] != df["Final Problem Category"]).astype(int)
    df["CallComplexity"]  = n_units_z + resp_time_z + cat_change
    df["ChangedCategory"] = cat_change

    # 3. Temporal signatures
    df["Hour"]      = df["Response Datetime"].dt.hour
    df["DayOfWeek"] = df["Response Datetime"].dt.dayofweek

    # Weekend & holiday flags
    df["IsWeekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)
    df["Hour_x_Weekend"] = df["Hour"] * df["IsWeekend"]

    if "holidays" not in artefacts:
        cal = USFederalHolidayCalendar()
        dr  = pd.DatetimeIndex(df["Response Datetime"].dt.normalize().unique())
        artefacts["holidays"] = set(cal.holidays(start=dr.min(), end=dr.max()))
    df["IsFedHoliday"] = df["Response Datetime"].dt.normalize().isin(artefacts["holidays"]).astype(int)

    # Season bucket (cold / mild / hot)
    df["Season"] = df["Response Datetime"].dt.month.map(_season)

    # Fine-grained cyclical encodings
    df["HourSin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["HourCos"] = np.cos(2 * np.pi * df["Hour"] / 24)
    df["WDaySin"] = np.sin(2 * np.pi * df["DayOfWeek"] / 7)
    df["WDayCos"] = np.cos(2 * np.pi * df["DayOfWeek"] / 7)

    # Night-shift & payday heuristics
    hrs = df["Hour"]
    df["IsNight"]  = ((hrs >= 23) | (hrs <= 6)).astype(int)
    dom = df["Response Datetime"].dt.day
    df["IsPayday"] = dom.isin([1, 15]).astype(int)

    # 4. Geo-spatial signal — census block-group clustering
    df["CensusNum"] = df["Census Block Group"].fillna(0).astype(float)
    if fit:
        km = KMeans(n_clusters=20, random_state=42)
        df["BG_Cluster"] = km.fit_predict(df[["CensusNum"]])
        artefacts["kmeans"] = km
    else:
        km = artefacts["kmeans"]
        df["BG_Cluster"] = km.predict(df[["CensusNum"]])
    df.drop(columns=["CensusNum"], inplace=True)

    # 5. Domain-knowledge categorical scores
    high_sector = {"Baker", "David", "George"}
    low_sector  = {"Airport", "Edward", "Henry"}

    def _sector_score(sec: str) -> int:
        if sec in high_sector:
            return 2
        if sec in low_sector:
            return 0
        return 1

    df["SectorScore"] = df["Sector"].astype(str).map(_sector_score)

    # Initial & Final descriptor heuristic scores
    hi_desc = {
        "Attempted Suicide Hs", "Check Welfare Service", "Check Welfare Urgent",
        "Disturbance Urgent", "Disturbance Other", "Suspicious Person",
        "Trespass Urgent",
    }
    lo_desc = {
        "Alarm Burglar", "Auto Theft Service", "Disturbance Hs",
        "Disturbance Vehicle Urgent", "Doc / C.o. Violation", "Found/Abandoned Hazardous",
        "Nature Unknown Urgent", "Traffic Hazard", "Traffic Hazard Hs",
    }

    def _desc_score(d: str) -> int:
        if d in hi_desc:
            return 2
        if d in lo_desc:
            return 0
        return 1

    hi_final_cat = {"Welfare Check"}
    lo_final_cat = {
        "Shoot/Stab", "Alarms", "Auto Theft", "Burglary", "Crashes",
        "Disorderly Conduct", "Suspicious Things", "Theft", "Traffic Stop/Hazard",
    }

    def _final_cat_score(cat: str) -> int:
        if cat in hi_final_cat:
            return 2
        if cat in lo_final_cat:
            return 0
        return 1

    hi_disp = {"No Report MH", "Report Written", "Report Written MH", "Supplement Written"}
    lo_disp = {"False Alarm", "No Report", "Unable To Locate"}

    def _disp_score(disp: str) -> int:
        if disp in hi_disp:
            return 2
        if disp in lo_disp:
            return 0
        return 1

    df["FinalProbCatScore"] = df["Final Problem Category"].astype(str).map(_final_cat_score)
    df["CallDispScore"]     = df["Call Disposition Description"].astype(str).map(_disp_score)
    df["InitProbDescScore"] = df["Initial Problem Description"].astype(str).map(_desc_score)

    # Escalation indicator (Priority 0/1 assumed "high")
    df["PriorityNum"]   = df["Priority Level"].str.extract(r"(\d+)").astype(float)
    df["EscalatedCall"] = (df["PriorityNum"] <= 1).astype(int)

    # 6. Narrative-based TF-IDF embeddings (Initial + Final description)
    text_corpus = (
        df["Initial Problem Description"].fillna("") + " " +
        df["Final Problem Description"].fillna("")
    ).tolist()

    if fit:
        tfidf = TfidfVectorizer(max_features=75, stop_words="english")
        tfidf_matrix = tfidf.fit_transform(text_corpus)
        artefacts["tfidf"] = tfidf
    else:
        tfidf = artefacts["tfidf"]
        tfidf_matrix = tfidf.transform(text_corpus)

    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        index=df.index,
        columns=[f"TXT_{i}" for i in range(tfidf_matrix.shape[1])],
    )
    df = pd.concat([df, tfidf_df], axis=1)

    mh_kws = (
        "suicid", "welfar", "overdos", "self-harm", "mental", "edp", "emergency detention"
    )

    def _mh_kw_flag(row) -> int:
        txt = f"{row['Initial Problem Description']} {row['Final Problem Description']}".lower()
        return int(any(kw in txt for kw in mh_kws))

    df["MH_KeywordFlag"] = df.apply(_mh_kw_flag, axis=1)

    # 7. One-off binary heuristics derived from delta-analysis
    df["IsWelfareCheck"] = (
        (df["Initial Problem Category"] == "Welfare Check") |
        (df["Final Problem Category"]   == "Welfare Check")
    ).astype(int)

    df["MH_DispoFlag"] = df["Call Disposition Description"].str.contains(r"\bMH\b", case=False, na=False).astype(int)

    df["IsAlarmCall"] = (
        (df["Initial Problem Category"] == "Alarms") |
        (df["Final Problem Category"]   == "Alarms")
    ).astype(int)

    df["IsTrafficHazard"] = (
        (df["Initial Problem Category"] == "Traffic Stop/Hazard") |
        (df["Final Problem Category"]   == "Traffic Stop/Hazard")
    ).astype(int)

    # 8. Days since previous call *within the same sector* (captures bursts)
    df.sort_values("Response Datetime", inplace=True)
    df["PrevCallTs"] = df.groupby("Sector")["Response Datetime"].shift(1)
    df["DaysSincePrevCall"] = (
        (df["Response Datetime"] - df["PrevCallTs"]).dt.total_seconds() / 86400
    ).fillna(30)  # fallback to 1-month if no history
    df.drop(columns=["PrevCallTs"], inplace=True)

    # 9. Word-count & injury/scene-duration diagnostics (existing)
    df["InitDescWC"]  = df["Initial Problem Description"].fillna("").str.split().str.len()
    df["FinalDescWC"] = df["Final Problem Description"].fillna("").str.split().str.len()

    df["TotalInjuries"] = (
        df["Officer Injured/Killed Count"].fillna(0) +
        df["Subject Injured/Killed Count"].fillna(0) +
        df["Other Injured/Killed Count"].fillna(0)
    )

    df["CallDuration"] = _secs(df["Call Closed Datetime"] - df["Response Datetime"])
    df["ServiceFraction"] = df["ArrivalToClose"] / (df["CallDuration"] + 1)

    to_drop = [
        # raw timestamps & IDs
        "Response Datetime", "First Unit Arrived Datetime", "Call Closed Datetime",
        "Geo ID", "Census Block Group",
        # raw text & categories
        "Initial Problem Description", "Final Problem Description",
        "Initial Problem Category", "Final Problem Category", "Call Disposition Description",
        # zero-variance
        "Officer Injured/Killed Count",
        "Subject Injured/Killed Count",
        "Other Injured/Killed Count",
    ]
    df = df.drop(columns=to_drop)

    return df, artefacts
