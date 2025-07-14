# MHCallIdentifier — Project Notes

An Optuna-tuned ensemble (PyTorch + LightGBM + CatBoost) to classify 9-1-1 mental-health calls based on Austin PD's dataset.

> **Goal**  
> Catch as many true MH calls as we can (high recall) **without** drowning human dispatchers in false alarms (so we still keep precision decent).

---

## Table of Contents
1. [Folder Layout](#folder-layout)
2. [Data Cheat-Sheet](#data-cheat-sheet)
3. [Pre-Processing Pipeline](#pre-processing-pipeline)
5. [Model Line-Up + Results](#model-line-up--results)
6. [Tuning & Thresholds](#tuning--thresholds)
8. [FAQ / Trade-Offs](#faq--trade-offs)
9. [Repro Steps](#repro-steps)

---

## Folder Layout

```text
project_root/
├── data/                       # raw & intermediate goodies
│   └── austin_dispatched_filtered.csv
├── features.py                 # feature-eng helpers
├── train.py                    # run this for train + test
├── requirements.txt            # pinned deps (GPU-agnostic)
└── README.md                   # you’re reading it
```

## Data Cheat-Sheet

Feature engineering was a large part of this project. Features were examined using ```explore.py``` which calculated deltas between MH=1 and MH=0 for numerical and categorical columns.

Here are the rows left out of the data:

| Column                      |     Keep?    | Why / Why Not                                            |
| --------------------------- | :----------: | -------------------------------------------------------- |
| **Response Datetime**       |      No      | Only used for time-split — leaks info if used as feature |
| **First Unit Arrived**      |      No      | Same leakage risk                                        |
| **Call Closed**             |      No      | Same                                                     |
| **Geo ID / CBG**            |      No      | Super sparse & high-cardinality                          |
| **Problem Desc / Category** | No (for now) | Rich text but slow (TF-IDF ↑ latency 1 → 5 s, F1 +0.01)  |
| **Injured / Killed counts** |      No      | All zeros — dead weight                                  |

In some cases, rows were dropped **after** they were used for high-impact engineered features based on my findings with ```explore.py```.

## Pre-Processing Pipeline

1. Time Split
```train < 2022-01-01, dev = 2022–2023, test = 2024+.```
Data before 2022 is ignored for now.
2. Features
```engineer_features()``` in ```features.py```
Adds 9 new engineered features to the dataframe, drops columns.
3. Target Encoding on Council District, Priority, Sector
4. Label Encoding
5. Standard Scaling for numeric cols (helps tabular NNs & LightGBM).
6. Class Imbalance
Ditched naïve up-sampling. Using scale_pos_weight in LightGBM.

## Model Line Up + Results
| Model        | Accuracy / Precision / Recall / F1          | Notes                               |
| ------------ | --------------------- | ----------------------------------- |
| **LogReg**   | 0.74 / 0.30 / 0.78 / 0.43           | 15 s sanity check                   |
| **LightGBM** | 0.84 / 0.43 / 0.63 / 0.51                  |  |
| **CatBoost** | ~+0.02 Precision, ~+0.01 Accuracy / F1 when stacked | Optional, ordered boosting          |

**Why LightGBM and CatBoost?**
LightGBM won most experiments because tree-based gradient boosting handles messy tabular data (mixed dtypes, lots of missing values) with almost zero preprocessing, trains quickly, and keeps inference sub-second; CatBoost's ordered boosting and built-in categorical encoding curb overfitting on our imbalanced dataset and learns a different “view” of the data, so stacking it with LightGBM bumps precision/F1/accuracy without extra latency.

**Why is the model optimized for recall?** 
Recall is relatively difficult to achieve with a high level of precision (precision baseline = 0.18) because there are very few examples of MH=1 in the dataset. It's possible to achieve >90% accuracy and >90% precision with low recall. However, police departments may rather know with greater certainty that a MH=1 event is a true positive than otherwise. A program that tags every event as MH is useless. Hence, the algorithm focuses on recall with a precision threshold rather than accuracy and precision.

## Tuning & Thresholds
* Optuna: 10 trials × 5-fold stratified CV.
* Max recall while keeping precision ≥ 0.50.
* Early-Stopping: patience 200 (7 min → 2 min per trial).
* Threshold Sweep: 0.05 … 0.95 on out-of-fold probs to pick the sweet spot.

## FAQ / Trade-Offs
Why duplicate positives in v1?
* `scale_pos_weight` is cleaner (no duplicate rows → less variance). 

Why yank text fields?
* Super sparse; LightGBM likes dense mats. TF-IDF barely helped and made inference ~5× slower. Suggestion: use a Hugging Face model.

Why force precision above a certain threshold?
* Dispatcher time is precious. Anything lower and they’d be buried in false alarms.
* Acknowledgement that there may be a place for prioritizing precision over recall e.g. if a dispatcher is primarily concerned with knowing if something is **not** a MH event.

## Repro Steps
**1. clone**
```git clone https://github.com/your_handle/apd-mh-flag.git && cd apd-mh-flag```

**2. make a fresh Python 3.11 env**
```python -m venv .venv && source .venv/bin/activate```
```pip install -r requirements.txt```

**3. drop in the raw CSV**
```mkdir -p data && cp /path/to/austin_dispatched_filtered.csv data/```
* This is a CSV accessible on the APD website (https://data.austintexas.gov/Public-Safety/APD-Computer-Aided-Dispatch-Incidents/22de-7rzg/data_preview) filtered for dispatched calls only and rows <5 years old.

**4. train & get predictions**
```python train.py```
**→ prints metrics + writes predictions.txt in project root**
