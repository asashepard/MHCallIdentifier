## if you see this, add a feature that measures the length of the numbers (i.e. number of decimals)
from typing import Any, Dict
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

from features import engineer_features

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "16"  # Limit to 16 cores for multiprocessing

class GlobalEncoders:
    fitted = False
    cat_cols_for_te = [
        'Council District',
        'Priority Level',
        'Sector',
    ]

    target_encodings: Dict[str, Dict[Any, float]] = {}

    @classmethod
    def fit(cls, df: pd.DataFrame, target_col: str):
        # Make a copy so we don’t clobber the original
        df = df.copy()

        # Append a row of "__UNKNOWN__" for each categorical col,
        # so unseen categories at transform time get handled gracefully
        for col in cls.cat_cols_for_te:
            if col in df:
                unknown = df.iloc[0].copy()
                unknown[col] = '__UNKNOWN__'
                df = pd.concat([df, pd.DataFrame([unknown])], ignore_index=True)

        for col in cls.cat_cols_for_te:
            if col in df.columns:
                unknown_row = df.iloc[0].copy()
                unknown_row[col] = '__UNKNOWN__'
                unknown_row[target_col] = df[target_col].mean()   # <-- NEW
                df = pd.concat([df, pd.DataFrame([unknown_row])],
                            ignore_index=True)

        # Compute mean(target) per category
        for col in cls.cat_cols_for_te:
            cls.target_encodings[col] = df.groupby(col)[target_col] \
                                          .mean().to_dict()

        cls.fitted = True

    @classmethod
    def transform(cls, df: pd.DataFrame):
        if not cls.fitted:
            raise RuntimeError(
                "GlobalEncoders.transform() called before fit(); "
                "call GlobalEncoders.fit(train_df, 'Mental Health Flag') first."
            )

        df = df.copy()

        # Map each categorical column to its learned mean
        for col in cls.cat_cols_for_te:
            global_mean = np.mean(list(cls.target_encodings[col].values()))
            df[col + '_TE'] = (
                df[col]
                  .astype(str)
                  .map(lambda x: cls.target_encodings[col]
                                   .get(x, global_mean))
            )

        return df

class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32], dropout=0.1):
        super(MLP, self).__init__()
        layers = []
        prev_dim = input_dim
        for hd in hidden_dims:
            layers.append(nn.Linear(prev_dim, hd))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hd
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(1)

def train_mlp(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for Xb, yb in dataloader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(Xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * Xb.size(0)
    return running_loss / len(dataloader.dataset)

def eval_mlp(model, dataloader, device):
    model.eval()
    preds = []
    reals = []
    with torch.no_grad():
        for Xb, yb in dataloader:
            Xb = Xb.to(device)
            out = model(Xb)
            pred = (out.sigmoid()>0.5).cpu().numpy()
            preds.extend(pred)
            reals.extend(yb.numpy())
    return preds, reals

if __name__ == "__main__":
    # Load data as DataFrames first
    df = pd.read_csv('./data/austin_dispatched_filtered.csv',
                  parse_dates=['Response Datetime',
                               'First Unit Arrived Datetime',
                               'Call Closed Datetime'])

    # split by date
    def get_subset(frame, mask, n=50_000, seed=42):
        subset = frame[mask]
        take   = min(n, len(subset))            # don’t error if < n rows
        return subset.sample(n=take, random_state=seed)

    train = get_subset(df, df['Response Datetime'] <  '2022-01-01')
    dev   = get_subset(df, (df['Response Datetime'] >= '2022-01-01') &
                        (df['Response Datetime'] <  '2024-01-01'))
    test  = get_subset(df, df['Response Datetime'] >= '2024-01-01')

    # Feature engineer train and dev separately, then combine
    train_df, artefacts = engineer_features(train, None, fit=True)
    dev_df, _          = engineer_features(dev, artefacts, fit=False)
    full = pd.concat([train_df, dev_df], keys=['train','dev'])

    # Fit global encoders on train set only
    train_idx = full.index.get_level_values(0)=='train'
    dev_idx = full.index.get_level_values(0)=='dev'

    train_df = full.loc[train_idx].copy()
    dev_df = full.loc[dev_idx].copy()

    GlobalEncoders.fit(train_df, target_col='Mental Health Flag')
    train_df = GlobalEncoders.transform(train_df)
    dev_df = GlobalEncoders.transform(dev_df)

    y_train = train_df['Mental Health Flag'].astype(int)
    y_dev = dev_df['Mental Health Flag'].astype(int)

    DROP_COLS = ['Mental Health Flag']
    for c in DROP_COLS:
        if c in train_df.columns:
            train_df.drop(columns=[c], inplace=True)
        if c in dev_df.columns:
            dev_df.drop(columns=[c], inplace=True)

    # Encode object columns
    object_cols = train_df.select_dtypes(include='object').columns.tolist()
    le_dict = {}

    for c in object_cols:
        le = LabelEncoder()
        unique_vals = train_df[c].unique().tolist()
        
        # Explicitly add '__UNKNOWN__' if not present
        if '__UNKNOWN__' not in unique_vals:
            unique_vals.append('__UNKNOWN__')
        
        le.fit(unique_vals)
        
        # Transform training data
        train_df[c] = le.transform(train_df[c].astype(str))
        le_dict[c] = le

    # Transform dev data
    for c in object_cols:
        known_categories = set(le_dict[c].classes_)
        # Map unknown values to '__UNKNOWN__'
        dev_df[c] = dev_df[c].astype(str).map(lambda x: x if x in known_categories else '__UNKNOWN__')
        # Transform
        dev_df[c] = le_dict[c].transform(dev_df[c])

    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    scaler = StandardScaler()
    for df_ in (train_df, dev_df):
        dt_cols = df_.select_dtypes(include=['datetime64[ns]', 'datetimetz']).columns
        df_.drop(columns=dt_cols, inplace=True)
    train_df[numeric_cols] = scaler.fit_transform(train_df[numeric_cols])
    dev_df[numeric_cols] = scaler.transform(dev_df[numeric_cols])
    train_df.fillna(0, inplace=True)
    dev_df.fillna(0, inplace=True)
    X_train = train_df
    X_dev = dev_df
    
    from catboost import CatBoostClassifier, Pool
    import optuna, lightgbm as lgb, numpy as np
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from lightgbm import early_stopping, log_evaluation

    # 1. helper: LightGBM custom eval that returns recall
    def recall_eval(preds, lgb_data):
        y_true = lgb_data.get_label()
        y_pred = (preds > 0.5).astype(int)          # fixed 0.5 inside booster
        return "recall", recall_score(y_true, y_pred), True # "True" = higher is better

    # 2. Optuna objective maximises OUT-OF-FOLD recall
    def objective(trial):
        params = {
            "objective": "binary",
            "metric":    "None",        # we supply custom feval
            "learning_rate": trial.suggest_float("lr", 0.01, 0.2, log=True),
            "num_leaves":    trial.suggest_int("num_leaves", 32, 256, log=True),
            "max_depth":     trial.suggest_int("max_depth",  -1, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 200),
            "subsample":     trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 5.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 5.0),
            "scale_pos_weight": (y_train==0).sum()/(y_train==1).sum(), # class-weight
            "verbosity": -1,
            "feature_pre_filter": False,
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        oof_pred = np.zeros(len(X_train))

        for tr_idx, va_idx in cv.split(X_train, y_train):
            dtrain = lgb.Dataset(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
            dvalid = lgb.Dataset(X_train.iloc[va_idx], y_train.iloc[va_idx])
            gbm = lgb.train(
                params,
                dtrain,
                num_boost_round=4000,
                valid_sets=[dvalid],
                feval=recall_eval,
                callbacks=[early_stopping(200, first_metric_only=True)],
            )
            oof_pred[va_idx] = gbm.predict(X_train.iloc[va_idx],
                                        num_iteration=gbm.best_iteration)

        # grid-search threshold that gives BEST recall on OOF predictions
        min_prec = 0.50

        best_recall, best_thr = -1, None
        for thr in np.arange(0.05, 0.96, 0.01):
            pred = (oof_pred > thr).astype(int)
            prec = precision_score(y_train, pred, zero_division=0)
            rec  = recall_score(y_train, pred, zero_division=0)

            # keep only thresholds that meet both floors
            if prec >= min_prec and rec > best_recall:
                best_recall, best_thr = rec, thr

        if best_thr is None:
            raise RuntimeError(
                "No threshold on dev meets precision≥0.50!"
            )

        trial.set_user_attr("best_threshold", best_thr)
        return 1.0 - best_recall # Optuna minimises

    # 3. run Optuna search (fewer trials = faster)
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=10, timeout=1800)   # ↓ trials for speed

    best_params = study.best_params
    best_thr    = study.best_trial.user_attrs["best_threshold"]

    final_params           = best_params.copy()
    final_params.update(                   # add the fixed keys we need
        objective="binary",
        metric="None",                     # we supply custom feval
        scale_pos_weight=(y_train==0).sum()/(y_train==1).sum(),
        verbosity=-1,
    )

    train_set  = lgb.Dataset(X_train, label=y_train)
    valid_set  = lgb.Dataset(X_dev,   label=y_dev)

    # 4. train FINAL model with best params
    gbm_final = lgb.train(
        params               = final_params,
        train_set            = train_set,
        num_boost_round      = 4000,
        valid_sets           = [valid_set],
        feval                = recall_eval,
        callbacks            = [
            early_stopping(200, first_metric_only=True),
            log_evaluation(period=50)
        ],
    )

    # A. Train CatBoost on the *numeric* matrix you have (categoricals are already 
    #    label-encoded, so we treat everything as numeric).
    cat_params = {
        "loss_function": "Logloss",
        "eval_metric":   "Recall",
        "iterations":    2000,
        "learning_rate": 0.05,
        "depth":         8,
        "random_seed":   42,
        "verbose":       200,
        "scale_pos_weight": (y_train==0).sum()/(y_train==1).sum()
    }

    train_pool = Pool(X_train, label=y_train)
    dev_pool   = Pool(X_dev,   label=y_dev)

    cb_clf = CatBoostClassifier(**cat_params)
    cb_clf.fit(train_pool,
            eval_set           = dev_pool,
            early_stopping_rounds = 200)

    # B. Get dev-set probabilities from BOTH models
    dev_proba_cb  = cb_clf.predict_proba(X_dev)[:, 1]           # CatBoost
    dev_proba_lgb = gbm_final.predict(X_dev,
                                    num_iteration=gbm_final.best_iteration)

    # C. Simple weighted average ensemble
    ALPHA = 0.2      # 0 = only LightGBM, 1 = only CatBoost
    dev_proba_ens = ALPHA * dev_proba_cb + (1 - ALPHA) * dev_proba_lgb

    # 5. evaluate on dev set
    dev_proba  = dev_proba_ens
    y_pred_dev = (dev_proba > best_thr).astype(int)

    final_metrics = {
        'accuracy':  accuracy_score(y_dev, y_pred_dev),
        'precision': precision_score(y_dev, y_pred_dev, zero_division=0),
        'recall':    recall_score(y_dev, y_pred_dev, zero_division=0),
        'f1':        f1_score(y_dev, y_pred_dev, zero_division=0)
    }
    print("\nLightGBM tuned for recall  (thr={:.2f})".format(best_thr))
    for k, v in final_metrics.items():
        print(f"{k.capitalize():9}: {v:.4f}")

    # TEST-SET PREDICTIONS  (keep engineering / alignment steps)

    test_df, _ = engineer_features(test)
    test_df = GlobalEncoders.transform(test_df)

    # drop high-cardinality text columns already defined in DROP_COLS
    for c in DROP_COLS:
        if c in test_df.columns:
            test_df.drop(columns=[c], inplace=True, errors='ignore')

    # label-encode object columns with same encoders as training
    object_cols_test = test_df.select_dtypes(include='object').columns.tolist()
    for c in object_cols_test:
        if c in le_dict:
            known = set(le_dict[c].classes_)
            test_df[c] = test_df[c].astype(str).map(
                lambda x: x if x in known else '__UNKNOWN__'
            )
            test_df[c] = le_dict[c].transform(test_df[c])

    # scale numerics, impute, and ensure numeric-only dataframe
    test_df[numeric_cols] = scaler.transform(test_df[numeric_cols])
    test_df.fillna(0, inplace=True)

    dt_cols = test_df.select_dtypes(include=['datetime64[ns]', 'datetimetz']).columns
    test_df.drop(columns=dt_cols, inplace=True)

    obj_cols_extra = test_df.select_dtypes(include='object').columns
    test_df.drop(columns=obj_cols_extra, inplace=True)

    for col in X_train.columns:
        if col not in test_df:
            test_df[col] = 0
    extra_cols = set(test_df.columns) - set(X_train.columns)
    test_df.drop(columns=list(extra_cols), inplace=True)
    test_df = test_df[X_train.columns]
    test_df.fillna(0, inplace=True)

    # Predict and write file
    test_proba_cb  = cb_clf.predict_proba(test_df)[:, 1]
    test_proba_lgb = gbm_final.predict(test_df, num_iteration=gbm_final.best_iteration)
    test_proba_ens = ALPHA * test_proba_cb + (1 - ALPHA) * test_proba_lgb
    final_test_pred = (test_proba_ens > best_thr).astype(int)

    with open('predictions.txt', 'w') as f:
        for p in final_test_pred:
            f.write(f"{p}\n")