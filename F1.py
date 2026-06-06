import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import KBinsDiscretizer

warnings.simplefilter("ignore")

class CFG:
    COMP_PATH = "/kaggle/input/competitions/playground-series-s6e5/"
    ORIG_PATH = "/kaggle/input/datasets/aadigupta1601/f1-strategy-dataset-pit-stop-prediction/"
    
    TARGET = "PitNextLap"
    N_SPLITS = 5
    SEED = 2026
    
    USE_ORIG = True
    ORIG_WEIGHT = 0.65
    USE_GROUP_CV = False
    
    TE_COMBOS = [("Race", "Compound"), ("Race", "Year")]
    TE_SMOOTH = 60.0
    
    LGB_WEIGHT = 0.40
    XGB_WEIGHT = 0.35
    CAT_WEIGHT = 0.25

    LGB_PARAMS = dict(
        objective="binary", metric="auc", learning_rate=0.025,
        num_leaves=63, max_depth=8, min_child_samples=60,
        subsample=0.85, subsample_freq=1, colsample_bytree=0.80,
        reg_alpha=0.3, reg_lambda=3.0, random_state=SEED,
        n_estimators=8000, verbose=-1, n_jobs=-1
    )
    
    XGB_PARAMS = dict(
        objective="binary:logistic", eval_metric="auc", learning_rate=0.025,
        max_depth=7, min_child_weight=5, subsample=0.85,
        colsample_bytree=0.80, reg_alpha=0.5, reg_lambda=4.0,
        tree_method="hist", early_stopping_rounds=300,
        random_state=SEED, n_estimators=8000, n_jobs=-1
    )
    
    CAT_PARAMS = dict(
        loss_function="Logloss", eval_metric="AUC", learning_rate=0.03,
        depth=6, l2_leaf_reg=5.0, random_seed=SEED,
        iterations=8000, early_stopping_rounds=300, verbose=0
    )


class FeatureEngineer:
    IMPORTANT_COMBOS = [("Race", "Compound"), ("Race", "Year")]
    BIN_CONFIG = {"RaceProgress": 150, "LapTime (s)": 7}
    
    def __init__(self, cat_cols: list, num_cols: list):
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self._state = {}
        self.combo_names = []
    
    def fit_transform(self, df):
        return self._transform(df, fit=True)
    
    def transform(self, df):
        return self._transform(df, fit=False)
    
    def _transform(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        df = df.copy()
        
        # 基础交互特征
        df["feat_RemainingRace"] = (1 - df["RaceProgress"]).astype("float32")
        df["feat_WearRate"] = (df["TyreLife"] / (df["LapNumber"] + 1)).astype("float32")
        df["feat_TyreLife_per_Stint"] = (df["TyreLife"] / (df["Stint"] + 1)).astype("float32")
        df["feat_TyreLife_x_RaceProgress"] = (df["TyreLife"] * df["RaceProgress"]).astype("float32")
        df["feat_LapTime_x_TyreLife"] = (df["LapTime (s)"] * df["TyreLife"]).astype("float32")
        df["feat_LapTime_x_Degradation"] = (df["LapTime (s)"] * df["Cumulative_Degradation"]).astype("float32")
        df["feat_LapTime_x_DegradAbs"] = (df["LapTime (s)"] * df["Cumulative_Degradation"].abs()).astype("float32")
        df["feat_LapTime_div_DegradAbs"] = (df["LapTime (s)"] / (df["Cumulative_Degradation"].abs() + 1e-6)).astype("float32")
        df["feat_Position_x_RaceProgress"] = (df["Position"] * df["RaceProgress"]).astype("float32")
        df["feat_LapNumber_div_RaceProgress"] = (df["LapNumber"] / (df["RaceProgress"] + 1e-6)).astype("float32")
        df["feat_TyreLife_div_LapNumber"] = (df["TyreLife"] / df["LapNumber"].clip(lower=1)).astype("float32")
        df["feat_Degrad_per_TyreLife"] = (df["Cumulative_Degradation"] / (df["TyreLife"] + 1)).astype("float32")
        df["feat_Position_per_Lap"] = (df["Position"] / df["LapNumber"].clip(lower=1)).astype("float32")
        
        # GM级别新增：轮胎衰减斜率与战术风险交叉
        df["feat_Degradation_Slope"] = (df["Cumulative_Degradation"] / (df["TyreLife"] + 1e-6)).astype("float32")
        df["feat_PitWindow_Risk"] = (df["RaceProgress"] * (21 - df["Position"])).astype("float32")
        
        extra_num = [
            "feat_LapNumber_div_RaceProgress", "feat_TyreLife_div_LapNumber",
            "feat_Degrad_per_TyreLife", "feat_Position_per_Lap",
            "feat_Degradation_Slope", "feat_PitWindow_Risk"
        ]
        
        for c in ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]:
            df[f"ohe_Compound_{c}"] = (df["Compound"].astype(str) == c).astype("int8")
        
        floor_cols = self.num_cols + extra_num
        for col in floor_cols:
            cname = f"floor_{col}"
            floored = np.floor(df[col])
            if fit:
                _, uniques = floored.factorize()
                self._state[f"fc_{col}"] = {v: i for i, v in enumerate(uniques)}
            df[cname] = floored.map(self._state[f"fc_{col}"]).fillna(-1).astype("int32")
        
        for col in self.cat_cols:
            cname = f"count_{col}"
            if fit:
                self._state[f"cnt_{col}"] = df[col].value_counts()
            df[cname] = df[col].map(self._state[f"cnt_{col}"]).fillna(0).astype("int32")
        
        for col, n_bins in self.BIN_CONFIG.items():
            bname = f"kbin_{col}_{n_bins}"
            if fit:
                kb = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile", subsample=None)
                self._state[f"kb_{col}"] = kb
                df[bname] = kb.fit_transform(df[[col]]).ravel().astype("int32")
            else:
                df[bname] = self._state[f"kb_{col}"].transform(df[[col]]).ravel().astype("int32")
        
        self.combo_names = []
        for cols in self.IMPORTANT_COMBOS:
            cname = "_x_".join(cols)
            self.combo_names.append(cname)
            df[cname] = df[cols[0]].astype(str) + "_" + df[cols[1]].astype(str)
        
        df = df.replace([np.inf, -np.inf], np.nan)
        return df

def weighted_smooth_target_encoding(tr_combo, y_tr, w_tr, va_combo, te_combo, cols, n_splits, smooth, seed):
    y_arr = np.asarray(y_tr).astype(float)
    w_arr = np.asarray(w_tr).astype(float)
    global_mean = np.average(y_arr, weights=w_arr)
    
    enc_tr, enc_va, enc_te = pd.DataFrame(index=tr_combo.index), pd.DataFrame(index=va_combo.index), pd.DataFrame(index=te_combo.index)
    inner = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    for col in cols:
        new_col = f"te_{col}"
        oof = np.zeros(len(tr_combo), dtype="float32")
        tmp_all = pd.DataFrame({"key": tr_combo[col].astype(str).values, "y": y_arr, "w": w_arr, "yw": y_arr * w_arr})
        
        for in_idx, out_idx in inner.split(tr_combo, y_arr):
            tmp = tmp_all.iloc[in_idx]
            stat = tmp.groupby("key").agg(yw=("yw", "sum"), w=("w", "sum"))
            mapping = (stat["yw"] + smooth * global_mean) / (stat["w"] + smooth)
            oof[out_idx] = tr_combo.iloc[out_idx][col].astype(str).map(mapping).fillna(global_mean).astype("float32").values
        
        stat = tmp_all.groupby("key").agg(yw=("yw", "sum"), w=("w", "sum"))
        mapping = (stat["yw"] + smooth * global_mean) / (stat["w"] + smooth)
        
        enc_tr[new_col] = oof
        enc_va[new_col] = va_combo[col].astype(str).map(mapping).fillna(global_mean).astype("float32").values
        enc_te[new_col] = te_combo[col].astype(str).map(mapping).fillna(global_mean).astype("float32").values
        
    return enc_tr, enc_va, enc_te

def main():
    print("=" * 60)
    print("Loading Data & Initiating Grandmaster Ensemble Framework")
    print("=" * 60)
    
    comp_train = pd.read_csv(CFG.COMP_PATH + "train.csv")
    test_raw = pd.read_csv(CFG.COMP_PATH + "test.csv")
    subm = pd.read_csv(CFG.COMP_PATH + "sample_submission.csv")
    
    comp_train["__is_orig"] = 0
    test_raw["__is_orig"] = 0
    
    try:
        orig_train = pd.read_csv(CFG.ORIG_PATH + "f1_strategy_dataset_v4.csv")
        orig_train = orig_train.drop(columns=["Normalized_TyreLife"], errors="ignore")
        orig_train = orig_train[[c for c in comp_train.columns if c in orig_train.columns]]
        orig_train["__is_orig"] = 1
        print(f"Competition train: {comp_train.shape} | Original train: {orig_train.shape}")
    except FileNotFoundError:
        orig_train = pd.DataFrame(columns=comp_train.columns)
        print("Original dataset missing. Comp-only execution.")
        
    CAT_COLS = test_raw.drop(columns=["id", "__is_orig"], errors="ignore").select_dtypes(include=["object", "category"]).columns.tolist()
    NUM_COLS = [c for c in test_raw.columns if c not in ["id"] and c not in CAT_COLS]
    
    y = comp_train[CFG.TARGET].copy()
    
    if CFG.USE_GROUP_CV:
        groups = comp_train["Race"].astype(str) + "_" + comp_train["Year"].astype(str)
        splitter = StratifiedGroupKFold(n_splits=CFG.N_SPLITS, shuffle=True, random_state=CFG.SEED)
        split_iter = splitter.split(comp_train, y, groups)
    else:
        splitter = StratifiedKFold(n_splits=CFG.N_SPLITS, shuffle=True, random_state=CFG.SEED)
        split_iter = splitter.split(comp_train, y)
        
    oof_preds = np.zeros(len(comp_train))
    test_preds = np.zeros(len(test_raw))
    fold_aucs = []
    
    for fold, (tr_idx, va_idx) in enumerate(split_iter, start=1):
        print(f"\n---  Blending Fold {fold}/{CFG.N_SPLITS} ---")
        
        tr_raw = comp_train.iloc[tr_idx].copy()
        va_raw = comp_train.iloc[va_idx].copy()
        
        if CFG.USE_ORIG and len(orig_train) > 0:
            tr_all_raw = pd.concat([tr_raw, orig_train], ignore_index=True)
        else:
            tr_all_raw = tr_raw.copy()
            
        y_tr_all = tr_all_raw[CFG.TARGET].copy()
        y_va = va_raw[CFG.TARGET].copy()
        w_tr_all = np.where(tr_all_raw["__is_orig"].values == 1, CFG.ORIG_WEIGHT, 1.0)
        
        fe = FeatureEngineer(CAT_COLS, NUM_COLS)
        tr_all_fe = fe.fit_transform(tr_all_raw)
        va_fe = fe.transform(va_raw)
        te_fe = fe.transform(test_raw)
        
        combo_names = fe.combo_names
        
        for col in CAT_COLS:
            known = pd.concat([tr_all_fe[col].astype(str), te_fe[col].astype(str)], ignore_index=True).unique()
            mp = {v: i for i, v in enumerate(known)}
            tr_all_fe[col] = tr_all_fe[col].astype(str).map(mp).fillna(-1).astype("int32")
            va_fe[col] = va_fe[col].astype(str).map(mp).fillna(-1).astype("int32")
            te_fe[col] = te_fe[col].astype(str).map(mp).fillna(-1).astype("int32")
            
        DROP = {"id", CFG.TARGET} | set(combo_names)
        FEAT = [c for c in tr_all_fe.columns if c not in DROP]
        
        X_tr = tr_all_fe[FEAT].copy()
        X_va = va_fe[FEAT].copy()
        X_te = te_fe[FEAT].copy()
        
        enc_tr, enc_va, enc_te = weighted_smooth_target_encoding(
            tr_combo=tr_all_fe[combo_names], y_tr=y_tr_all, w_tr=w_tr_all,
            va_combo=va_fe[combo_names], te_combo=te_fe[combo_names],
            cols=combo_names, n_splits=CFG.N_SPLITS, smooth=CFG.TE_SMOOTH, seed=CFG.SEED + fold
        )
        
        X_tr = pd.concat([X_tr, enc_tr], axis=1)
        X_va = pd.concat([X_va, enc_va], axis=1)
        X_te = pd.concat([X_te, enc_te], axis=1)
        
        #[Model 1] LightGBM
        model_lgb = lgb.LGBMClassifier(**CFG.LGB_PARAMS)
        model_lgb.fit(X_tr, y_tr_all, sample_weight=w_tr_all, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(300, verbose=False)])
        val_p_lgb = model_lgb.predict_proba(X_va)[:, 1]
        test_p_lgb = model_lgb.predict_proba(X_te)[:, 1]
        
        #[Model 2] XGBoost 
        model_xgb = xgb.XGBClassifier(**CFG.XGB_PARAMS)
        model_xgb.fit(X_tr, y_tr_all, sample_weight=w_tr_all, eval_set=[(X_va, y_va)], verbose=False)
        val_p_xgb = model_xgb.predict_proba(X_va)[:, 1]
        test_p_xgb = model_xgb.predict_proba(X_te)[:, 1]
        
        # [Model 3] CatBoost
        model_cat = CatBoostClassifier(**CFG.CAT_PARAMS)
        model_cat.fit(X_tr, y_tr_all, sample_weight=w_tr_all, eval_set=(X_va, y_va), use_best_model=True)
        val_p_cat = model_cat.predict_proba(X_va)[:, 1]
        test_p_cat = model_cat.predict_proba(X_te)[:, 1]
        
        #强行执行折内多模异构线性融合
        fold_val_blend = (val_p_lgb * CFG.LGB_WEIGHT) + (val_p_xgb * CFG.XGB_WEIGHT) + (val_p_cat * CFG.CAT_WEIGHT)
        fold_test_blend = (test_p_lgb * CFG.LGB_WEIGHT) + (test_p_xgb * CFG.XGB_WEIGHT) + (test_p_cat * CFG.CAT_WEIGHT)
        
        oof_preds[va_idx] = fold_val_blend
        test_preds += fold_test_blend / CFG.N_SPLITS
        
        auc = roc_auc_score(y_va, fold_val_blend)
        fold_aucs.append(auc)
        print(f"LGB AUC: {roc_auc_score(y_va, val_p_lgb):.5f} | XGB AUC: {roc_auc_score(y_va, val_p_xgb):.5f} | Cat AUC: {roc_auc_score(y_va, val_p_cat):.5f}")
        print(f"--> Blended Fold {fold} AUC: {auc:.5f}")
        
    print("\n" + "=" * 60)
    print(" Grandmaster Heterogeneous Ensemble OOF Results")
    print("=" * 60)
    print(f"Fold AUCs       : {[f'{s:.5f}' for s in fold_aucs]}")
    print(f"Mean + Std      : {np.mean(fold_aucs):.5f} ± {np.std(fold_aucs):.5f}")
    print(f"Overall OOF AUC : {roc_auc_score(y, oof_preds):.5f}")
    
    pd.DataFrame({"id": comp_train["id"], CFG.TARGET: oof_preds}).to_csv("oof_gm_ensemble.csv", index=False)
    subm[CFG.TARGET] = test_preds
    subm.to_csv("submission.csv", index=False)
    print("\nFinal Outputs Saved Successfully: submission.csv & oof_gm_ensemble.csv")

if __name__ == "__main__":
    main()
