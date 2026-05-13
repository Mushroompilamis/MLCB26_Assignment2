#This script is deticated to the creation of the pipeline of the Repeated Nested Cross-Validation (rnCV) class 
#for heart disease classification, with task 3 not focusing in feature selection
###################################################################
# Importing the proper libraries
import warnings
import numpy as np
import pandas as pd
import optuna
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.base import clone, BaseEstimator
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import (matthews_corrcoef, roc_auc_score, balanced_accuracy_score,
                             f1_score, recall_score, precision_score,average_precision_score, confusion_matrix)

optuna.logging.set_verbosity(optuna.logging.WARNING)

###################################################################
# Redifining the columns to use them in the ncCV
# Logic behind separation of the types, since they need different preprocessing steps.
# Regarding continuous features --> Scaled
# Binary --> 0/1 kept
# Categorical --> one-hot encoded
# Ordinal --> Kept as ordered numerical variables

continuous_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
binary_cols = ["sex", "fbs", "exang"]
categorical_cols = ["cp", "restecg", "thal"]
ordinal_cols = ["slope", "ca"]
ALL_FEATURE_COLS = continuous_cols + binary_cols + categorical_cols + ordinal_cols
###################################################################

#Building the preprocessor which returns a ColumnTransformer that imputes, scales, and encodes features.

def build_preprocessor(present_columns: Optional[List[str]] = None) -> ColumnTransformer:

    if present_columns is not None:
        present_set = set(present_columns)
        cont_cols = [c for c in continuous_cols if c in present_set]
        bin_cols = [c for c in binary_cols if c in present_set]
        cat_cols = [c for c in categorical_cols if c in present_set]
        ord_cols = [c for c in ordinal_cols if c in present_set]
    else:
        cont_cols = list(continuous_cols)
        bin_cols = list(binary_cols)
        cat_cols = list(categorical_cols)
        ord_cols = list(ordinal_cols)

    # Preprocessing part
    continuous_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),("scaler", StandardScaler())])
    binary_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))])
    categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    ordinal_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])

    # Including only the non empty colymn groups
    transformers = []
    if cont_cols:
        transformers.append(("continuous", continuous_pipe, cont_cols))
    if bin_cols:
        transformers.append(("binary", binary_pipe, bin_cols))
    if cat_cols:
        transformers.append(("categorical", categorical_pipe, cat_cols))
    if ord_cols:
        transformers.append(("ordinal", ordinal_pipe, ord_cols))

    return ColumnTransformer(transformers=transformers,remainder="drop")



###################################################################
# Computation of the metrics 
def compute_metrics(y_true: np.ndarray,y_pred: np.ndarray,y_prob: np.ndarray) -> Dict[str, float]:
# Computation for the outer test fold
    #Confusion matrix & Specificity
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "MCC":         matthews_corrcoef(y_true, y_pred),
        "AUC":         roc_auc_score(y_true, y_prob),
        "BA":          balanced_accuracy_score(y_true, y_pred),
        "F1":          f1_score(y_true, y_pred, zero_division=0),
        "Recall":      recall_score(y_true, y_pred, zero_division=0),
        "Specificity": specificity,
        "Precision":   precision_score(y_true, y_pred, zero_division=0),
        "PRAUC":       average_precision_score(y_true, y_prob),
    }
###################################################################
###################################################################

# Repeated Nested Cross-Validation (rnCV) pipeline.
class RepeatedNestedCV:
    def __init__(self, estimators: List[Tuple[str, BaseEstimator]],param_spaces: Dict[str, Any],
        n_repeats: int = 10, n_outer: int = 5, n_inner: int = 3, n_trials: int = 50, base_seed: int = 42,
        tune: bool = True, feature_selection: bool = False, fs_n_features: Optional[int] = None, inner_metric: str = "roc_auc",):
     
        self.estimators = estimators
        self.param_spaces = param_spaces
        self.n_repeats = n_repeats
        self.n_outer = n_outer
        self.n_inner = n_inner
        self.n_trials = n_trials
        self.base_seed = base_seed
        self.tune = tune
        self.feature_selection = feature_selection
        self.fs_n_features = fs_n_features
        self.inner_metric = inner_metric

        # Dict with results stored after fit().
        self.results_: Dict[str, List[Dict[str, float]]] = {}

        # Dict for storing the feature selecions with if feature_selection=True
        self.feature_freq_: Dict[str, Dict[str, float]] = {}

    ###################################################################
    # Creating function for reproducible seeds
    def _make_seed(self, repeat: int, fold: int, extra: int = 0) -> int:
        return self.base_seed + repeat * 1000 + fold * 100 + extra
    ###################################################################
    # Selector function for the extraction of the positive class scores
    def _get_scores(self, fitted_model: BaseEstimator, X) -> np.ndarray:
        if hasattr(fitted_model, "predict_proba"):
            return fitted_model.predict_proba(X)[:, 1]

        if hasattr(fitted_model, "decision_function"):
            return fitted_model.decision_function(X)
        return fitted_model.predict(X)
    ###################################################################
    # Feature selector function (task 4 correspondance)
    def _make_selector(self, seed: int, n_features: int) -> SelectKBest:
        if self.fs_n_features is None:
            k = max(1, n_features // 2)
        else:
            k = min(self.fs_n_features, n_features)

        def mi_score(X, y):
            return mutual_info_classif(X, y, random_state=seed)

        return SelectKBest(score_func=mi_score, k=k)
    ###################################################################
    def _tune_hyperparams(self, clf_name: str, clf: BaseEstimator, X_train: pd.DataFrame, y_train: np.ndarray, seed: int, ) -> BaseEstimator:
    # Inner loop --> Optuna hyperparameter search. 
        if clf_name not in self.param_spaces or not self.tune:
            return clone(clf)
    #####
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError(
                f"Expects a pd.DataFrame, got {type(X_train)}. ")
    #####
        present_cols = list(X_train.columns)
        param_fn = self.param_spaces[clf_name]

        def objective(trial):
            params = param_fn(trial)
            candidate = clone(clf).set_params(**params)
            # The build_preprocessor receives present_cols so it only references 
            # columns that actually exist in X_train
            pipe = Pipeline([("preprocessor", build_preprocessor(present_cols)),("clf", candidate),])
            inner_cv = StratifiedKFold(n_splits=self.n_inner, shuffle=True, random_state=seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores = cross_val_score(pipe, X_train, y_train, cv=inner_cv,scoring=self.inner_metric, n_jobs=-1,)
            return scores.mean()

        study = optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler(seed=seed),)
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        return clone(clf).set_params(**study.best_params)
        ###################################################################
               
    def _select_features(self, selector: SelectFromModel, X_train_pp: np.ndarray, y_train: np.ndarray,X_test_pp: np.ndarray,
        feature_names: List[str],) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        selector.fit(X_train_pp, y_train)
        mask = selector.get_support()
        selected = [n for n, m in zip(feature_names, mask) if m]
        return selector.transform(X_train_pp), selector.transform(X_test_pp), selected

    ###################################################################
    ###################################################################
    #Running the rnCV pipeline
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RepeatedNestedCV":
        
        X = X.reset_index(drop=True)
        y_arr = y.reset_index(drop=True).values
        
        feature_names_raw = list(X.columns)
        for clf_name, _ in self.estimators:
            self.results_[clf_name] = []

        # Setting the number of outer fold evaluations per estimator.
        n_total_outer_folds = self.n_repeats * self.n_outer


        # Repeated outer CV loop
        for r in range(self.n_repeats):
            outer_cv = StratifiedKFold(n_splits=self.n_outer,shuffle=True,random_state=self._make_seed(r, 0))

            # Outer fold loop
            for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y_arr)):

                # Outer train & test split, with the second one remaing untouched until final evaluation
                X_tr = X.iloc[train_idx].reset_index(drop=True)
                X_te = X.iloc[test_idx].reset_index(drop=True)
                y_tr = y_arr[train_idx]
                y_te = y_arr[test_idx]

               
                for clf_name, clf in self.estimators:
                    inner_seed = self._make_seed(r, fold_idx, 1)
                    # Inner loop & hyperparameter tuning on outer train only

                    best_clf = self._tune_hyperparams(clf_name=clf_name, clf=clf, X_train=X_tr, y_train=y_tr, seed=inner_seed)


                    if self.feature_selection:
                        # Fitting preprocessing only on the outer train set.
                        preprocessor = build_preprocessor(feature_names_raw)
                        X_tr_pp = preprocessor.fit_transform(X_tr, y_tr)
                        X_te_pp = preprocessor.transform(X_te)

                        feature_names_pp = list(preprocessor.get_feature_names_out())
                        if clf_name not in self.feature_freq_:
                            self.feature_freq_[clf_name] = {
                                f: 0 for f in feature_names_pp
                            }

                        # Feature selection --> fitted only on the outer training set.
                        selector = self._make_selector(seed=inner_seed,n_features=X_tr_pp.shape[1])
                        selector.fit(X_tr_pp, y_tr)
                        X_tr_sel = selector.transform(X_tr_pp)
                        X_te_sel = selector.transform(X_te_pp)
                        selected_mask = selector.get_support()
                        selected_features = [fname for fname, keep in zip(feature_names_pp, selected_mask) if keep ]

                        for fname in selected_features:
                            self.feature_freq_[clf_name][fname] += 1

                        # Fitting the best classifier on the selected outer-train features.
                        final_clf = clone(best_clf)
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            final_clf.fit(X_tr_sel, y_tr)

                        # Doing prediction on the selected outer test features.
                        y_pred = final_clf.predict(X_te_sel)
                        y_score = self._get_scores(final_clf, X_te_sel)

                    ###################################################################
                    # Path of no feature selection (task 3 correspondance) 
                    else:
                        # Building the final full pipeline using the best hyperparameters, with the pipeline fitted only on the outer train set.
                        # Regarding outer test set --> Used only here, for final evaluation.
                        pipe = Pipeline([("preprocessor", build_preprocessor(feature_names_raw)),("clf", clone(best_clf))])

                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            pipe.fit(X_tr, y_tr)
                        y_pred = pipe.predict(X_te)
                        y_score = self._get_scores(pipe, X_te)

                   # Computing & storing outer metrics
                    metrics = compute_metrics(y_te, y_pred, y_score)

                    metrics["repeat"] = r
                    metrics["fold"] = fold_idx

                    self.results_[clf_name].append(metrics)

       
        if self.feature_selection:
            for clf_name in self.feature_freq_:
                for fname in self.feature_freq_[clf_name]:
                    self.feature_freq_[clf_name][fname] /= n_total_outer_folds

        return self
    
    ###################################################################
    #Function for creation of a summary table with median and bootstrapped confidence intervals for every metric & estimator.
    def summary(self, ci_level: float = 0.95) -> pd.DataFrame:
        
        from scipy.stats import bootstrap as scipy_bootstrap
        metric_keys = ["MCC", "AUC", "BA", "F1", "Recall","Specificity", "Precision", "PRAUC"]
        rng = np.random.default_rng(self.base_seed)
        rows = []

        for clf_name, fold_records in self.results_.items():
            df_folds = pd.DataFrame(fold_records)
            row = {"Estimator": clf_name}
            for mk in metric_keys:
                vals = df_folds[mk].values
                median_val = np.median(vals)
                #Bootstrap confidence interval for median
                bs = scipy_bootstrap((vals,), statistic=np.median, n_resamples=2000, confidence_level=ci_level,random_state=rng, method="percentile",)
                row[f"{mk}_median"] = round(median_val, 4)
                row[f"{mk}_CI_lo"]  = round(bs.confidence_interval.low,  4)
                row[f"{mk}_CI_hi"]  = round(bs.confidence_interval.high, 4)
            rows.append(row)

        return pd.DataFrame(rows).set_index("Estimator")


    ###################################################################
    #Function of dataframe creation for lated visualization and inspect of the results
    # Saving the per-fold metrics for a specific estimator.
    def get_fold_results(self, clf_name: str) -> pd.DataFrame:
        if clf_name not in self.results_:
            raise KeyError(f"No results found for estimator: {clf_name}")

        return pd.DataFrame(self.results_[clf_name])
    
    #For task4 returning the feature selection frequencies for each estimator
    def feature_selection_report(self) -> pd.DataFrame:
        if not self.feature_freq_:
            raise RuntimeError("Run with feature_selection=True first.")
        return pd.DataFrame(self.feature_freq_).T

    