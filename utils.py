import time
from typing import Optional, Union, List, Callable, Dict, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold

import optuna
from phik import phik_matrix


def evaluate_and_record(
    name: str,
    model: ClassifierMixin,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    start_time: float,
    threshold: float = 0.5,
    proba: Optional[np.ndarray] = None,
    extra_info: Any = "N/A",
    results_list: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """
    Evaluates a classification model on validation data, logs metrics, and appends results to a list.

    Args:
        name (str): Name of the model for reporting.
        model (ClassifierMixin): Trained classifier with predict_proba().
        X_val (pd.DataFrame): Validation feature set.
        y_val (pd.Series): Validation labels.
        start_time (float): Time when training started (for runtime calculation).
        threshold (float, optional): Classification threshold for converting probabilities to labels. Default is 0.5.
        proba (np.ndarray, optional): Predicted probabilities for the positive class. If None, computed internally.
        extra_info (Any, optional): Additional info (e.g. hyperparameters) to store with results.
        results_list (List[Dict], optional): List to append the evaluation summary to.
    """
    if proba is None:
        proba = model.predict_proba(X_val)[:, 1]

    preds = (proba >= threshold).astype(int)

    print(f"\nEvaluation — {name}")
    print(classification_report(y_val, preds))

    if results_list is not None:
        results_list.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_val, preds),
                "ROC AUC": roc_auc_score(y_val, proba),
                "PR AUC": average_precision_score(y_val, proba),
                "F1": f1_score(y_val, preds),
                "Precision": precision_score(y_val, preds),
                "Recall": recall_score(y_val, preds),
                "Best Params": extra_info,
                "Runtime (s)": round(time.time() - start_time, 2),
            }
        )

    # Confusion matrix
    cm = confusion_matrix(y_val, preds)
    plt.figure(figsize=(4, 3))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["False", "True"],
        yticklabels=["False", "True"],
    )
    plt.title(f"Confusion Matrix — {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_val, proba)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_val, proba):.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title(f"ROC Curve — {name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_transported_distribution(df: pd.DataFrame) -> None:
    """
    Plots the distribution of the 'Transported' column with percentage annotations.
    """
    colors = ["skyblue", "crimson"]
    counts = df["Transported"].value_counts()
    percentages = counts / counts.sum() * 100

    plt.figure(figsize=(6, 4))
    ax = sns.countplot(
        data=df, x="Transported", hue="Transported", palette=colors, legend=False
    )

    for i, count in enumerate(counts):
        percent = percentages.iloc[i]
        ax.text(
            i,
            count + 10,
            f"{percent:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    plt.title("Transported Distribution with Percentages")
    plt.xlabel("Transported")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def plot_feature_vs_transported(
    df: pd.DataFrame,
    feature: str,
    bins: Optional[int] = None,
    max_unique: int = 15,
    bar_width: float = 1.0,
) -> None:
    """
    Plots a feature against 'Transported', supporting both categorical and binned numeric features.
    """
    if feature not in df.columns:
        print(f"Feature '{feature}' not found.")
        return

    if pd.api.types.is_numeric_dtype(df[feature]) and bins is not None:
        binned = pd.cut(df[feature], bins=bins)
        stats = (
            df.groupby(binned, observed=True)
            .agg(
                passenger_count=("Transported", "count"),
                transported_ratio=("Transported", "mean"),
            )
            .reset_index()
        )
        x = stats[binned.name].apply(lambda interval: interval.mid)
        xlabel = f"{feature} (binned)"
    else:
        if df[feature].nunique() > max_unique:
            return
        stats = (
            df.groupby(feature, observed=True)
            .agg(
                passenger_count=("Transported", "count"),
                transported_ratio=("Transported", "mean"),
            )
            .reset_index()
        )
        x = stats[feature]
        xlabel = feature

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.bar(
        x,
        stats["passenger_count"],
        width=bar_width,
        color="skyblue",
        label="Passenger Count",
    )
    ax1.set_ylabel("Passenger Count", color="skyblue")
    ax1.set_xlabel(xlabel)
    ax1.tick_params(axis="y", labelcolor="skyblue")

    ax2 = ax1.twinx()
    ax2.plot(
        x,
        stats["transported_ratio"],
        color="crimson",
        marker="o",
        label="Transported Ratio",
    )
    ax2.set_ylabel("Transported Ratio", color="crimson")
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis="y", labelcolor="crimson")

    plt.title(f"{xlabel} vs Transported")
    fig.tight_layout()
    plt.show()


def plot_spending_vs_transport(
    df: pd.DataFrame,
    cryo_filter: Optional[bool] = None,
    title_suffix: str = "All Passengers",
) -> None:
    """
    Plots spending behavior vs. transport status, optionally filtered by CryoSleep status.
    """
    amenity_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    df_plot = df.copy()
    df_plot["SpentAnything"] = df_plot[amenity_cols].fillna(0).sum(axis=1) > 0

    if cryo_filter is not None:
        df_plot = df_plot[df_plot["CryoSleep"] == cryo_filter]

    grouped = (
        df_plot.groupby(["SpentAnything", "Transported"])
        .size()
        .reset_index(name="Count")
    )

    plt.figure(figsize=(8, 5))
    sns.barplot(
        x="SpentAnything",
        y="Count",
        hue="Transported",
        data=grouped,
        palette=["skyblue", "crimson"],
    )
    plt.title(f"{title_suffix}: Spent on Amenities vs. Transported")
    plt.xlabel("Spent on Amenities")
    plt.ylabel("Passenger Count")
    plt.xticks([0, 1], ["No", "Yes"])
    plt.tight_layout()
    plt.show()


def plot_phik_correlation(df: pd.DataFrame, interval_cols: List[str]) -> None:
    """
    Plots a Phi-k correlation heatmap of features sorted by their relationship to 'Transported'.
    """
    phik_corr = phik_matrix(df, interval_cols=interval_cols)
    sorted_cols = phik_corr["Transported"].sort_values(ascending=False).index
    phik_sorted = phik_corr.loc[sorted_cols, sorted_cols]
    mask = np.triu(np.ones_like(phik_sorted, dtype=bool))

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        phik_sorted,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        square=True,
        linewidths=0.5,
    )
    plt.title(
        "Phi-k Correlation Heatmap (Lower Triangle)\nSorted by Correlation with 'Transported'"
    )
    plt.tight_layout()
    plt.show()


def plot_class_balance(
    y_full: Union[pd.Series, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    y_val: Union[pd.Series, np.ndarray],
) -> None:
    """
    Plots the class balance of 'Transported' across full, training, and validation datasets.
    """
    colors = ["skyblue", "crimson"]
    datasets = {
        "Full Dataset": y_full,
        "Training Set (80%)": y_train,
        "Validation Set (20%)": y_val,
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, (label, target_data) in zip(axes, datasets.items()):
        sns.countplot(x=target_data, palette=colors, ax=ax)
        ax.set_title(label)
        ax.set_xlabel("Transported")
        ax.set_ylabel("Count")

        total = len(target_data)
        for p in ax.patches:
            count = p.get_height()
            pct = f"{100 * count / total:.1f}%"
            ax.annotate(
                pct, (p.get_x() + p.get_width() / 2, count), ha="center", va="bottom"
            )

    plt.suptitle("Class Balance in Full, Train, and Validation Sets")
    plt.tight_layout()
    plt.show()


def plot_permutation_importance(
    model, X_val: pd.DataFrame, y_val: Union[pd.Series, np.ndarray], top_n: int = 25
) -> None:
    """
    Plots the top N permutation importances for a fitted model using validation data.
    """
    perm_result = permutation_importance(
        model,
        X_val,
        y_val,
        scoring="accuracy",
        n_repeats=20,
        random_state=42,
        n_jobs=-1,
    )
    sorted_idx = perm_result.importances_mean.argsort()[::-1]
    top_features = X_val.columns[sorted_idx][:top_n]
    importances = perm_result.importances_mean[sorted_idx][:top_n]

    plt.figure(figsize=(10, 6))
    plt.barh(top_features, importances)
    plt.gca().invert_yaxis()
    plt.title(f"Permutation Feature Importance")
    plt.xlabel("Mean Decrease in Accuracy")
    plt.tight_layout()
    plt.show()


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for feature engineering on the passenger dataset.
    """

    def __init__(self):
        self.amenity_cols = [
            "RoomService",
            "FoodCourt",
            "ShoppingMall",
            "Spa",
            "VRDeck",
        ]

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if "PassengerId" in X.columns:
            X[["GroupID", "GroupMember"]] = X["PassengerId"].str.split("_", expand=True)
        if "Cabin" in X.columns:
            X[["Deck", "CabinNum", "Side"]] = X["Cabin"].str.split("/", expand=True)
        if "Name" in X.columns:
            X[["FirstName", "LastName"]] = (
                X["Name"].fillna(" ").str.split(" ", n=1, expand=True)
            )
        if "GroupID" in X.columns:
            X["GroupID"] = X["GroupID"].astype(str)
            group_sizes = X["GroupID"].value_counts()
            X["GroupSize"] = X["GroupID"].map(group_sizes)
        X["SpentAnything"] = X[self.amenity_cols].fillna(0).sum(axis=1) > 0
        return X.drop(columns=["PassengerId", "Name", "Cabin"], errors="ignore")


def tune_and_evaluate_with_optuna(
    name: str,
    cfg: Dict[str, Callable[[optuna.Trial], Dict[str, Any]]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    cv: StratifiedKFold,
    n_trials: int = 10,
    results_list: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[str, ClassifierMixin]:
    """
    Tune hyperparameters using Optuna, fit the best model, evaluate it, and return the result.

    Args:
        name (str): Name of the model.
        cfg (Dict): Dictionary with two keys:
            - "space": A function that defines the hyperparameter search space (takes a trial).
            - "build": A function that builds the model given a parameter dict.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        X_val (pd.DataFrame): Validation features.
        y_val (pd.Series): Validation labels.
        cv (StratifiedKFold): Cross-validation splitter.
        n_trials (int, optional): Number of Optuna trials to run. Default is 10.
        results_list (List[Dict], optional): List to append evaluation metrics to.

    Returns:
        Tuple[str, ClassifierMixin]: Name of the tuned model and the trained model instance.
    """
    start = time.time()

    def objective(trial: optuna.Trial) -> float:
        params = cfg["space"](trial)
        model = cfg["build"](params)
        scores = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
            preds = model.predict(X_train.iloc[val_idx])
            scores.append(accuracy_score(y_train.iloc[val_idx], preds))
        return np.mean(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    final_model = cfg["build"](best_params)
    final_model.fit(X_train, y_train)

    model_name = f"{name} (Optuna)"
    val_proba = final_model.predict_proba(X_val)[:, 1]

    evaluate_and_record(
        name=model_name,
        model=final_model,
        X_val=X_val,
        y_val=y_val,
        start_time=start,
        threshold=0.5,
        proba=val_proba,
        extra_info=best_params,
        results_list=results_list,
    )

    return model_name, final_model
