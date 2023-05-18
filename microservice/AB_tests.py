from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from scipy import stats
import json
import numpy as np

from load_data import Preprocessor
from files_utils import BASE_PREDICTIONS_FILE, KNN_PREDICTIONS_FILE


def get_collected_predictions():
    base_predictions = json.load(BASE_PREDICTIONS_FILE)
    knn_predictions = json.load(KNN_PREDICTIONS_FILE)
    return (base_predictions, knn_predictions)


def split_set_into_groups(X, y, test_size=0.5, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def get_z_score(X_group_A, X_group_B, p_A, p_B):
    # proportions
    n_A = len(X_group_A)
    n_B = len(X_group_B)

    # standard error
    se_A = np.sqrt(p_A * (1 - p_A) / n_A)
    se_B = np.sqrt(p_B * (1 - p_B) / n_B)

    # z-score
    return (p_B - p_A) / np.sqrt(se_A**2 + se_B**2)


def AB_test(alpha=0.05):
    # hypothesis H0: Base Model is better than KNN - t < T_ALPHA = 1.96 (slides - please check if you see it the same as me)
    # hypothesis H1: alternative
    base_predictions, knn_predictions = get_collected_predictions()
    X, y = Preprocessor.process(...)  # TODO fix when Filip fixes preprocessor
    X_group_A, X_group_B, y_group_A, y_group_B = split_set_into_groups(X, y)

    # base model:
    y_pred_A = [elem["prediction"] for elem in base_predictions]
    f1_score_A = f1_score(y_group_A, y_pred_A)

    # knn model:
    y_pred_B = [elem["prediction"] for elem in knn_predictions]
    f1_score_B = f1_score(y_group_B, y_pred_B)

    z_score = get_z_score(X_group_A, X_group_B, p_A=f1_score_A, p_B=f1_score_B)
    # p-value
    p_value = 1 - stats.norm.cdf(z_score)

    if p_value < alpha:
        return "Reject H0: Model B (KNN) performs significantly better than A (base)"
    else:
        return "Fail to reject H0: No significant difference in performance between A and B"
