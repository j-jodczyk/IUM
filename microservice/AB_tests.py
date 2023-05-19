from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from scipy import stats
import json
import numpy as np

from load_data import Preprocessor
from files_utils import PREDICTIONS_FILE, BASE_NAME, KNN_NAME


def get_collected_predictions():
    with open(PREDICTIONS_FILE, "r") as f:
        predictions = json.load(open(PREDICTIONS_FILE))
    return predictions


def sort_predictions(predictions):
    y_actual_A = []
    y_pred_A = []
    y_actual_B = []
    y_pred_B = []
    for prediction in predictions:
        if prediction["model"] == BASE_NAME:
            y_actual_A.append(prediction["actual"])
            y_pred_A.append(prediction["prediction"])
        elif prediction["model"] == KNN_NAME:
            y_actual_B.append(prediction["actual"])
            y_pred_B.append(prediction["prediction"])
    return (y_actual_A, y_pred_A, y_actual_B, y_pred_B)


def split_set_into_groups(X, y, test_size=0.5, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def AB_test(alpha=0.05):
    # hypothesis H0: Base Model is better than KNN - t < T_ALPHA = 1.96 (slides - please check if you see it the same as me)
    # hypothesis H1: alternative
    predictions = get_collected_predictions()
    y_actual_A, y_pred_A, y_actual_B, y_pred_B = sort_predictions(predictions)

    # base model:
    y_pred_A = [
        elem["prediction"] for elem in predictions if elem["model"] == BASE_NAME
    ]
    f1_score_A = f1_score(y_actual_A, y_pred_A)

    # knn model:
    y_pred_B = [elem["prediction"] for elem in predictions if elem["model"] == KNN_NAME]
    f1_score_B = f1_score(y_actual_B, y_pred_B)

    t_stat, p_value = stats.ttest_ind(f1_score_B, f1_score_A)

    if p_value < alpha:
        return "Reject H0: Model B (KNN) performs significantly better than A (base)"
    else:
        return "Fail to reject H0: No significant difference in performance between A and B"
