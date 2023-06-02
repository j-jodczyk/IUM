import json
from sklearn.model_selection import train_test_split

KNN_FILEPATH = "./microservice/saved_models/KNN_model.sav"
BASE_FILEPATH = "./microservice/saved_models/base_model.sav"

PREDICTIONS_FILE = "./microservice/saved_models/predictions.json"

BASE_NAME = "base"
KNN_NAME = "knn"


def update_prediction_file(user_id: int, prediction: int, model_name: str, actual: int):
    with open(PREDICTIONS_FILE, "r") as f:
        file_data = json.load(f)
    file_data.append(
        {
            "user": user_id,
            "prediction": prediction,
            "model": model_name,
            "actual": actual,
        }
    )
    with open(PREDICTIONS_FILE, "w") as f:
        json.dump(file_data, f)


def add_prediction(user_id: int, prediction: int, model_name: str):
    update_prediction_file(user_id, prediction, model_name, 0)


def add_actual(user_id: int, actual: int):
    with open(PREDICTIONS_FILE, "r") as f:
        file_data = json.load(f)
    for user in file_data:
        if user["user"] == user_id:
            user["actual"] = actual
            break
    with open(PREDICTIONS_FILE, "w") as f:
        json.dump(file_data, f)


def randomly_split_group(df, proportion=0.5):
    df1, df2 = train_test_split(df, test_size=1 - proportion)
    df1.reset_index(drop=True, inplace=True)
    df2.reset_index(drop=True, inplace=True)
    return df1, df2
