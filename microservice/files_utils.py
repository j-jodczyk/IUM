import json

KNN_FILEPATH = "./models/KNN_model.sav"
BASE_FILEPATH = "./models/base_model.sav"

BASE_PREDICTIONS_FILE = "./test/base.json"
KNN_PREDICTIONS_FILE = "./test/knn.json"


def update_prediction_file(user_id: int, prediction: int, file: str):
    file_data = json.load(file)
    file_data.update({"user": user_id, "prediction": prediction})
    json.dump(file_data, BASE_PREDICTIONS_FILE)
