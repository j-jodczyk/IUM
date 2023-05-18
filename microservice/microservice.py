import pickle
import fastapi
import uvicorn
import random
import json

from fastapi.responses import JSONResponse
from AB_tests import AB_test
from models.models import NaiveModel

from user import User
from files_utils import (
    update_prediction_file,
    KNN_PREDICTIONS_FILE,
    BASE_PREDICTIONS_FILE,
    KNN_FILEPATH,
    BASE_FILEPATH,
)


knn = pickle.load(open(KNN_FILEPATH, "rb"))
base = pickle.load(open(BASE_FILEPATH, "rb"))

models = {
    "KNN": knn,
    "base": base,
}

app = fastapi.FastAPI()


@app.post("/predict/{model_name}")
def get_will_buy_premium(model_name: str, user: User) -> JSONResponse:
    if model_name not in models.keys():
        raise fastapi.HTTPException(status_code=404, detail="Unknown model")
    if user is None:
        raise fastapi.HTTPException(status_code=400, detail="Empty request body")
    model = models[model_name]

    return {"will_buy_premium": int(model.predict([user.to_vector()])[0])}


@app.post("/add_to_test_ab")
def add_to_test_ab(user: User) -> JSONResponse:
    to_A = random.randint(0, 1)
    prediction = None
    if to_A:
        prediction = models["base"].predict([user.to_vector()])[0]
        update_prediction_file(user.user_id, prediction, BASE_PREDICTIONS_FILE)
    else:
        prediction = models["KNN"].predict([user.to_vector()])[0]
        update_prediction_file(user.user_id, prediction, KNN_PREDICTIONS_FILE)
    return {"will_buy_premium": prediction}


@app.get("/test_ab_results")
def test_ab_results() -> JSONResponse:
    return {"AB_test_verdict": AB_test}


uvicorn.run(app)
