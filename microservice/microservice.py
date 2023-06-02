import pickle
import fastapi
import uvicorn
import random
import json

from fastapi.responses import JSONResponse
import pandas as pd
from AB_tests import AB_test
from models import NaiveModel
from load_data import Preprocessor, DataModel

from user import User
from session import Sessions
from actual import Actual
from files_utils import (
    add_prediction,
    add_actual,
    BASE_NAME,
    KNN_NAME,
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


def update_df():
    base_model = DataModel()
    df = base_model.get_merged_dfs()
    df = Preprocessor.run(df, drop_user_id=False)
    return df


df = update_df()


@app.post("/predict-with/{model_name}")
def predict_with(model_name: str, user: User, test: bool = False) -> JSONResponse:
    if model_name not in models.keys():
        raise fastapi.HTTPException(status_code=404, detail="Unknown model")
    if user is None:
        raise fastapi.HTTPException(status_code=400, detail="Empty request body")
    model = models[model_name]
    user_vec = user.to_vector(df).drop("user_id", axis=1)
    prediction = int(model.predict(user_vec))
    if test:
        add_prediction(user.user_id, prediction, model_name)

    return {"will_buy_premium": prediction}


# predicts with randomly selected model and saves to file
@app.post("/predict")
def predict(user: User) -> JSONResponse:
    to_A = random.randint(0, 1)
    prediction = None
    user_vec = user.to_vector(df).drop("user_id", axis=1)
    if to_A:
        prediction = models["base"].predict(user_vec)[0]
        add_prediction(user.user_id, prediction, BASE_NAME)
    else:
        prediction = models["KNN"].predict(user_vec)[0]
        add_prediction(user.user_id, prediction, KNN_NAME)
    return {"will_buy_premium": prediction}


@app.post("/add-sessions")
def add_sessions(sessions: Sessions) -> JSONResponse:
    global df
    with open("./microservice/data/sessions.json", "r+") as f:
        data = json.load(f)
    for session in sessions.sessions:
        data.append(session.dict())
    with open("./microservice/data/sessions.json", "w") as f:
        json.dump(data, f, default=str)
    df = update_df()


# lets user update the prediction file if user has knowlege about prediction
@app.post("/submit-actual")
def submit_actual(actual: Actual) -> JSONResponse:
    add_actual(actual.user_id, actual.actual)
    return {"response": "OK"}


@app.get("/test_ab_results")
def test_ab_results() -> JSONResponse:
    return {"AB_test_verdict": AB_test()}


uvicorn.run(app)
