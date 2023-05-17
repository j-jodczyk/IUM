import pickle
import fastapi
import uvicorn

from fastapi.responses import JSONResponse

from user import User

filename = "finalized_kneigh.sav"
model = pickle.load(open(filename, "rb"))

app = fastapi.FastAPI()


@app.get("/will_buy_premium/")
def get_will_buy_premium(user: User) -> JSONResponse:
    if user is None:
        raise fastapi.HTTPException(status_code=400, detail="Empty request")

    return {"will_buy_premium": int(model.predict([user.to_vector()])[0])}


uvicorn.run(app)
