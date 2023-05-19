from pydantic import BaseModel


class Actual(BaseModel):
    user_id: int
    actual: int
