from pydantic import BaseModel
from typing import List
import pandas as pd


class User(BaseModel):
    user_id: int
    name: str
    city: str
    street: str
    favourite_genres: List[str]
    premium_user: bool

    def to_vector(self, df):
        user_df = df[df["user_id"] == self.user_id]
        return list(user_df.iloc[0, :])
