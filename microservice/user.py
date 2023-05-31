from pydantic import BaseModel
from typing import List
from load_data import Preprocessor, DataModel
import pandas as pd

class User(BaseModel):
    user_id: int
    name: str
    city: str
    street: str
    favourite_genres: List[str]
    premium_user: bool

    def to_vector(self):
        base_model = DataModel()
        df = base_model.get_merged_dfs()
        df = Preprocessor.run(df)
        return list(df.iloc[0, :])