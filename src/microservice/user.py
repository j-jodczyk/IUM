from pydantic import BaseModel
from typing import List
from load_data import Preprocessor
import pandas as pd

EVENT_TYPES = ["ADVERTISEMENT", "LIKE", "PLAY", "SKIP"]
CITIES = ["Gdynia", "Kraków", "Poznań", "Radom", "Szczecin", "Warszawa", "Wrocław"]
GENRES = [
    "adult standards",
    "album rock",
    "alternative metal",
    "alternative rock",
    "argentine rock",
    "art rock",
    "blues rock",
    "brill building pop",
    "c-pop",
    "classic rock",
    "country rock",
    "dance pop",
    "europop",
    "folk",
    "folk rock",
    "funk",
    "hard rock",
    "hoerspiel",
    "italian adult pop",
    "j-pop",
    "latin",
    "latin alternative",
    "latin pop",
    "latin rock",
    "lounge",
    "mandopop",
    "mellow gold",
    "metal",
    "modern rock",
    "motown",
    "mpb",
    "new romantic",
    "new wave",
    "new wave pop",
    "permanent wave",
    "pop",
    "pop rock",
    "post-teen pop",
    "psychedelic rock",
    "quiet storm",
    "ranchera",
    "regional mexican",
    "rock",
    "rock en espanol",
    "roots rock",
    "singer-songwriter",
    "soft rock",
    "soul",
    "tropical",
    "vocal jazz",
]


class User(BaseModel):
    user_id: int
    name: str
    city: str
    street: str
    favourite_genres: List[str]
    premium_user: bool

    def get_city_vector(self):
        return [1 if self.city == city else 0 for city in CITIES]

    def get_ads_ratio(self):
        pass

    def get_events_vector(self):
        events_vector = []
        return events_vector

    def get_adds_after_fav_ratio(self):
        # TODO
        return 0

    # TODO: make sure order is the same as the MLB we used in preprocessing
    def get_genres_vector(self):
        return [1 if genre in self.favourite_genres else 0 for genre in GENRES]

    def to_vector(self):
        user_vector = self.get_city_vector()
        user_vector += [self.get_ads_ratio()]
        user_vector += self.get_events_vector()
        user_vector += [self.get_adds_after_fav_ratio()]
        user_vector += self.get_genres_vector()
        return user_vector
