from pydantic import BaseModel
from typing import List
from session import Session
from load_data import Preprocessor


class User(BaseModel):
    CITIES = ["Gdynia", "Kraków", "Poznań", "Radom", "Szczecin", "Warszawa", "Wrocław"]
    GENRES = [
        "adds_after_fav_ratio",
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

    user_id: int
    name: str
    city: str
    sessions: List[Session]
    favourite_genres: List[str]

    def __init__(self, preprocessor: Preprocessor):
        super.__init__()
        self.preprocessor = preprocessor

    def get_city_vector(self):
        cities = User.CITIES
        return [1 if self.city in cities else 0 for city in cities]

    def get_ads_ratio(self):
        return self.preprocessor.calculate_ads_time(
            [s.__dict__() for s in self.sessions]
        )

    def get_events_vector(self):
        events_vector = [
            len([s for s in self.sessions if s["event_type"] == event])
            for event in Session.EVENT_TYPES
        ]
        return events_vector

    def get_adds_after_fav_ratio(self):
        # TODO
        return 0

    def get_genres_vector(self):
        genres = User.GENRES
        return [1 if genre in self.favourite_genres else 0 for genre in genres]

    def to_vector(self):
        user_vector = [self.user_id, self.name]
        city_vector = self.get_city_vector()
        user_vector += city_vector
        user_vector += [self.get_ads_ratio()]
        user_vector += self.get_events_vector()
        user_vector += [self.get_adds_after_fav_ratio()]
        user_vector += self.get_genres_vector()
        return user_vector
