from pydantic import BaseModel
import datetime


class Session(BaseModel):
    EVENT_TYPES = ["ADVERTISEMENT", "LIKE", "PLAY", "SKIP"]

    timestamp: datetime.datetime
    user_id: int
    track_id: str
    event_type: str
    session_id: int
