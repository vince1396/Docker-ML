from pydantic import BaseModel


class Booking(BaseModel):
    tweet: str
