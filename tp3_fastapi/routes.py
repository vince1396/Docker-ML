import joblib
from Booking import Booking
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

model = joblib.load("../trained_classifier.joblib")
app = FastAPI()

origins = [
    "http://localhost:80",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Hello World !"}


@app.post("/predict/")
async def predict(booking: Booking):
    post = booking.dict()
    tweet = post.get("tweet")
    print(type(proba(tweet)))
    text = proba(tweet)
    return {"text": text}


def proba(tweet):
    for each in model.predict_proba([tweet]):
        tweet = str(''.join([tweet, each]))
    return tweet
