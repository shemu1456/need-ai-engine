from fastapi import FastAPI
from pydantic import BaseModel
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Initialize FastAPI app
app = FastAPI()

# Download the AI model data
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

class ReviewRequest(BaseModel):
    text: str

@app.get("/")
def home():
    return {"status": "NEED AI Engine is online"}

@app.post("/analyze")
def analyze_sentiment(request: ReviewRequest):
    # The AI performs the analysis here
    scores = sia.polarity_scores(request.text)
    compound = scores['compound']
    
    # Classify based on the AI's score
    if compound >= 0.05:
        sentiment = "Positive"
    elif compound <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
        
    return {
        "sentiment": sentiment,
        "score": compound
    }
