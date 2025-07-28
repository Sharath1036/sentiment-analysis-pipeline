from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from typing import List
import pandas as pd
from analyzer import CollegeReviewAnalyzer
import io
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException

app = FastAPI()

# Load analyzer and model once at startup (use a small dummy CSV for init, will not use its data)
analyzer = CollegeReviewAnalyzer(df_path='datasets/collegereview2021.csv')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ReviewRequest(BaseModel):
    review: str

class BatchReviewRequest(BaseModel):
    reviews: List[str]

class SentimentResponse(BaseModel):
    sentiment: str

class BatchSentimentResponse(BaseModel):
    sentiments: List[str]

@app.post("/analyze-sentiment", response_model=SentimentResponse)
def analyze_sentiment(request: ReviewRequest):
    sentiment = analyzer.get_sentiment(analyzer.clean_text(request.review))
    return {"sentiment": sentiment}

@app.post("/analyze-sentiment-batch", response_model=BatchSentimentResponse)
def analyze_sentiment_batch(request: BatchReviewRequest):
    sentiments = [analyzer.get_sentiment(analyzer.clean_text(r)) for r in request.reviews]
    return {"sentiments": sentiments}

@app.post("/analyze-sentiment-csv")
def analyze_sentiment_csv(
    file: UploadFile = File(...),
    text_column: str = Form(...)
):
    try:
        contents = file.file.read()
        df = pd.read_csv(io.BytesIO(contents))
        if text_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"CSV must have a '{text_column}' column.")
        temp_analyzer = CollegeReviewAnalyzer(df_path='datasets/collegereview2021.csv')  # Use a dummy file
        temp_analyzer.df = df
        temp_analyzer.add_sentiment_columns(
            review_column=text_column,
            cleaned_column='cleaned_text',
            sentiment_column='sentiment'
        )
        return temp_analyzer.df[[text_column, 'sentiment']].to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
