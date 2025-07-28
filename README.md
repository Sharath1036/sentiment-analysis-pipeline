# Sentiment Analysis Pip

## Setup 
Operating System: `Windows 11`

Clone the repository 
```
git clone https://github.com/Sharath1036/sentiment-analysis-pipeline.git
```

Install virtual environment and activate it
```
python -m venv myenv
myenv\Scripts\activate
```

Installing dependencies
```
pip install -r requirements.txt
```

## Testing the code locally with the given dataset: `datasets/collegereview2021.csv`
```
python analyzer.py
```

## Testing the FASTAPI Endpoint
```
uvicorn main:app --reload
```

Access the application at `localhost:8000/docs`.

### 1. **POST `/analyze-sentiment`**

**Purpose:**  
Analyze sentiment for a single review.

**Request Body (JSON):**
```json
{
  "review": "Your review text here"
}
```

**Example:**
```json
{
  "review": "The college has great faculty and excellent facilities."
}
```

**Response:**
```json
{
  "sentiment": "positive"
}
```

---

### 2. **POST `/analyze-sentiment-batch`**

**Purpose:**  
Analyze sentiment for a list of reviews.

**Request Body (JSON):**
```json
{
  "reviews": [
    "The college has great faculty and excellent facilities.",
    "The food in the canteen is terrible.",
    "The campus is okay, but the administration is slow."
  ]
}
```

**Response:**
```json
{
  "sentiments": ["positive", "negative", "neutral"]
}
```

---

### 3. **POST `/analyze-sentiment-csv`**

**Purpose:**  
Upload a CSV file with a `review` column and get back the data with sentiment analysis.

**Request:**  .
- Upload a file (CSV) with at least a one column of sentences.
- Name of the column containing the sentences

**Example CSV content:**

You can use the dataset `datasets/review.csv` of this project which contains thr `review` column.
```
review
The college has great faculty and excellent facilities.
The food in the canteen is terrible.
The campus is okay, but the administration is slow.
```

text_string: `review`

**How to send:**  
- Use a tool like Postman, curl, or a frontend with a file upload field.
- The form field name must be `file`.

**Response:**  
A JSON array of records, each with the original review and its sentiment:
```json
[
  {
    "review": "The college has great faculty and excellent facilities.",
    "cleaned_reviews": "The college has great faculty and excellent facilities ",
    "sentiment": "positive"
  },
  {
    "review": "The food in the canteen is terrible.",
    "cleaned_reviews": "The food in the canteen is terrible ",
    "sentiment": "negative"
  },
  ...
]
```

---

**Summary Table:**

| Endpoint                    | Method | Input Type   | Required Field(s)         | Example Value/Format         |
|-----------------------------|--------|--------------|---------------------------|------------------------------|
| /analyze-sentiment          | POST   | JSON         | review                    | {"review": "text"}           |
| /analyze-sentiment-batch    | POST   | JSON         | reviews (list of strings) | {"reviews": ["text1", ...]}  |
| /analyze-sentiment-csv      | POST   | form-data    | file (CSV with 'review')  | Upload file                  |