import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import re

# --- APPLICATION ---
app = FastAPI(
    title="Air Paradis Sentiment API", 
    description="API de prédiction de sentiment (LSTM)",
    version="1.0.0"
)


# Variable globale pour le modèle (chargé au premier appel)
model = None

def get_model():
    global model
    if model is None:
        import nltk
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('punkt')
        nltk.download('punkt_tab')
        from services.LSTM_service import LSTMService
        PATH_MODEL = "./Modeles/LSTM/modele_lstm" 
        PATH_TOKENIZER = "./Modeles/LSTM/tokenizer.json"
        model = LSTMService(PATH_MODEL, PATH_TOKENIZER)
    return model

class TweetInput(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API Air Paradis. Utilisez /predict pour analyser un tweet."}

@app.post("/predict")
def predict_sentiment(input_data: TweetInput):
    try:
        raw_text = input_data.text
        prediction = get_model().predict(raw_text)
        sentiment_label = "Positif" if prediction > 0.5 else "Négatif"
        return {
            "tweet_original": raw_text,
            "sentiment": sentiment_label,
            "score": float(prediction)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
