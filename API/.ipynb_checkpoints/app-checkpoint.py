import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle
import numpy as np
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')

from services.LSTM_service import LSTMService

MAX_LEN = 100
MAX_WORDS = 10000
PATH_MODEL = "./Modeles/LSTM/modele_lstm" 
PATH_TOKENIZER = "./Modeles/LSTM/tokenizer.json"

model = LSTMService(PATH_MODEL, PATH_TOKENIZER)

# --- 3. DÉFINITION DE L'APPLICATION ---
app = FastAPI(
    title="Air Paradis Sentiment API", 
    description="API de prédiction de sentiment (LSTM)",
    version="1.0.0"
)

# On définit le format attendu des données (Un JSON avec un champ "text")
class TweetInput(BaseModel):
    text: str

# Fonction de nettoyage (Doit être la même que celle de ton entraînement !)
def nettoyer_texte(texte):
    texte = texte.lower()
    texte = re.sub(r'<.*?>', '', texte)
    texte = re.sub(r'https?://\S+|www\.\S+', '', texte)
    texte = re.sub(r'[^a-zA-Z\s]', '', texte) # On garde lettres et espaces
    return texte

# --- 4. LES ENDPOINTS (Les routes) ---

@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API nommée Air Paradis. Utilisez /predict pour analyser un tweet."}

@app.post("/predict")
def predict_sentiment(input_data: TweetInput):
    try:
        # A. Récupération et nettoyage
        raw_text = input_data.text

        # D. Prédiction
        # Le modèle renvoie un tableau (ex: [[0.85]]), on prend la valeur float
        prediction = model.predict(raw_text)
        
        # E. Interprétation
        sentiment_label = "Positif" if prediction > 0.5 else "Négatif"
        
        # F. Réponse JSON
        return {
            "tweet_original": raw_text,
            "sentiment": sentiment_label,
            "score": float(prediction) # On convertit en float python standard
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
