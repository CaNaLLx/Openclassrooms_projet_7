import os

#NETTOYAGE
import string
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json
import pickle
import re



class LSTMService:
    def __init__(self, model_path, tokenizer_path):
        print(f"🔄 Chargement du modèle exporté depuis : {model_path}")
        
        # 1. Chargement du Tokenizer
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError("Tokenizer introuvable")
            
        """with open(tokenizer_path, 'rb') as handle:
            self.tokenizer = pickle.load(handle)"""
        with open(tokenizer_path, 'r') as f:
            tokenizer_json = json.load(f)
        self.tokenizer = tokenizer_from_json(tokenizer_json)

        # 2. Chargement du Modèle (Spécifique à model.export)
        # On charge l'artefact bas niveau
        self.imported_model = tf.saved_model.load(model_path)
        
        # On récupère la fonction d'inférence par défaut ("serving_default")
        self.inference_func = self.imported_model.signatures["serving_default"]
        
        # ⚠️ DOIT être la même valeur que lors de l'entraînement
        self.max_len = 100 

    def nettoyer_texte(self,texte):
        if not isinstance(texte, str):
            return ""


        stop_words = set(stopwords.words('english'))

        for neg in ['not', 'no', 'never', "n't", 'nor']:
            if neg in stop_words:
                stop_words.remove(neg)
        lemmatizer = WordNetLemmatizer()
    
        #Passer en minuscule tout le texte
        texte = texte.lower()
    
        #Supprimer des éléments frequents dans des tweets, mais jugés ininteressants pour l'analyse
        texte = re.sub(r'<.*?>', '', texte)
        texte = re.sub(r'@\w+', '', texte)
        texte = re.sub(r'\d+', '', texte)
        texte = re.sub(r'https?://\S+|www\.\S+', '', texte)
    
        #Supprimer les ponctuactions
        texte = texte.translate(str.maketrans("","",string.punctuation))
    
        texte = re.sub(r'\s+', ' ', texte)
    
    
        #TOKENISATION
        tokens = texte.split()
    
        #Parcours des tokens:
        clean_tokens = [
            lemmatizer.lemmatize(token)
            for token in tokens
            if token not in stop_words and len(token) > 2
        ]
    
        #Renvoie des elements à joindre
        return " ".join(clean_tokens)

    def predict(self, text):
        clean_text = self.nettoyer_texte(text)
        
        # 3. Pré-traitement (Identique à l'entraînement)
        seq = self.tokenizer.texts_to_sequences([clean_text])
        padded = pad_sequences(seq, maxlen=self.max_len, padding='post', truncating='post')
        
        # 4. Conversion en Tenseur (CRUCIAL pour un modèle exporté)
        # Les modèles exportés attendent des tf.Tensor, pas des numpy arrays
        # On force le type float32 ou int32 selon la première couche de ton modèle
        input_tensor = tf.constant(padded, dtype=tf.float32) 
        
        # 5. Inférence via la signature
        # Note : Les modèles exportés attendent souvent un dictionnaire ou un argument nommé.
        # Par défaut, l'entrée s'appelle souvent "inputs" ou le nom de la première couche (ex: "embedding_input")
        # L'astuce ci-dessous trouve le nom de l'input automatiquement :
        
        results = self.inference_func(input_tensor)
        
        # 6. Extraction du résultat
        # results est un dictionnaire (ex: {'dense_1': <tf.Tensor: ...>})
        # On prend la première valeur (la prédiction)
        key = list(results.keys())[0]
        prediction_tensor = results[key]
        
        prob = float(prediction_tensor.numpy()[0][0])
        
        return prob
