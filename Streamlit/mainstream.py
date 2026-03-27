import streamlit as st
import requests
import logging
from opencensus.ext.azure.log_exporter import AzureLogHandler

# --- 1. CONFIGURATION DE LA PAGE (DOIT ÊTRE EN PREMIER) ---
st.set_page_config(page_title="Test API & Validation", page_icon="🐦")

# --- 2. VARIABLES ---
API_URL = "https://airparadis-api4.azurewebsites.net/predict" 

CONNECTION_STRING = "InstrumentationKey=fe2bfada-b7b9-4a2b-bcee-ad17f7e23f62;IngestionEndpoint=https://westeurope-5.in.applicationinsights.azure.com/;LiveEndpoint=https://westeurope.livediagnostics.monitor.azure.com/;ApplicationId=c421c20b-7a9f-4e3c-a8c8-fe4912d3b80a"

# --- 3. SETUP DU LOGGER AZURE ---
logger = logging.getLogger(__name__)

# On configure le logger UNIQUEMENT si une Connection String est fournie
if "InstrumentationKey" in CONNECTION_STRING:
    if not any(isinstance(h, AzureLogHandler) for h in logger.handlers):
        try:
            azure_handler = AzureLogHandler(connection_string=CONNECTION_STRING)
            logger.addHandler(azure_handler)
            logger.setLevel(logging.INFO)
            print("✅ Logger Azure connecté")
        except Exception as e:
            st.error(f"Erreur de connexion Azure Insights: {e}")
else:
    # Mode silencieux si pas de clé (pour éviter les erreurs en local sans Azure)
    print("Aucune Connection String Azure détectée. Le logging est désactivé.")

# --- 4. INTERFACE STREAMLIT ---
st.title("Air Paradis - Analyse de sentiments")
st.markdown("Testez le modèle et validez (ou non) les prédictions pour améliorer le système.")

# Zone de saisie
tweet_text = st.text_area("Saisissez le tweet à analyser :", height=100)

# Initialisation de l'état (Session State)
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'tweet_analyzed' not in st.session_state:
    st.session_state.tweet_analyzed = ""

# --- ETAPE 5 : PRÉDICTION ---
if st.button("Analyser le sentiment du tweet"):
    if tweet_text:
        with st.spinner("Analyse en cours..."):
            try:
                # Appel à ton API
                response = requests.post(API_URL, json={"text": tweet_text})
                
                if response.status_code == 200:
                    result = response.json()
                    # On stocke le résultat en mémoire session
                    st.session_state.prediction = result
                    st.session_state.tweet_analyzed = tweet_text
                else:
                    st.error(f"Erreur API : {response.status_code}")
                    st.error(response.text)
            except requests.exceptions.ConnectionError:
                st.error("ATTENTION : Impossible de contacter l'API. Vérifie que 'uvicorn' tourne bien dans l'autre terminal !")
            except Exception as e:
                st.error(f"Erreur inattendue : {e}")
    else:
        st.warning("Veuillez écrire un tweet d'abord.")

# --- ETAPE 6 : AFFICHAGE & VALIDATION ---
if st.session_state.prediction:
    st.divider()
    st.subheader("Résultat de l'IA")
    
    # Récupération des données (Adapté à ton API FastAPI)
    sentiment = st.session_state.prediction.get('sentiment', 'Inconnu')
    # CORRECTION ICI : Ton API renvoie 'score', pas 'probability'
    score = st.session_state.prediction.get('score', 0.0) 
    
    col1, col2 = st.columns(2)
    with col1:
        color = "green" if sentiment == "Positif" else "red"
        st.markdown(f"Sentiment : **:{color}[{sentiment}]**")
    with col2:
        st.metric("Score de positivité", f"{score:.2%}")

    st.write("---")
    st.write("**Cette prédiction est-elle correcte ?**")
    
    col_yes, col_no = st.columns([1, 4])
    
    # BOUTON OUI (Validation)
    with col_yes:
        if st.button("Oui"):
            st.success("La prédiction est bonne ! Trop bien :)")
            st.session_state.prediction = None # Reset

    # BOUTON NON (Erreur -> Envoi Azure)
    with col_no:
        if st.button("Non"):
            st.error("La prédiction est mauvaise ! Le feedback a été envoyé à l'équipe technique. Merci pour votre retour..")
            
            # --- ENVOI DU TRACE À AZURE ---
            properties = {
                'custom_dimensions': {
                    'tweet_content': st.session_state.tweet_analyzed,
                    'model_prediction': sentiment,
                    'model_confidence': str(score),
                    'user_feedback': 'incorrect_prediction'
                }
            }
            
            # On loggue l'erreur pour Azure
            if "InstrumentationKey" in CONNECTION_STRING:
                logger.warning("Mauvaise prédiction signalée par l'utilisateur", extra=properties)
                print("Log envoyé à Azure.")
            else:
                print("Simulation d'envoi Azure (Pas de clé configurée)")
            
            # Reset
            st.session_state.prediction = None
