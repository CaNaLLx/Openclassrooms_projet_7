Auteur : Allan CHARDON
Projet : AirParadis


ARBORESCENCE PROJET :

├── API/                        # API
│   ├── app.py                  # Point d'entrée de l'API
│   ├── services/               # Chargement du modèle et prédiction
│   ├── Modeles/                # Modèle LSTM + tokenizer sérialisés
│   ├── tests/                  # Tests unitaires (pytest)
│   └── requirements.txt        # Dépendances de l'API
│
├── "Data/"                     # Jeu de données (inexistant car fichier trop volumineux pour export)
│
├── Notebooks/                  # Notebooks de modélisation
│   └── modelisation.ipynb      # Prétraitement, entraînement, tracking MLflow
│
├── mainstream.py               # Interface de test Streamlit
│
├── .github/workflows/
│   └── deploy.yml              # Pipeline (GitHub Actions)
│
└── README.md


GitHub COMMANDES POUR CLONER LE PROJET :
git clone https://github.com/CaNaLLx/Openclassrooms_projet_7.git
cd Openclassrooms_projet_7



LANCER L'API (depuis la racine du projet) :
cd API
uvicorn app:app --reload

=> API accessible depuis http://127.0.0.1:8000


LANCER UNE PREDICTION POUR TESTER  :
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "<texte_a_entrer>"}'

LANCER LES TESTS :
cd API
pytest


RESSOURCES UTILISEES PAR L'API (LIBRAIRIES) présentes dans le fichier "requirements.txt", situé dans le dossier API