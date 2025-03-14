from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import numpy as np
import random
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import logging
import ssl

# Configuration des logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Désactiver la vérification SSL pour le téléchargement NLTK
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

app = Flask(__name__)
CORS(app)

try:
    # Télécharger toutes les ressources NLTK nécessaires
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('universal_tagset', quiet=True)
    
    logger.info("Ressources NLTK téléchargées avec succès")

    lemmatizer = WordNetLemmatizer()

    # Charger les fichiers nécessaires
    with open('data/intents.json') as file:
        intents = json.load(file)

    words = pickle.load(open('model/words.pkl', 'rb'))
    classes = pickle.load(open('model/classes.pkl', 'rb'))
    model = load_model('model/chatbot_model.keras')
    
    logger.info(f"Nombre de mots dans le vocabulaire: {len(words)}")
    logger.info(f"Nombre de classes: {len(classes)}")
    logger.info("Toutes les ressources ont été chargées avec succès")
except Exception as e:
    logger.error(f"Erreur lors du chargement des ressources: {str(e)}")
    raise

def clean_up_sentence(sentence):
    try:
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        logger.info(f"Mots traités : {sentence_words}")
        return sentence_words
    except Exception as e:
        logger.error(f"Erreur dans clean_up_sentence: {str(e)}")
        raise

def bag_of_words(sentence_words):
    try:
        bag = [0] * len(words)
        for w in sentence_words:
            for i, word in enumerate(words):
                if word == w:
                    bag[i] = 1
        return np.array(bag)
    except Exception as e:
        logger.error(f"Erreur dans bag_of_words: {str(e)}")
        raise

def predict_class(sentence):
    try:
        sentence_words = clean_up_sentence(sentence)
        bow = bag_of_words(sentence_words)
        res = model.predict(np.array([bow]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        predicted = classes[results[0][0]] if results else None
        logger.info(f"Classe prédite : {predicted}")
        return predicted
    except Exception as e:
        logger.error(f"Erreur dans predict_class: {str(e)}")
        raise

def get_response(tag):
    try:
        if not tag:
            return "Je ne comprends pas votre question."
        
        for intent in intents['intents']:
            if intent['tag'] == tag:
                response = random.choice(intent['responses'])
                logger.info(f"Réponse choisie : {response}")
                return response
        return "Je ne comprends pas votre question."
    except Exception as e:
        logger.error(f"Erreur dans get_response: {str(e)}")
        raise

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No JSON data received'}), 400
        
        message = data.get('message', '')
        logger.info(f"Message reçu : {message}")
        
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        
        tag = predict_class(message)
        response = get_response(tag)
        
        logger.info(f"Tag : {tag}")
        logger.info(f"Réponse envoyée : {response}")
        
        return jsonify({
            'response': response,
            'tag': tag
        })
    except Exception as e:
        logger.error(f"Erreur dans la route /api/chat: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("Démarrage du serveur Flask...")
    app.run(host='0.0.0.0', debug=True) 