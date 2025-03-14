import json
import numpy as np
import random
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()

# Charger les fichiers
with open('data/intents.json') as file:
    intents = json.load(file)

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('model/chatbot_model.keras')
# model = load_model('model/chatbot_model.keras')


# Fonction pour transformer une phrase en bag of words
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return [1 if w in sentence_words else 0 for w in words]


# Prédiction de l'intention
def predict_class(sentence):
    bow = np.array(clean_up_sentence(sentence))
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)

    return classes[results[0][0]] if results else None


# Obtenir une réponse
def get_response(tag):
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "Je ne comprends pas."


# Chatbot interactif
print("Chatbot prêt ! Tape 'quit' pour arrêter.")

while True:
    message = input("Vous : ")
    if message.lower() == "quit":
        break
    tag = predict_class(message)
    response = get_response(tag)
    print(f"Bot : {response}")
