# Import necessary libraries
from flask import Flask, request, jsonify
import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import random
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load intents data
intents = json.loads(open('intents (3).json').read())

# Load preprocessed data
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Load the trained model
from keras.models import load_model
model = load_model('chatbot_model.h5')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Function to clean up sentences
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Function to convert sentence to bag of words
def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)

# Function to predict class based on input sentence
def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

# Function to get response based on predicted class
def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

# Function to respond to user input
def chatbot_response(text):
    ints = predict_class(text, model)
    res = getResponse(ints, intents)
    return res

# Route for handling incoming messages
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    message = data['message']
    response = chatbot_response(message)

    # Storing the conversation
    with open("conversation.log", "a") as f:
        f.write("User: " + message + "\n")
        f.write("Bot: " + response + "\n\n")

    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)
