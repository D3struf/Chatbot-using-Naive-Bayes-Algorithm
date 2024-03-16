import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import json
import string
import numpy as np
import random

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from nltk_utils import tokenize, stem

# Load intents from JSON file
with open("intents.json", 'r') as file:
    intents = json.load(file)

# Define the set of ignore words
ignore_words = ['?', '!', '.', ',']

# Initialize lists to store all words, tags, and training data
all_words = []
tags = []
X_train = []
y_train = []

# Iterate through each intent
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    
    # Iterate through each pattern in the intent
    for pattern in intent['patterns']:
        # Tokenize and stem the words, ignoring certain words
        words = [stem(word.lower()) for word in word_tokenize(pattern) if word not in ignore_words]
        all_words.extend(words)
        # Join the stemmed words back into sentences
        sentence = ' '.join(words)
        X_train.append(sentence)
        y_train.append(tag)

# Remove duplicates and sort the lists
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# Vectorize the training data
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)

# Train the Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_counts, y_train)

# Example usage:
bot_name = 'LRT GANG:'
print("Let's chat! type 'quit' to exit")
while True:
    input_sentence = input("Prompt: ")
    if input_sentence == "quit":
        break
    
    input_features = vectorizer.transform([input_sentence])
    predicted_intent = nb_classifier.predict(input_features)[0]

    for intent in intents["intents"]:
        if predicted_intent == intent["tag"]:
            result = random.choice(intent["responses"])
            print(f"{bot_name}: {result}")
            break
