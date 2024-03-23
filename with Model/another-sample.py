from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import json
import random

# Import functions from nltk_utils.py
from nltk_utils import tokenize, stem

# Load intents from JSON file
with open("kaggle-dataset.json", 'r') as file:
    intents = json.load(file)

# Define the set of ignore words
ignore_words = ['?', '!', '.', ',', ':', ';', "'", "=", "(", ")"]

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
        words = [stem(word.lower()) for word in tokenize(pattern) if word not in ignore_words]
        all_words.extend(words)
        # Join the stemmed words back into sentences
        sentence = ' '.join(words)
        X_train.append(sentence)
        y_train.append(tag)
        
bow_vectorizer = CountVectorizer()
training_vectors = bow_vectorizer.fit_transform(X_train)

classifier = MultinomialNB()
classifier.fit(training_vectors, y_train)

class ChatBot:
    exit_commands = ("quit", "pause", "exit", "goodbye", "bye", "later", "stop")

    def start_chat(self):
        user_response = input("Hi, I'm a chatbot trained on random dialogs!!\n")
        self.chat(user_response)
    
    def chat(self, reply):
        while not self.make_exit(reply):
            reply = input(self.generate_response(reply) + "\n")
        return
    
    def generate_response(self, sentence):
        input_vector = bow_vectorizer.transform([sentence])
        predict = classifier.predict(input_vector)[0]  # Ensure to access the first element
        index = tags.index(predict)  # Find the index of the predicted tag
        probabilities = classifier.predict_proba(input_vector)[0]
        accuracy = str(probabilities[index] * 100)[:5] + "%"  # Calculate accuracy
        print("Accurate:", accuracy)
        
        for intent in intents["intents"]:
            if predict == intent["tag"]:
                result = random.choice(intent["responses"])
                # print(f"LRT GANG: {result}")
                break
        return result
    
    def make_exit(self, reply):
        for exit_command in self.exit_commands:
            if exit_command in reply:
                print("Ok, have a great day!")
                return True
        return False

etcetera = ChatBot()
etcetera.start_chat()