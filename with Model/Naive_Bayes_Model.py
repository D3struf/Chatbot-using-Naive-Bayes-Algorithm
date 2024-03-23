from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import json
import random

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

from nltk_utils import tokenize, stem, bag_of_words

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
        words = [stem(word.lower()) for word in word_tokenize(pattern) if word not in ignore_words]
        all_words.extend(words)
        # Join the stemmed words back into sentences
        sentence = ' '.join(words)
        X_train.append(sentence)
        y_train.append(tag)

# Remove duplicates and sort the lists
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Vectorize the training data
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)

# Train the Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_counts, y_train)

# Vectorize the testing data
X_test_counts = vectorizer.transform(X_test)

# Predict labels for the testing data
y_pred = nb_classifier.predict(X_test_counts)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Save the trained model
# joblib.dump(nb_classifier, 'naive_bayes_model.pkl')

# Example usage:
bot_name = 'LRT GANG:'
print("👋 Hello there! Welcome to our virtual assistant designed to enhance your experience with academic management at TUP-Manila! Whether you have questions about course schedules, exam dates, or anything in between, I'm here to help. Just ask away, and let's make your academic journey smoother together! 📚✨")
print("Just type quit to exit")
while True:
    input_sentence = input("Prompt: ")
    if input_sentence == "quit":
        break
    
    preprocessed_input = tokenize(input_sentence)
    input_features = bag_of_words(preprocessed_input, all_words)
    input_features = input_features.reshape(1, -1)
    print(input_features)
    # Vectorize the input sentence
    input_features = vectorizer.transform([input_features])

    # Reshape the input features if necessary
    if input_features.shape[0] == 1:
        input_features = input_features.reshape(1, -1)
        
    predicted_intent = nb_classifier.predict(input_features)[0]
    print('Predicted Intent: ', predicted_intent)

    for intent in intents["intents"]:
        if predicted_intent == intent["tag"]:
            result = random.choice(intent["responses"])
            print(f"{bot_name}: {result}")
            break
