"""
    FOR SOFTWARE ENGINEERING WITH NAIVE BAYES
"""


import json
import joblib
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Download NLTK resources (if not already downloaded)
# List of NLTK resources to check/download
resources = ['punkt', 'stopwords', 'wordnet']

# Check and download each resource if not already downloaded
for resource in resources:
    if not nltk.download(resource, quiet=True):
        print(f"'{resource}' is already downloaded.")
    else:
        print(f"'{resource}' is downloaded now.")

# Load dataset from JSON
with open('intents.json', 'r', encoding='utf-8') as file:
    dataset = json.load(file)

# Extract patterns and intents from the dataset
patterns = []
intents = []
for intent in dataset['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        intents.append(intent['tag'])

# Initialize stemmer and lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Preprocess patterns (tokenization, stopword removal, stemming, and lemmatization)
def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Convert text to lowercase and tokenize
    stop_words = set(stopwords.words('english'))
    # tokens = [token for token in tokens if token not in stop_words]  # Remove stopwords
    stemmed_tokens = [stemmer.stem(token) for token in tokens]  # Stemming
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in stemmed_tokens]  # Lemmatization
    return ' '.join(lemmatized_tokens)  # Join tokens back into a single string

preprocessed_patterns = [preprocess_text(pattern) for pattern in patterns]

# Train Naive Bayes classifier
classifier = Pipeline([
    ('bow', CountVectorizer()),  # Convert text to Bag-of-Words features
    ('clf', MultinomialNB()),    # Naive Bayes classifier
])
classifier.fit(preprocessed_patterns, intents)

# Save the trained model
joblib.dump(classifier, 'naive_bayes_model.pkl')
print("Training Successful!!!")