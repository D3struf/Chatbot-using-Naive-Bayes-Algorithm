"""
    
"""

# Import necessary libraries
import json
import string
import random
import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet

# Get the data from JSON
with open('./kaggle-dataset.json') as file:
    dataset = json.load(file)

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def text_cleaning(text):
    text = re.sub(r'(.)\1+', r'\1', text)
    text.strip()  # Remove leading and trailing spaces
    tokens = word_tokenize(text.lower())  # Convert text to lowercase and tokenize
    tokens = [char for char in tokens if char not in string.punctuation]
    # cleaned_tokens = [token for token in tokens if token not in stop_words]  # Remove stopwords
    tokens = [stemmer.stem(token) for token in tokens]  # Stemming
    tokens = [lemmatizer.lemmatize(token, wordnet.VERB) for token in tokens]  # Lemmatization
    return tokens

def get_responses(input_text):
    # Preprocess user input
    input_tokens = set(text_cleaning(input_text))
    print(input_tokens)
    max_intersection = 0
    # Iterate through intents
    for intent in dataset['intents']:
        # Check if any pattern matches user input
        for pattern in intent['patterns']:
            pattern_tokens = set(text_cleaning(pattern))
            intersection_size = len(input_tokens.intersection(pattern_tokens))
            if intersection_size >= max_intersection:
                max_intersection = intersection_size
                responses = random.choice(intent['responses'])
    
    # If no matches found, return default response
    if max_intersection == 0:
        responses = "I'm sorry, I didn't understand that."
    
    return responses

user_input = "where is the location of the college?"
response = get_responses(user_input)
print(response)