import json
import string
import random
import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet

# Add a library for named entity recognition
import spacy

# Load the NLP model for entity recognition
nlp = spacy.load("en_core_web_sm")

# Get the data from JSON
with open('without Model\intent_data.json') as file:
    dataset = json.load(file)

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def text_cleaning(text):
    text = re.sub(r'(.)\1+', r'\1', text)
    text.strip()  # Remove leading and trailing spaces
    tokens = word_tokenize(text.lower())  # Convert text to lowercase and tokenize
    tokens = [char for char in tokens if char not in string.punctuation]
    # tokens = [token for token in tokens if token not in stop_words]  # Remove stopwords
    tokens = [stemmer.stem(token) for token in tokens]  # Stemming
    tokens = [lemmatizer.lemmatize(token, wordnet.VERB) for token in tokens]  # Lemmatization
    return tokens

def extract_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def get_responses(input_text):
    input_tokens = set(text_cleaning(input_text))
    input_entities = extract_entities(input_text)

    max_intersection = 0
    matched_intent = None
    matched_entities = None

    # Iterate through intents
    for intent in dataset['intents']:
        # Compare input tokens with intent patterns
        for pattern in intent['patterns']:
            pattern_tokens = set(text_cleaning(pattern))
            intersection_size = len(input_tokens.intersection(pattern_tokens))

            # Check for entity matches
            pattern_entities = extract_entities(pattern)
            if input_entities == pattern_entities:
                intersection_size += 1  # Bonus for entity match

            if intersection_size >= max_intersection:
                max_intersection = intersection_size
                matched_intent = intent
                matched_entities = pattern_entities

    # Construct response using entities if matched
    if matched_intent:
        responses = random.choice(matched_intent['responses'])
        if matched_entities:
            for entity in matched_entities:
                responses = responses.replace("{" + entity[1] + "}", entity[0])
        return responses
    else:
        return "I'm sorry, I didn't understand that."

bot_name = 'LRT GANG'
print("👋 Hello there! Welcome to our virtual assistant designed to enhance your experience with academic management at TUP-Manila! Whether you have questions about course schedules, exam dates, or anything in between, I'm here to help. Just ask away, and let's make your academic journey smoother together! 📚✨")
print("Just type quit to exit")

while True:
    user_input = input("Prompt: ")
    if user_input == "quit":
        break
    response = get_responses(user_input)
    print(response)