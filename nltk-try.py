import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MaxAbsScaler

# Download NLTK resources (if not already downloaded)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load intents from JSON file
with open('intents.json', 'r') as file:
    intents = json.load(file)

# Initialize NLTK components
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Preprocess input text
def preprocess_input(text):
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in stop_words and len(token) > 1]
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    stemmed_tokens = [stemmer.stem(token) for token in lemmatized_tokens]
    return ' '.join(stemmed_tokens)

# Initialize the TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Extract patterns from intents and preprocess them
patterns = []
intent_tags = []
for intent in intents['intents']:
    intent_tags.append(intent['tag'])
    patterns.extend([preprocess_input(pattern) for pattern in intent['patterns']])

# Fit the vectorizer to the patterns and transform the text data into TF-IDF vectors
X = vectorizer.fit_transform(patterns)

# Scale the data to ensure all features are on the same scale
scaler = MaxAbsScaler()
X_scaled = scaler.fit_transform(X.toarray())

# Perform hierarchical clustering using AgglomerativeClustering
n_clusters = 5  # Specify the number of clusters
hierarchical_clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
clusters = hierarchical_clustering.fit_predict(X_scaled)

# Print the identified clusters
print("Clusters:")
print(clusters)

def get_user_input():
    user_input = input("You: ")
    return user_input

def get_cluster_response(user_input, clusters, vectorizer, intents):
    preprocessed_user_input = preprocess_input(user_input)
    user_input_vector = vectorizer.transform([preprocessed_user_input])
    similarities = cosine_similarity(user_input_vector, X)
    best_intent_index = similarities.argmax()

    if best_intent_index < len(intent_tags):
        best_intent_tag = intent_tags[best_intent_index]
        return f"Chatbot: {best_intent_tag}"
    else:
        return "Chatbot: Sorry, I couldn't understand your input. Please try again."

def main():
    print("Welcome to the chatbot!")
    while True:
        user_input = get_user_input()
        if user_input.lower() == "quit":
            break
        response = get_cluster_response(user_input, clusters, vectorizer, intents)
        print(response)

if __name__ == "__main__":
    main()