import random
import joblib
import json

from sample import preprocess_text

# Load dataset from JSON
with open('kaggle-dataset.json') as file:
    dataset = json.load(file)

# Load the model
classifier = joblib.load('naive_bayes_model.pkl')

# Function to preprocess user input during inference
def preprocess_user_input(user_input):
    preprocessed_input = preprocess_text(user_input)
    return preprocessed_input

bot_name = 'LRT GANG:'
print("👋 Hello there! Welcome to our virtual assistant designed to enhance your experience with academic management at TUP-Manila! Whether you have questions about course schedules, exam dates, or anything in between, I'm here to help. Just ask away, and let's make your academic journey smoother together! 📚✨")
print("Just type quit to exit")

while True:
    user_input = input("Prompt: ")
    if user_input == "quit":
        break
    
    preprocessed_input = preprocess_user_input(user_input)
    predicted_intent = classifier.predict([preprocessed_input])[0]
    print("Predicted intent:", predicted_intent)
    
    for intent in dataset["intents"]:
        if predicted_intent == intent["tag"]:
            result = random.choice(intent["responses"])
            print(f"{bot_name}: {result}")
            break