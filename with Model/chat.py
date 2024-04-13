"""
    FOR SOFTWARE ENGINEERING WITH NAIVE BAYES
"""

import random
import joblib
import json

from sample import preprocess_text

# Load dataset from JSON
with open('intents.json', 'r', encoding='utf-8') as file:
    dataset = json.load(file)

# Load the model
classifier = joblib.load('naive_bayes_model.pkl')

# Function to preprocess user input during inference
def preprocess_user_input(user_input):
    preprocessed_input = preprocess_text(user_input)
    return preprocessed_input

bot_name = 'TekBot'
print("👋 Hello there! Welcome to our virtual assistant designed to enhance your experience with academic management at TUP-Manila! Whether you have questions about course schedules, exam dates, or anything in between, I'm here to help. Just ask away, and let's make your academic journey smoother together! 📚✨")
print("Just type quit to exit")


while True:
    try:
        user_input = input("Prompt: ")
        if user_input == "quit":
            break
        
        preprocessed_input = preprocess_user_input(user_input)
        predicted_intent = classifier.predict([preprocessed_input])[0]
        # predicted_intent_int = int(predicted_intent)
        # print("Predicted intent:", predicted_intent, " | ", dataset["intents"][predicted_intent_int]["tag"])
        # for index, intent in enumerate(dataset["intents"]):
        #     if index == predicted_intent_int:
        #         result = random.choice(intent["responses"])
        #         print(f"{bot_name}: {result}")
        #         break
        for intent in dataset["intents"]:
            if predicted_intent == intent['tag']:
                result = random.choice(intent["responses"])
                print(f"{bot_name}: {result}")
                break
    except KeyboardInterrupt:
        print(f"\n{bot_name}'s signing off...")
        break