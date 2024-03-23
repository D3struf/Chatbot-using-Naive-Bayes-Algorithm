from nlp import get_responses, text_cleaning
import sys

bot_name = 'LRT GANG'
print("👋 Hello there! Welcome to our virtual assistant designed to enhance your experience with academic management at TUP-Manila! Whether you have questions about course admission, graduation, schedules, exam dates, or anything in between, I'm here to help. Just ask away, and let's make your academic journey smoother together! 📚✨")
print("Just type quit to exit")

while True:
    try:
        user_input = input("Prompt: ")
        if user_input == "quit":
            break
        
        print(text_cleaning(user_input))
        responses = get_responses(user_input)
        print(f"{bot_name}: {responses}")
    except KeyboardInterrupt:
        print("\nExiting program.")
        sys.exit()