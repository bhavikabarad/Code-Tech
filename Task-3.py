# ai_chatbot.py

import nltk
from nltk.chat.util import Chat, reflections

# Download nltk resources (only once)
nltk.download('punkt')

# Define chatbot pairs (input-output patterns)
pairs = [
    [
        r"hi|hello|hey",
        ["Hello! How can I assist you today?", "Hi there! Need any help?"]
    ],
    [
        r"what is your name?",
        ["I am a simple chatbot created using NLTK!", "You can call me NLP Bot."]
    ],
    [
        r"how are you?",
        ["I'm just code, but I'm functioning as expected!", "All good! Ready to help."]
    ],
    [
        r"what can you do?",
        ["I can answer basic questions. Try asking me something!", "I respond to greetings and general queries."]
    ],
    [
        r"(.*) your name?",
        ["I'm an AI chatbot built using Python and NLTK."]
    ],
    [
        r"bye|exit|quit",
        ["Goodbye! Have a great day!", "See you soon!"]
    ],
    [
        r"(.*)",
        ["I'm sorry, I didn't understand that. Can you rephrase?"]
    ]
]

# Create chatbot instance
chatbot = Chat(pairs, reflections)

# Run the chatbot
def run_chatbot():
    print("Hi! I'm your AI Chatbot. Type 'quit' to exit.")
    chatbot.converse()

if __name__ == "__main__":
    run_chatbot()
