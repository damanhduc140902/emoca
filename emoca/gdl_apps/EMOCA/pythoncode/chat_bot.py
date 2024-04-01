import json
import random

class EmotionChatBot:
    def __init__(self):
        with open('responses.json', 'r', encoding='utf-8') as file:
            self.responses = json.load(file)
    
    def get_response(self, emotion):
        return random.choice(self.responses[emotion])
        