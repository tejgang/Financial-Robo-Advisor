import re

class DialogueManager:
    def __init__(self, chatbot):
        self.chatbot = chatbot
        self.state = "greeting"
        
    def handle_input(self, text):
        if self.state == "greeting":
            self.state = "profile_building"
            return "Welcome! Let's start with your financial profile. What's your age?"
            
        elif self.state == "profile_building":
            return self._extract_profile_info(text)
            
        response = self.chatbot.process_message(text)
        return response
    
    def _extract_profile_info(self, text):
        # Use regex/NLP to extract numerical values
        if 'age' not in self.chatbot.user_profile:
            age = re.search(r'\d+', text)
            if age:
                self.chatbot.user_profile['age'] = int(age.group())
                return "What's your annual income?"
        # Continue for other profile fields 