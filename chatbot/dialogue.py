import re

class DialogueManager:
    def __init__(self, chatbot):
        self.chatbot = chatbot
        self.state = "greeting"
        self.profile_fields = [
            ('age', "What's your age?"),
            ('income', "What's your annual income?"),
            ('savings', "How much do you have in savings?"),
            ('risk_tolerance', "On a scale of 1-5, how much risk can you tolerate? (1=Low, 5=High)"),
            ('investment_horizon', "How many years do you plan to invest?"),
            ('debt', "Do you have any outstanding debt? If so, how much?")
        ]
        self.current_field = 0
        
    def handle_input(self, text):
        if self.state == "greeting":
            self.state = "profile_building"
            return "Welcome to FinBot! Let's start with your financial profile.\n" + self.profile_fields[0][1]
            
        elif self.state == "profile_building":
            return self._extract_profile_info(text)
            
        response = self.chatbot.process_message(text)
        return response
    
    def _extract_profile_info(self, text):
        field_name, prompt = self.profile_fields[self.current_field]
        
        # Extract numerical value using regex
        numbers = re.findall(r'\d+', text)
        if numbers:
            value = float(numbers[0])
            self.chatbot.user_profile[field_name] = value
            self.current_field += 1
            
            if self.current_field < len(self.profile_fields):
                return self.profile_fields[self.current_field][1]
            else:
                self.state = "ready"
                return "Profile complete! How can I assist you today?\nYou can ask about:\n- Portfolio recommendations\n- Risk assessment\n- Debt management\n- Retirement planning"
        
        return f"Please provide a numerical value for {field_name.replace('_', ' ')}. {prompt}" 