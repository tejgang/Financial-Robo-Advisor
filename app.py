from flask import Flask, request, jsonify, render_template_string
from chatbot.model import FinancialIntentRecognizer, PortfolioRecommender, ChatBot
from chatbot.dialogue import DialogueManager
import torch
import pickle

app = Flask(__name__)

# Load tokenizer
with open('models/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load intent model
checkpoint = torch.load('models/intent_model.pth')
vocab_size = checkpoint['vocab_size']
intent_model = FinancialIntentRecognizer(vocab_size).to(device)
intent_model.load_state_dict(checkpoint['state_dict'])
intent_model.eval()

# Load recommender
recommender = PortfolioRecommender().to(device)
recommender.load_state_dict(torch.load('models/recommender.pth'))
recommender.eval()

# Initialize chatbot
chatbot = ChatBot(intent_model, recommender, tokenizer)
dialogue_manager = DialogueManager(chatbot)

# Add a default route with a simple HTML interface
@app.route('/')
def index():
    html = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Financial Advisor Chatbot</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            #chat-container { height: 400px; border: 1px solid #ccc; padding: 10px; overflow-y: scroll; margin-bottom: 10px; }
            #user-input { width: 80%; padding: 8px; }
            button { padding: 8px 15px; background: #4CAF50; color: white; border: none; cursor: pointer; }
            .bot-message { background: #f1f1f1; padding: 8px; border-radius: 5px; margin: 5px 0; }
            .user-message { background: #e3f2fd; padding: 8px; border-radius: 5px; margin: 5px 0; text-align: right; }
        </style>
    </head>
    <body>
        <h1>Financial Advisor Chatbot</h1>
        <div id="chat-container"></div>
        <input type="text" id="user-input" placeholder="Type your message...">
        <button onclick="sendMessage()">Send</button>

        <script>
            const chatContainer = document.getElementById('chat-container');
            const userInput = document.getElementById('user-input');
            
            // Add initial bot message
            addBotMessage("Welcome! I'm your financial advisor bot. How can I help you today?");
            
            function addUserMessage(message) {
                const div = document.createElement('div');
                div.className = 'user-message';
                div.textContent = message;
                chatContainer.appendChild(div);
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
            
            function addBotMessage(message) {
                const div = document.createElement('div');
                div.className = 'bot-message';
                div.textContent = message;
                chatContainer.appendChild(div);
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
            
            function sendMessage() {
                const message = userInput.value.trim();
                if (message) {
                    addUserMessage(message);
                    userInput.value = '';
                    
                    // Send to API
                    fetch('/chat', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            message: message
                        }),
                    })
                    .then(response => response.json())
                    .then(data => {
                        addBotMessage(data.response);
                    })
                    .catch((error) => {
                        console.error('Error:', error);
                        addBotMessage("Sorry, I encountered an error processing your request.");
                    });
                }
            }
            
            // Allow Enter key to send message
            userInput.addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    sendMessage();
                }
            });
        </script>
    </body>
    </html>
    '''
    return render_template_string(html)

@app.route('/chat', methods=['POST'])
def handle_chat():
    data = request.json
    user_id = data.get('user_id', 'default_user')
    message = data.get('message', '')
    
    response = dialogue_manager.handle_input(message)
    
    return jsonify({
        'user_id': user_id,
        'response': response,
        'profile_complete': dialogue_manager.state == "ready"
    })

@app.route('/portfolio', methods=['POST'])
def get_portfolio():
    data = request.json
    user_id = data.get('user_id', 'default_user')
    
    if not chatbot.user_profile:
        return jsonify({'error': 'Profile not complete'}), 400
        
    allocation = chatbot.recommender(
        torch.tensor([[
            chatbot.user_profile['age'],
            chatbot.user_profile['income'],
            chatbot.user_profile['savings'],
            chatbot.user_profile['risk_tolerance'],
            chatbot.user_profile['investment_horizon'],
            chatbot.user_profile['debt']
        ]]).float().to(device)
    ).cpu().detach().numpy().squeeze()
    
    return jsonify({
        'user_id': user_id,
        'allocation': {
            'stocks': float(allocation[0]),
            'bonds': float(allocation[1]),
            'crypto': float(allocation[2]),
            'real_estate': float(allocation[3]),
            'cash': float(allocation[4])
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 