# 0. Import Required Libraries
import pandas as pd
from openai import OpenAI
import os

# NEW: Import Flask and jsonify
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app) # Enable CORS for frontend communication

# 1. Load your csv file
try:
    df = pd.read_csv("MUMS 2.0_dataset.csv")
except FileNotFoundError:
    print("Error: The CSV file was not found. Please check the file path.")
    exit()

# 2. Connect to OpenAI
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set.")
client = OpenAI(api_key=api_key)

# ==== Search students (simple word match) ====
def search_students(query):
    query_lower = query.lower()
    mask = df.apply(
        lambda row: query_lower in str(row).lower(),
        axis=1
    )
    return df[mask]

# ==== Main chatbot logic ====
def get_bot_response(user_message):
    matches = search_students(user_message)

    context = matches.to_string(index=False)
    
    prompt_template = f"""You are a helpful assistant for IIIT Bhubaneswar that answers questions by referring to the provided participant data table.
students Data Table:
{context}

Based on the table, answer the following question. If a student's ID, name, or branch is mentioned, always include their ID, name, and branch in your response. If no matching information is found, state that no matching students were found.

Question: {user_message}
Answer:
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-nano-2025-04-14",
            messages=[
                {"role": "system", "content": prompt_template},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7
        )

        bot_reply = response.choices[0].message.content.strip()
    except Exception as e:
        bot_reply = f"Error: An error occurred with the AI model. Please try again later. Details: {str(e)}"

    return bot_reply

# The HTML and CSS for the frontend are stored in a Python string
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IIIT Bhubaneswar Student Info</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Roboto', sans-serif;
            background-image: linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.5)), url('{{ url_for('static', filename='international_institute_of_information_technology_bhubaneswar_cover.jpeg') }}');
            background-size: cover;
            background-position: center center;
            background-attachment: fixed;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
            color: #fff;
        }
        .chat-container {
            width: 100%;
            max-width: 768px;
            height: 90vh;
            background: rgba(255, 255, 255, 0.9);
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        .chat-header {
            background-color: #03528d;
            color: white;
            padding: 18px;
            text-align: center;
            font-size: 1.25rem;
            font-weight: 500;
        }
        .chat-log {
            flex-grow: 1;
            padding: 20px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        .message {
            display: flex;
            align-items: flex-start;
            gap: 10px;
        }
        .message-icon {
            width: 30px;
            height: 30px;
            border-radius: 50%;
            display: flex;
            justify-content: center;
            align-items: center;
            font-size: 1.1rem;
            font-weight: 700;
        }
        .user-message .message-icon {
            background-color: #007bff;
            color: white;
        }
        .bot-message .message-icon {
            background-color: #f0f0f0;
            color: #333;
        }
        .message-bubble {
            padding: 12px 18px;
            border-radius: 20px;
            max-width: 80%;
            line-height: 1.5;
        }
        .user-message .message-bubble {
            background-color: #007bff;
            color: white;
            border-bottom-right-radius: 0;
        }
        .bot-message .message-bubble {
            background-color: #f0f0f0;
            color: #333;
            border-bottom-left-radius: 0;
        }
        .chat-input-container {
            padding: 20px;
            border-top: 1px solid #e0e0e0;
            display: flex;
            gap: 10px;
        }
        .chat-input {
            flex-grow: 1;
            padding: 12px 20px;
            border: 1px solid #e0e0e0;
            border-radius: 25px;
            font-size: 1rem;
            outline: none;
        }
        .chat-input:focus {
            border-color: #007bff;
        }
        .chat-send-btn {
            background-color: #03528d;
            color: white;
            border: none;
            border-radius: 25px;
            padding: 12px 20px;
            font-size: 1rem;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        .chat-send-btn:hover {
            background-color: #0056b3;
        }
        .welcome-message {
            text-align: center;
            color: #666;
            margin-top: auto;
            margin-bottom: auto;
            padding: 0 20px;
        }
        .welcome-message h2 {
            font-size: 1.5rem;
            font-weight: 500;
            margin-bottom: 10px;
        }
    </style>
</head>
<body>

<div class="chat-container">
    <div class="chat-header">
        IIIT Bhubaneswar Student Info
    </div>
    <div class="chat-log" id="chat-log">
        <div class="welcome-message">
            <h2>Welcome to IIIT Bhubaneswar Student Info! 👋</h2>
            <p>I can help you with common questions about the institute.</p>
        </div>
    </div>
    <div class="chat-input-container">
        <input type="text" id="user-input" class="chat-input" placeholder="Ask me anything..." onkeydown="handleKey(event)">
        <button id="send-btn" class="chat-send-btn">Send</button>
    </div>
</div>

<script>
    const chatLog = document.getElementById('chat-log');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const welcomeMessage = document.querySelector('.welcome-message');

    function createMessage(message, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', `${sender}-message`);
        
        const iconDiv = document.createElement('div');
        iconDiv.classList.add('message-icon');
        iconDiv.textContent = sender === 'user' ? 'U' : 'AI';

        const bubbleDiv = document.createElement('div');
        bubbleDiv.classList.add('message-bubble');
        bubbleDiv.textContent = message;

        if (sender === 'user') {
            messageDiv.appendChild(bubbleDiv);
            messageDiv.appendChild(iconDiv);
            messageDiv.style.justifyContent = 'flex-end';
        } else {
            messageDiv.appendChild(iconDiv);
            messageDiv.appendChild(bubbleDiv);
            messageDiv.style.justifyContent = 'flex-start';
        }

        chatLog.appendChild(messageDiv);
        chatLog.scrollTop = chatLog.scrollHeight;
    }

    async function sendMessage() {
        const message = userInput.value.trim();
        if (!message) return;

        // Clear welcome message if present
        if (welcomeMessage) {
            welcomeMessage.remove();
        }

        // Display user message
        createMessage(message, 'user');
        userInput.value = '';

        // Send message to backend and get response
        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ message: message })
            });

            const data = await response.json();
            
            // Display bot response
            createMessage(data.response, 'bot');
        } catch (error) {
            console.error('Error:', error);
            createMessage('Sorry, I am unable to connect right now. Please try again later.', 'bot');
        }
    }

    function handleKey(event) {
        if (event.key === 'Enter') {
            sendMessage();
        }
    }

    sendBtn.addEventListener('click', sendMessage);
</script>

</body>
</html>
"""

# ==== New Flask Endpoint to serve the frontend ====
@app.route('/')
def home():
    """Renders the HTML template for the chatbot interface."""
    return render_template_string(HTML_TEMPLATE)

# ==== New Flask Endpoint to process chat messages ====
@app.route('/chat', methods=['POST'])
def chat():
    """Handles incoming messages and provides a response."""
    data = request.json
    user_message = data.get('message', '')
    
    if not user_message:
        return jsonify({"response": "Please provide a message."}), 400
        
    bot_response = get_bot_response(user_message)
    return jsonify({"response": bot_response})

# ==== Run the app ====
if __name__ == "__main__":
    app.run(port=5000, debug=True)