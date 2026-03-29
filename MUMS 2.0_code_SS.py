import pandas as pd
import numpy as np
import faiss
import os
from openai import OpenAI
import gradio as gr

# ==== Load Student Dataset (CSV) ====
df = pd.read_csv(r"C:\Users\ADMIN\OneDrive\Desktop\AIML.BSRS\AIChatbot\R_Experiment\MUMS 2.0_Chatbot\MUMS 2.0_dataset.csv")

# ==== OpenAI Client ====
api_key = os.getenv("AI_API_KEY")
client = OpenAI(api_key=api_key)

# ==== Convert each row to text ====
def row_to_text(row):
    # Adjust column names based on your CSV (example: ID, Name, Branch, Year, Email)
    return f"ID: {row['ID']}, Name: {row['Name']}, Branch: {row['Branch']}"

row_texts = [row_to_text(row) for _, row in df.iterrows()]

# ==== Embedding Function ====
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(response.data[0].embedding, dtype="float32")

# ==== Generate embeddings for all rows ====
embeddings = np.vstack([get_embedding(text) for text in row_texts])

# ==== Store in FAISS index ====
d = embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(embeddings)

# ==== Chatbot Logic ====
def chatbot_response(user_message, history):
    # Create embedding for query
    query_emb = get_embedding(user_message).reshape(1, -1)

    # Search top k similar rows
    k = 3
    distances, indices = index.search(query_emb, k)
    retrieved = [row_texts[i] for i in indices[0]]

    # Build context
    context = "\n".join(retrieved)

    # Prompt for LLM
    prompt = f"""
    You are a helpful assistant for IIIT Bhubaneswar.
    Answer student queries using the following student information.
    If you don’t know, say "I don’t know."

    Context:
    {context}

    Question: {user_message}
    Answer:
    Always include ID, Name, and Branch when giving student info.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-nano-2025-04-14",  # lightweight + cost-efficient
            messages=[
                {"role": "system", "content": "You are a helpful assistant for student information queries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        bot_reply = response.choices[0].message.content.strip()
    except Exception as e:
        bot_reply = f"Error: {str(e)}"

    history.append((user_message, bot_reply))
    return history

# ==== Gradio Chatbot UI ====
with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Ask about a student (e.g., 'Tell me about ID 101')...")
    clear = gr.Button("Clear")

    def user_input(message, history):
        return chatbot_response(message, history)

    msg.submit(user_input, [msg, chatbot], chatbot)
    clear.click(lambda: None, None, chatbot, queue=False)

demo.launch()
