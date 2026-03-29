# 0. Import Required Libraries
import pandas as pd                # Required to load dataset
from openai import OpenAI          # Required to use openai_api_key
import os                          # Required to access environment variables
import gradio as gr                # Required to access ui interface

# 1. Load your csv file
df = pd.read_csv(r"C:\Users\ADMIN\OneDrive\Desktop\AIML.BSRS\AIChatbot\R_Experiment\MUMS 2.0_Chatbot\MUMS 2.0_dataset.csv")

# 2. Connect to OpenAI
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# ==== Search students (simple word match) ====
def search_students(query):
    query_lower = query.lower()                           # Converts the input by the user into lower case (if any upper case exists)
    mask = df.apply(                                      # Applies the follwing to the dataframe
        lambda row: query_lower in str(row).lower(),      # search for the query in the complete data table provided
        axis=1                                            # Applies the lambda function to each rows as axis = 1 (if axis = 0 - applies function to each columns)
    )
    return df[mask]                                       # returns the rows where mask == True

# ==== Main chatbot logic ====
def chatbot_response(user_message, history):
    matches = search_students(user_message)           # Stores matching row of the mask

    # if matches.empty:                                     # if matches is empty then
    #     bot_reply = "No matching students found."     # "No matching students found" is going to be printed and stored in the bot_reply
    #     history.append((user_message, bot_reply))         # we are maintaining the history by appending the last (user_input, bot_reply) into history
    #     return history

    context = matches.to_string(index=False)              # Converting matches into stirng and storing it in context
                                                          # system_prompt is written
    prompt = f"""You are a assistant in IIIT Bhubaneswar answering the questions with your knowledge and also take help of the students table provided answering by interlinking all the three columns Code,Name and Branch.
Students:{context}
Question: {user_message}
Answer:"""

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-nano-2025-04-14",              # Small, free-friendly model
            messages=[
                # {"role": "system", "content": "You answer questions based only on the provided participants table."},
                {"role" : "system", "content" : prompt},
                #{"role": "user", "content": prompt}
                {"role" : "user", "content" : user_message}
            ],
            temperature=0.7                               # by setting temperature to 0 it gives the accurate facts with being random
        )

        bot_reply = response.choices[0].message.content.strip() # this response text is only stored in the bot_reply
    except Exception as e:
        bot_reply = f"Error: {str(e)}"

    history.append((user_message, bot_reply))             # as mentioned earlier history gets appended
    return history

# === Build Gradio UI ===
with gr.Blocks() as demo:
    gr.Markdown("## 📋 Participant Q&A Chatbot")
    chatbot = gr.Chatbot(label="Chat with Participant Data")
    msg = gr.Textbox(label="Your question", placeholder="Type your question here...")
    clear = gr.Button("Clear Chat")

    msg.submit(chatbot_response, [msg, chatbot], chatbot)
    clear.click(lambda: [], None, chatbot, queue=False)

# ==== Run the app ====
if __name__ == "__main__":                               # it is a type of entry-check-point
    demo.launch(share=True)                              # demo is an object in block class of gradio
                                                         # share=True allows to make the link public any provides access to app for anyone