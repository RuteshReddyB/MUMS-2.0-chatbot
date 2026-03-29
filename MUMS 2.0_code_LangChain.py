import pandas as pd
import gradio as gr
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# ---- Load CSV ----
df = pd.read_csv(r"C:\Users\ADMIN\OneDrive\Desktop\AIML.BSRS\AIChatbot\R_Experiment\MUMS 2.0_Chatbot\MUMS 2.0_dataset.csv")

# ---- Setup LLM + Agent ----
llm = ChatOpenAI(model="gpt-4.1-nano-2025-04-14", temperature=0.7)
agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True)

# ---- Chatbot Function ----
def chatbot(query, history=[]):
    try:
        response = agent.run(query)
    except Exception as e:
        response = f"⚠️ Error: {str(e)}"
    return response

# ---- Gradio Interface ----
with gr.Blocks() as demo:
    gr.Markdown("## 🎓 Student Info Chatbot")
    gr.Markdown("Ask questions about students in the CSV file (e.g., *Who has GPA > 3.5?*)")

    chatbot_ui = gr.ChatInterface(
        fn=chatbot,
        title="Student Info Chatbot",
        theme="soft",
    )

demo.launch()
