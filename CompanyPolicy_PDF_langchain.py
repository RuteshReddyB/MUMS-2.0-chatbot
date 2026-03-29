from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS

# load the policy pdf
loader = PyPDFLoader("CompanyPolicy.pdf")
documents = loader.load()

# split into smaller chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)
docs = splitter.split_documents(documents)

# create embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Store in the FAISS vector database
db = FAISS.from_documents(docs, embeddings)

# Create a retriever
retriever = db.as_retriever(search_kwargs={"k": 3})

# Define an LLM
llm = ChatOpenAI(model="gpt-4.1-nano-2025-04-14")

# Prompt building function
def ask_policy_bot(query: str):
    # retrieve relevant chunks
    retrieved_docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    # build dynamic prompt
    prompt = f"""
    You are a helpful HR assistant.
    Use ONLY the context below to answer the question.
    If the answer is not in the context, say "I don't know."

    Context:
    {context}

    Question: {query}
    Answer:
    """

    return llm.predict(prompt)