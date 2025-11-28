import os
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from groq import Groq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from pathlib import Path

BASEDIR = Path(__file__).parent
load_dotenv(BASEDIR / ".env")
app = Flask(__name__)

# -----------------------------
# 1. Groq Setup
# -----------------------------
groq_api_key = os.getenv("GROQ_API_KEY")
print(f"API Key loaded: {'YES' if groq_api_key else 'NO'}")  # Will show NO currently
print(f"Full key preview: {groq_api_key[:10] if groq_api_key else 'None'}...")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found! Check .env file.")

client = Groq(api_key=groq_api_key)

# -----------------------------
# 2. Load FAISS Vectorstore
# -----------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# vectorstore folder must sit inside your project directory
vectorstore = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

# -----------------------------
# 3. Model Options for Dropdown
# -----------------------------
MODELS = {
    "Llama 4 Maverick 17B": "meta-llama/llama-4-maverick-17b-128e-instruct",
    "Llama 3.3 70B": "llama-3.3-70b-versatile",
    "Llama 3.1 8B": "llama-3.1-8b-instant",
    "Moonshot Kimi K2": "moonshotai/kimi-k2-instruct",
    "Groq Compound": "groq/compound",
    "Groq Compound Mini": "groq/compound-mini",
    "OpenAI GPT OSS 120B": "openai/gpt-oss-120b",
    "OpenAI GPT OSS 20B": "openai/gpt-oss-20b",
    "Qwen 3 32B": "qwen/qwen3-32b"
}


# -----------------------------
# 4. Home Page
# -----------------------------
@app.route("/")
def index():
    return render_template("index.html", models=MODELS)


# -----------------------------
# 5. Dropdown Model Selection Page (optional)
# -----------------------------
@app.route("/select_model", methods=["POST"])
def select_model():
    selected_model = request.form.get("model")
    input_text = request.form.get("input_text")

    # Retrieve relevant chunks
    results = vectorstore.similarity_search(input_text, k=3)
    context = "\n".join(doc.page_content for doc in results)

    prompt = f"""
    You are Irla, the Technov8 AI assistant.
    Be friendly, accurate, and helpful.
    Do NOT introduce yourself unless the user asks. 
    Answer naturally, clearly, and based ONLY on the context.

    CONTEXT:
    {context}

    QUESTION:
    {input_text}
    """

    try:
        reply = client.chat.completions.create(
            model=MODELS[selected_model],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )
        answer = reply.choices[0].message.content
    except Exception as e:
        answer = f"Error: {str(e)}"

    return render_template("index.html", models=MODELS,
                           model=selected_model,
                           input_text=input_text,
                           result=answer)


# -----------------------------
# 6. Chatbot API (used by JS)
# -----------------------------
@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "")

    # Retrieve similar context from vector store
    results = vectorstore.similarity_search(user_message, k=3)
    context = "\n".join([doc.page_content for doc in results])

    prompt = f"""
    You are Irla, the Technov8 AI assistant.
    Do NOT introduce yourself unless the user asks.
    Provide clear, friendly answers based ONLY on the context.

    CONTEXT:
    {context}

    USER:
    {user_message}
    """

    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # default model for chatbot
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300
        )
        bot_reply = completion.choices[0].message.content.strip()
    except Exception as e:
        bot_reply = f"Error: {e}"

    return jsonify({"response": bot_reply})


# -----------------------------
# 7. Run App
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
