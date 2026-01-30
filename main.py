from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import webbrowser
import threading
import openai
import os, requests
import json
import re

load_dotenv()

app = FastAPI()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MAKE_WEBHOOK_URL = os.getenv("MAKE_WEBHOOK_URL")
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------
# Load Vector Database for RAG
# -----------------------------
VECTOR_DB_PATH = "rag/vector_store"

print("Loading vector database...")
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=OPENAI_API_KEY
)
vector_db = FAISS.load_local(
    VECTOR_DB_PATH, 
    embeddings,
    allow_dangerous_deserialization=True
)
print("[SUCCESS] Vector database loaded!")

# -----------------------------
# Retrieve relevant context from knowledge base
# -----------------------------
def retrieve_context(query: str, top_k: int = 5) -> str:
    """
    Search vector database for relevant restaurant info
    """
    try:
        docs = vector_db.similarity_search(query, k=top_k)
        context = "\n\n".join([doc.page_content for doc in docs])
        print("=== Retrieved Context ===")
        print(context[:500] + "..." if len(context) > 500 else context)
        print("=========================")
        return context
    except Exception as e:
        print(f"Retrieval error: {e}")
        return ""

# -----------------------------
# System prompt for restaurant chatbot
# -----------------------------
BASE_SYSTEM_PROMPT = """You are a friendly customer service chatbot for "Taste Of Punjab" - a home-style Punjabi food & tiffin service based in San Leandro, CA.

Your job is to:
1. Answer customer questions about menu, pricing, delivery, and tiffin service
2. Help customers place orders by collecting their details
3. Be warm, helpful, and patient

MENU RULE (VERY IMPORTANT):
When user asks for "menu", "what do you have", "what's available", "show menu", or anything similar - ALWAYS show the COMPLETE menu with ALL of these:
- All Curries with prices
- All Rice & Biryani items with prices
- All Breads with prices  
- Small Tiffin Box with contents and price
- Large Tiffin Box with contents and price
Never show just curries alone. Always show the full menu.


ORDER COLLECTION:
- For orders, collect: name, phone, delivery address, items/tiffin type, and preferred delivery date.
- Once ALL order details are complete, output as JSON:
  {{"action":"order","name":"...","phone":"...","address":"...","items":"...","date":"..."}}
  Wrap it in triple backticks with json.

CONTEXT FROM KNOWLEDGE BASE:
{context}
"""

def build_system_prompt(context: str) -> dict:
    return {
        "role": "system",
        "content": BASE_SYSTEM_PROMPT.format(context=context)
    }

# Conversation history (without system prompt - added dynamically)
chat_history = []

# -----------------------------
# Serve chat.html
# -----------------------------
@app.get("/")
async def home():
    return FileResponse("chat.html")

# -----------------------------
# OpenAI API call with RAG context
# -----------------------------
def call_openai_with_context(context: str):
    try:
        # Build messages with dynamic system prompt containing context
        system_prompt = build_system_prompt(context)
        messages = [system_prompt] + chat_history
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.3,
            max_tokens=1000
        )
        ai_text = response.choices[0].message.content.strip()
        print("=== AI Response ===")
        print(ai_text)
        print("==================")
        return ai_text
    except Exception as e:
        print("OpenAI API error:", e)
        return "Sorry, something went wrong. Please try again."

# -----------------------------
# Make.com webhook call
# -----------------------------
def send_to_make(data):
    try:
        print("=== Sending data to Make.com ===")
        print(json.dumps(data, indent=2))
        response = requests.post(
            MAKE_WEBHOOK_URL,
            json=data,
            headers={"Content-Type": "application/json"}
        )
        print("Make.com response code:", response.status_code)
        print("Make.com response body:", response.text)
        print("===============================")
        return response.status_code == 200
    except Exception as e:
        print("Make.com error:", e)
        return False

# -----------------------------
# Extract JSON from AI response safely
# -----------------------------
def extract_order_data(ai_text):
    """
    Extract JSON between ```json ... ``` or curly braces from AI response
    """
    match = re.search(r"```json(.*?)```", ai_text, re.DOTALL | re.IGNORECASE)
    if not match:
        match = re.search(r"\{.*\}", ai_text, re.DOTALL)
    if match:
        try:
            json_text = match.group(1).strip() if '```json' in match.group(0) else match.group(0)
            data = json.loads(json_text)
            # Check for order action
            if data.get("action") == "order":
                required_fields = ["name", "phone", "address", "items", "date"]
                if all(field in data and data[field] for field in required_fields):
                    print("=== Extracted Order Data ===")
                    print(json.dumps(data, indent=2))
                    print("============================")
                    return data
        except Exception as e:
            print("JSON parsing error:", e)
            return None
    return None

# -----------------------------
# Chat endpoint with RAG
# -----------------------------
@app.post("/chat")
async def chat_endpoint(request: Request):
    body = await request.json()
    user_message = body.get("message", "")
    print("=== User Message ===")
    print(user_message)
    print("===================")

    # Step 1: Retrieve relevant context from knowledge base
    context = retrieve_context(user_message)

    # Step 2: Add user message to history
    chat_history.append({"role": "user", "content": user_message})

    # Step 3: Call OpenAI with RAG context
    ai_response_text = call_openai_with_context(context)

    # Step 4: Check if AI returned a complete order JSON
    order_data = extract_order_data(ai_response_text)
    if order_data:
        success = send_to_make(order_data)
        if success:
            reply_text = f"Your order for {order_data['items']} has been placed! We'll deliver to {order_data['address']} on {order_data['date']}. Thank you!"
        else:
            reply_text = "Something went wrong while placing your order. Please try again or call us directly."
    else:
        reply_text = ai_response_text  # normal chat

    # Step 5: Add AI reply to history
    chat_history.append({"role": "assistant", "content": reply_text})

    print("=== Reply Sent to User ===")
    print(reply_text)
    print("==========================")

    return {"reply": reply_text}

# -----------------------------
# Reset chat endpoint
# -----------------------------
@app.post("/reset")
async def reset_chat():
    global chat_history
    chat_history = []
    return {"status": "Chat history cleared"}

# -----------------------------
# Open browser automatically
# -----------------------------
def open_browser():
    webbrowser.open("http://127.0.0.1:8000")

if __name__ == "__main__":
    threading.Timer(1.0, open_browser).start()
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
