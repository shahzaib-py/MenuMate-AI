# 🍽️ MenuMate AI - AI-Powered Restaurant Assistant

An intelligent customer service chatbot for restaurants and food delivery services, powered by RAG (Retrieval-Augmented Generation). It answers customer questions, shows menus, and helps place orders — all using your own knowledge base.

## What it does

- **Answers questions** about menu, pricing, delivery areas, and tiffin service
- **Shows the full menu** with curries, rice, breads, and tiffin boxes
- **Collects orders** and sends them to Make.com webhook for automation
- **Uses RAG** so you can update info anytime without retraining

## Tech Stack

- **Backend:** FastAPI + Python
- **LLM:** OpenAI GPT-3.5-turbo
- **Vector DB:** FAISS (local, free)
- **Embeddings:** OpenAI text-embedding-3-small
- **Automation:** Make.com webhook

## Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/yourusername/foodbot-rag.git
cd foodbot-rag

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Add your API keys

Create a `.env` file in the root:

```
OPENAI_API_KEY=sk-your-key-here
MAKE_WEBHOOK_URL=https://hook.make.com/your-webhook
```

### 3. Build the knowledge base

```bash
python rag/build_vector_db.py
```

### 4. Run the server

```bash
python main.py
```

Open http://127.0.0.1:8000 in your browser.

## Project Structure

```
├── main.py                 # FastAPI server + chat logic
├── chat.html               # Chat UI
├── data/
│   └── restaurant_knowledge.txt   # Your knowledge base
├── rag/
│   ├── build_vector_db.py  # Builds FAISS vector database
│   └── vector_store/       # Generated embeddings
├── requirements.txt
└── .env                    # API keys (not committed)
```

## How RAG Works

```
User asks question
      ↓
Convert to embedding
      ↓
Search vector database for relevant info
      ↓
Pass context + question to LLM
      ↓
LLM generates answer using your data
```

**Why RAG?** Update `restaurant_knowledge.txt` anytime and rebuild vectors — no model retraining needed.

## Updating the Menu

1. Edit `data/restaurant_knowledge.txt`
2. Run `python rag/build_vector_db.py`
3. Restart the server

That's it — your chatbot now knows the new info.

## License

MIT — use it however you want.
