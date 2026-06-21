# NileTel Support Assistant

A hybrid-retrieval (RAG) support assistant for a telecom use case, with automated ticket dispatch. A user question is answered from a Markdown knowledge base using combined lexical + semantic retrieval and a Groq-hosted LLM. When the assistant determines an action is required, it triggers an n8n workflow that logs a support ticket to Google Sheets.

## Architecture

```
User -> Streamlit UI -> FastAPI /ask -> TelecomRAG
                                     |- Retrieval: FAISS (dense) + BM25Okapi (lexical)
                                     |- Generation: Groq LLM
                                     |- needs_action? --yes--> n8n webhook --> Google Sheets (ticket)
```

## Tech Stack

- **Retrieval (hybrid):** FAISS dense index over `intfloat/multilingual-e5-large` embeddings + `BM25Okapi` lexical search
- **Chunking:** LangChain `RecursiveCharacterTextSplitter` (700 chars, 150 overlap), document title injected into each chunk
- **LLM:** Groq API
- **Backend:** FastAPI (`/ask` endpoint)
- **Frontend:** Streamlit (query box + live ticket table)
- **Automation:** n8n webhook -> Google Sheets ticket log

## Project Structure

- `rag_class.py` — `TelecomRAG`: document loading/chunking, FAISS + BM25 index building, retrieval, and LLM answering
- `mains2.py` — FastAPI app; runs the RAG pipeline and calls the n8n webhook when an action is needed
- `streams2.py` — Streamlit UI for asking questions and viewing dispatched tickets
- `test.py` — scratch / testing

## Setup

1. Install dependencies:
   ```bash
   pip install fastapi uvicorn streamlit faiss-cpu sentence-transformers rank-bm25 langchain-text-splitters groq python-dotenv requests pandas
   ```
2. Create a `.env` with your Groq key:
   ```env
   GROQ_API_KEY=your_key_here
   ```
3. Point `DATA_PATH` in `rag_class.py` to a folder of `.md` knowledge-base files.
4. Run the API: `uvicorn mains2:app --reload`
5. Run the UI: `streamlit run streams2.py`

## Notes

- The knowledge base is a folder of Markdown documents (not committed to this repo).
- Configure the n8n webhook URL in `mains2.py` and the Google Sheet CSV export URL in `streams2.py`.
- This is a functional prototype; formal retrieval/answer evaluation is not yet implemented.
