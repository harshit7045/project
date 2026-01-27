# Harrison Medical Bot

A RAG (Retrieval-Augmented Generation) chatbot that answers medical questions based on "Harrison's Principles of Internal Medicine".

## Tech Stack
- **Python**
- **LangChain**
- **Pinecone** (Vector Database)
- **Google Gemini** (LLM & Embeddings)
- **Streamlit** (UI)

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Environment Variables**:
    - Copy `.env.example` to `.env`.
    - Fill in your `GOOGLE_API_KEY` and `PINECONE_API_KEY`.

3.  **PDF Ingestion**:
    - Place `harrison_book.pdf` in the root directory.
    - Run the ingestion script:
      ```bash
      python ingest.py
      ```
    - *Note:* The script supports resuming. If it stops, just run it again, and it will pick up from the last saved page.

## Running the App

Run the Streamlit app:
```bash
streamlit run app.py
```
Open your browser to the URL provided in the terminal (usually `http://localhost:8501`).
