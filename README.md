# YouTube video chat

Paste a YouTube video URL into the Streamlit app to load its English captions
with LangChain's `YoutubeLoader`, then you can chat using the FAISS retrieval chain.

## Setup

1. Create and activate a Python 3.10+ virtual environment.
2. Install the dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and set `OPENAI_API_KEY` to your own OpenAI API
   key. Do not commit `.env`.

## Run

Start the Streamlit app:

```powershell
streamlit run streamlit_app.py
```
The video must have English captions. Transcript retrieval, embeddings, and chat
responses require network access and may incur OpenAI API charges. `YoutubeLoader`
uses the YouTube transcript service underneath, so it may still be affected by
YouTube rate limits (HTTP 429); wait and try again later. The loader cannot
bypass that YouTube limit.
