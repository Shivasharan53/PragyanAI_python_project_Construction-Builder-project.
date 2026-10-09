# BuildWise — Construction Builder RAG

Streamlit app that crawls a small number of public HTML pages from a builder website, splits the extracted text into chunks, embeds the chunks, indexes them in FAISS, retrieves relevant passages, and uses Groq's chat-completions API to generate an answer grounded in those passages.

## Deploy directly through GitHub and Streamlit Community Cloud

1. Download and extract this ZIP.
2. Create a GitHub repository at https://github.com/new.
3. Upload `app.py`, `requirements.txt`, and `README.md` to the repository root.
4. Open https://share.streamlit.io/ and create an app from that repository.
5. Set the main file path to `app.py`, then deploy.
6. Create a Groq API key from https://console.groq.com/ (check the provider's current access and usage limits).
7. In Streamlit Cloud, open your app's **Settings → Secrets** and add:

```toml
GROQ_API_KEY = "paste_your_groq_api_key_here"
```

Save the secret and reboot/redeploy the app if required. Do not commit the key to GitHub.

## How to use

1. Open the app's sidebar.
2. Enter a builder's official website URL, e.g. `https://www.brigadegroup.com/`.
3. Select the maximum pages to crawl and click **Crawl & index website**.
4. Open **RAG Assistant** and ask a question about the indexed site.
5. Inspect the retrieved passages and source links shown under the answer.

## Stack

- Streamlit UI
- Requests + BeautifulSoup for basic HTML extraction
- `sentence-transformers/all-MiniLM-L6-v2` for lightweight sentence embeddings
- FAISS `IndexFlatIP` with normalized embeddings for cosine-similarity-style retrieval
- Groq API with `llama-3.3-70b-versatile` for answer generation

The lighter embedding model is chosen to be more practical for a cloud-hosted demo than a large embedding model. You can change `EMBEDDING_MODEL_NAME` in `app.py` if your deployment has sufficient memory.

## Important limitations

- This is a student-project MVP, not a production web crawler.
- It fetches a limited number of HTML pages from one domain; it does not execute JavaScript, bypass login, or access private pages. Some sites may block automated requests or serve content that cannot be extracted.
- Respect the site's terms, robots.txt, rate limits, and applicable copyright rules. Index only pages you are allowed to use.
- Indexes are kept in Streamlit session state and may be lost when the session/app restarts. Re-index the website when needed. For a persistent multi-user app, move indexes/documents to durable storage.
- RAG reduces unsupported answers but cannot guarantee perfect factual accuracy. Check the source passages for important claims.


## Property Finder UI additions

- Builder name dropdown
- Requirement / BHK dropdown
- City dropdown
- Property budget range slider
- Filtered property cards
- Sidebar builder selector with official website URL and one-click crawl/index for RAG

**Important:** Property records are illustrative demo data for layout and filter testing only, not verified live listings. Connect sourced, current inventory before presenting them as real properties.
