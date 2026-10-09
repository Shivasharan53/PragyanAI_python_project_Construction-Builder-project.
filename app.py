import os
import re
import time
import hashlib
from collections import deque
from urllib.parse import urljoin, urlparse, urldefrag

import faiss
import numpy as np
import requests
import streamlit as st
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="BuildWise | Construction RAG",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- Styling ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #f5f7fb; }
[data-testid="stSidebar"] { background: #101b2d; }
[data-testid="stSidebar"] * { color: #eaf0fa; }
[data-testid="stSidebar"] input { color: #172033 !important; }
.block-container { padding-top: 1.5rem; max-width: 1440px; }
.hero { background: linear-gradient(120deg,#14243b,#1c3d5a 62%,#176b70); padding:28px 30px; border-radius:20px; color:white; margin-bottom:22px; }
.hero h1 { font-family:'Manrope',sans-serif; font-size:2rem; font-weight:800; margin:0 0 8px; color:white; }
.hero p { color:#d7e5f3; margin:0; }
.eyebrow { color:#8ee4d1; font-size:.75rem; font-weight:700; letter-spacing:.15em; text-transform:uppercase; margin-bottom:8px; }
.metric { background:white; border:1px solid #e7ebf2; border-radius:16px; padding:17px 18px; min-height:105px; }
.metric-label { color:#6b778c; font-size:.83rem; font-weight:600; }
.metric-value { color:#17263d; font-family:'Manrope',sans-serif; font-size:1.8rem; font-weight:800; margin-top:8px; }
.section-title { font-family:'Manrope',sans-serif; font-size:1.18rem; color:#18283f; font-weight:800; margin:8px 0 3px; }
.section-sub { color:#78859a; font-size:.86rem; margin-bottom:15px; }
.panel { background:white; border:1px solid #e7ebf2; border-radius:16px; padding:20px; color:#17263d !important; }
.property-card { background:#ffffff; border:1px solid #dfe6f0; border-radius:18px; padding:22px; margin:4px 0 18px; min-height:255px; box-shadow:0 8px 24px rgba(20,36,59,.07); color:#17263d !important; }
.property-card * { color:#17263d !important; }
.property-tag { display:inline-block; background:#e8f7f3; color:#087e70 !important; padding:5px 10px; border-radius:20px; font-size:.72rem; font-weight:800; letter-spacing:.04em; margin-bottom:13px; }
.property-name { font-family:'Manrope',sans-serif; font-size:1.25rem; line-height:1.35; font-weight:800; margin-bottom:6px; color:#14243b !important; }
.property-builder { font-size:.94rem; font-weight:700; color:#476078 !important; margin-bottom:13px; }
.property-detail { font-size:.9rem; color:#526176 !important; margin:8px 0; }
.property-price { font-family:'Manrope',sans-serif; font-size:1.55rem; font-weight:800; color:#087e70 !important; margin:17px 0 12px; }
.property-link { color:#1769d2 !important; font-weight:700; text-decoration:none; }
.property-link:hover { text-decoration:underline; }
.small-muted { color:#7a879a; font-size:.78rem; }
div[data-testid="stChatMessage"] { background:white; border:1px solid #e7ebf2; border-radius:14px; }
.stButton button { border-radius:10px; font-weight:700; }
</style>
""", unsafe_allow_html=True)

# ---------------- Starter builder directory ----------------
COMPANIES = [
    {"name":"L&T Construction","type":"Infrastructure & Engineering","location":"Pan-India","url":"https://www.lntecc.com/"},
    {"name":"Prestige Group","type":"Residential & Commercial Real Estate","location":"Bengaluru, Karnataka","url":"https://www.prestigeltd.in/"},
    {"name":"Brigade Group","type":"Residential, Office & Hospitality","location":"Bengaluru, Karnataka","url":"https://www.brigadegroup.com/"},
    {"name":"SOBHA Limited","type":"Residential & Contractual Projects","location":"Bengaluru, Karnataka","url":"https://www.sobha.com/"},
    {"name":"Puravankara","type":"Residential Property Development","location":"Bengaluru, Karnataka","url":"https://www.puravankara.com/"},
    {"name":"Sattva Group","type":"Residential & Commercial Development","location":"Bengaluru, Karnataka","url":"https://sattvagroup.com/"},
    {"name":"Afcons Infrastructure","type":"Infrastructure & EPC","location":"Pan-India","url":"https://afcons.com/"},
    {"name":"Godrej Properties","type":"Residential & Commercial Real Estate","location":"Pan-India","url":"https://www.godrejproperties.com/"},
    {"name":"Shriram Properties","type":"Residential Property Development","location":"Bengaluru, Karnataka","url":"https://www.shriramproperties.com/"},
    {"name":"Assetz Property Group","type":"Residential & Commercial Real Estate","location":"Bengaluru, Karnataka","url":"https://www.assetzproperty.com/"},
]

# Demo records for the UI only; not verified live inventory.
PROPERTIES = [
    {"name":"Sample Residency","builder":"Prestige Group","city":"Bengaluru","bhk":2,"price_lakh":95,"area":"Whitefield","url":"https://www.prestigeltd.in/"},
    {"name":"Sample Heights","builder":"Prestige Group","city":"Bengaluru","bhk":3,"price_lakh":145,"area":"Sarjapur Road","url":"https://www.prestigeltd.in/"},
    {"name":"Sample Park","builder":"Brigade Group","city":"Bengaluru","bhk":2,"price_lakh":88,"area":"North Bengaluru","url":"https://www.brigadegroup.com/"},
    {"name":"Sample Enclave","builder":"Brigade Group","city":"Bengaluru","bhk":3,"price_lakh":135,"area":"Whitefield","url":"https://www.brigadegroup.com/"},
    {"name":"Sample Dream Acres","builder":"SOBHA Limited","city":"Bengaluru","bhk":2,"price_lakh":82,"area":"Panathur","url":"https://www.sobha.com/"},
    {"name":"Sample Signature","builder":"SOBHA Limited","city":"Bengaluru","bhk":3,"price_lakh":160,"area":"North Bengaluru","url":"https://www.sobha.com/"},
    {"name":"Sample Garden","builder":"Puravankara","city":"Bengaluru","bhk":2,"price_lakh":78,"area":"Electronic City","url":"https://www.puravankara.com/"},
    {"name":"Sample Vista","builder":"Puravankara","city":"Chennai","bhk":3,"price_lakh":115,"area":"South Chennai","url":"https://www.puravankara.com/"},
    {"name":"Sample Lakefront","builder":"Sattva Group","city":"Bengaluru","bhk":3,"price_lakh":125,"area":"Hebbal","url":"https://sattvagroup.com/"},
    {"name":"Sample Greenview","builder":"Godrej Properties","city":"Mumbai","bhk":2,"price_lakh":180,"area":"Eastern Suburbs","url":"https://www.godrejproperties.com/"},
    {"name":"Sample Gateway","builder":"Shriram Properties","city":"Bengaluru","bhk":2,"price_lakh":72,"area":"Yelahanka","url":"https://www.shriramproperties.com/"},
    {"name":"Sample Earth","builder":"Assetz Property Group","city":"Bengaluru","bhk":3,"price_lakh":155,"area":"Sarjapur Road","url":"https://www.assetzproperty.com/"},
]

USER_AGENT = "BuildWiseRAG/1.0 (educational project; respectful website indexing)"
REQUEST_TIMEOUT = 12
MAX_PAGES_PER_SITE = 8
MAX_CHARS_PER_PAGE = 18000
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 4
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

@st.cache_resource(show_spinner="Loading embedding model (first load can take a minute)...")
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

def normalize_url(url: str) -> str:
    url = url.strip()
    if not url.startswith(("https://", "http://")):
        url = "https://" + url
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("Enter a valid http:// or https:// website URL.")
    # Remove fragments and common tracking query strings. Keep path and query otherwise.
    clean, _ = urldefrag(url)
    return clean.rstrip("/") + "/"

def same_site(url: str, base_netloc: str) -> bool:
    try:
        host = urlparse(url).netloc.lower().split(":")[0]
        base = base_netloc.lower().split(":")[0]
        return host == base or host.endswith("." + base) or base.endswith("." + host)
    except Exception:
        return False

def fetch_page(url: str):
    headers = {"User-Agent": USER_AGENT}
    response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT, allow_redirects=True)
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "").lower()
    if "text/html" not in content_type:
        return "", []
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "nav", "footer", "header", "form", "iframe"]):
        tag.decompose()
    title = soup.title.get_text(" ", strip=True) if soup.title else url
    main = soup.find("main") or soup.find("article") or soup.body or soup
    text = re.sub(r"\s+", " ", main.get_text(" ", strip=True)).strip()
    links = []
    for a in soup.find_all("a", href=True):
        href = a.get("href", "").strip()
        if not href or href.startswith(("mailto:", "tel:", "javascript:")):
            continue
        absolute = urljoin(url, href)
        absolute, _ = urldefrag(absolute)
        parsed = urlparse(absolute)
        if parsed.scheme in ("http", "https") and parsed.netloc:
            # Drop query parameters to avoid crawling tracking variants.
            absolute = parsed._replace(query="", fragment="").geturl()
            links.append(absolute.rstrip("/") + "/")
    return f"Page title: {title}\nSource URL: {url}\n\n{text[:MAX_CHARS_PER_PAGE]}", links

def chunk_text(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        # Try to end on a sentence/space boundary.
        if end < len(text):
            boundary = max(text.rfind(". ", start, end), text.rfind("; ", start, end), text.rfind(" ", start, end))
            if boundary > start + int(size * 0.6):
                end = boundary + 1
        piece = text[start:end].strip()
        if len(piece) > 80:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks

def crawl_website(start_url: str, max_pages: int):
    start_url = normalize_url(start_url)
    base_netloc = urlparse(start_url).netloc
    queue = deque([start_url])
    seen = set()
    pages = []
    errors = []
    session_count = 0
    while queue and len(pages) < max_pages:
        url = queue.popleft()
        if url in seen:
            continue
        seen.add(url)
        # Only fetch pages from the starting host/domain.
        if not same_site(url, base_netloc):
            continue
        try:
            page_text, links = fetch_page(url)
            session_count += 1
            if page_text and len(page_text) > 100:
                pages.append({"url": url, "text": page_text})
            for link in links:
                if same_site(link, base_netloc) and link not in seen and len(seen) + len(queue) < max_pages * 12:
                    queue.append(link)
            time.sleep(0.15)  # Small pause between requests.
        except Exception as exc:
            errors.append(f"{url}: {str(exc)[:180]}")
    return pages, errors

def build_index(pages):
    model = load_embedding_model()
    chunks, metadata = [], []
    for page in pages:
        for idx, chunk in enumerate(chunk_text(page["text"])):
            chunks.append(chunk)
            metadata.append({"source": page["url"], "chunk": idx})
    if not chunks:
        raise ValueError("No usable text was extracted. Try a different official website URL.")
    vectors = model.encode(
        chunks,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype("float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return {"index": index, "chunks": chunks, "metadata": metadata, "pages": pages}

def groq_answer(question: str, retrieved):
    api_key = st.secrets.get("GROQ_API_JNTU_KEY", "") if hasattr(st, "secrets") else ""
    api_key = api_key or os.getenv("GROQ_API_JNTU_KEY", "")
    if not api_key:
        raise RuntimeError("GROQ_API_JNTU_KEY is missing. Add it in Streamlit Cloud → App settings → Secrets.")
    context = "\n\n".join(
        f"[Source {i+1}] URL: {item['source']}\nContent: {item['text']}"
        for i, item in enumerate(retrieved)
    )
    system_prompt = (
        "You are BuildWise, a careful construction-company research assistant. "
        "Answer using only the supplied retrieved website context. If the context does not contain the answer, "
        "say you could not find it in the indexed pages. Do not invent project names, prices, addresses, approvals, "
        "phone numbers, or claims. Cite supporting sources inline as [Source 1], [Source 2]. "
        "Keep the answer clear and useful."
    )
    payload = {
        "model": "llama-3.3-70b-versatile",
        "temperature": 0.1,
        "max_tokens": 900,
        "messages": [
            {"role":"system","content":system_prompt},
            {"role":"user","content":f"Retrieved website context:\n{context}\n\nQuestion: {question}\n\nAnswer only from the context and cite sources."}
        ],
    }
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type":"application/json"},
        json=payload,
        timeout=60,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"Groq API error ({response.status_code}): {response.text[:500]}")
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()

def retrieve(question: str, store, top_k=TOP_K):
    model = load_embedding_model()
    query_vec = model.encode([question], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    k = min(top_k, len(store["chunks"]))
    scores, ids = store["index"].search(query_vec, k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        results.append({
            "text": store["chunks"][idx],
            "source": store["metadata"][idx]["source"],
            "score": float(score),
        })
    return results

# ---------------- State ----------------
if "knowledge_stores" not in st.session_state:
    st.session_state.knowledge_stores = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_retrieved" not in st.session_state:
    st.session_state.last_retrieved = []

# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("## 🏗️ BuildWise")
    st.caption("Construction intelligence · RAG workspace")
    page = st.radio("WORKSPACE", ["Property Finder", "Overview", "Builder Directory", "Knowledge Sources", "RAG Assistant"])
    st.markdown("---")
    st.markdown("### Add / index a builder website")
    builder_options = ["Select builder"] + [c["name"] for c in COMPANIES] + ["Other / custom website"]
    sidebar_builder = st.selectbox("Builder name", builder_options, key="sidebar_builder")
    default_url = next((c["url"] for c in COMPANIES if c["name"] == sidebar_builder), "")
    source_url = st.text_input("Website URL", value=default_url, placeholder="https://example.com", key="sidebar_url")
    max_pages = st.slider("Maximum pages to crawl", min_value=1, max_value=MAX_PAGES_PER_SITE, value=5)
    if st.button("＋ Add website & index", type="primary", use_container_width=True):
        if source_url.strip():
            try:
                normalized = normalize_url(source_url)
                with st.spinner("Crawling pages and building the vector index..."):
                    pages, crawl_errors = crawl_website(normalized, max_pages=max_pages)
                    if not pages:
                        st.error("No readable pages were found. The site may block automated requests or render content with JavaScript.")
                    else:
                        store = build_index(pages)
                        domain = urlparse(normalized).netloc
                        st.session_state.knowledge_stores[domain] = store
                        if "indexed_builder_names" not in st.session_state:
                            st.session_state.indexed_builder_names = {}
                        st.session_state.indexed_builder_names[domain] = sidebar_builder if sidebar_builder not in ("Select builder", "Other / custom website") else domain
                        st.success(f"Indexed {len(pages)} pages and {len(store['chunks'])} chunks.")
                        if crawl_errors:
                            st.caption(f"{len(crawl_errors)} page(s) could not be read; other pages were indexed.")
            except Exception as exc:
                st.error(str(exc))
        else:
            st.warning("Choose a builder and enter a website URL.")
    st.markdown("---")
    st.caption("Indexes are kept in this session. Re-index after app restarts. Respect website terms and rate limits.")

# ---------------- Header ----------------
st.markdown("""
<div class="hero">
  <div class="eyebrow">Construction intelligence workspace</div>
  <h1>Find the right builder. Faster.</h1>
  <p>Index official builder websites, retrieve relevant information, and ask grounded questions using Retrieval-Augmented Generation.</p>
</div>
""", unsafe_allow_html=True)

all_stores = st.session_state.knowledge_stores
total_pages = sum(len(s["pages"]) for s in all_stores.values())
total_chunks = sum(len(s["chunks"]) for s in all_stores.values())

if page == "Property Finder":
    st.markdown('<div class="section-title">Find your property</div><div class="section-sub">Choose a builder, BHK requirement, city, and budget range.</div>', unsafe_allow_html=True)
    f1, f2, f3, f4 = st.columns([1.25, 1, 1, 1.35])
    with f1:
        selected_builder = st.selectbox("Builder name", ["All builders"] + [c["name"] for c in COMPANIES], key="property_builder")
    with f2:
        selected_bhk = st.selectbox("Requirement · BHK", ["Any", "1 BHK", "2 BHK", "3 BHK", "4 BHK", "5+ BHK"], key="property_bhk")
    with f3:
        selected_city = st.selectbox("Select city", ["Any city", "Bengaluru", "Chennai", "Hyderabad", "Mumbai", "Pune", "Delhi NCR"], key="property_city")
    with f4:
        budget = st.select_slider("Property range (₹ lakh)", options=[25, 50, 75, 100, 150, 200, 300, 500], value=(50, 200), key="property_budget")

    filtered_properties = []
    for prop in PROPERTIES:
        if selected_builder != "All builders" and prop["builder"] != selected_builder:
            continue
        if selected_bhk != "Any":
            requested_bhk = 5 if selected_bhk == "5+ BHK" else int(selected_bhk.split()[0])
            if prop["bhk"] != requested_bhk:
                continue
        if selected_city != "Any city" and prop["city"] != selected_city:
            continue
        if not (budget[0] <= prop["price_lakh"] <= budget[1]):
            continue
        filtered_properties.append(prop)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f'<div class="metric"><div class="metric-label">Matching properties</div><div class="metric-value">{len(filtered_properties)}</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric"><div class="metric-label">Selected builder</div><div class="metric-value" style="font-size:1.15rem;">{selected_builder}</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric"><div class="metric-label">Budget range</div><div class="metric-value" style="font-size:1.4rem;">₹{budget[0]}L–₹{budget[1]}L</div></div>', unsafe_allow_html=True)

    st.markdown("")
    st.markdown('<div class="section-title">Property results</div><div class="section-sub">Demo property records for UI testing only — not verified live listings or current offers.</div>', unsafe_allow_html=True)
    if filtered_properties:
        cols = st.columns(3)
        for i, prop in enumerate(filtered_properties):
            with cols[i % 3]:
                st.markdown(f"""<div class="property-card">
                  <div class="property-tag">🏠 PROPERTY</div>
                  <div class="property-name">{prop['name']}</div>
                  <div class="property-builder">{prop['builder']}</div>
                  <div class="property-detail">📍 {prop['area']}, {prop['city']}</div>
                  <div class="property-detail">🛏️ {prop['bhk']} BHK · Apartment</div>
                  <div class="property-price">₹{prop['price_lakh']} lakh</div>
                  <a class="property-link" href="{prop['url']}" target="_blank" rel="noopener noreferrer">Visit builder website ↗</a>
                </div>""", unsafe_allow_html=True)
    else:
        st.info("No sample properties match these filters. Try a wider budget or choose Any city / Any BHK.")

    st.markdown("---")
    st.markdown("**Add website for RAG:** use the sidebar to select a builder, confirm the website URL, and click **Add website & index**. Then ask questions in RAG Assistant.")

if page == "Property Finder":
    st.markdown('<div class="section-title">Find your property</div><div class="section-sub">Choose a builder, BHK requirement, city, and budget range.</div>', unsafe_allow_html=True)
    f1, f2, f3, f4 = st.columns([1.25, 1, 1, 1.35])
    with f1:
        selected_builder = st.selectbox("Builder name", ["All builders"] + [c["name"] for c in COMPANIES], key="property_builder")
    with f2:
        selected_bhk = st.selectbox("Requirement · BHK", ["Any", "1 BHK", "2 BHK", "3 BHK", "4 BHK", "5+ BHK"], key="property_bhk")
    with f3:
        selected_city = st.selectbox("Select city", ["Any city", "Bengaluru", "Chennai", "Hyderabad", "Mumbai", "Pune", "Delhi NCR"], key="property_city")
    with f4:
        budget = st.select_slider("Property range (₹ lakh)", options=[25, 50, 75, 100, 150, 200, 300, 500], value=(50, 200), key="property_budget")

    filtered_properties = []
    for prop in PROPERTIES:
        if selected_builder != "All builders" and prop["builder"] != selected_builder:
            continue
        if selected_bhk != "Any":
            requested_bhk = 5 if selected_bhk == "5+ BHK" else int(selected_bhk.split()[0])
            if prop["bhk"] != requested_bhk:
                continue
        if selected_city != "Any city" and prop["city"] != selected_city:
            continue
        if not (budget[0] <= prop["price_lakh"] <= budget[1]):
            continue
        filtered_properties.append(prop)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f'<div class="metric"><div class="metric-label">Matching properties</div><div class="metric-value">{len(filtered_properties)}</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric"><div class="metric-label">Selected builder</div><div class="metric-value" style="font-size:1.15rem;">{selected_builder}</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric"><div class="metric-label">Budget range</div><div class="metric-value" style="font-size:1.4rem;">₹{budget[0]}L–₹{budget[1]}L</div></div>', unsafe_allow_html=True)

    st.markdown("")
    st.markdown('<div class="section-title">Property results</div><div class="section-sub">Demo property records for UI testing only — not verified live listings.</div>', unsafe_allow_html=True)
    if filtered_properties:
        cols = st.columns(3)
        for i, prop in enumerate(filtered_properties):
            with cols[i % 3]:
                st.markdown(f"""<div class="property-card">
                  <div class="property-tag">🏠 PROPERTY</div>
                  <div class="property-name">{prop['name']}</div>
                  <div class="property-builder">{prop['builder']}</div>
                  <div class="property-detail">📍 {prop['area']}, {prop['city']}</div>
                  <div class="property-detail">🛏️ {prop['bhk']} BHK · Apartment</div>
                  <div class="property-price">₹{prop['price_lakh']} lakh</div>
                  <a class="property-link" href="{prop['url']}" target="_blank" rel="noopener noreferrer">Visit builder website ↗</a>
                </div>""", unsafe_allow_html=True)
    else:
        st.info("No sample properties match these filters. Try a wider budget or choose Any city / Any BHK.")

    st.markdown("---")
    st.markdown("**Add website for RAG:** use the sidebar to select a builder, confirm the website URL, and click **Add website & index**. Then ask questions in RAG Assistant.")

if page == "Overview":
    st.markdown('<div class="section-title">Workspace overview</div><div class="section-sub">Live status of your current RAG session.</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (c1, "Builder profiles", str(len(COMPANIES))),
        (c2, "Indexed websites", str(len(all_stores))),
        (c3, "Pages indexed", str(total_pages)),
        (c4, "Vector chunks", str(total_chunks)),
    ]
    for col, label, value in cards:
        with col:
            st.markdown(f'<div class="metric"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)
    st.write("")
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown('<div class="panel"><div class="section-title">Starter builder directory</div><div class="section-sub">Use official sites as starting points for your knowledge base.</div>', unsafe_allow_html=True)
        for company in COMPANIES[:6]:
            st.markdown(f"- **{company['name']}** — {company['type']} · [Official website]({company['url']})")
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel"><div class="section-title">RAG pipeline</div><div class="section-sub">The connected retrieval flow</div>', unsafe_allow_html=True)
        for n, title, desc in [
            ("01","Crawl","Fetch a small set of public pages from one domain"),
            ("02","Chunk + embed","Split text and generate sentence embeddings"),
            ("03","FAISS retrieval","Find chunks most similar to the question"),
            ("04","Grounded generation","Groq LLM answers from retrieved context and cites sources"),
        ]:
            st.markdown(f"**{n} · {title}**  \n<span class='small-muted'>{desc}</span>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    st.info("To begin, enter an official builder website in the sidebar and click **Crawl & index website**. Add your Groq key in Streamlit Secrets before asking questions.")

elif page == "Builder Directory":
    st.markdown('<div class="section-title">Builder directory</div><div class="section-sub">Browse starter companies and open their official websites.</div>', unsafe_allow_html=True)
    query = st.text_input("Search companies", placeholder="Company, location, or construction type")
    filtered = [c for c in COMPANIES if query.lower() in f"{c['name']} {c['type']} {c['location']}".lower()]
    cols = st.columns(2)
    for i, company in enumerate(filtered):
        with cols[i % 2]:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown(f"### 🏢 {company['name']}")
            st.caption(f"{company['type']} · {company['location']}")
            st.markdown(f"[Visit official website ↗]({company['url']})")
            st.markdown("</div>", unsafe_allow_html=True)
            st.write("")
    if not filtered:
        st.warning("No matching companies.")

elif page == "Knowledge Sources":
    st.markdown('<div class="section-title">Knowledge sources</div><div class="section-sub">Index websites using the sidebar form. The current vector indexes live in this app session.</div>', unsafe_allow_html=True)
    if all_stores:
        for domain, store in all_stores.items():
            with st.expander(f"🌐 {domain} — {len(store['pages'])} pages, {len(store['chunks'])} chunks", expanded=True):
                for p in store["pages"]:
                    st.markdown(f"- [{p['url']}]({p['url']})")
                if st.button(f"Remove index for {domain}", key=f"remove_{domain}"):
                    del st.session_state.knowledge_stores[domain]
                    st.rerun()
    else:
        st.info("No websites indexed yet. Add an official builder URL from the sidebar.")
    st.warning("This MVP keeps FAISS indexes in Streamlit session state. For durable multi-user storage, connect a persistent database or object storage later.")

elif page == "RAG Assistant":
    st.markdown('<div class="section-title">Builder RAG assistant</div><div class="section-sub">Answers are generated from retrieved website chunks, not from the LLM alone.</div>', unsafe_allow_html=True)
    if not all_stores:
        st.warning("First index a builder website using the sidebar.")
    else:
        domains = list(all_stores.keys())
        selected_domain = st.selectbox("Search within indexed website", ["All indexed websites"] + domains)
        example_cols = st.columns(3)
        examples = [
            "What does this company do?",
            "What projects or services are described on the website?",
            "What contact or location information can be found?",
        ]
        for i, prompt in enumerate(examples):
            with example_cols[i]:
                if st.button(prompt, key=f"example_{i}", use_container_width=True):
                    st.session_state.pending_question = prompt

        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        question = st.chat_input("Ask about projects, services, locations, or company details...")
        if st.session_state.get("pending_question"):
            question = st.session_state.pending_question
            st.session_state.pending_question = ""
        if question:
            st.session_state.chat_history.append({"role":"user","content":question})
            with st.chat_message("user"):
                st.markdown(question)
            try:
                candidate_stores = all_stores if selected_domain == "All indexed websites" else {selected_domain: all_stores[selected_domain]}
                retrieved = []
                for domain, store in candidate_stores.items():
                    for item in retrieve(question, store, top_k=TOP_K):
                        item["domain"] = domain
                        retrieved.append(item)
                retrieved.sort(key=lambda x: x["score"], reverse=True)
                retrieved = retrieved[:TOP_K]
                if not retrieved or max(x["score"] for x in retrieved) < 0.15:
                    answer = "I couldn't find relevant information for that question in the indexed pages. Try indexing more pages or ask about information shown on the website."
                else:
                    answer = groq_answer(question, retrieved)
                    st.session_state.last_retrieved = retrieved
                st.session_state.chat_history.append({"role":"assistant","content":answer})
                with st.chat_message("assistant"):
                    st.markdown(answer)
                    if retrieved:
                        with st.expander("Retrieved source passages"):
                            for i, item in enumerate(retrieved, 1):
                                st.markdown(f"**Source {i} · Similarity {item['score']:.3f}**")
                                st.markdown(f"[{item['source']}]({item['source']})")
                                st.write(item["text"][:1200] + ("…" if len(item["text"]) > 1200 else ""))
            except Exception as exc:
                st.error(f"Could not generate an answer: {exc}")
                st.info("Check that GROQ_API_JNTU_KEY is configured in Streamlit Cloud Secrets and that the API quota is available.")

st.markdown("---")
st.markdown("<div style='text-align:center;color:#8a96a8;font-size:.78rem;'>BuildWise · Construction Builder RAG · Streamlit</div>", unsafe_allow_html=True)
