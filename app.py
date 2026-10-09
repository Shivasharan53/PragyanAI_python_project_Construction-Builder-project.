import os
import re
import time
from collections import deque
from urllib.parse import urlparse, urljoin, urldefrag, quote_plus

import faiss
import folium
import numpy as np
import requests
import streamlit as st
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from streamlit_folium import st_folium


# ============================================================
# 1. CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="BuildWise | Construction RAG",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

USER_AGENT = "BuildWiseConstructionResearch/1.0"
REQUEST_TIMEOUT = 15
MAX_CHARS_PER_PAGE = 18000
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 4
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CITIES = [
    "Bengaluru", "Mysuru", "Mangaluru", "Hubballi", "Dharwad",
    "Belagavi", "Kalaburagi", "Tumakuru", "Davanagere",
    "Shivamogga", "Ballari", "Hosapete", "Raichur", "Bidar",
    "Vijayapura", "Bagalkote", "Gadag", "Haveri", "Hassan",
    "Mandya", "Udupi", "Chikkamagaluru", "Chitradurga", "Kolar",
    "Ramanagara", "Yadgir", "Koppal", "Madikeri", "Karwar",
    "Chikkaballapura",
]

COMPANIES = [
    {
        "name": "L&T Construction",
        "type": "Infrastructure & Engineering",
        "location": "Pan-India",
        "url": "https://www.lntecc.com/",
    },
    {
        "name": "Prestige Group",
        "type": "Residential & Commercial Real Estate",
        "location": "Bengaluru, Karnataka",
        "url": "https://www.prestigeltd.in/",
    },
    {
        "name": "Brigade Group",
        "type": "Residential, Office & Hospitality",
        "location": "Bengaluru, Karnataka",
        "url": "https://www.brigadegroup.com/",
    },
    {
        "name": "SOBHA Limited",
        "type": "Residential & Contractual Projects",
        "location": "Bengaluru, Karnataka",
        "url": "https://www.sobha.com/",
    },
    {
        "name": "Puravankara",
        "type": "Residential Property Development",
        "location": "Bengaluru, Karnataka",
        "url": "https://www.puravankara.com/",
    },
    {
        "name": "Sattva Group",
        "type": "Residential & Commercial Development",
        "location": "Bengaluru, Karnataka",
        "url": "https://sattvagroup.com/",
    },
    {
        "name": "Afcons Infrastructure",
        "type": "Infrastructure & EPC",
        "location": "Pan-India",
        "url": "https://afcons.com/",
    },
    {
        "name": "Godrej Properties",
        "type": "Residential & Commercial Real Estate",
        "location": "Pan-India",
        "url": "https://www.godrejproperties.com/",
    },
]

# ============================================================
# 2. CUSTOM DASHBOARD STYLING
# ============================================================

st.markdown(
    """
    <style>
    @import url(
      'https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap'
    );

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background: #f4f7fb;
        color: #17263d;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    [data-testid="stSidebar"] {
        background: #101b2d;
    }

    [data-testid="stSidebar"] * {
        color: #eaf0fa;
    }

    [data-testid="stSidebar"] input {
        color: #17263d !important;
    }

    .hero {
        background: linear-gradient(
            120deg, #14243b, #1c3d5a 62%, #176b70
        );
        padding: 30px;
        border-radius: 20px;
        color: white;
        margin-bottom: 24px;
    }

    .hero h1 {
        color: white !important;
        font-family: 'Manrope', sans-serif;
        font-weight: 800;
        font-size: 2rem;
        margin-bottom: 8px;
    }

    .hero p {
        color: #d7e5f3 !important;
        margin-bottom: 0;
    }

    .eyebrow {
        color: #8ee4d1;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.15em;
        margin-bottom: 8px;
    }

    .section-title {
        color: #17263d !important;
        font-family: 'Manrope', sans-serif;
        font-size: 1.25rem;
        font-weight: 800;
        margin: 12px 0 5px;
    }

    .section-sub {
        color: #65748b !important;
        margin-bottom: 18px;
        font-size: 0.9rem;
    }

    .metric-card {
        background: #ffffff;
        border: 1px solid #e1e8f0;
        border-radius: 16px;
        padding: 20px;
        min-height: 112px;
    }

    .metric-label {
        color: #64748b !important;
        font-size: 0.85rem;
    }

    .metric-value {
        color: #17263d !important;
        font-family: 'Manrope', sans-serif;
        font-size: 1.8rem;
        font-weight: 800;
        margin-top: 8px;
    }

    .property-card {
        background: #ffffff !important;
        border: 1px solid #dce5ef;
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 18px;
        min-height: 180px;
        box-shadow: 0 4px 14px rgba(20, 36, 59, 0.05);
    }

    .property-name {
        color: #17263d !important;
        font-family: 'Manrope', sans-serif;
        font-size: 1.15rem;
        font-weight: 800;
        margin: 8px 0;
    }

    .property-detail {
        color: #334155 !important;
        font-size: 0.9rem;
        margin: 8px 0;
    }

    .panel {
        background: white;
        border: 1px solid #e1e8f0;
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 14px;
    }

    a {
        color: #0878d1 !important;
    }

    .stButton button {
        border-radius: 10px;
        font-weight: 700;
    }

    div[data-testid="stChatMessage"] {
        background: white;
        border: 1px solid #e1e8f0;
        border-radius: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. SESSION STATE
# ============================================================

if "knowledge_stores" not in st.session_state:
    st.session_state.knowledge_stores = {}

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = ""

if "current_map_builders" not in st.session_state:
    st.session_state.current_map_builders = []

if "current_map_city" not in st.session_state:
    st.session_state.current_map_city = ""


# ============================================================
# 4. HUGGING FACE EMBEDDINGS
# ============================================================

@st.cache_resource(show_spinner="Loading Hugging Face embedding model...")
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


# ============================================================
# 5. FREE OPENSTREETMAP CITY SEARCH
# ============================================================

@st.cache_data(ttl=1800, show_spinner=False)
def get_city_coordinates(city):
    response = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={
            "q": f"{city}, Karnataka, India",
            "format": "jsonv2",
            "limit": 1,
        },
        headers={"User-Agent": USER_AGENT},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    results = response.json()

    if not results:
        return None

    return float(results[0]["lat"]), float(results[0]["lon"])


@st.cache_data(ttl=1800, show_spinner=False)
def get_city_builders(city):
    coords = get_city_coordinates(city)

    if not coords:
        return [], "Could not locate the selected city."

    lat, lon = coords

    query = f"""
    [out:json][timeout:20];
    (
      nwr(around:15000,{lat},{lon})["craft"="builder"];
      nwr(around:15000,{lat},{lon})["office"="construction"];
      nwr(around:15000,{lat},{lon})["industrial"="construction"];
    );
    out center tags;
    """

    endpoints = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://overpass.private.coffee/api/interpreter",
    ]

    errors = []

    for endpoint in endpoints:
        try:
            response = requests.post(
                endpoint,
                data={"data": query},
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "application/json",
                },
                timeout=(8, 30),
            )

            if response.status_code in (429, 500, 502, 503, 504):
                errors.append(
                    f"{urlparse(endpoint).netloc}: "
                    f"HTTP {response.status_code}"
                )
                continue

            response.raise_for_status()
            data = response.json()

            builders = []
            seen = set()

            for item in data.get("elements", []):
                tags = item.get("tags", {})
                name = tags.get("name", "").strip()

                if not name or name.casefold() in seen:
                    continue

                seen.add(name.casefold())

                center = item.get("center", {})
                item_lat = item.get("lat", center.get("lat"))
                item_lon = item.get("lon", center.get("lon"))

                address_parts = [
                    tags.get("addr:street"),
                    tags.get("addr:city"),
                    tags.get("addr:postcode"),
                ]

                address = ", ".join(
                    part for part in address_parts if part
                )

                builders.append({
                    "name": name,
                    "address": address or city,
                    "phone": (
                        tags.get("contact:phone")
                        or tags.get("phone", "")
                    ),
                    "website": (
                        tags.get("contact:website")
                        or tags.get("website", "")
                    ),
                    "lat": item_lat,
                    "lon": item_lon,
                })

            return builders, None

        except (requests.RequestException, ValueError, TypeError) as exc:
            errors.append(
                f"{urlparse(endpoint).netloc}: {str(exc)[:120]}"
            )

    return [], (
        "Public map services are temporarily unavailable. "
        + " | ".join(errors)
    )


def render_builder_map(city, builders):
    coords = get_city_coordinates(city)

    if not coords:
        st.warning("Could not load coordinates for this city.")
        return

    builder_map = folium.Map(
        location=list(coords),
        zoom_start=12,
        tiles="OpenStreetMap",
    )

    folium.Marker(
        list(coords),
        tooltip=city,
        popup=f"City centre: {city}",
        icon=folium.Icon(color="blue", icon="home"),
    ).add_to(builder_map)

    plotted = 0

    for builder in builders:
        lat = builder.get("lat")
        lon = builder.get("lon")

        if lat is None or lon is None:
            continue

        popup_html = (
            f"<b>{builder['name']}</b><br>"
            f"{builder['address']}"
        )

        folium.Marker(
            [lat, lon],
            tooltip=builder["name"],
            popup=folium.Popup(popup_html, max_width=300),
            icon=folium.Icon(color="green", icon="building"),
        ).add_to(builder_map)

        plotted += 1

    st_folium(
        builder_map,
        use_container_width=True,
        height=450,
        key=f"builder_map_{city}",
    )

    st.caption(f"{plotted} builder locations plotted on the map.")


# ============================================================
# 6. RAG WEBSITE FETCHING
# ============================================================

def normalize_url(url):
    url = url.strip()

    if not url:
        raise ValueError("Enter a website URL.")

    if not url.startswith(("https://", "http://")):
        url = "https://" + url

    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("Enter a valid website URL.")

    clean, _ = urldefrag(url)
    return clean.rstrip("/") + "/"


def fetch_page(url):
    response = requests.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "").lower()

    if "text/html" not in content_type:
        return "", []

    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup([
        "script", "style", "noscript", "svg",
        "nav", "footer", "header", "form", "iframe",
    ]):
        tag.decompose()

    title = soup.title.get_text(" ", strip=True) if soup.title else url
    main = soup.find("main") or soup.find("article") or soup.body or soup

    text = re.sub(
        r"\s+", " ", main.get_text(" ", strip=True)
    ).strip()

    links = []
    base_host = urlparse(url).netloc.lower()

    for anchor in soup.find_all("a", href=True):
        href = anchor.get("href", "").strip()

        if href.startswith(("mailto:", "tel:", "javascript:")):
            continue

        absolute = urljoin(url, href)
        absolute, _ = urldefrag(absolute)
        parsed = urlparse(absolute)

        if (
            parsed.scheme in ("http", "https")
            and parsed.netloc.lower() == base_host
        ):
            absolute = parsed._replace(query="", fragment="").geturl()
            links.append(absolute.rstrip("/") + "/")

    page_text = (
        f"Page title: {title}\n"
        f"Source URL: {url}\n\n"
        f"{text[:MAX_CHARS_PER_PAGE]}"
    )

    return page_text, links


def crawl_website(start_url, max_pages):
    start_url = normalize_url(start_url)
    base_host = urlparse(start_url).netloc.lower()

    queue = deque([start_url])
    seen = set()
    pages = []
    errors = []

    while queue and len(pages) < max_pages:
        url = queue.popleft()

        if url in seen:
            continue

        seen.add(url)

        if urlparse(url).netloc.lower() != base_host:
            continue

        try:
            page_text, links = fetch_page(url)

            if page_text and len(page_text) > 100:
                pages.append({
                    "url": url,
                    "text": page_text,
                })

            for link in links:
                if (
                    urlparse(link).netloc.lower() == base_host
                    and link not in seen
                    and len(queue) < max_pages * 10
                ):
                    queue.append(link)

            time.sleep(0.2)

        except Exception as exc:
            errors.append(f"{url}: {str(exc)[:120]}")

    return pages, errors


# ============================================================
# 7. TEXT CHUNKING AND FAISS INDEX
# ============================================================

def chunk_text(text):
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(len(text), start + CHUNK_SIZE)

        if end < len(text):
            boundary = text.rfind(" ", start, end)

            if boundary > start + CHUNK_SIZE * 0.6:
                end = boundary

        piece = text[start:end].strip()

        if len(piece) > 80:
            chunks.append(piece)

        if end >= len(text):
            break

        start = max(end - CHUNK_OVERLAP, start + 1)

    return chunks


def build_index(pages):
    model = load_embedding_model()
    chunks = []
    metadata = []

    for page in pages:
        for chunk in chunk_text(page["text"]):
            chunks.append(chunk)
            metadata.append({"source": page["url"]})

    if not chunks:
        raise ValueError(
            "No usable text found. Try another official website."
        )

    vectors = model.encode(
        chunks,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype("float32")

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    return {
        "index": index,
        "chunks": chunks,
        "metadata": metadata,
        "pages": pages,
    }


def retrieve(question, store):
    model = load_embedding_model()

    vector = model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    k = min(TOP_K, len(store["chunks"]))
    scores, ids = store["index"].search(vector, k)

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


# ============================================================
# 8. GROQ API
# ============================================================

def groq_answer(question, retrieved):
    try:
        api_key = st.secrets.get("GROQ_API_JNTU_KEY", "")
    except Exception:
        api_key = ""

    api_key = api_key or os.getenv("GROQ_API_JNTU_KEY", "")

    if not api_key:
        raise RuntimeError(
            "Add GROQ_API_JNTU_KEY to Streamlit Cloud Secrets."
        )

    context = "\n\n".join(
        f"[Source {i + 1}] URL: {item['source']}\n"
        f"Content: {item['text']}"
        for i, item in enumerate(retrieved)
    )

    payload = {
        "model": "llama-3.3-70b-versatile",
        "temperature": 0.1,
        "max_tokens": 900,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are BuildWise, a construction research assistant. "
                    "Answer using only the retrieved website context. "
                    "Do not invent property prices, projects, addresses, "
                    "approvals, or contact information. If the answer is "
                    "missing, say so. Cite evidence as [Source 1], etc."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question: {question}\n\n"
                    "Answer using the context and cite sources."
                ),
            },
        ],
    }

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=60,
    )

    if response.status_code >= 400:
        raise RuntimeError(
            f"Groq API error ({response.status_code}). "
            "Check your key, model availability, and API quota."
        )

    return response.json()["choices"][0]["message"]["content"].strip()


# ============================================================
# 9. SIDEBAR NAVIGATION AND WEBSITE INDEXING
# ============================================================

with st.sidebar:
    st.markdown("## 🏗️ BuildWise")
    st.caption("Construction intelligence · RAG workspace")

    page = st.radio(
        "WORKSPACE",
        [
            "Property Finder",
            "Overview",
            "Builder Directory",
            "Knowledge Sources",
            "RAG Assistant",
        ],
        key="workspace_page",
    )

    st.divider()
    st.markdown("### Add / index a builder website")

    builder_names = [company["name"] for company in COMPANIES]

    sidebar_builder = st.selectbox(
        "Builder name",
        builder_names,
        key="sidebar_builder_select",
    )

    selected_company = next(
        company for company in COMPANIES
        if company["name"] == sidebar_builder
    )

    source_url = st.text_input(
        "Website URL",
        value=selected_company["url"],
        key="sidebar_website_url",
    )

    max_pages = st.slider(
        "Maximum pages to crawl",
        min_value=1,
        max_value=8,
        value=5,
        key="crawl_page_limit",
    )

    if st.button(
        "＋ Add website & index",
        type="primary",
        use_container_width=True,
        key="index_website_button",
    ):
        try:
            normalized = normalize_url(source_url)

            with st.spinner("Reading website and building index..."):
                pages, errors = crawl_website(normalized, max_pages)

                if not pages:
                    st.error(
                        "No readable pages found. The website may block "
                        "automated requests or use JavaScript rendering."
                    )
                else:
                    store = build_index(pages)
                    domain = urlparse(normalized).netloc

                    st.session_state.knowledge_stores[domain] = store
                    st.session_state.chat_history = []

                    st.success(
                        f"Indexed {len(pages)} pages and "
                        f"{len(store['chunks'])} text chunks."
                    )

                    if errors:
                        st.caption(
                            f"{len(errors)} page(s) could not be read."
                        )

        except Exception as exc:
            st.error(str(exc))

    st.divider()
    st.caption(
        "Indexes are held in the current session. Respect website terms "
        "and crawling restrictions."
    )


# ============================================================
# 10. MAIN HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">CONSTRUCTION INTELLIGENCE WORKSPACE</div>
        <h1>Find the right builder. Faster.</h1>
        <p>
            Explore property requirements, find builders on a map,
            index official websites, and ask grounded questions using RAG.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 11. PROPERTY FINDER — NO FAKE PROPERTY RECORDS
# ============================================================

if page == "Property Finder":

    st.markdown(
        '<div class="section-title">Find your property</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-sub">'
        'Choose your requirements and open a property portal.'
        '</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns([1.25, 1, 1, 1.2])

    with col1:
        selected_builder = st.selectbox(
            "Builder name",
            ["Any builder"] + builder_names,
            key="property_builder_filter",
        )

    with col2:
        selected_bhk = st.selectbox(
            "Requirement · BHK",
            ["Any BHK", "1 BHK", "2 BHK", "3 BHK", "4 BHK"],
            key="property_bhk_filter",
        )

    with col3:
        selected_city = st.selectbox(
            "Select city",
            CITIES,
            key="property_city_filter",
        )

    with col4:
        budget = st.slider(
            "Budget range (₹ lakh)",
            min_value=5,
            max_value=500,
            value=(25, 200),
            step=5,
            key="property_budget_filter",
        )

    m1, m2, m3 = st.columns(3)

    with m1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Property data</div>
                <div class="metric-value">Portal search</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Selected city</div>
                <div class="metric-value" style="font-size:1.25rem">
                    {selected_city}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Budget range</div>
                <div class="metric-value" style="font-size:1.25rem">
                    ₹{budget[0]}L–₹{budget[1]}L
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown(
        '<div class="section-title">Property portals</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Your selected filters are shown below. On the portal, enter "
        "your city, BHK, builder and budget to see its current listings."
    )

    st.markdown(
        f"""
        <div class="panel">
            <b>Search preferences</b><br><br>
            City: {selected_city}<br>
            Builder: {selected_builder}<br>
            Requirement: {selected_bhk}<br>
            Budget: ₹{budget[0]} lakh – ₹{budget[1]} lakh
        </div>
        """,
        unsafe_allow_html=True,
    )

    portals = [
        ("99acres", "https://www.99acres.com/"),
        ("Magicbricks", "https://www.magicbricks.com/"),
        ("Housing.com", "https://housing.com/"),
        ("NoBroker", "https://www.nobroker.in/"),
    ]

    portal_columns = st.columns(4)

    for col, (name, url) in zip(portal_columns, portals):
        with col:
            st.markdown(
                f"""
                <div class="property-card">
                    <div style="font-size:2rem">🏠</div>
                    <div class="property-name">{name}</div>
                    <div class="property-detail">{selected_city}</div>
                    <div class="property-detail">{selected_bhk}</div>
                    <div class="property-detail">
                        ₹{budget[0]}L–₹{budget[1]}L
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.link_button(
                f"Open {name}",
                url,
                use_container_width=True,
            )

    st.info(
        "No fake property records are shown. These links open the actual "
        "portals; BuildWise does not yet have a structured live property "
        "feed, so individual prices and availability are not verified."
    )


# ============================================================
# 12. OVERVIEW
# ============================================================

elif page == "Overview":

    stores = st.session_state.knowledge_stores
    total_pages = sum(len(s["pages"]) for s in stores.values())
    total_chunks = sum(len(s["chunks"]) for s in stores.values())

    st.markdown(
        '<div class="section-title">Workspace overview</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    metrics = [
        (c1, "Builder profiles", len(COMPANIES)),
        (c2, "Indexed websites", len(stores)),
        (c3, "Pages indexed", total_pages),
        (c4, "Vector chunks", total_chunks),
    ]

    for col, label, value in metrics:
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")
    st.markdown("### Getting started")

    st.markdown(
        """
        1. Select **Builder Directory** to search mapped businesses.
        2. Select a city and display the interactive builder map.
        3. Use **Property Finder** to open property portals.
        4. Select a builder in the sidebar and click **Add website & index**.
        5. Open **Knowledge Sources** to review indexed pages.
        6. Open **RAG Assistant** to ask questions about indexed content.
        """
    )


# ============================================================
# 13. BUILDER DIRECTORY + INTERACTIVE MAP
# ============================================================

elif page == "Builder Directory":

    st.markdown(
        '<div class="section-title">Builder Directory</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-sub">'
        'Search public map data for mapped construction businesses.'
        '</div>',
        unsafe_allow_html=True,
    )

    city = st.selectbox(
        "Select city",
        CITIES,
        key="directory_city",
    )

    st.link_button(
        f"Find more builders in {city}",
        "https://www.google.com/maps/search/"
        + quote_plus(f"house construction builders in {city}, Karnataka"),
        use_container_width=True,
    )

    if st.button(
        "Search mapped builders",
        type="primary",
        key="directory_map_search",
    ):
        with st.spinner("Searching public map data..."):
            builders, error = get_city_builders(city)

        if error:
            st.error(error)
        else:
            st.session_state.current_map_builders = builders
            st.session_state.current_map_city = city

    if st.session_state.current_map_city == city:
        builders = st.session_state.current_map_builders

        st.markdown("### Builder locations")
        render_builder_map(city, builders)

        if not builders:
            st.info(
                "No matching businesses were found in public map data. "
                "Try the Google Maps search above."
            )

        for builder in builders:
            with st.container(border=True):
                st.markdown(
                    f'<div class="property-name">{builder["name"]}</div>',
                    unsafe_allow_html=True,
                )

                st.write(f"Location: {builder['address']}")

                if builder["phone"]:
                    st.write(f"Phone: {builder['phone']}")

                if builder["website"]:
                    st.markdown(
                        f"[Company website]({builder['website']})"
                    )

                st.caption(
                    "Availability: Contact the business to confirm."
                )

    st.divider()
    st.markdown("### Builder company directory")

    search = st.text_input(
        "Search company profiles",
        placeholder="Company name, location, or construction type",
        key="company_directory_search",
    )

    matching_companies = [
        company for company in COMPANIES
        if search.lower() in (
            company["name"] + " "
            + company["type"] + " "
            + company["location"]
        ).lower()
    ]

    for start in range(0, len(matching_companies), 2):
        columns = st.columns(2)

        for col, company in zip(
            columns, matching_companies[start:start + 2]
        ):
            with col:
                with st.container(border=True):
                    st.markdown(
                        f'<div class="property-name">'
                        f'{company["name"]}</div>',
                        unsafe_allow_html=True,
                    )
                    st.write(company["type"])
                    st.write(company["location"])
                    st.link_button(
                        "Visit official website",
                        company["url"],
                        use_container_width=True,
                    )


# ============================================================
# 14. KNOWLEDGE SOURCES
# ============================================================

elif page == "Knowledge Sources":

    stores = st.session_state.knowledge_stores

    st.markdown(
        '<div class="section-title">Knowledge Sources</div>',
        unsafe_allow_html=True,
    )

    if not stores:
        st.info(
            "No websites indexed yet. Choose a builder in the sidebar "
            "and click Add website & index."
        )

    for domain, store in list(stores.items()):
        with st.expander(
            f"{domain} · {len(store['pages'])} pages · "
            f"{len(store['chunks'])} chunks"
        ):
            for page_data in store["pages"]:
                st.markdown(
                    f"- [{page_data['url']}]({page_data['url']})"
                )

            if st.button(
                f"Remove index: {domain}",
                key=f"remove_index_{domain}",
            ):
                del st.session_state.knowledge_stores[domain]
                st.rerun()

    st.caption(
        "Indexes are held in session memory and may disappear "
        "when the session resets or the app restarts."
    )


# ============================================================
# 15. RAG ASSISTANT
# ============================================================

elif page == "RAG Assistant":

    stores = st.session_state.knowledge_stores

    st.markdown(
        '<div class="section-title">Builder RAG Assistant</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-sub">'
        'Ask questions about public website pages that you indexed.'
        '</div>',
        unsafe_allow_html=True,
    )

    if not stores:
        st.warning(
            "First index a builder website using the sidebar."
        )

    else:
        domains = list(stores.keys())

        selected_domain = st.selectbox(
            "Search within indexed website",
            ["All indexed websites"] + domains,
            key="rag_domain_filter",
        )

        examples = [
            "What does this company do?",
            "What projects or services are described?",
            "What contact information is available?",
        ]

        cols = st.columns(3)

        for i, prompt in enumerate(examples):
            with cols[i]:
                if st.button(
                    prompt,
                    key=f"rag_example_{i}",
                    use_container_width=True,
                ):
                    st.session_state.pending_question = prompt

        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        question = st.chat_input(
            "Ask about projects, services, or company details..."
        )

        if st.session_state.pending_question:
            question = st.session_state.pending_question
            st.session_state.pending_question = ""

        if question:
            st.session_state.chat_history.append({
                "role": "user",
                "content": question,
            })

            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                try:
                    if selected_domain == "All indexed websites":
                        candidate_stores = stores
                    else:
                        candidate_stores = {
                            selected_domain: stores[selected_domain]
                        }

                    retrieved = []

                    for store in candidate_stores.values():
                        retrieved.extend(retrieve(question, store))

                    retrieved.sort(
                        key=lambda item: item["score"],
                        reverse=True,
                    )

                    retrieved = retrieved[:TOP_K]

                    if (
                        not retrieved
                        or retrieved[0]["score"] < 0.15
                    ):
                        answer = (
                            "I couldn't find relevant information in "
                            "the indexed pages. Try indexing more pages "
                            "or asking a more specific question."
                        )
                    else:
                        answer = groq_answer(question, retrieved)

                    st.markdown(answer)

                    if retrieved:
                        with st.expander("Retrieved source passages"):
                            seen_sources = set()

                            for i, item in enumerate(retrieved, 1):
                                st.markdown(
                                    f"**Source {i} · "
                                    f"Similarity {item['score']:.3f}**"
                                )

                                if item["source"] not in seen_sources:
                                    seen_sources.add(item["source"])
                                    st.markdown(
                                        f"[{item['source']}]"
                                        f"({item['source']})"
                                    )

                                st.write(item["text"][:1000])

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                    })

                except Exception as exc:
                    st.error(
                        f"Could not generate an answer: {exc}"
                    )
                    st.info(
                        "Check GROQ_API_JNTU_KEY in Streamlit Cloud "
                        "Secrets and verify your API quota."
                    )


# ============================================================
# 16. FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="
        text-align:center;
        color:#64748b;
        font-size:0.8rem;
        padding:10px;
    ">
        BuildWise · Construction Intelligence · Hugging Face · FAISS · Groq
    </div>
    """,
    unsafe_allow_html=True,
)
