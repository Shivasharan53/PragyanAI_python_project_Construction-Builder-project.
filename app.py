import os
import re
import time
from collections import deque
from urllib.parse import urljoin, urlparse, urldefrag

import faiss
import numpy as np
import requests
import streamlit as st
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="BuildWise | Construction RAG",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS — READABLE TEXT AND PROPERTY CARDS
# ============================================================

st.markdown("""
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
    min-height: 285px;
    box-shadow: 0 4px 14px rgba(20, 36, 59, 0.05);
}

.property-name {
    color: #17263d !important;
    font-family: 'Manrope', sans-serif;
    font-size: 1.15rem;
    font-weight: 800;
    margin: 8px 0;
}

.property-builder {
    color: #475569 !important;
    font-size: 0.9rem;
    margin-bottom: 12px;
}

.property-detail {
    color: #334155 !important;
    font-size: 0.9rem;
    margin: 8px 0;
}

.property-price {
    color: #087f5b !important;
    font-family: 'Manrope', sans-serif;
    font-size: 1.45rem;
    font-weight: 800;
    margin: 15px 0;
}

.property-badge {
    display: inline-block;
    color: #155e75 !important;
    background: #e0f2fe;
    padding: 5px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
}

.panel {
    background: white;
    border: 1px solid #e1e8f0;
    border-radius: 16px;
    padding: 20px;
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
""", unsafe_allow_html=True)


# ============================================================
# BUILDER DIRECTORY
# ============================================================

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
# DEMO PROPERTY DATA
# These are examples, NOT verified live property listings.
# ============================================================

PROPERTIES = [
    {
        "name": "Sample Residency",
        "builder": "Prestige Group",
        "city": "Bengaluru",
        "bhk": 2,
        "type": "Apartment",
        "price": 95,
        "url": "https://www.prestigeltd.in/",
    },
    {
        "name": "Sample Heights",
        "builder": "Prestige Group",
        "city": "Bengaluru",
        "bhk": 3,
        "type": "Apartment",
        "price": 145,
        "url": "https://www.prestigeltd.in/",
    },
    {
        "name": "Sample Park",
        "builder": "Brigade Group",
        "city": "Bengaluru",
        "bhk": 2,
        "type": "Apartment",
        "price": 88,
        "url": "https://www.brigadegroup.com/",
    },
    {
        "name": "Sample Gardens",
        "builder": "SOBHA Limited",
        "city": "Bengaluru",
        "bhk": 4,
        "type": "Apartment",
        "price": 220,
        "url": "https://www.sobha.com/",
    },
    {
        "name": "Sample City Homes",
        "builder": "Godrej Properties",
        "city": "Hyderabad",
        "bhk": 2,
        "type": "Apartment",
        "price": 75,
        "url": "https://www.godrejproperties.com/",
    },
    {
        "name": "Sample Grand Towers",
        "builder": "Godrej Properties",
        "city": "Hyderabad",
        "bhk": 3,
        "type": "Apartment",
        "price": 130,
        "url": "https://www.godrejproperties.com/",
    },
    {
        "name": "Sample Urban Living",
        "builder": "L&T Construction",
        "city": "Hyderabad",
        "bhk": 4,
        "type": "Apartment",
        "price": 180,
        "url": "https://www.lntecc.com/",
    },
    {
        "name": "Sample Green Villas",
        "builder": "Puravankara",
        "city": "Chennai",
        "bhk": 3,
        "type": "Villa",
        "price": 115,
        "url": "https://www.puravankara.com/",
    },
    {
        "name": "Sample Smart Homes",
        "builder": "Sattva Group",
        "city": "Bengaluru",
        "bhk": 2,
        "type": "Apartment",
        "price": 70,
        "url": "https://sattvagroup.com/",
    },
    {
        "name": "Sample Premium Homes",
        "builder": "Brigade Group",
        "city": "Hyderabad",
        "bhk": 3,
        "type": "Apartment",
        "price": 160,
        "url": "https://www.brigadegroup.com/",
    },
    {
        "name": "Sample Family Villas",
        "builder": "SOBHA Limited",
        "city": "Chennai",
        "bhk": 4,
        "type": "Villa",
        "price": 250,
        "url": "https://www.sobha.com/",
    },
    {
        "name": "Sample City Residency",
        "builder": "L&T Construction",
        "city": "Bengaluru",
        "bhk": 3,
        "type": "Apartment",
        "price": 150,
        "url": "https://www.lntecc.com/",
    },
]


# ============================================================
# RAG CONFIGURATION
# ============================================================

USER_AGENT = "BuildWiseRAG/1.0 (educational project)"
REQUEST_TIMEOUT = 12
MAX_CHARS_PER_PAGE = 18000
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 4
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@st.cache_resource(show_spinner="Loading Hugging Face embedding model...")
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def normalize_url(url):
    url = url.strip()

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

    if "text/html" not in response.headers.get(
        "Content-Type", ""
    ).lower():
        return "", []

    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup([
        "script", "style", "noscript", "svg",
        "nav", "footer", "header", "form", "iframe"
    ]):
        tag.decompose()

    title = soup.title.get_text(" ", strip=True) if soup.title else url
    main = soup.find("main") or soup.find("article") or soup.body or soup

    text = re.sub(
        r"\s+", " ", main.get_text(" ", strip=True)
    ).strip()

    links = []

    for anchor in soup.find_all("a", href=True):
        href = anchor.get("href", "").strip()

        if href.startswith(("mailto:", "tel:", "javascript:")):
            continue

        absolute = urljoin(url, href)
        absolute, _ = urldefrag(absolute)
        parsed = urlparse(absolute)

        if parsed.scheme in ("http", "https") and parsed.netloc:
            absolute = parsed._replace(
                query="", fragment=""
            ).geturl()
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

        parsed_host = urlparse(url).netloc.lower()

        if parsed_host != base_host:
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

            time.sleep(0.15)

        except Exception as exc:
            errors.append(f"{url}: {str(exc)[:150]}")

    return pages, errors


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

    for page in
