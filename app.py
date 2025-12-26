import os, io, re, time, tempfile
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests import exceptions as req_exc
import pandas as pd
import numpy as np
from openai import OpenAI
import os, streamlit as st
# ---- OpenAI (Responses API) ----

api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
if not api_key:
    st.error("Missing OPENAI_API_KEY (set env var or add to Streamlit secrets).")
    st.stop()

client = OpenAI(api_key=api_key)

with st.expander("🔍 OpenAI connection diagnostics", expanded=False):
    # 1) Is the key visible?
    key_src = "st.secrets" if "OPENAI_API_KEY" in st.secrets else "env"
    mask = lambda s: (s[:7] + "..." + s[-4:]) if s and len(s) > 12 else "unset"
    st.write("Key source:", key_src)
    st.write("Key (masked):", mask(api_key))

    # 2) Simple status ping
    try:
        _ = client.models.list()
        st.success("OpenAI client initialised successfully.")
    except Exception as e:
        st.error(f"OpenAI client initialisation failed: {e}")

# Custom model used for reasoning via Responses API
OPENAI_MODEL = "gpt-4.1-mini"

#======================
# Basic settings
#======================

HOME = "https://www.bseindia.com/"
CORP = "https://www.bseindia.com/corporates/ann.html"

# --- Order-related keyword logic for filtering ---
ORDER_KEYWORDS = ["order", "contract", "bagged", "supply", "purchase order"]
ORDER_REGEX = re.compile(r"\b(?:" + "|".join(map(re.escape, ORDER_KEYWORDS)) + r")\b", re.IGNORECASE)

# --- Capex / expansion keywords (to seed the search) ---
CAPEX_KEYWORDS = [
    "commercial production",
    "commencement of commercial production",
    "commenced commercial production",
    "capex", "capital expenditure", "capacity expansion",
    "new plant", "manufacturing facility", "brownfield", "greenfield",
    "setting up a plant", "increase in capacity", "expansion"
]
CAPEX_REGEX = re.compile("|".join(map(re.escape, CAPEX_KEYWORDS)), re.IGNORECASE)

MAX_CAPEX_ROWS_OPENAI = 200

# BSE endpoints (older + newer)
ENDPOINTS = [
    "https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w",
    "https://api.bseindia.com/BseIndiaAPI/api/AnnGetData/w",
]

BASE_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": HOME,
    "Origin": "https://www.bseindia.com",
    "X-Requested-With": "XMLHttpRequest",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}

#======================
# Small helpers
#======================

def _norm(x):
    if x is None:
        return ""
    return str(x).strip()

def _first_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

#======================
# BSE fetch (Company Update / M&A etc.)
#======================

def fetch_bse_announcements_strict(start_yyyymmdd: str,
                                   end_yyyymmdd: str,
                                   verbose: bool = True,
                                   request_timeout: int = 25) -> pd.DataFrame:
    """Fetches raw announcements, then filters:
    Category='Company Update' AND subcategory contains any:
    Acquisition | Amalgamation / Merger | Scheme of Arrangement | Joint Venture
    """
    assert len(start_yyyymmdd) == 8 and len(end_yyyymmdd) == 8
    assert start_yyyymmdd <= end_yyyymmdd
    base_page = "https://www.bseindia.com/corporates/ann.html"
    url = "https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w"

    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": base_page,
        "X-Requested-With": "XMLHttpRequest",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    })

    try:
        s.get(base_page, timeout=15)
    except Exception:
        pass

    variants = [
        {"subcategory": "-1", "strSearch": "P"},
        {"subcategory": "-1", "strSearch": ""},
        {"subcategory": "",   "strSearch": "P"},
        {"subcategory": "",   "strSearch": ""},
    ]

    all_rows = []
    for v in variants:
        params = {
            "pageno": 1, "strCat": "-1", "subcategory": v["subcategory"],
            "strPrevDate": start_yyyymmdd, "strToDate": end_yyyymmdd,
            "strSearch": v["strSearch"], "strscrip": "", "strType": "C",
        }
        rows, total, page = [], None, 1
        while True:
            try:
                r = s.get(url, params=params, timeout=request_timeout)
            except req_exc.ReadTimeout as e:
                if verbose:
                    st.warning(f"[variant {v}] ReadTimeout on page {page}: {e}")
                rows = []
                break
            except req_exc.RequestException as e:
                if verbose:
                    st.warning(f"[variant {v}] Request error on page {page}: {e}")
                rows = []
                break

            ct = r.headers.get("content-type","")
            if "application/json" not in ct:
                if verbose:
                    st.warning(f"[variant {v}] non-JSON on page {page} (ct={ct}).")
                break

            data = r.json()
            table = data.get("Table") or []
            rows.extend(table)

            if total is None:
                try:
                    total = int((data.get("Table1") or [{}])[0].get("ROWCNT") or 0)
                except Exception:
                    total = None

            if not table:
                break

            params["pageno"] += 1
            page += 1
            time.sleep(0.25)

            if total and len(rows) >= total:
                break

        if rows:
            all_rows = rows; break

    if not all_rows: return pd.DataFrame()

    all_keys = set()
    for r in all_rows: all_keys.update(r.keys())
    df = pd.DataFrame(all_rows, columns=list(all_keys))

    # Filter to Company Update
    def filter_announcements(df_in: pd.DataFrame, category_filter="Company Update") -> pd.DataFrame:
        if df_in.empty: return df_in.copy()
        cat_col = _first_col(df_in, ["CATEGORYNAME", "CATEGORY", "NEWS_CAT", "NEWSCATEGORY", "NEWS_CATEGORY"])
        if not cat_col: return df_in.copy()
        df2 = df_in.copy()
        df2["_cat_norm"] = df2[cat_col].map(lambda x: _norm(x).lower())
        return df2.loc[df2["_cat_norm"] == _norm(category_filter).lower()].drop(columns=["_cat_norm"])

    df_filtered = filter_announcements(df, category_filter="Company Update")
    if df_filtered.empty:
        return df_filtered

    # Filter to those subcategories
    df_filtered = df_filtered.loc[
        df_filtered
        .filter(["NEWSSUB", "SUBCATEGORY", "SUBCATEGORYNAME", "NEWS_SUBCATEGORY", "NEWS_SUB"], axis=1)
        .astype(str)
        .apply(
            lambda col: col.str.contains(
                r"(Acquisition|Amalgamation\s*/\s*Merger|Scheme of Arrangement|Joint Venture)",
                case=False,
                na=False,
            )
        )
        .any(axis=1)
    ]

    return df_filtered

#======================
# Announcement filters (Orders / Capex)
#======================

def enrich_orders(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    mask = df["HEADLINE"].fillna("").str.contains(ORDER_REGEX)
    out = df.loc[
        mask,
        ["SLONGNAME", "HEADLINE", "NEWSSUB", "NEWS_DT", "ATTACHMENTNAME", "NSURL"],
    ].copy()
    out.columns = ["Company", "Announcement", "Details", "Date", "Attachment", "Link"]
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce", dayfirst=True)
    return out.sort_values("Date", ascending=False).reset_index(drop=True)

def enrich_capex_headline(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    combined = df["HEADLINE"].fillna("") + " " + df["NEWSSUB"].fillna("")
    mask = combined.str.contains(CAPEX_REGEX, na=False)
    out = df.loc[
        mask,
        ["SLONGNAME", "HEADLINE", "NEWSSUB", "NEWS_DT", "ATTACHMENTNAME", "NSURL"],
    ].copy()
    out.columns = ["Company", "Announcement", "Details", "Date", "Attachment", "Link"]
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce", dayfirst=True)
    return out.sort_values("Date", ascending=False).reset_index(drop=True)

#======================
# PDF handling
#======================

def _candidate_pdf_urls(row: dict) -> list[str]:
    cands = []
    att = str(row.get("ATTACHMENTNAME") or "").strip()
    if att:
        cands += [
            f"https://www.bseindia.com/xml-data/corpfiling/AttachHis/{att}",
            f"https://www.bseindia.com/xml-data/corpfiling/Attach/{att}",
            f"https://www.bseindia.com/xml-data/corpfiling/AttachLive/{att}",
        ]
    ns = str(row.get("NSURL") or "").strip()
    if ".pdf" in ns.lower():
        cands.append(ns if ns.lower().startswith("http") else HOME + ns.lstrip("/"))

    seen, out = set(), []
    for u in cands:
        if u and u not in seen:
            out.append(u)
            seen.add(u)
    return out

def _download_pdf(url: str, timeout=25) -> bytes:
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/pdf,application/octet-stream,*/*",
        "Referer": CORP,
    })
    r = s.get(url, timeout=timeout, allow_redirects=True, stream=True)
    if r.status_code != 200:
        return b""
    data = r.content
    if not data or len(data) < 500:
        return b""
    return data

def _upload_pdf_to_openai(pdf_bytes: bytes, fname: str = "doc.pdf"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp.flush()
        f = client.files.create(file=open(tmp.name, "rb"), purpose="assistants")
    return f

#======================
# OpenAI helpers: JSON call
#======================

def _call_openai_json(prompt: str,
                      file_id: str | None = None,
                      max_tokens: int = 600,
                      temperature: float = 0.2) -> dict | None:
    content = [{"type": "input_text", "text": prompt}]
    if file_id:
        content.append({"type": "input_file", "file_id": file_id})

    try:
        resp = client.responses.create(
            model=OPENAI_MODEL,
            temperature=temperature,
            max_output_tokens=max_tokens,
            tools=[{"type": "web_search"}],
            input=[{"role": "user", "content": content}],
        )
    except Exception as e:
        st.warning(f"OpenAI JSON call failed: {e}")
        return None

    txt = (getattr(resp, "output_text", None) or "").strip()
    if not txt:
        return None

    cleaned = (
        txt.strip()
        .replace("```json", "")
        .replace("```", "")
        .replace("“", '"')
        .replace("”", '"')
        .replace("’", "'")
    )

    if "{" in cleaned and "}" in cleaned:
        cleaned = cleaned[cleaned.find("{") : cleaned.rfind("}") + 1]

    try:
        return json.loads(cleaned)
    except Exception as e:
        st.warning(f"JSON parse error: {e}")
        return None

#======================
# Enrich Orders (TTM, MCap, Order Amount, OB, Execution Timeline)
#======================

import json

def enrich_orders_with_openai(orders_df: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    if orders_df.empty:
        return orders_df

    df = orders_df.copy()

    df["TTM / Latest Revenue (₹ Cr)"] = np.nan
    df["Market Cap (₹ Cr)"] = np.nan
    df["Order Amount (₹ Cr)"] = np.nan
    df["Current Order Book (₹ Cr)"] = np.nan
    df["Order Book / Sales (x)"] = np.nan
    df["Execution Timeline"] = ""

    raw_key = raw_df.set_index(["SLONGNAME", "HEADLINE"])

    for idx, row in df.iterrows():
        company = str(row["Company"])
        ann     = str(row["Announcement"])
        details = str(row.get("Details") or "")
        date_val = str(row["Date"].date()) if pd.notnull(row["Date"]) else ""

        try:
            raw_row = raw_key.loc[(company, ann)]
            if isinstance(raw_row, pd.DataFrame):
                raw_row = raw_row.iloc[0]
        except Exception:
            raw_row = None

        file_id = None
        urls = _candidate_pdf_urls(raw_row) if raw_row is not None else []

        for u in urls:
            pdf_bytes = _download_pdf(u)
            if pdf_bytes:
                try:
                    fobj = _upload_pdf_to_openai(pdf_bytes, fname="order.pdf")
                    file_id = fobj.id
                    break
                except Exception:
                    pass
            time.sleep(0.2)

        prompt = f"""
You are a fundamental equity analyst specialising in Indian listed companies.

For the company and announcement below, extract the following:

1) Use web_search to fetch the company's **latest reported revenue**
   (most recent full-year OR trailing twelve months)
   AND **current Market Capitalisation**.
   - Prefer reliable financial sources (Screener.in, exchanges, Moneycontrol etc.)
   - Convert BOTH into ₹ Crore (₹ Cr)

2) Determine the **Order Amount (₹ Cr)** for THIS specific order.
   - Use ONLY the attached PDF filing and/or the announcement text.
   - This must represent the value of THIS order.
   - Convert to ₹ Crore.
   - If truly not disclosed, return null.

3) Estimate the **Current Total Order Book (₹ Cr)** AFTER including this order.
   - Use web_search and official disclosures if needed.
   - Convert to ₹ Crore.

4) Determine the **Execution Timeline** for the order book burn-down horizon:
   - FIRST look inside the PDF and announcement text.
   - IF (and only if) the filing does NOT disclose a timeline,
     THEN infer a reasonable forecast based on:
       - industry norms
       - commentary found online
       - order-book / revenue ratio
   - Output a SHORT human-readable string like:
       "12–18 months" or "over 2–3 years"

Return ONLY valid JSON in this structure:

{{
  "ttm_revenue_cr": <number or null>,
  "market_cap_cr": <number or null>,
  "order_amount_cr": <number or null>,
  "current_order_book_cr": <number or null>,
  "execution_timeline": <string or null>
}}

Company: {company}
Headline: {ann}
Details: {details}
Date: {date_val}
"""

        data = _call_openai_json(prompt, file_id=file_id, max_tokens=750, temperature=0.1)
        if not data:
            continue

        def _f(x):
            if x is None:
                return np.nan
            try:
                return float(str(x).replace(",", ""))
            except Exception:
                return np.nan

        revenue = _f(data.get("ttm_revenue_cr"))
        mcap    = _f(data.get("market_cap_cr"))
        order   = _f(data.get("order_amount_cr"))
        ob_cur  = _f(data.get("current_order_book_cr"))

        df.at[idx, "TTM / Latest Revenue (₹ Cr)"] = revenue
        df.at[idx, "Market Cap (₹ Cr)"] = mcap
        df.at[idx, "Order Amount (₹ Cr)"] = order
        df.at[idx, "Current Order Book (₹ Cr)"] = ob_cur

        if pd.notna(revenue) and revenue > 0 and pd.notna(ob_cur):
            df.at[idx, "Order Book / Sales (x)"] = ob_cur / revenue

        timeline = data.get("execution_timeline")
        if timeline:
            df.at[idx, "Execution Timeline"] = str(timeline).strip()

    return df

#======================
# Enrich Capex (Impact paragraph)
#======================

def enrich_capex_with_openai(capex_df: pd.DataFrame) -> pd.DataFrame:
    if capex_df.empty:
        return capex_df

    df = capex_df.copy()
    df["Impact"] = ""

    for idx, row in df.head(MAX_CAPEX_ROWS_OPENAI).iterrows():
        company = str(row["Company"])
        ann = str(row["Announcement"])
        details = str(row.get("Details") or "")

        pseudo_raw = {
            "ATTACHMENTNAME": row.get("Attachment"),
            "NSURL": row.get("Link"),
        }
        urls = _candidate_pdf_urls(pseudo_raw)

        file_id = None
        for u in urls:
            pdf_bytes = _download_pdf(u)
            if pdf_bytes:
                try:
                    fobj = _upload_pdf_to_openai(pdf_bytes, fname="capex.pdf")
                    file_id = fobj.id
                    break
                except Exception:
                    pass
            time.sleep(0.2)

        keyword_list_str = ", ".join(CAPEX_KEYWORDS)

        prompt = f"""
You are a sell-side equity research analyst.

You are given:
- a BSE announcement (headline + details),
- and, where available, the full PDF filing for this announcement.

First, SEARCH INSIDE the attached PDF filing (if present) for capex/expansion-related
phrases, focusing on these keywords (and close variants):

{keyword_list_str}

Use both:
- the PDF content, and
- the headline + details text below

to determine the nature of the capex or capacity expansion, including:

- what is being done (new plant / brownfield / greenfield / line expansion / debottlenecking),
- product or segment involved,
- location and capacity (if disclosed),
- total capex outlay in ₹ crore (if disclosed or reasonably inferable).

Then write a concise, investor-focused Impact paragraph (3–6 sentences, max ~140 words)
answering:

- How does this capex / expansion change the company's growth and margin trajectory
  vs its existing business?
- What is the rough revenue and EBITDA potential (₹ crore range) if it ramps up as planned?
- What are the key execution / market risks during ramp-up?

If the filing is clearly NOT about capex / capacity / plant / expansion, return a short
sentence like "This filing does not relate to capex or capacity expansion." instead.

Tone: neutral, analytical, no hype. Output plain text only (no bullets).

Company: {company}
Headline: {ann}
Details: {details}
"""

        content = [{"type": "input_text", "text": prompt}]
        if file_id:
            content.append({"type": "input_file", "file_id": file_id})

        try:
            resp = client.responses.create(
                model=OPENAI_MODEL,
                temperature=0.25,
                max_output_tokens=280,
                input=[{"role": "user", "content": content}],
            )
            impact = (getattr(resp, "output_text", None) or "").strip()
        except Exception as e:
            impact = f"(OpenAI error while generating impact: {e})"

        df.at[idx, "Impact"] = impact

    return df

#======================
# Streamlit UI
#======================

st.set_page_config(
    page_title="BSE Order & Capex (OpenAI-enriched)", layout="wide"
)
st.title("📣 BSE Order & Capex Announcements — OpenAI + Web Search")

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=datetime(2025, 1, 1).date())
with col2:
    end_date = st.date_input("End Date", value=datetime.today().date())

run = st.button("🔎 Fetch & Enrich", use_container_width=True)

if run:
    if start_date > end_date:
        st.error("Start Date cannot be after End Date.")
        st.stop()

    ds = start_date.strftime("%Y%m%d")
    de = end_date.strftime("%Y%m%d")

    with st.spinner("Fetching BSE announcements (Company Update + M&A/JV)..."):
        df_raw = fetch_bse_announcements_strict(ds, de, verbose=True)

    orders_df = enrich_orders(df_raw)
    capex_df = enrich_capex_headline(df_raw)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Announcements", len(df_raw))
    c2.metric("Order Announcements", len(orders_df))
    c3.metric(
        "Capex / Expansion Candidates (headline-keyword filtered)", len(capex_df)
    )

    if df_raw.empty:
        st.warning("No announcements found for this date range.")
        st.stop()

    if not orders_df.empty:
        with st.spinner("Enriching ALL order announcements via internet + PDF..."):
            orders_df = enrich_orders_with_openai(orders_df, df_raw)

    if not capex_df.empty:
        with st.spinner(
            f"Generating 'Impact' commentary for up to {min(MAX_CAPEX_ROWS_OPENAI, len(capex_df))} capex / expansion filings via PDF keyword search..."
        ):
            capex_df = enrich_capex_with_openai(capex_df)

    tab_orders, tab_capex, tab_all = st.tabs(
        ["📦 Orders (Enriched)", "🏭 Capex / Expansion (Impact)", "📄 All Raw"]
    )

    with tab_orders:
        st.caption(
            "Latest Revenue and Market Cap are fetched via OpenAI web_search. "
            "Order Amount (₹ Cr) is extracted from the filing PDF / announcement where disclosed. "
            "Current Order Book (₹ Cr) is estimated from disclosures plus web_search. "
            "Order Book / Sales (x) is computed as Current Order Book ÷ Revenue. "
            "Execution Timeline is taken from the filing wherever available, and otherwise forecast by the model."
        )
        st.dataframe(orders_df, use_container_width=True)

    with tab_capex:
        st.caption(
            "Capex / expansion candidates are first filtered on headline keywords "
            "(commercial production, capex, capital expenditure, capacity expansion, "
            "new plant, brownfield/greenfield, etc.), then OpenAI reads the PDF filing "
            "and searches for these keywords inside the content to generate the Impact commentary."
        )
        st.dataframe(capex_df, use_container_width=True)

    with tab_all:
        st.dataframe(df_raw, use_container_width=True)
