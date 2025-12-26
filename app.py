import os
import json
import tempfile
import re
import time
from datetime import date

import requests
import pandas as pd
import numpy as np
import streamlit as st
from openai import OpenAI

# =========================================
# Config
# =========================================

HOME = "https://www.bseindia.com/"
CORP = "https://www.bseindia.com/corporates/ann.html"

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

OPENAI_MODEL = "gpt-4.1-mini"
MAX_OPENAI_ROWS = 30  # limit cost

# =========================================
# OpenAI helpers
# =========================================

def get_openai_client() -> OpenAI:
    api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
    if not api_key:
        st.error("Missing OPENAI_API_KEY (env var or Streamlit secrets).")
        st.stop()
    return OpenAI(api_key=api_key)

def call_openai_json(
    client: OpenAI,
    prompt: str,
    file_id: str | None = None,
    max_tokens: int = 600,
    temperature: float = 0.2,
) -> dict | None:
    """
    Call Responses API with web_search (and optional PDF file) and
    return parsed JSON dict (or None on failure).
    """
    content = [{"type": "input_text", "text": prompt}]
    if file_id:
        content.append({"type": "input_file", "file_id": file_id})

    resp = client.responses.create(
        model=OPENAI_MODEL,
        temperature=temperature,
        max_output_tokens=max_tokens,
        tools=[{"type": "web_search"}],
        input=[
            {
                "role": "user",
                "content": content,
            }
        ],
    )

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
    except Exception:
        return None

def upload_pdf_to_openai(client: OpenAI, pdf_bytes: bytes, fname: str = "doc.pdf"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp.flush()
        f = client.files.create(file=open(tmp.name, "rb"), purpose="assistants")
    return f

# =========================================
# PDF helpers
# =========================================

def candidate_pdf_urls(row) -> list[str]:
    """
    Build possible PDF URLs from ATTACHMENTNAME + NSURL.
    """
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

def primary_bse_url(row) -> str:
    ns = str(row.get("NSURL") or "").strip()
    if not ns:
        return ""
    return ns if ns.lower().startswith("http") else HOME + ns.lstrip("/")

def download_pdf(url: str, timeout: int = 25) -> bytes | None:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/pdf,application/octet-stream,*/*",
            "Referer": CORP,
        }
    )
    r = s.get(url, timeout=timeout, allow_redirects=True)
    if r.status_code != 200:
        return None
    data = r.content
    if not data or len(data) < 500:
        return None
    return data

# =========================================
# Backend: BSE fetcher
# =========================================

def _call_once(s: requests.Session, url: str, params: dict):
    """One guarded call; returns (rows, total, meta)."""
    r = s.get(url, params=params, timeout=30)
    ct = r.headers.get("content-type", "")
    if "application/json" not in ct:
        return [], None, {"blocked": True, "ct": ct, "status": r.status_code}
    data = r.json()
    rows = data.get("Table") or []
    total = None
    try:
        total = int((data.get("Table1") or [{}])[0].get("ROWCNT") or 0)
    except Exception:
        pass
    return rows, total, {}

def _fetch_single_range(s: requests.Session, d1: str, d2: str, log):
    """
    Fetch a single date range (usually one day) using multiple parameter
    permutations to cope with BSE quirks.
    """
    search_opts = ["", "P"]
    seg_opts = ["C", "E"]
    subcat_opts = ["", "-1"]
    pageno_keys = ["pageno", "Pageno"]
    scrip_keys = ["strScrip", "strscrip"]

    for ep in ENDPOINTS:
        for strType in seg_opts:
            for strSearch in search_opts:
                for subcategory in subcat_opts:
                    for pageno_key in pageno_keys:
                        for scrip_key in scrip_keys:

                            params = {
                                pageno_key: 1,
                                "strCat": "-1",
                                "strPrevDate": d1,
                                "strToDate": d2,
                                scrip_key: "",
                                "strSearch": strSearch,
                                "strType": strType,
                                "subcategory": subcategory,
                            }

                            log.append(
                                f"Trying {ep} | {pageno_key} | {scrip_key} | Type={strType} | {d1}..{d2}"
                            )

                            rows_acc = []
                            page = 1

                            while True:
                                rows, total, meta = _call_once(s, ep, params)

                                if meta.get("blocked"):
                                    log.append("Blocked: retry warmup")
                                    try:
                                        s.get(HOME, timeout=10)
                                        s.get(CORP, timeout=10)
                                    except Exception:
                                        pass
                                    rows, total, meta = _call_once(s, ep, params)
                                    if meta.get("blocked"):
                                        break

                                if page == 1 and total == 0 and not rows:
                                    break

                                if not rows:
                                    break

                                rows_acc.extend(rows)
                                params[pageno_key] += 1
                                page += 1

                                if total and len(rows_acc) >= total:
                                    break

                            if rows_acc:
                                return rows_acc

    return []

def fetch_bse_announcements_strict(start_yyyymmdd: str, end_yyyymmdd: str, log=None):
    """
    Fetch announcements between start_yyyymmdd and end_yyyymmdd (inclusive).

    BSE's API is most reliable when strPrevDate == strToDate, so we iterate
    day-by-day over the range, call _fetch_single_range for each day, and
    then aggregate + de-duplicate.
    """
    if log is None:
        log = []

    # Session & warmup once
    s = requests.Session()
    s.headers.update(BASE_HEADERS)
    try:
        s.get(HOME, timeout=15)
        s.get(CORP, timeout=15)
    except Exception:
        pass

    start_dt = pd.to_datetime(start_yyyymmdd, format="%Y%m%d")
    end_dt = pd.to_datetime(end_yyyymmdd, format="%Y%m%d")

    if end_dt < start_dt:
        return pd.DataFrame(
            columns=[
                "SCRIP_CD",
                "SLONGNAME",
                "HEADLINE",
                "NEWSSUB",
                "NEWS_DT",
                "ATTACHMENTNAME",
                "NSURL",
            ]
        )

    all_rows: list[dict] = []

    cur = start_dt
    while cur <= end_dt:
        d_str = cur.strftime("%Y%m%d")
        log.append(f"Day chunk fetch: {d_str}..{d_str}")
        rows = _fetch_single_range(s, d_str, d_str, log)
        if rows:
            all_rows.extend(rows)
        cur += pd.Timedelta(days=1)

    if not all_rows:
        return pd.DataFrame(
            columns=[
                "SCRIP_CD",
                "SLONGNAME",
                "HEADLINE",
                "NEWSSUB",
                "NEWS_DT",
                "ATTACHMENTNAME",
                "NSURL",
            ]
        )

    base_cols = [
        "SCRIP_CD",
        "SLONGNAME",
        "HEADLINE",
        "NEWSSUB",
        "NEWS_DT",
        "ATTACHMENTNAME",
        "NSURL",
        "NEWSID",
    ]

    seen = set(base_cols)
    extra_cols = []
    for r in all_rows:
        for k in r.keys():
            if k not in seen:
                extra_cols.append(k)
                seen.add(k)

    df = pd.DataFrame(all_rows, columns=base_cols + extra_cols)

    keys = ["NSURL", "NEWSID", "ATTACHMENTNAME", "HEADLINE"]
    keys = [k for k in keys if k in df.columns]
    if keys:
        df = df.drop_duplicates(subset=keys)

    if "NEWS_DT" in df.columns:
        df["_NEWS_DT_PARSED"] = pd.to_datetime(
            df["NEWS_DT"], errors="coerce", dayfirst=True
        )
        df = (
            df.sort_values("_NEWS_DT_PARSED", ascending=False)
            .drop(columns=["_NEWS_DT_PARSED"])
            .reset_index(drop=True)
        )

    return df

# =========================================
# Filters: Orders + Capex
# =========================================

ORDER_KEYWORDS = ["order", "contract", "bagged", "supply", "purchase order"]
ORDER_REGEX = re.compile(
    r"\b(?:" + "|".join(map(re.escape, ORDER_KEYWORDS)) + r")\b", re.IGNORECASE
)

# Focused on commercial production capex
CAPEX_KEYWORDS = [
    "commercial production",
    "commencement of commercial production",
    "commenced commercial production",
]
CAPEX_REGEX = re.compile("|".join(CAPEX_KEYWORDS), re.IGNORECASE)

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

def enrich_capex(df: pd.DataFrame) -> pd.DataFrame:
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

# =========================================
# OpenAI enrichment: Orders
# =========================================

def enrich_orders_with_openai(
    orders_df: pd.DataFrame, raw_df: pd.DataFrame, client: OpenAI
) -> pd.DataFrame:
    """
    For each order announcement (top MAX_OPENAI_ROWS):
      - Use filing PDF (if available) for Existing Order Book (₹ Cr).
      - Use web_search for TTM Revenue, Market Cap, Current Order Book (₹ Cr).
    """
    if orders_df.empty:
        return orders_df

    df = orders_df.copy()
    df["TTM Revenue (₹ Cr)"] = np.nan
    df["Market Cap (₹ Cr)"] = np.nan
    df["Existing Order Book (₹ Cr)"] = np.nan
    df["Current Order Book (₹ Cr)"] = np.nan

    # rough join back to raw df by company + headline
    raw_key = raw_df.set_index(["SLONGNAME", "HEADLINE"])

    for idx, row in df.head(MAX_OPENAI_ROWS).iterrows():
        company = str(row["Company"])
        ann = str(row["Announcement"])
        details = str(row.get("Details") or "")
        date_val = str(row["Date"].date()) if pd.notnull(row["Date"]) else ""

        try:
            raw_row = raw_key.loc[(company, ann)]
            if isinstance(raw_row, pd.DataFrame):
                raw_row = raw_row.iloc[0]
        except Exception:
            raw_row = None

        file_id = None
        urls = candidate_pdf_urls(raw_row) if raw_row is not None else []

        for u in urls:
            pdf_bytes = download_pdf(u)
            if pdf_bytes:
                try:
                    fobj = upload_pdf_to_openai(client, pdf_bytes, fname="order.pdf")
                    file_id = fobj.id
                    break
                except Exception:
                    file_id = None
            time.sleep(0.3)

        prompt = f"""
You are a fundamental equity analyst specialising in Indian listed companies.

Tasks for the company and announcement below:

1) Use web_search to fetch latest **TTM revenue (Sales TTM)** and **Market Capitalisation**.
   - Prefer reliable sites (Screener.in, exchanges, Moneycontrol etc.).
   - Convert both into **INR Crore (₹ Cr)**.
   - If you still cannot find a reliable value, return null.

2) Determine **Existing Order Book (₹ Cr)** *before* this order:
   - Use ONLY the attached PDF filing (if provided) and the announcement text.
   - If not clearly stated, return null.

3) Estimate **Current Order Book (₹ Cr)** *after* including this order:
   - Use web_search if needed (presentations, concalls etc.).
   - Combine existing order book and value of this new order when possible.
   - If you cannot estimate, return null.

Return ONLY valid JSON in this exact structure:

{{
  "ttm_revenue_cr": <number or null>,
  "market_cap_cr": <number or null>,
  "existing_order_book_cr": <number or null>,
  "current_order_book_cr": <number or null>
}}

Company: {company}
Announcement headline: {ann}
Announcement details: {details}
Date: {date_val}
"""

        data = call_openai_json(client, prompt, file_id=file_id, max_tokens=650, temperature=0.1)
        if not data:
            continue

        def _as_float(x):
            if x is None:
                return np.nan
            try:
                return float(str(x).replace(",", ""))
            except Exception:
                return np.nan

        df.at[idx, "TTM Revenue (₹ Cr)"] = _as_float(data.get("ttm_revenue_cr"))
        df.at[idx, "Market Cap (₹ Cr)"] = _as_float(data.get("market_cap_cr"))
        df.at[idx, "Existing Order Book (₹ Cr)"] = _as_float(
            data.get("existing_order_book_cr")
        )
        df.at[idx, "Current Order Book (₹ Cr)"] = _as_float(
            data.get("current_order_book_cr")
        )

    return df

# =========================================
# OpenAI enrichment: Capex Impact
# =========================================

def enrich_capex_with_openai(capex_df: pd.DataFrame, client: OpenAI) -> pd.DataFrame:
    """
    Add an 'Impact' paragraph for capex announcements, especially those
    mentioning 'commercial production'.
    """
    if capex_df.empty:
        return capex_df

    df = capex_df.copy()
    df["Impact"] = ""

    for idx, row in df.head(MAX_OPENAI_ROWS).iterrows():
        company = str(row["Company"])
        ann = str(row["Announcement"])
        details = str(row.get("Details") or "")

        text_for_filter = (ann + " " + details).lower()
        if "commercial production" not in text_for_filter:
            continue

        prompt = f"""
You are a sell-side equity research analyst.

Write a concise, investor-focused Impact paragraph (3–6 sentences, max ~140 words)
for the following capex / commercial production announcement.

Cover:
- what has commenced (product, capacity, plant/location),
- how it changes growth and margin trajectory vs existing business,
- rough revenue and EBITDA potential range in INR crore if inferable,
- key execution or market risks during ramp-up.

Tone: neutral, analytical, no hype. Output plain text only.

Company: {company}
Headline: {ann}
Details: {details}
"""

        # For Impact we just want text; do a simple non-JSON call
        resp = client.responses.create(
            model=OPENAI_MODEL,
            temperature=0.25,
            max_output_tokens=280,
            input=prompt,
        )
        impact = (getattr(resp, "output_text", None) or "").strip()
        df.at[idx, "Impact"] = impact

    return df

# =========================================
# Streamlit UI
# =========================================

st.set_page_config(
    page_title="BSE Order & Capex (OpenAI-enriched)", layout="wide"
)
st.title("📣 BSE Order & Capex Announcements — OpenAI + Web Search")

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=date(2025, 1, 1))
with col2:
    end_date = st.date_input("End Date", value=date.today())

run = st.button("🔎 Fetch & Enrich", use_container_width=True)

if run:
    if start_date > end_date:
        st.error("Start Date cannot be after End Date.")
        st.stop()

    ds = start_date.strftime("%Y%m%d")
    de = end_date.strftime("%Y%m%d")
    logs: list[str] = []

    with st.spinner("Fetching BSE announcements..."):
        df_raw = fetch_bse_announcements_strict(ds, de, log=logs)

    orders_df = enrich_orders(df_raw)
    capex_df = enrich_capex(df_raw)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Announcements", len(df_raw))
    c2.metric("Order Announcements", len(orders_df))
    c3.metric(
        "Capex Announcements (commercial production–filtered)", len(capex_df)
    )

    if df_raw.empty:
        st.warning("No announcements found for this date range.")
        st.stop()

    client = get_openai_client()

    if not orders_df.empty:
        with st.spinner(
            f"Enriching top {min(MAX_OPENAI_ROWS, len(orders_df))} order announcements via internet + PDF..."
        ):
            orders_df = enrich_orders_with_openai(orders_df, df_raw, client)

    if not capex_df.empty:
        with st.spinner(
            "Generating 'Impact' commentary for capex announcements (commercial production)..."
        ):
            capex_df = enrich_capex_with_openai(capex_df, client)

    tab_orders, tab_capex, tab_all, tab_logs = st.tabs(
        ["📦 Orders (Enriched)", "🏭 Capex (Impact)", "📄 All Raw", "🧪 Fetch Logs"]
    )

    with tab_orders:
        st.caption(
            "TTM Revenue, Market Cap, and Current Order Book are fetched via OpenAI web_search. "
            "Existing Order Book is extracted from the filing PDF where available."
        )
        st.dataframe(orders_df, use_container_width=True)

    with tab_capex:
        st.caption(
            "Capex announcements filtered on 'commercial production' keywords. "
            "Impact column is generated by OpenAI."
        )
        st.dataframe(capex_df, use_container_width=True)

    with tab_all:
        st.dataframe(df_raw, use_container_width=True)

    with tab_logs:
        for line in logs:
            st.text(line)
