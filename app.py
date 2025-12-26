import os
import requests, pandas as pd, time, re
from datetime import datetime, date
import streamlit as st
from openai import OpenAI
import json
import numpy as np

# --------------------
# Config
# --------------------

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

# how many rows (per table) you allow OpenAI to enrich
MAX_OPENAI_ROWS = 30
OPENAI_MODEL = "gpt-4.1-mini"

# --------------------
# Backend (resilient fetcher)
# --------------------

def _call_once(s: requests.Session, url: str, params: dict):
    """One guarded call; returns (rows, total, meta)."""
    r = s.get(url, params=params, timeout=30)
    ct = r.headers.get("content-type","")
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

def _fetch_single_range(s, d1: str, d2: str, log):
    """Fetch full date range without chunking."""
    search_opts = ["", "P"]
    seg_opts    = ["C", "E"]
    subcat_opts = ["", "-1"]
    pageno_keys = ["pageno", "Pageno"]
    scrip_keys  = ["strScrip", "strscrip"]

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

                            log.append(f"Trying {ep} | {pageno_key} | {scrip_key} | Type={strType}")

                            rows_acc = []
                            page = 1

                            while True:
                                rows, total, meta = _call_once(s, ep, params)

                                if meta.get("blocked"):
                                    log.append("Blocked: retry warmup")
                                    try:
                                        s.get(HOME, timeout=10)
                                        s.get(CORP, timeout=10)
                                    except:
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
    """Fetch full date range once — NO throttle, NO chunks."""
    if log is None:
        log = []

    s = requests.Session()
    s.headers.update(BASE_HEADERS)

    # warmup
    try:
        s.get(HOME, timeout=15)
        s.get(CORP, timeout=15)
    except:
        pass

    log.append(f"Full fetch: {start_yyyymmdd}..{end_yyyymmdd}")

    all_rows = _fetch_single_range(s, start_yyyymmdd, end_yyyymmdd, log)

    if not all_rows:
        return pd.DataFrame(columns=[
            "SCRIP_CD","SLONGNAME","HEADLINE","NEWSSUB",
            "NEWS_DT","ATTACHMENTNAME","NSURL"
        ])

    base_cols = ["SCRIP_CD","SLONGNAME","HEADLINE","NEWSSUB",
                 "NEWS_DT","ATTACHMENTNAME","NSURL","NEWSID"]

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
        df["_NEWS_DT_PARSED"] = pd.to_datetime(df["NEWS_DT"], errors="coerce", dayfirst=True)
        df = (
            df.sort_values("_NEWS_DT_PARSED", ascending=False)
              .drop(columns=["_NEWS_DT_PARSED"])
              .reset_index(drop=True)
        )

    return df

# --------------------
# Filters: Orders + Capex
# --------------------

ORDER_KEYWORDS = ["order","contract","bagged","supply","purchase order"]
ORDER_REGEX = re.compile(r"\b(?:" + "|".join(map(re.escape, ORDER_KEYWORDS)) + r")\b", re.IGNORECASE)

CAPEX_KEYWORDS = [
    "capex","capital expenditure","capacity expansion",
    "new plant","manufacturing facility","brownfield","greenfield",
    "setting up a plant","increase in capacity","expansion",
    "commercial production"  # added as requested
]
CAPEX_REGEX = re.compile("|".join(CAPEX_KEYWORDS), re.IGNORECASE)

def enrich_orders(df):
    if df.empty: 
        return df
    mask = df["HEADLINE"].fillna("").str.contains(ORDER_REGEX)
    out = df.loc[mask, ["SLONGNAME","HEADLINE","NEWSSUB","NEWS_DT","NSURL"]].copy()
    out.columns = ["Company","Announcement","Details","Date","Link"]
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce", dayfirst=True)
    return out.sort_values("Date", ascending=False).reset_index(drop=True)

def enrich_capex(df):
    if df.empty: 
        return df
    combined = (df["HEADLINE"].fillna("") + " " + df["NEWSSUB"].fillna(""))
    mask = combined.str.contains(CAPEX_REGEX, na=False)
    out = df.loc[mask, ["SLONGNAME","HEADLINE","NEWSSUB","NEWS_DT","NSURL"]].copy()
    out.columns = ["Company","Announcement","Details","Date","Link"]
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce", dayfirst=True)
    return out.sort_values("Date", ascending=False).reset_index(drop=True)

# --------------------
# OpenAI helpers
# --------------------

def get_openai_client():
    api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
    if not api_key:
        st.error("Missing OPENAI_API_KEY (env var or Streamlit secrets).")
        st.stop()
    return OpenAI(api_key=api_key)

def call_openai(client: OpenAI, prompt: str, max_tokens: int = 400, temperature: float = 0.15) -> str:
    """Thin wrapper around Responses API returning plain text."""
    resp = client.responses.create(
        model=OPENAI_MODEL,
        temperature=temperature,
        max_output_tokens=max_tokens,
        input=prompt,
    )
    # new SDK exposes helper for text
    return (resp.output_text or "").strip()

def enrich_orders_with_openai(orders_df: pd.DataFrame, client: OpenAI) -> pd.DataFrame:
    """Add TTM Revenue, Market Cap, Existing & Current Order Book via OpenAI."""
    if orders_df.empty:
        return orders_df

    df = orders_df.copy()
    cols = [
        "TTM Revenue (₹ Cr)",
        "Market Cap (₹ Cr)",
        "Existing Order Book (₹ Cr)",
        "Current Order Book (₹ Cr)",
    ]
    for c in cols:
        df[c] = np.nan

    for idx, row in df.head(MAX_OPENAI_ROWS).iterrows():
        company = str(row["Company"])
        ann = str(row["Announcement"])
        details = str(row.get("Details") or "")

        prompt = f"""
You are a fundamental equity analyst.

Using only your prior financial knowledge plus the announcement text below, do your best to estimate the following for the listed Indian company. 
If you genuinely cannot estimate a value, set it to null (not a string).

Return ONLY valid JSON in this exact structure, no commentary, no markdown:

{{
  "ttm_revenue_cr": <number or null>,
  "market_cap_cr": <number or null>,
  "existing_order_book_cr": <number or null>,
  "current_order_book_cr": <number or null>
}}

Company: {company}
Announcement headline: {ann}
Announcement details: {details}
"""

        try:
            txt = call_openai(client, prompt, max_tokens=350)
            data = json.loads(
                txt.strip()
                .replace("```json", "")
                .replace("```", "")
            )
        except Exception:
            # if anything goes wrong, just leave NaNs and continue
            continue

        df.at[idx, "TTM Revenue (₹ Cr)"] = data.get("ttm_revenue_cr")
        df.at[idx, "Market Cap (₹ Cr)"] = data.get("market_cap_cr")
        df.at[idx, "Existing Order Book (₹ Cr)"] = data.get("existing_order_book_cr")
        df.at[idx, "Current Order Book (₹ Cr)"] = data.get("current_order_book_cr")

    return df

def enrich_capex_with_openai(capex_df: pd.DataFrame, client: OpenAI) -> pd.DataFrame:
    """Add an 'Impact' column, focusing on commercial production–type capex."""
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
            # you said: filter capex announcements using keywords like commercial production
            continue

        prompt = f"""
You are a sell-side equity research analyst writing a short impact commentary for a capex / plant-commissioning announcement.

Write a crisp impact paragraph (3–5 sentences, max ~130 words) explaining:
- what has commenced (e.g., commercial production of which product / capacity),
- how this shifts the company's margin / growth trajectory versus the core business,
- approximate revenue/EBITDA potential range if it is clear from the text (in INR crores, not percentages),
- key execution / market risks that an investor should monitor.

Tone: analytical, neutral-positive, no hype, no bullet points, no headings, just a single paragraph of plain text.

Company: {company}
Announcement: {ann}
Details: {details}
"""

        try:
            impact = call_openai(client, prompt, max_tokens=220, temperature=0.25)
        except Exception:
            impact = ""

        df.at[idx, "Impact"] = impact

    return df

# --------------------
# Streamlit UI
# --------------------

st.set_page_config(page_title="BSE Order & Capex Announcements (OpenAI-enriched)", layout="wide")
st.title("📣 BSE Order & Capex Announcements Finder (with OpenAI Enrichment)")

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=date(2025,1,1))
with col2:
    end_date = st.date_input("End Date", value=date.today())

run = st.button("🔎 Fetch & Enrich", use_container_width=True)

if run:
    if start_date > end_date:
        st.error("Start Date cannot be after End Date.")
        st.stop()

    ds = start_date.strftime("%Y%m%d")
    de = end_date.strftime("%Y%m%d")
    logs = []

    with st.spinner("Fetching BSE announcements..."):
        df = fetch_bse_announcements_strict(ds, de, log=logs)

    orders_df = enrich_orders(df)
    capex_df = enrich_capex(df)

    st.subheader("High-level counts")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Announcements", len(df))
    c2.metric("Order Announcements", len(orders_df))
    c3.metric("Capex Announcements (incl. commercial production)", len(capex_df))

    if df.empty:
        st.warning("No announcements found for this range.")
        st.stop()

    # OpenAI client (only if we actually have rows to enrich)
    client = None
    if not orders_df.empty or not capex_df.empty:
        client = get_openai_client()

    if client is not None and not orders_df.empty:
        with st.spinner(f"Enriching top {min(MAX_OPENAI_ROWS, len(orders_df))} order announcements with OpenAI..."):
            orders_df = enrich_orders_with_openai(orders_df, client)

    if client is not None and not capex_df.empty:
        with st.spinner(f"Generating 'Impact' for top {min(MAX_OPENAI_ROWS, len(capex_df))} capex announcements (commercial production focused)..."):
            capex_df = enrich_capex_with_openai(capex_df, client)

    tab_orders, tab_capex, tab_all, tab_logs = st.tabs(["📦 Orders (Enriched)", "🏭 Capex (Impact)", "📄 All Raw", "🧪 Fetch Logs"])

    with tab_orders:
        st.caption("Includes OpenAI-estimated TTM Revenue, Market Cap and Order Book metrics (where available).")
        st.dataframe(orders_df, use_container_width=True)

    with tab_capex:
        st.caption("Filtered using capex keywords including 'commercial production'. Impact column is generated via OpenAI for top rows.")
        st.dataframe(capex_df, use_container_width=True)

    with tab_all:
        st.dataframe(df, use_container_width=True)

    with tab_logs:
        st.write("Fetcher debug log:")
        for line in logs:
            st.text(line)
