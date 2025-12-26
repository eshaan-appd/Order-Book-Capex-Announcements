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

# Two BSE endpoints with slightly different parameter contracts.
ENDPOINTS = [
    "https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w",
    "https://api.bseindia.com/BseIndiaAPI/api/AnnGetData/w",
]

# Common headers to avoid being blocked as a bot.
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
MAX_OPENAI_ROWS = 200  # cap mainly used for capex; orders enrich ALL rows now

# =========================================
# OpenAI helpers
# =========================================

def get_openai_client() -> OpenAI:
    """
    Initialise the OpenAI client, reading the API key from
    Streamlit secrets or environment variable.
    """
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
    Call the Responses API with:
      - web_search tool enabled
      - optional PDF file attached (file_id)
    Expect a JSON string back and return it as a Python dict.

    If parsing fails, return None and let the caller fall back / skip.
    """
    # Build multi-part content: text + optional file.
    content = [{"type": "input_text", "text": prompt}]
    if file_id:
        content.append({"type": "input_file", "file_id": file_id})

    resp = client.responses.create(
        model=OPENAI_MODEL,
        temperature=temperature,
        max_output_tokens=max_tokens,
        tools=[{"type": "web_search"}],
        input=[{"role": "user", "content": content}],
    )

    txt = (getattr(resp, "output_text", None) or "").strip()
    if not txt:
        return None

    # Clean wrappers / smart quotes that sometimes appear in model output.
    cleaned = (
        txt.strip()
        .replace("```json", "")
        .replace("```", "")
        .replace("“", '"')
        .replace("”", '"')
        .replace("’", "'")
    )

    # Try to isolate the JSON object region explicitly.
    if "{" in cleaned and "}" in cleaned:
        cleaned = cleaned[cleaned.find("{") : cleaned.rfind("}") + 1]

    try:
        return json.loads(cleaned)
    except Exception:
        return None

def upload_pdf_to_openai(client: OpenAI, pdf_bytes: bytes, fname: str = "doc.pdf"):
    """
    Persist a PDF to OpenAI so it can be referenced in the Responses call.
    """
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
    Build a list of possible PDF URLs from ATTACHMENTNAME + NSURL for a row.
    We try multiple base paths because BSE moves attachments between folders.
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

    # De-duplicate while preserving order.
    seen, out = set(), []
    for u in cands:
        if u and u not in seen:
            out.append(u)
            seen.add(u)
    return out

def primary_bse_url(row) -> str:
    """
    Compose a full BSE URL from the NSURL field.
    """
    ns = str(row.get("NSURL") or "").strip()
    if not ns:
        return ""
    return ns if ns.lower().startswith("http") else HOME + ns.lstrip("/")

def download_pdf(url: str, timeout: int = 25) -> bytes | None:
    """
    Download PDF bytes from a URL. Return None if we fail or content is too small.
    """
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
    if not data or len(data) < 500:  # tiny PDFs are usually error pages
        return None
    return data

# =========================================
# Backend: BSE fetcher
# =========================================

def _call_once(s: requests.Session, url: str, params: dict):
    """
    Single low-level call to the BSE API.
    Returns (rows, total, meta) where meta captures 'blocked' state.
    """
    r = s.get(url, params=params, timeout=30)
    ct = r.headers.get("content-type", "")
    if "application/json" not in ct:
        # If we see HTML (block page / error), mark this attempt as blocked.
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
    permutations to deal with BSE's inconsistent API behaviour.
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

                                # If we got HTML / block page, warm up and retry once.
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
                                    break  # no results for this configuration

                                if not rows:
                                    break  # exhausted pages

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

    The BSE API behaves best when strPrevDate == strToDate, so we:
      1. Loop day-by-day over the range.
      2. Call _fetch_single_range for each day.
      3. Concatenate & de-duplicate the results.
    """
    if log is None:
        log = []

    # Single session + warmup to reduce TLS overhead.
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
        # Defensive: inverted range -> empty DF.
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

    # Build DataFrame with dynamic extra columns if present.
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

    # De-duplicate using reasonably stable keys.
    keys = ["NSURL", "NEWSID", "ATTACHMENTNAME", "HEADLINE"]
    keys = [k for k in keys if k in df.columns]
    if keys:
        df = df.drop_duplicates(subset=keys)

    # Sort by NEWS_DT descending (most recent first).
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

# Keywords that mark an announcement as "order / contract".
ORDER_KEYWORDS = ["order", "contract", "bagged", "supply", "purchase order"]
ORDER_REGEX = re.compile(
    r"\b(?:" + "|".join(map(re.escape, ORDER_KEYWORDS)) + r")\b", re.IGNORECASE
)

# Capex / capacity / plant / expansion keywords (broader set).
CAPEX_KEYWORDS = [
    "commercial production",
    "commencement of commercial production",
    "commenced commercial production",
    "capex", "capital expenditure", "capacity expansion",
    "new plant", "manufacturing facility", "brownfield", "greenfield",
    "setting up a plant", "increase in capacity", "expansion"
]
CAPEX_REGEX = re.compile("|".join(map(re.escape, CAPEX_KEYWORDS)), re.IGNORECASE)

def enrich_orders(df: pd.DataFrame) -> pd.DataFrame:
    """
    Slice the raw announcements down to order-related items
    and rename columns into a clean front-end schema.
    """
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
    """
    Filter to announcements whose HEADLINE or NEWSSUB contain
    capex/expansion-related keywords, then map into a clean schema.

    The actual confirmation + impact for capex is done in
    enrich_capex_with_openai by reading the filing content.
    """
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
    Enrich order announcements with:
      - Latest Revenue (₹ Cr)         [internet via web_search]
      - Market Cap (₹ Cr)             [internet via web_search]
      - Order Amount (₹ Cr)           [PDF filing + text ONLY]
      - Current Order Book (₹ Cr)     [internet + filings/text]
      - Order Book / Sales (x)        [derived metric = OB / Revenue]
      - Execution Timeline            [PDF first → else forecast]

    NOTE: This now enriches **all rows** in orders_df (no head() truncation).
    """

    if orders_df.empty:
        return orders_df

    df = orders_df.copy()

    # ---------- initialise enrichment columns ----------
    df["TTM / Latest Revenue (₹ Cr)"] = np.nan
    df["Market Cap (₹ Cr)"] = np.nan
    df["Order Amount (₹ Cr)"] = np.nan
    df["Current Order Book (₹ Cr)"] = np.nan
    df["Order Book / Sales (x)"] = np.nan
    df["Execution Timeline"] = ""   # may come from PDF OR forecast

    # Join back to raw BSE row for attachment lookup
    raw_key = raw_df.set_index(["SLONGNAME", "HEADLINE"])

    # >>> ENRICH ALL ROWS (no .head(MAX_OPENAI_ROWS)) <<<
    for idx, row in df.iterrows():

        company = str(row["Company"])
        ann     = str(row["Announcement"])
        details = str(row.get("Details") or "")
        date_val = str(row["Date"].date()) if pd.notnull(row["Date"]) else ""

        # ---------- Locate PDF ----------
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
                    pass
            time.sleep(0.2)

        # ---------- PROMPT ----------
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

        data = call_openai_json(client, prompt, file_id=file_id, max_tokens=750, temperature=0.1)
        if not data:
            continue

        # ---------- helpers ----------
        def _f(x):
            if x is None:
                return np.nan
            try:
                return float(str(x).replace(",", ""))
            except Exception:
                return np.nan

        # ---------- assign ----------
        revenue = _f(data.get("ttm_revenue_cr"))
        mcap    = _f(data.get("market_cap_cr"))
        order   = _f(data.get("order_amount_cr"))
        ob_cur  = _f(data.get("current_order_book_cr"))

        df.at[idx, "TTM / Latest Revenue (₹ Cr)"] = revenue
        df.at[idx, "Market Cap (₹ Cr)"] = mcap
        df.at[idx, "Order Amount (₹ Cr)"] = order
        df.at[idx, "Current Order Book (₹ Cr)"] = ob_cur

        # ---------- derived ----------
        if pd.notna(revenue) and revenue > 0 and pd.notna(ob_cur):
            df.at[idx, "Order Book / Sales (x)"] = ob_cur / revenue

        # ---------- execution timeline ----------
        timeline = data.get("execution_timeline")
        if timeline:
            df.at[idx, "Execution Timeline"] = str(timeline).strip()

    return df

# =========================================
# OpenAI enrichment: Capex Impact (PDF keyword search)
# =========================================

def enrich_capex_with_openai(capex_df: pd.DataFrame, client: OpenAI) -> pd.DataFrame:
    """
    Add an 'Impact' paragraph for capex / expansion announcements.

    For each candidate:
      - Upload the filing PDF (if available).
      - Ask OpenAI to SEARCH INSIDE the filing for the CAPEX_KEYWORDS list:
        ["commercial production", "capex", "capacity expansion", "new plant",
         "manufacturing facility", "brownfield", "greenfield", etc.]
      - Use those matches + context to write an impact paragraph.

    This still limits to MAX_OPENAI_ROWS rows per run for cost control.
    """
    if capex_df.empty:
        return capex_df

    df = capex_df.copy()
    df["Impact"] = ""

    for idx, row in df.head(MAX_OPENAI_ROWS).iterrows():
        company = str(row["Company"])
        ann = str(row["Announcement"])
        details = str(row.get("Details") or "")

        # ---------- Locate PDF, using Attachment + Link from capex_df ----------
        pseudo_raw = {
            "ATTACHMENTNAME": row.get("Attachment"),
            "NSURL": row.get("Link"),
        }
        urls = candidate_pdf_urls(pseudo_raw)

        file_id = None
        for u in urls:
            pdf_bytes = download_pdf(u)
            if pdf_bytes:
                try:
                    fobj = upload_pdf_to_openai(client, pdf_bytes, fname="capex.pdf")
                    file_id = fobj.id
                    break
                except Exception:
                    pass
            time.sleep(0.2)

        # Build the keyword list string for the prompt.
        keyword_list_str = ", ".join(CAPEX_KEYWORDS)

        # ---------- PROMPT: explicitly instruct model to search keywords in filing ----------
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

        # For Impact we want text; attach PDF if available.
        content = [{"type": "input_text", "text": prompt}]
        if file_id:
            content.append({"type": "input_file", "file_id": file_id})

        resp = client.responses.create(
            model=OPENAI_MODEL,
            temperature=0.25,
            max_output_tokens=280,
            input=[{"role": "user", "content": content}],
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

# Date range inputs.
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=date(2025, 1, 1))
with col2:
    end_date = st.date_input("End Date", value=date.today())

run = st.button("🔎 Fetch & Enrich", use_container_width=True)

if run:
    # Basic sanity check.
    if start_date > end_date:
        st.error("Start Date cannot be after End Date.")
        st.stop()

    ds = start_date.strftime("%Y%m%d")
    de = end_date.strftime("%Y%m%d")
    logs: list[str] = []

    # ---------------------------
    # 1) Fetch raw BSE data
    # ---------------------------
    with st.spinner("Fetching BSE announcements..."):
        df_raw = fetch_bse_announcements_strict(ds, de, log=logs)

    # Pre-filter into Orders and Capex sets (headline-based).
    orders_df = enrich_orders(df_raw)
    capex_df = enrich_capex(df_raw)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Announcements", len(df_raw))
    c2.metric("Order Announcements", len(orders_df))
    c3.metric(
        "Capex / Expansion Candidates (headline-keyword filtered)", len(capex_df)
    )

    if df_raw.empty:
        st.warning("No announcements found for this date range.")
        st.stop()

    # ---------------------------
    # 2) OpenAI enrichment
    # ---------------------------
    client = get_openai_client()

    if not orders_df.empty:
        with st.spinner(
            "Enriching ALL order announcements via internet + PDF..."
        ):
            orders_df = enrich_orders_with_openai(orders_df, df_raw, client)

    if not capex_df.empty:
        with st.spinner(
            f"Generating 'Impact' commentary for up to {min(MAX_OPENAI_ROWS, len(capex_df))} capex / expansion filings via PDF keyword search..."
        ):
            capex_df = enrich_capex_with_openai(capex_df, client)

    # ---------------------------
    # 3) Tabs & display
    # ---------------------------
    tab_orders, tab_capex, tab_all, tab_logs = st.tabs(
        ["📦 Orders (Enriched)", "🏭 Capex / Expansion (Impact)", "📄 All Raw", "🧪 Fetch Logs"]
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

    with tab_logs:
        for line in logs:
            st.text(line)
