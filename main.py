import streamlit as st
import camelot
import pdfplumber
import contextlib
import pandas as pd
import tempfile
import re, os, io, collections
from openai import OpenAI
from dotenv import load_dotenv
import fitz  # PyMuPDF

# Load environment variables and initialize OpenAI client if API key is provided.
# Ensure `client` exists (possibly None) so callers can safely check `if not client:`
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = None
if OPENAI_API_KEY:
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        # If initialization fails, leave client as None and allow graceful degradation.
        client = None


# Improve dark-theme readability: force table/cell text to a dark color and ensure
# row background styles remain visible. This helps when the Streamlit theme
# applies white text on light-ish backgrounds making content unreadable.
_DARK_FRIENDLY_CSS = """
<style>
/* Force dark/readable text on all tables rendered in the app */
/* Covers Streamlit dataframes, styled pandas tables, markdown tables and expanders */
.stApp table td, .stApp table th,
div[data-testid="stDataFrame"] table td, div[data-testid="stDataFrame"] table th,
.stDataFrame table td, .stDataFrame table th,
.streamlit-expander table td, .streamlit-expander table th,
div[data-testid="stMarkdownContainer"] table td, div[data-testid="stMarkdownContainer"] table th,
table.dataframe td, table.dataframe th,
table td, table th {
    color: #0b1220 !important; /* enforce dark (black-ish) text */
}

/* Ensure download/button text is readable */
button[data-baseweb="button"], .stButton>button, .stDownloadButton>button {
    color: inherit !important;
}

/* Reduce overly-bright grid borders in dark themes */
.stDataFrame table, table.dataframe {
    border-color: rgba(15,18,32,0.06) !important;
}

/* Also ensure styled cell backgrounds don't force white text via inline styles */
div[data-testid="stDataFrame"] td, div[data-testid="stDataFrame"] th,
.stDataFrame td, .stDataFrame th, table.dataframe td, table.dataframe th {
    color: #0b1220 !important;
}
</style>
"""

# Inject CSS once (safe to call even if Streamlit re-renders)
try:
        st.markdown(_DARK_FRIENDLY_CSS, unsafe_allow_html=True)
except Exception:
        # If called in non-Streamlit context (unit tests), ignore
        pass


def detect_repeating_lines(pdf, top_n=8, bottom_n=3, min_repeat=2):
    """Detect repeating top/bottom lines across pages — likely headers or footers."""
    header_counter, footer_counter = collections.Counter(), collections.Counter()
    for page in pdf.pages:
        text = page.extract_text() or ""
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for ln in lines[:top_n]:
            header_counter[ln] += 1
        for ln in lines[-bottom_n:]:
            footer_counter[ln] += 1
    headers = {ln for ln, c in header_counter.items() if c >= min_repeat}
    footers = {ln for ln, c in footer_counter.items() if c >= min_repeat}
    return headers, footers


def extract_pdfplumber_pagewise(path):
    """Smart pdfplumber extractor removing headers & footers dynamically."""
    txn_pattern = re.compile(r"^S\d+|\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}")
    all_pages = []

    # Suppress noisy PDF parser stderr output (e.g. incorrect startxref pointer)
    with contextlib.redirect_stderr(io.StringIO()):
        with pdfplumber.open(path) as pdf:
            headers, footers = detect_repeating_lines(pdf)

            for page_idx, page in enumerate(pdf.pages, start=1):
                rows, current = [], ""
                text = page.extract_text() or ""
                lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

                for ln in lines:
                    if ln in headers or ln in footers:
                        continue

                    if page_idx > 1 and lines.index(ln) < 4 and not txn_pattern.match(ln):
                        continue

                    if txn_pattern.match(ln):
                        if current:
                            rows.append(current.strip())
                        current = ln
                    else:
                        current += " " + ln

                if current:
                    rows.append(current.strip())
                all_pages.append(pd.DataFrame(rows, columns=["Raw"]))

    return all_pages

def ensure_canonical_deposits_column(df: pd.DataFrame, num_re: re.Pattern) -> str:
    '''Normalize deposit-like columns so a single 'Deposits' column is always present.'''
    if df.empty:
        return ''

    dep_re = re.compile(r"(deposit|deposits|cr|credit)", re.I)
    insert_loc = max(len(df.columns) - 1, 0)

    deposit_candidates = [c for c in df.columns if dep_re.search(str(c))]
    canonical_col = None

    if deposit_candidates:
        canonical_col = deposit_candidates[0]
        if canonical_col != 'Deposits':
            if 'Deposits' in df.columns:
                canonical_col = 'Deposits'
            else:
                df.rename(columns={canonical_col: 'Deposits'}, inplace=True)
                canonical_col = 'Deposits'

        for other in deposit_candidates[1:]:
            if other == canonical_col or other not in df.columns:
                continue
            for idx in df.index:
                try:
                    target_val = str(df.at[idx, canonical_col]) if canonical_col in df.columns else ''
                except Exception:
                    target_val = ''
                try:
                    other_val = str(df.at[idx, other])
                except Exception:
                    other_val = ''
                if (not target_val or target_val.strip().lower() == 'nan') and other_val.strip().lower() not in ('', 'nan'):
                    df.at[idx, canonical_col] = other_val
            try:
                df.drop(columns=[other], inplace=True)
            except Exception:
                pass
    else:
        empty_cols = [c for c in df.columns if str(c).strip() == '']
        if empty_cols:
            canonical_col = empty_cols[0]
            df.rename(columns={canonical_col: 'Deposits'}, inplace=True)
            canonical_col = 'Deposits'
        else:
            base_name = 'Deposits'
            canonical_col = base_name
            idx_counter = 1
            while canonical_col in df.columns:
                canonical_col = f"{base_name}_{idx_counter}"
                idx_counter += 1
            df.insert(insert_loc, canonical_col, [''] * len(df))

    try:
        dep_like = [c for c in df.columns if dep_re.search(str(c))]
        cols_list = list(df.columns)
        for c in dep_like:
            try:
                col_vals = df[c].astype(str).str.strip().replace('nan', '')
            except Exception:
                continue
            if col_vals.eq('').all():
                idx = cols_list.index(c)
                if idx < len(cols_list) - 1:
                    next_col = cols_list[idx + 1]
                    try:
                        has_numeric = df[next_col].astype(str).str.contains(num_re).any()
                    except Exception:
                        has_numeric = False
                    if has_numeric:
                        try:
                            df.drop(columns=[c], inplace=True)
                        except Exception:
                            pass
                        if 'Deposits' not in df.columns:
                            try:
                                df.rename(columns={next_col: 'Deposits'}, inplace=True)
                            except Exception:
                                pass
                        cols_list = list(df.columns)
                        break
    except Exception:
        pass

    try:
        dep_like = [c for c in df.columns if dep_re.search(str(c))]
        if dep_like:
            canonical = None
            for c in dep_like:
                if str(c).strip().lower() == 'deposits':
                    canonical = c
                    break
            if canonical is None:
                canonical = dep_like[0]
                if canonical != 'Deposits' and 'Deposits' not in df.columns:
                    try:
                        df.rename(columns={canonical: 'Deposits'}, inplace=True)
                        canonical = 'Deposits'
                    except Exception:
                        pass

            for other in dep_like:
                if other == canonical or other not in df.columns:
                    continue
                for idx in df.index:
                    try:
                        v_can = str(df.at[idx, canonical]) if canonical in df.columns else ''
                    except Exception:
                        v_can = ''
                    try:
                        v_oth = str(df.at[idx, other])
                    except Exception:
                        v_oth = ''
                    if (not v_can or v_can.strip().lower() in ('', 'nan')) and v_oth.strip().lower() not in ('', 'nan'):
                        df.at[idx, canonical] = v_oth
                try:
                    df.drop(columns=[other], inplace=True)
                except Exception:
                    pass

            if canonical != 'Deposits' and 'Deposits' not in df.columns:
                try:
                    df.rename(columns={canonical: 'Deposits'}, inplace=True)
                except Exception:
                    pass
    except Exception:
        pass

    if 'Deposits' not in df.columns:
        insert_loc = max(len(df.columns) - 1, 0)
        df.insert(insert_loc, 'Deposits', [''] * len(df))
        canonical_col = 'Deposits'

    return 'Deposits' if 'Deposits' in df.columns else (canonical_col or '')


def extract_header_info(path):
    """Extract Bank Name (even if logo-only), Account Name, and Account Number automatically."""
    bank_name = ""
    account_name = ""
    account_number = ""

    try:
        # -------- 1️⃣  Extract all text from first 2 pages --------
        with fitz.open(path) as doc:
            text_all = ""
            for page in doc[:2]:
                text_all += page.get_text("text") + "\n"

            # -------- 2️⃣  BANK NAME detection --------
            # (a) Try direct textual extraction
            m_bank = re.search(r"([A-Z][A-Za-z\s]+Bank\s+of\s+[A-Za-z]+|Bank\s+of\s+[A-Za-z]+)", text_all, re.I)
            if m_bank:
                bank_name = m_bank.group(1).strip().title()

            # (b) Infer from IFSC prefix if not found
            if not bank_name:
                m_ifsc = re.search(r"IFSC\s*Code\s*[:\-]?\s*([A-Z]{4}\d{7})", text_all, re.I)
                if m_ifsc:
                    prefix = m_ifsc.group(1)[:4].upper()
                    bank_map = {
                        "BKID": "Bank of India",
                        "SBIN": "State Bank of India",
                        "HDFC": "HDFC Bank",
                        "ICIC": "ICICI Bank",
                        "UTIB": "Axis Bank",
                        "PUNB": "Punjab National Bank",
                        "KARB": "Karnataka Bank",
                        "YESB": "Yes Bank",
                        "IDFB": "IDFC FIRST Bank",
                        "BARB": "Bank of Baroda",
                    }
                    bank_name = bank_map.get(prefix, f"Unknown Bank ({prefix})")

            # (c) If still blank, detect from image metadata
            if not bank_name:
                for page in doc:
                    for img in page.get_images(full=True):
                        try:
                            base = doc.extract_image(img[0])
                            imgname = base.get("name", "").lower()
                            if any(k in imgname for k in ["boi", "bankofindia", "bki"]):
                                bank_name = "Bank of India"
                                break
                        except Exception:
                            continue
                    if bank_name:
                        break

        # -------- 3️⃣  ACCOUNT NUMBER detection --------
        # Capture long numeric strings after “Account No” or “A/c No”
        acc_match = re.search(r"(Account\s*(?:No\.?|Number|#)\s*[:\-]?\s*)(\d{6,})", text_all, re.I)
        if acc_match:
            account_number = acc_match.group(2).strip()
        else:
            # Fallback: look for standalone 12–18 digit patterns near 'Account' keyword
            acc_line = re.search(r"Account\s*[:\-]?\s*[A-Z0-9\s]*?(\d{10,18})", text_all, re.I)
            if acc_line:
                account_number = acc_line.group(1).strip()

        # -------- 4️⃣  ACCOUNT NAME detection --------
        name_candidates = re.findall(r"Name\s*[:–-]\s*([A-Z0-9,&\.\-\s\(\)]+)", text_all, re.I)
        if name_candidates:
            for cand in reversed(name_candidates):
                cand = re.sub(r"\s+", " ", cand.strip())
                if len(cand) > 3 and not re.search(r"infotech|traders|co\s", cand, re.I):
                    account_name = cand
                    break

        # -------- 5️⃣  PdfPlumber fallback --------
        if not account_name:
            with contextlib.redirect_stderr(io.StringIO()):
                with pdfplumber.open(path) as pdf:
                    for page in pdf.pages[:2]:
                        text = page.extract_text() or ""
                        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
                        for ln in lines:
                            if re.match(r"^[A-Z\s,&\.]{10,}$", ln) and not re.search(r"(branch|address|statement|date|customer|ifsc|account)", ln, re.I):
                                account_name = ln.strip()
                                break
                        if account_name:
                            break

    except Exception as e:
        print("Header extraction error:", e)

    return bank_name, account_name, account_number






def _is_metadata_line(s: str) -> bool:
    """Return True for lines that look like combined metadata (Date:, Address:, Customer ID, File:, Account No etc.).

    This is used to avoid showing a single long metadata blob in the UI.
    """
    if s is None:
        return False
    t = str(s).strip()
    if not t:
        return False
    # Common metadata tokens that indicate a combined metadata blob
    if re.search(r"\bDate\b\s*:|\bAddress\b\s*:|Customer\s*ID|\bAccount\b\s*(?:No|Number|:)|\bFile\b\s*:", t, re.I):
        return True
    # Very long single-line strings often are full metadata blobs
    if len(t) > 180:
        return True
    return False


def credit_debit_checker(current, previous, return_type: str = "label"):
    """
    Determine whether the movement from `previous` to `current` is a credit or a debit.

    Notes:
    - This function accepts numeric values or strings containing numbers (commas, parentheses,
      negative signs, extra spaces). It will try to parse the first numeric token found.
    - Calculation: diff = current - previous
        * diff > 0 => Credit
        * diff < 0 => Debit
        * diff == 0 => NoChange
    - return_type controls the output:
        * "label" (default) -> returns one of: "Credit", "Debit", "NoChange", or None if N/A
        * "sign" -> returns 1 (credit), -1 (debit), 0 (no change), or None
        * "diff" -> returns the numeric difference (float) or None if parsing failed

    This function is defined for future use and is not applied automatically.
    """

    def _to_num(x):
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip()
        if s == "":
            return None
        # remove non-breaking spaces and zero-width spaces
        s = s.replace('\u200b', '').replace('\xa0', ' ')
        # detect parentheses style negative: (1,234.56)
        is_paren_negative = False
        if s.startswith('(') and s.endswith(')'):
            is_paren_negative = True
            s = s[1:-1]
        # find first numeric token (allow commas and optional decimals)
        m = re.search(r"-?[\d,]+(?:\.\d+)?", s)
        if not m:
            return None
        num_str = m.group(0).replace(',', '')
        try:
            val = float(num_str)
        except Exception:
            return None
        if is_paren_negative and val > 0:
            val = -val
        return val

    curr = _to_num(current)
    prev = _to_num(previous)

    if curr is None or prev is None:
        return None if return_type in ("label", "sign") else None

    diff = curr - prev
    if return_type == "diff":
        return diff
    if diff > 0:
        return ("Credit" if return_type == "label" else 1)
    if diff < 0:
        return ("Debit" if return_type == "label" else -1)
    return ("NoChange" if return_type == "label" else 0)


def extract_camelot_pagewise(path, flavor):
    """Extract tables page-wise using Camelot."""
    all_pages = []
    try:
        # Suppress noisy PDF parser stderr output when checking page count
        with contextlib.redirect_stderr(io.StringIO()):
            with pdfplumber.open(path) as pdf:
                total_pages = len(pdf.pages)
        for i in range(1, total_pages + 1):
            try:
                # Suppress any stderr noise emitted by Camelot/pdf parsing
                with contextlib.redirect_stderr(io.StringIO()):
                    tables = camelot.read_pdf(path, pages=str(i), flavor=flavor)
                if tables and len(tables) > 0:
                    df = pd.concat([t.df for t in tables], ignore_index=True)
                else:
                    df = pd.DataFrame()
            except Exception:
                df = pd.DataFrame()
            # Post-process: detect and split merged Deposit+Balance values that Camelot sometimes puts
            # into a single cell (commonly the last column). We look for two numeric tokens in the last
            # column (e.g. "13,48,348.00 -3,49,48,64,718.53") and split them into a new Deposits column
            # and a cleaned Balance column. We keep original formatting (commas/decimals) intact.
            try:
                if not df.empty:
                    # Ensure strings for regex processing
                    df = df.astype(str)
                    # Normalize column names (collapse whitespace/newlines) to make header detection robust
                    df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]

                    # If Camelot left the header row as the first data row, promote it.
                    # Dynamic detection: examine the first row cells and decide if they look like headers
                    # based on (a) presence of alphabetic tokens vs numeric tokens and (b) optional keyword matches.
                    try:
                        first_row_raw = df.iloc[0].astype(str).replace('\n', ' ')
                        first_row = first_row_raw.str.strip()

                        # common header keywords to help detection (used as hints, not hard enforcement)
                        common_kw = ['date', 'txn', 'transaction', 'description', 'narration', 'particulars',
                                     'cheque', 'ref', 'sl', 'no', 'withdraw', 'deposit', 'balance', 'value']

                        def looks_like_header_cell(s: str) -> bool:
                            s = str(s).strip()
                            if not s:
                                return False
                            # if contains letters and is relatively short -> likely header
                            letters = sum(ch.isalpha() for ch in s)
                            digits = sum(ch.isdigit() for ch in s)
                            # treat cells with more letters than digits as header-like
                            if letters >= max(1, digits):
                                return True
                            # or if contains a common keyword
                            low = s.lower()
                            for kw in common_kw:
                                if kw in low:
                                    return True
                            return False

                        header_like_count = sum(1 for v in first_row.values if looks_like_header_cell(v))
                        # threshold: consider it a header row if a reasonable fraction of cells look header-like
                        threshold = max(2, int(len(df.columns) * 0.3))
                        if header_like_count >= threshold:
                            # promote first row to header
                            new_cols = [re.sub(r"\s+", " ", str(x)).strip() for x in df.iloc[0].values]
                            df = df[1:].reset_index(drop=True)
                            df.columns = new_cols
                            # normalize again
                            df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]
                    except Exception:
                        pass
                    num_re = re.compile(r"-?[\d,]+\.\d{2}")
                    last_col_name = df.columns[-1]

                                        # Rows where last column contains 2+ numeric tokens
                    merged_mask = df[last_col_name].apply(lambda s: len(num_re.findall(s)) >= 2)
                    deposits_vals = []
                    balance_vals = []
                    if merged_mask.any():
                        for val in df[last_col_name].astype(str):
                            nums = num_re.findall(val)
                            if len(nums) >= 2:
                                deposits_vals.append(nums[0])
                                balance_vals.append(nums[-1])
                            else:
                                deposits_vals.append("")
                                balance_vals.append(val)

                    canonical_col = ensure_canonical_deposits_column(df, num_re) or ("Deposits" if "Deposits" in df.columns else None)

                    if merged_mask.any() and canonical_col:
                        for idx, row_idx in enumerate(df.index):
                            if idx >= len(deposits_vals):
                                continue
                            if not merged_mask.iloc[idx]:
                                continue
                            try:
                                existing = str(df.at[row_idx, canonical_col])
                            except Exception:
                                existing = ""
                            if existing.strip() in ("", "nan"):
                                df.at[row_idx, canonical_col] = deposits_vals[idx]

                        try:
                            df[last_col_name] = pd.Series(balance_vals, index=df.index)
                        except Exception:
                            df[last_col_name] = balance_vals
            except Exception:
                # Don't fail extraction for individual pages; leave df as-is on error
                pass
            all_pages.append(df)
    except Exception as e:
        st.warning(f"Camelot failed: {e}")
    return all_pages


# Remove duplicate/empty ai_enhance above and replace the implementation below.


def render_metric_card(value, label):
    """Render a styled metric card."""
    st.markdown(
        f"""
        <div class='metric-card'>
            <div class='metric-value'>{value}</div>
            <div class='metric-label'>{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_badge(status):
    """Render a color-coded status badge."""
    if "Match" in status and "✅" in status:
        return '<span class="success-badge">✅ Match</span>'
    elif "Mismatch" in status and "❌" in status:
        return '<span class="warning-badge">⚠️ Mismatch</span>'
    else:
        return '<span class="info-box">ℹ️ No Data</span>'


def ai_enhance(source_data, source_type: str = "text"):
    """AI table reconstruction.

    source_data: either a raw text string, a pandas.DataFrame, or CSV string.
    source_type: 'text' (default) or 'table' to hint model.
    Returns: pandas.DataFrame on success, empty DataFrame on failure.
    """
    if not client:
        st.warning("OpenAI API key not configured.")
        return pd.DataFrame()

    # Normalize input into a textual payload for the model.
    payload_text = ""
    if isinstance(source_data, pd.DataFrame):
        try:
            payload_text = source_data.to_csv(index=False)
            source_type = "table"
        except Exception:
            payload_text = "\n".join(source_data.astype(str).apply(lambda r: ", ".join(r), axis=1).tolist())
    elif isinstance(source_data, (list, tuple)):
        payload_text = "\n".join([str(x) for x in source_data])
    else:
        payload_text = str(source_data)

    # If passing a CSV-like table, tell the model the input is CSV.
    input_hint = "The following input is a CSV-style table." if source_type == "table" else "The following input is extracted page text."

    # Use the user-provided prompt template (strict CSV with fixed headers and rules).
    prompt = f"""
You are an expert financial document parser.
Your task is to extract only the transaction tables from a bank statement PDF (multi-page text).
Output the result as a clean, consistent CSV table with all rows and columns preserved exactly as they appear in the document.
 
OBJECTIVE
Extract all rows belonging to tabular transaction data, regardless of format or bank template.
Ignore headers, footers, disclaimers, summaries, or any non-transactional text.
 
DETECTING TABLE STRUCTURE
 
Identify the main transaction table(s) by locating recurring patterns of data in tabular form.
Common headers (may vary):
 
Date / Txn Date / Value Dt
 
Description / Narration / Particulars
 
Chq./Ref.No. / Transaction ID
 
Withdrawal / Debit / Dr Amount
 
Deposit / Credit / Cr Amount
 
Balance / Closing Balance
 
Determine the correct number and order of columns based on header structure on that page.
Use the same structure for all subsequent rows.
 
If the document has no explicit header on every page, reuse the previous detected structure.
 
HANDLING MULTI-LINE OR UNALIGNED DATA
 
If a cell's text continues on the next line but no new row pattern is detected, treat it as part of the same cell.
Example: multi-line narrations or beneficiary names.
 
If multiple entries exist under one header (e.g., multiple transactions for one date), treat each as a separate row, even if the date is repeated or omitted in some rows.
 
Do not merge distinct rows into one, even if spacing is inconsistent.
When in doubt, split based on natural tabular alignment or consistent repetition of numeric columns.
 
Maintain consistent column count for every row.
If some fields are missing, leave them blank ("").
 
DATA INTEGRITY RULES
 
Never merge numeric columns (e.g., Withdrawal, Deposit, Balance must always be separate).
 
Never duplicate or drop rows.
 
Never hallucinate or infer missing data.
 
Preserve the order of transactions exactly as in the PDF.
 
Keep numeric values and text exactly as in source (no rounding, guessing, or formatting).
 
Skip all non-transactional content like summaries, totals, account info, disclaimers, etc.
 
OUTPUT FORMAT (STRICT CSV)
 
You are required to output only CSV-formatted data.
The CSV must contain exactly 11 columns, with the following headers (in this exact order):
Sr. No., Bank Name, Account Number, File Name, Page Number, Date, Transaction Details, Cheque No., Amount Debit, Amount Credit, Balance
The output must:
Contain no extra columns beyond these 11.
Contain no markdown syntax (e.g., no csv or ``` marks).
Contain no explanations, commentary, or text before or after the CSV data.
Begin immediately with the header row followed by the corresponding data rows.
 
First line = headers (only once).
 
Each subsequent line = one transaction.
 
Use commas (,) as separators.
 
Wrap text fields in quotes (") if they contain commas.
 
Example:
Date,Description,Ref No,Value Date,Withdrawal,Deposit,Balance
01/01/25,"CORRESPONDENT BANK CHARGES","E33EBRL242540001",31/12/24,21348.47,,62119.14
03/01/25,"CORRESPONDENT BANK CHARGES","E33EBRL242650002",02/01/25,22242.33,,39876.81
03/01/25,"CORRESPONDENT BANK CHARGES","E33EBRL242540003",02/01/25,20629.63,,19247.18
06/01/25,"NEFT CR - PUNB0200200 - DCM NOUVELLE LTD UNIT DCM TEXTILES","PUNBZ25006921805",06/01/25,,50000,69247.18
13/01/25,"CORRESPONDENT BANK CHARGES","E33EBRL242650005",10/01/25,21808.31,,47438.87
 
VALIDATION REQUIREMENTS
Every output row corresponds to exactly one transaction from the input.
Same number of columns for all rows.
No merged or misaligned numeric columns.
Multi-line narrations stay within their correct cells.
Blank cells remain blank.
No invented or omitted data.
 
FINAL INSTRUCTION
If uncertain about a value, column boundary, or row alignment:
 
leave the cell blank ("")
 
never merge unrelated data
 
and never fabricate content.
 
If the table has no visible boundaries, infer column separation based on consistent spacing, alignment, or repeated numeric patterns.
Do not merge cells even if spacing appears inconsistent.
Assume each row contains one logical transaction record with the same number of fields.
 
Please generate my output in csv format with fixed columns in output - transaction date, Transaction Details, Amount Credited, Amount Debited, Balance

Text/CSV below:
{payload_text}
"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=4000,
        )
        result = response.choices[0].message.content.strip()
        # Attempt to parse CSV result into DataFrame
        try:
            df = pd.read_csv(io.StringIO(result))
            return df
        except Exception:
            # If direct CSV parsing fails, try to extract CSV-like lines
            try:
                # find first line with comma separators (header)
                lines = [ln for ln in result.splitlines() if ln.strip()]
                csv_text = "\n".join(lines)
                df = pd.read_csv(io.StringIO(csv_text))
                return df
            except Exception as e:
                st.warning(f"AI enhancement parsing failed: {e}")
                return pd.DataFrame()
    except Exception as e:
        st.warning(f"AI enhancement failed: {e}")
        return pd.DataFrame()


# -------------------- Main --------------------
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
# Always use Camelot lattice flavor for more consistent table detection
flavor = "lattice"



if uploaded_files:
    overall_summary = []
    all_extracted_data = []  # store everything for final combined export
    
    # Create a progress placeholder for overall file processing
    progress_bar = st.progress(0)
    status_text = st.empty()

    for file_idx, file in enumerate(uploaded_files):
        # Update overall progress
        progress = (file_idx + 0.1) / len(uploaded_files)
        progress_bar.progress(min(progress, 0.99))
        status_text.text(f"Processing file {file_idx + 1} of {len(uploaded_files)}: {file.name}")
        
        st.markdown(f"## 📘 File: `{file.name}`")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        # Extract header info (bank name / account name) to display above the tables
        bank_name, account_name, account_number = extract_header_info(tmp_path)
        # Show file-level headings (not part of the table)
        meta_lines = []
        if bank_name:
            meta_lines.append(f"**Bank:** {bank_name}")
        if account_name:
            meta_lines.append(f"**Account:** {account_name}")
        meta_lines.append(f"**File:** {file.name}")
        # Filter out any combined metadata blob lines (e.g., 'Bank: Date: ... File:')
        filtered = [m for m in meta_lines if not _is_metadata_line(m)]
        if filtered:
            st.markdown("  ".join(filtered))

        with st.spinner(f"Extracting data from {file.name} ..."):
            camelot_pages = extract_camelot_pagewise(tmp_path, flavor)
            pdfp_pages = extract_pdfplumber_pagewise(tmp_path)

        total_pages = max(len(camelot_pages), len(pdfp_pages))
        results = []

        for i in range(total_pages):
            # Create expander for each page to reduce clutter
            with st.expander(f"📄 Page {i + 1} — Details", expanded=(i == 0)):
                col1, col2 = st.columns(2)

            camelot_df = camelot_pages[i] if i < len(camelot_pages) else pd.DataFrame()
            pdfp_df = pdfp_pages[i] if i < len(pdfp_pages) else pd.DataFrame()

            with col1:
                st.markdown(f"**🐫 Camelot (Rows: {len(camelot_df)})**")
                if not camelot_df.empty:
                    st.dataframe(camelot_df, width='stretch')
                else:
                    st.info("ℹ️ No data detected by Camelot")

            with col2:
                st.markdown(f"**🧾 PdfPlumber (Rows: {len(pdfp_df)})**")
                if not pdfp_df.empty:
                    st.dataframe(pdfp_df, width='stretch')
                else:
                    st.info("ℹ️ No data detected by PdfPlumber")

            match = "✅ Match" if len(camelot_df) == len(pdfp_df) else "❌ Mismatch"
            match_html = render_status_badge(match)
            st.markdown(f"**Row Count:** {match_html} (Camelot: **{len(camelot_df)}** rows | PdfPlumber: **{len(pdfp_df)}** rows)", unsafe_allow_html=True)

            # Collect full extracted data
            if not camelot_df.empty:
                camelot_df["Source"] = "Camelot"
                camelot_df["Bank"] = bank_name
                camelot_df["Account"] = account_name
                camelot_df["Account No"] = account_number
                camelot_df["File"] = file.name
                camelot_df["Page"] = i + 1
                all_extracted_data.append(camelot_df)

            if not pdfp_df.empty:
                pdfp_df["Source"] = "PdfPlumber"
                pdfp_df["Bank"] = bank_name
                pdfp_df["Account"] = account_name
                pdfp_df["File"] = file.name
                pdfp_df["Page"] = i + 1
                all_extracted_data.append(pdfp_df)

            # (No per-page AI) — previews only. AI reconstruction will be available per-file at the end

            results.append({
                "File": file.name,
                "Page": i + 1,
                "Camelot Rows": len(camelot_df),
                "PdfPlumber Rows": len(pdfp_df),
                "Status": match,
            })

        summary_df = pd.DataFrame(results)
        st.markdown(f"### 📊 Summary — {file.name}")
        st.dataframe(summary_df, width='stretch')
        st.download_button(
            f"📥 Download Summary CSV for {file.name}",
            summary_df.to_csv(index=False),
            f"{file.name}_summary.csv"
        )

        overall_summary.append(summary_df)

    # Update progress to complete
    progress_bar.progress(1.0)
    status_text.text("✅ Processing complete!")
    
    # ========== ENHANCED METRICS DASHBOARD ==========
    st.markdown("---")
    st.markdown("## 📊 Extraction Metrics Dashboard")
    
    total_files = len(uploaded_files)
    total_pages = sum(len(summarize_df) for summarize_df in overall_summary)
    combined_all = pd.concat(overall_summary, ignore_index=True)
    total_rows_camelot = combined_all["Camelot Rows"].sum()
    total_rows_pdfp = combined_all["PdfPlumber Rows"].sum()
    match_count = len(combined_all[combined_all["Status"].str.contains("✅", na=False)])
    mismatch_count = len(combined_all) - match_count
    success_rate = round((match_count / len(combined_all) * 100) if len(combined_all) > 0 else 0, 1)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        render_metric_card(total_files, "Files Processed")
    with col2:
        render_metric_card(total_pages, "Total Pages")
    with col3:
        render_metric_card(total_rows_camelot, "Rows (Camelot)")
    with col4:
        render_metric_card(f"{success_rate}%", "Match Rate")
    with col5:
        render_metric_card(match_count, "Matches")

    # Combine all summaries
    final_summary = pd.concat(overall_summary, ignore_index=True)
    st.markdown("## 🧾 Combined Summary — All Files")
    st.dataframe(final_summary, width='stretch')
    
    col_dl1, col_dl2, col_dl3 = st.columns(3)
    with col_dl1:
        st.download_button(
            "📥 Download Summary (CSV)", 
            final_summary.to_csv(index=False), 
            "extraction_summary.csv",
            key="summary_csv"
        )
    with col_dl2:
        try:
            import openpyxl
            output = io.BytesIO()
            final_summary.to_excel(output, index=False, engine='openpyxl')
            output.seek(0)
            st.download_button(
                "📊 Download Summary (Excel)",
                output.getvalue(),
                "extraction_summary.xlsx",
                key="summary_xlsx"
            )
        except ImportError:
            st.info("💡 Install openpyxl to export Excel files: `pip install openpyxl`")

    # -------------------- NEW: Full Extracted Data Download (Per-File) --------------------
    if all_extracted_data:
        st.markdown("---")
        st.markdown("## 🧩 Full Extracted Data — Per File Preview & Download")

        # Normalize columns across all extracted DataFrames before concatenation
        all_extracted_data_normalized = []
        all_columns = set()
        for df in all_extracted_data:
            all_columns.update(df.columns)
        for df in all_extracted_data:
            for col in all_columns:
                if col not in df.columns:
                    df[col] = ""
            all_extracted_data_normalized.append(df)

        combined_all = pd.concat(all_extracted_data_normalized, ignore_index=True)

        # Show separate preview & download for each file
        files = combined_all["File"].unique() if "File" in combined_all.columns else []
        for fname in files:
            st.markdown(f"### 📄 File: `{fname}`")
            df_file_all = combined_all[combined_all["File"] == fname].reset_index(drop=True)

            # Sanitize None/NaN literals for display but keep Source column so user can view both extractor outputs
            try:
                df_file_all = df_file_all.fillna("")
                df_file_all = df_file_all.replace(r'^\s*None\s*$', '', regex=True)
                df_file_all = df_file_all.replace(r'^\s*nan\s*$', '', regex=True)
            except Exception:
                df_file_all = df_file_all.astype(str).replace('None', '').replace('nan', '')

            # Split by source and show in tabs: Combined (both), Camelot-only, PdfPlumber-only
            tab_combined, tab_camelot, tab_pdfp = st.tabs(["Combined", "Camelot", "PdfPlumber"])

            with tab_combined:
                st.markdown("**Combined Extracted Data (Camelot + PdfPlumber)**")
                try:
                    st.dataframe(df_file_all, width='stretch', height=350)
                except Exception:
                    st.dataframe(df_file_all.astype(str), width='stretch', height=350)

            df_cam = df_file_all[df_file_all.get("Source", "") == "Camelot"].reset_index(drop=True)
            with tab_camelot:
                st.markdown(f"**Camelot Extracted ({len(df_cam)} rows)**")
                if not df_cam.empty:
                    st.dataframe(df_cam, width='stretch', height=300)
                else:
                    st.info("No Camelot extracted rows for this file.")

            df_pdfp = df_file_all[df_file_all.get("Source", "") == "PdfPlumber"].reset_index(drop=True)
            with tab_pdfp:
                st.markdown(f"**PdfPlumber Extracted ({len(df_pdfp)} rows)**")
                if not df_pdfp.empty:
                    st.dataframe(df_pdfp, width='stretch', height=300)
                else:
                    st.info("No PdfPlumber extracted rows for this file.")

            # Downloads for each view
            col_csv, col_xl, col_json = st.columns(3)
            with col_csv:
                st.download_button(
                    f"📥 Download Combined CSV — {fname}",
                    df_file_all.to_csv(index=False),
                    f"{fname}_combined_extracted.csv",
                    key=f"csv_combined_{fname}"
                )
            with col_xl:
                try:
                    import openpyxl
                    output = io.BytesIO()
                    df_file_all.to_excel(output, index=False, engine='openpyxl')
                    output.seek(0)
                    st.download_button(
                        f"📊 Download Combined Excel — {fname}",
                        output.getvalue(),
                        f"{fname}_combined_extracted.xlsx",
                        key=f"xl_combined_{fname}"
                    )
                except ImportError:
                    st.caption("💡 Excel export requires: pip install openpyxl")
            with col_json:
                st.download_button(
                    f"� Download Combined JSON — {fname}",
                    df_file_all.to_json(orient='records'),
                    f"{fname}_combined_extracted.json",
                    key=f"json_combined_{fname}"
                )

            # Single AI reconstruction per-file (merged inputs)
            if st.button(f"🔁 Run AI reconstruction (Merged) — {fname}", key=f"ai_merge_{fname}"):
                st.info(f"Running AI reconstruction on merged Camelot+PdfPlumber data for {fname}...")
                # Use the combined DataFrame as input to AI (serialize to CSV)
                ai_input_df = df_file_all.copy()
                # Prefer passing the CSV/table form to the AI
                ai_result = ai_enhance(ai_input_df, source_type="table")
                if not ai_result.empty:
                    st.markdown("#### 🤖 AI-Reconstructed Merged Table")
                    st.dataframe(ai_result, width='stretch', height=400)
                    st.download_button(
                        f"� Download AI-Reconstructed CSV — {fname}",
                        ai_result.to_csv(index=False),
                        f"{fname}_ai_reconstructed.csv",
                        key=f"dl_ai_merge_{fname}"
                    )
                else:
                    st.warning("AI did not return a reconstructed table. Check API key, rate limits, or try again.")

else:
    st.markdown("---")
    st.markdown("""
    ### 👋 Welcome to Bank Statement Extractor
    
    **Quick Start:**
    1. 📤 Upload one or more bank statement PDF files using the uploader above
    2. ⏳ Wait for automatic extraction using Camelot and PdfPlumber
    3. 📊 Review extracted tables with side-by-side comparison
    4. 📥 Download results in CSV, Excel, or JSON format
    
    **Features:**
    - 🐫 **Camelot (Lattice)**: Extracts tables using grid structure
    - 🧾 **PdfPlumber**: Extracts text-based transaction records
    - 🤖 **AI Fallback**: Uses GPT-4 when automated extraction fails
    - 📈 **Metrics**: Track extraction success rate and data quality
    - 💾 **Multiple Formats**: Export as CSV, Excel, or JSON
    - 🔍 **Search & Filter**: Find specific transactions across files
    
    **Tips:**
    - ✅ Works best with standard bank statement formats
    - 📋 Set `OPENAI_API_KEY` environment variable for AI features
    - 🎯 Review row counts to identify extraction issues
    """)
    
    st.info("📤 Upload PDF files to begin extraction →")
