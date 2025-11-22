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


def credit_debit_checker(current, previous, return_type: str = "label", debit_val=None, credit_val=None):
    """
    Determine whether the movement from `previous` to `current` is a credit or a debit.
    Validates and corrects/appends values when discrepancies are found.

    Parameters:
    - current: ending balance
    - previous: starting balance
    - return_type: "label" (Credit/Debit/NoChange), "sign" (1/-1/0), "diff" (numeric difference), or "detailed" (dict with all info)
    - debit_val: provided debit amount (for validation and correction)
    - credit_val: provided credit amount (for validation and correction)

    Notes:
    - This function accepts numeric values or strings containing numbers (commas, parentheses,
      negative signs, extra spaces). It will try to parse the first numeric token found.
    - Calculation: diff = current - previous
        * diff > 0 => Credit
        * diff < 0 => Debit
        * diff == 0 => NoChange
    - When debit_val and credit_val are provided, validates them against the calculated diff.
    - Flags errors and suggests corrections when discrepancies found.
    
    Returns:
    - If return_type="label": "Credit", "Debit", "NoChange", or None
    - If return_type="sign": 1 (credit), -1 (debit), 0 (no change), or None
    - If return_type="diff": numeric difference (float) or None
    - If return_type="detailed": dict with full validation info including corrections
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
        if return_type == "detailed":
            return {"valid": False, "error": "Could not parse current or previous balance", "corrected_debit": None, "corrected_credit": None}
        return None if return_type in ("label", "sign") else None

    diff = curr - prev
    diff_abs = abs(diff)
    
    # Determine transaction type
    if diff > 0:
        txn_type = "Credit"
        sign = 1
    elif diff < 0:
        txn_type = "Debit"
        sign = -1
    else:
        txn_type = "NoChange"
        sign = 0
    
    # Validate and correct debit/credit values if provided
    corrected_debit = None
    corrected_credit = None
    validation_errors = []
    
    if debit_val is not None or credit_val is not None:
        debit_num = _to_num(debit_val) if debit_val is not None else 0
        credit_num = _to_num(credit_val) if credit_val is not None else 0
        
        # Check if provided amounts sum correctly to the diff
        provided_diff = credit_num - debit_num
        
        if abs(provided_diff - diff) > 0.01:  # Allow small floating point tolerance
            validation_errors.append(f"Mismatch: expected diff={diff:.2f}, got credit({credit_num:.2f}) - debit({debit_num:.2f}) = {provided_diff:.2f}")
            
            # Auto-correct based on transaction type
            if txn_type == "Credit":
                corrected_credit = diff_abs
                corrected_debit = 0
                validation_errors.append(f"CORRECTED: Debit should be 0, Credit should be {diff_abs:.2f}")
            elif txn_type == "Debit":
                corrected_debit = diff_abs
                corrected_credit = 0
                validation_errors.append(f"CORRECTED: Debit should be {diff_abs:.2f}, Credit should be 0")
            else:
                corrected_debit = 0
                corrected_credit = 0
                validation_errors.append("CORRECTED: No transaction (both should be 0)")
        else:
            # Values are correct
            corrected_debit = debit_num
            corrected_credit = credit_num
    else:
        # No validation values provided, use calculated diff
        if txn_type == "Credit":
            corrected_credit = diff_abs
            corrected_debit = 0
        elif txn_type == "Debit":
            corrected_debit = diff_abs
            corrected_credit = 0
        else:
            corrected_debit = 0
            corrected_credit = 0
    
    # Return based on requested type
    if return_type == "detailed":
        return {
            "valid": len(validation_errors) == 0,
            "transaction_type": txn_type,
            "sign": sign,
            "diff": diff,
            "diff_abs": diff_abs,
            "corrected_debit": corrected_debit,
            "corrected_credit": corrected_credit,
            "errors": validation_errors,
            "previous_balance": prev,
            "current_balance": curr,
        }
    elif return_type == "diff":
        return diff
    elif return_type == "sign":
        return sign
    else:  # "label"
        return txn_type


def validate_transaction_date(date_val):
    """
    Validate transaction date.
    
    Rules:
    - Cannot be blank or empty
    - Must be in valid date format (DD/MM/YY, DD-MM-YYYY, YYYY-MM-DD, etc.)
    
    Returns: (is_valid: bool, parsed_date: str or None, error_msg: str or None)
    """
    if date_val is None or date_val == "":
        return False, None, "Date cannot be blank"
    
    date_str = str(date_val).strip()
    if not date_str:
        return False, None, "Date cannot be blank"
    
    # Common date patterns to try
    date_patterns = [
        r"^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$",  # DD/MM/YY or DD-MM-YYYY
        r"^\d{4}[/-]\d{1,2}[/-]\d{1,2}$",    # YYYY-MM-DD or YYYY/MM/DD
        r"^\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s\d{2,4}$",  # DD MMM YYYY
        r"^\d{1,2}-\d{1,2}-\d{4}$",           # DD-MM-YYYY
    ]
    
    is_date_format = any(re.match(pattern, date_str, re.I) for pattern in date_patterns)
    
    if not is_date_format:
        return False, None, f"Invalid date format: '{date_str}'. Expected formats: DD/MM/YY, DD-MM-YYYY, YYYY-MM-DD, etc."
    
    return True, date_str, None


def validate_balance(balance_val):
    """
    Validate balance.
    
    Rules:
    - Cannot be blank or empty
    - Must be a numeric value (supports commas and decimals)
    
    Returns: (is_valid: bool, parsed_balance: float or None, error_msg: str or None)
    """
    if balance_val is None or balance_val == "":
        return False, None, "Balance cannot be blank"
    
    balance_str = str(balance_val).strip()
    if not balance_str or balance_str.lower() in ("nan", "none", ""):
        return False, None, "Balance cannot be blank"
    
    # Try to parse numeric value
    m = re.search(r"-?[\d,]+(?:\.\d+)?", balance_str)
    if not m:
        return False, None, f"Balance is not numeric: '{balance_str}'"
    
    try:
        num_str = m.group(0).replace(',', '')
        balance_num = float(num_str)
        return True, balance_num, None
    except Exception as e:
        return False, None, f"Failed to parse balance: {str(e)}"


def validate_debit_credit(debit_val, credit_val):
    """
    Validate debit and credit columns.
    
    Rules:
    - At least one of debit or credit must have a value
    - Both cannot be blank/empty
    - Both cannot be zero
    
    Returns: (is_valid: bool, parsed_debit: float or None, parsed_credit: float or None, error_msg: str or None)
    """
    def _to_num(x):
        if x is None or x == "":
            return None
        s = str(x).strip()
        if not s or s.lower() in ("nan", "none", ""):
            return None
        m = re.search(r"-?[\d,]+(?:\.\d+)?", s)
        if not m:
            return None
        try:
            return float(m.group(0).replace(',', ''))
        except Exception:
            return None
    
    debit_num = _to_num(debit_val)
    credit_num = _to_num(credit_val)
    
    # Check if at least one value exists
    if debit_num is None and credit_num is None:
        return False, None, None, "At least one of Debit or Credit must have a value"
    
    # Check if both are zero
    if debit_num == 0 and credit_num == 0:
        return False, 0, 0, "Both Debit and Credit cannot be zero"
    
    # Set zeros for missing values
    if debit_num is None:
        debit_num = 0
    if credit_num is None:
        credit_num = 0
    
    return True, debit_num, credit_num, None


def validate_transaction_row(row_dict):
    """
    Comprehensive validation for a single transaction row.
    
    Validates:
    1. Transaction date is present and in valid format
    2. Balance is present and numeric
    3. At least one of Debit or Credit has a value
    
    Parameters:
    - row_dict: dictionary with keys like 'Date', 'Balance', 'Amount Debit', 'Amount Credit' (case-insensitive)
    
    Returns: {
        "valid": bool,
        "errors": list of error messages,
        "warnings": list of warning messages,
        "corrected_values": dict with any corrected/parsed values,
        "details": {
            "date_valid": bool,
            "balance_valid": bool,
            "debit_credit_valid": bool,
        }
    }
    """
    errors = []
    warnings = []
    corrected_values = {}
    details = {
        "date_valid": False,
        "balance_valid": False,
        "debit_credit_valid": False,
    }
    
    # Find column names (case-insensitive)
    col_map = {k.lower(): k for k in row_dict.keys()}
    
    # 1️⃣ Validate Date
    date_key = None
    for key in col_map:
        if 'date' in key:
            date_key = col_map[key]
            break
    
    if date_key:
        date_val = row_dict.get(date_key)
        is_valid, parsed_date, error_msg = validate_transaction_date(date_val)
        details["date_valid"] = is_valid
        if not is_valid:
            errors.append(f"Date Error: {error_msg}")
        else:
            corrected_values['Date'] = parsed_date
    else:
        errors.append("Date column not found in row")
    
    # 2️⃣ Validate Balance
    balance_key = None
    for key in col_map:
        if 'balance' in key:
            balance_key = col_map[key]
            break
    
    if balance_key:
        balance_val = row_dict.get(balance_key)
        is_valid, parsed_balance, error_msg = validate_balance(balance_val)
        details["balance_valid"] = is_valid
        if not is_valid:
            errors.append(f"Balance Error: {error_msg}")
        else:
            corrected_values['Balance'] = parsed_balance
    else:
        errors.append("Balance column not found in row")
    
    # 3️⃣ Validate Debit/Credit
    debit_key = None
    credit_key = None
    for key in col_map:
        if 'debit' in key or 'dr' in key or 'withdrawal' in key:
            debit_key = col_map[key]
        if 'credit' in key or 'cr' in key or 'deposit' in key:
            credit_key = col_map[key]
    
    if debit_key or credit_key:
        debit_val = row_dict.get(debit_key) if debit_key else None
        credit_val = row_dict.get(credit_key) if credit_key else None
        
        is_valid, parsed_debit, parsed_credit, error_msg = validate_debit_credit(debit_val, credit_val)
        details["debit_credit_valid"] = is_valid
        if not is_valid:
            errors.append(f"Debit/Credit Error: {error_msg}")
        else:
            if debit_key:
                corrected_values['Debit'] = parsed_debit if parsed_debit else ""
            if credit_key:
                corrected_values['Credit'] = parsed_credit if parsed_credit else ""
    else:
        errors.append("Neither Debit nor Credit column found in row")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "corrected_values": corrected_values,
        "details": details,
    }


def validate_dataframe(df):
    """
    Validate entire transaction DataFrame.
    
    Returns: {
        "total_rows": int,
        "valid_rows": int,
        "invalid_rows": int,
        "row_validations": list of validation results per row,
        "summary": {
            "date_errors": count,
            "balance_errors": count,
            "debit_credit_errors": count,
        }
    }
    """
    if df.empty:
        return {
            "total_rows": 0,
            "valid_rows": 0,
            "invalid_rows": 0,
            "row_validations": [],
            "summary": {"date_errors": 0, "balance_errors": 0, "debit_credit_errors": 0},
        }
    
    row_validations = []
    valid_count = 0
    date_error_count = 0
    balance_error_count = 0
    debit_credit_error_count = 0
    
    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        validation = validate_transaction_row(row_dict)
        
        # Add row index for reference
        validation["row_index"] = idx
        row_validations.append(validation)
        
        if validation["valid"]:
            valid_count += 1
        else:
            # Count error types
            if not validation["details"]["date_valid"]:
                date_error_count += 1
            if not validation["details"]["balance_valid"]:
                balance_error_count += 1
            if not validation["details"]["debit_credit_valid"]:
                debit_credit_error_count += 1
    
    invalid_count = len(df) - valid_count
    
    return {
        "total_rows": len(df),
        "valid_rows": valid_count,
        "invalid_rows": invalid_count,
        "row_validations": row_validations,
        "summary": {
            "date_errors": date_error_count,
            "balance_errors": balance_error_count,
            "debit_credit_errors": debit_credit_error_count,
        },
    }


def detect_change_prompt_needed(df):
    """
    Detect if page extraction needs a different prompt/strategy.
    
    Flags that indicate need for Change Prompt:
    - Multiple columns that don't align properly
    - Merged cells (debit+credit in one column)
    - Inconsistent structure across rows
    - Very wide tables with many columns
    - Poor alignment/formatting
    
    Returns: (needs_change_prompt: bool, reason: str)
    """
    if df.empty:
        return False, ""
    
    reasons = []
    
    # Check for too many columns (> 8 suggests merged or multi-column format)
    if len(df.columns) > 8:
        reasons.append(f"Too many columns ({len(df.columns)}) - possible merged cells")
    
    # Check for inconsistent non-empty cell patterns
    try:
        filled_cells_per_row = df.astype(str).ne('').sum(axis=1)
        if filled_cells_per_row.std() > 2:
            reasons.append("Highly inconsistent column fills - possible merged cells")
    except:
        pass
    
    # Check for numeric patterns suggesting merged debit/credit
    try:
        last_col = df.iloc[:, -1].astype(str)
        merged_count = last_col.str.contains(r'\d+\s+\d+', regex=True).sum()
        if merged_count > len(df) * 0.3:  # More than 30% have multiple numbers
            reasons.append("Multiple numeric values in single cells - possible merged columns")
    except:
        pass
    
    # Check column name clarity
    col_names = [str(c).lower() for c in df.columns]
    has_date = any('date' in c for c in col_names)
    has_balance = any('balance' in c for c in col_names)
    has_debit_or_credit = any('debit' in c or 'credit' in c or 'dr' in c or 'cr' in c for c in col_names)
    
    if not (has_date and has_balance and has_debit_or_credit):
        reasons.append("Missing critical columns (Date/Balance/Debit-Credit)")
    
    needs_change = len(reasons) > 0
    reason_text = " | ".join(reasons) if reasons else ""
    
    return needs_change, reason_text


def refine_and_validate_data(df, prev_balance=None):
    """
    Refine extracted data by validating and attempting to fix issues.
    
    Process:
    1. Validate each row against rules
    2. For invalid rows, attempt to auto-correct using balance arithmetic
    3. Use credit_debit_checker to infer missing debit/credit values
    
    Returns: {
        "refined_df": DataFrame with corrected values,
        "validation_summary": validation results,
        "refinement_count": number of rows refined,
        "fixable_issues": list of fixes applied,
    }
    """
    refined_df = df.copy()
    all_validations = []
    refinement_count = 0
    fixable_issues = []
    
    # Validate all rows first
    for idx, row in refined_df.iterrows():
        row_dict = row.to_dict()
        validation = validate_transaction_row(row_dict)
        validation["row_index"] = idx
        all_validations.append(validation)
        
        if not validation["valid"]:
            # Try to fix common issues
            
            # Find column keys (case-insensitive)
            col_map = {k.lower(): k for k in row.index}
            
            # Issue 1: Missing Balance
            balance_key = None
            for key in col_map:
                if 'balance' in key:
                    balance_key = col_map[key]
                    break
            
            if balance_key and (row[balance_key] == "" or pd.isna(row[balance_key])):
                # Try to infer from previous and debit/credit
                debit_key = credit_key = None
                for key in col_map:
                    if 'debit' in key or 'dr' in key:
                        debit_key = col_map[key]
                    if 'credit' in key or 'cr' in key:
                        credit_key = col_map[key]
                
                if prev_balance is not None and (debit_key or credit_key):
                    debit_val = row.get(debit_key, 0) if debit_key else 0
                    credit_val = row.get(credit_key, 0) if credit_key else 0
                    
                    try:
                        debit_num = float(str(debit_val).replace(',', '')) if debit_val else 0
                        credit_num = float(str(credit_val).replace(',', '')) if credit_val else 0
                        inferred_balance = prev_balance + credit_num - debit_num
                        refined_df.at[idx, balance_key] = inferred_balance
                        fixable_issues.append(f"Row {idx}: Inferred balance from debit/credit")
                        refinement_count += 1
                    except:
                        pass
            
            # Issue 2: Missing Debit/Credit
            debit_key = credit_key = None
            for key in col_map:
                if 'debit' in key or 'dr' in key:
                    debit_key = col_map[key]
                if 'credit' in key or 'cr' in key:
                    credit_key = col_map[key]
            
            debit_val = row.get(debit_key) if debit_key else None
            credit_val = row.get(credit_key) if credit_key else None
            
            # If both are empty/zero, try to infer from balance change
            if (debit_val is None or debit_val == "" or debit_val == 0) and \
               (credit_val is None or credit_val == "" or credit_val == 0) and \
               prev_balance is not None and balance_key:
                try:
                    current_balance = float(str(row[balance_key]).replace(',', ''))
                    diff = current_balance - prev_balance
                    
                    if diff > 0:  # Credit
                        if credit_key:
                            refined_df.at[idx, credit_key] = diff
                        if debit_key:
                            refined_df.at[idx, debit_key] = 0
                        fixable_issues.append(f"Row {idx}: Inferred Credit ({diff:.2f}) from balance")
                        refinement_count += 1
                    elif diff < 0:  # Debit
                        if debit_key:
                            refined_df.at[idx, debit_key] = abs(diff)
                        if credit_key:
                            refined_df.at[idx, credit_key] = 0
                        fixable_issues.append(f"Row {idx}: Inferred Debit ({abs(diff):.2f}) from balance")
                        refinement_count += 1
                except:
                    pass
        
        # Update previous balance for next iteration
        if validation["details"]["balance_valid"]:
            try:
                prev_balance = float(str(row.get(balance_key, 0)).replace(',', ''))
            except:
                pass
    
    # Re-validate after refinement
    final_validation = validate_dataframe(refined_df)
    
    return {
        "refined_df": refined_df,
        "validation_summary": final_validation,
        "refinement_count": refinement_count,
        "fixable_issues": fixable_issues,
        "all_validations": all_validations,
    }


def extract_page_with_strategy(path, page_num, primary_flavor="lattice", use_ai_fallback=True):
    """
    Extract single page using smart strategy:
    1. Try primary flavor (Camelot lattice)
    2. Validate extraction
    3. If fails validation and needs change prompt → use AI
    4. If minimal issues → try to refine
    
    Returns: {
        "extracted_df": DataFrame,
        "method_used": "Camelot" or "AI",
        "extraction_quality": score 0-100,
        "notes": list of notes,
        "validation": validation results,
    }
    """
    notes = []
    
    try:
        # Step 1: Try Camelot extraction
        with contextlib.redirect_stderr(io.StringIO()):
            tables = camelot.read_pdf(path, pages=str(page_num), flavor=primary_flavor)
        
        if tables and len(tables) > 0:
            extracted_df = pd.concat([t.df for t in tables], ignore_index=True)
            extracted_df = extracted_df.astype(str)
            extracted_df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in extracted_df.columns]
            
            notes.append(f"Camelot extraction: {len(extracted_df)} rows, {len(extracted_df.columns)} columns")
            
            # Step 2: Check if needs change prompt
            needs_change, reason = detect_change_prompt_needed(extracted_df)
            
            if needs_change and use_ai_fallback:
                notes.append(f"Change Prompt detected: {reason}")
                notes.append("Switching to AI enhancement...")
                
                # Use AI fallback
                ai_result = ai_enhance(extracted_df, source_type="table")
                if not ai_result.empty:
                    extracted_df = ai_result
                    method_used = "AI"
                    notes.append("Successfully refined using AI")
                else:
                    method_used = "Camelot"
                    notes.append("AI fallback unsuccessful, using Camelot results")
            else:
                method_used = "Camelot"
            
            # Step 3: Validate and refine
            refinement_result = refine_and_validate_data(extracted_df)
            refined_df = refinement_result["refined_df"]
            validation = refinement_result["validation_summary"]
            
            if refinement_result["refinement_count"] > 0:
                notes.append(f"Refinement: Fixed {refinement_result['refinement_count']} rows")
                for issue in refinement_result["fixable_issues"][:3]:  # Show first 3
                    notes.append(f"  - {issue}")
            
            # Calculate extraction quality
            if validation["total_rows"] > 0:
                quality_score = int((validation["valid_rows"] / validation["total_rows"]) * 100)
            else:
                quality_score = 0
            
            return {
                "extracted_df": refined_df,
                "method_used": method_used,
                "extraction_quality": quality_score,
                "notes": notes,
                "validation": validation,
            }
        else:
            notes.append("Camelot extraction: No tables found")
            return {
                "extracted_df": pd.DataFrame(),
                "method_used": "None",
                "extraction_quality": 0,
                "notes": notes,
                "validation": None,
            }
    
    except Exception as e:
        notes.append(f"Extraction error: {str(e)}")
        return {
            "extracted_df": pd.DataFrame(),
            "method_used": "Error",
            "extraction_quality": 0,
            "notes": notes,
            "validation": None,
        }


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
You are a precise and reliable financial data extraction system.
 
Your task is to extract only the transaction data from a multi-page bank statement text (converted from PDF).  
The output must be a clean, structured CSV table with all transaction rows and columns exactly as they appear in the document.
 
OBJECTIVE
Extract all rows that represent financial transactions, regardless of bank layout or template.  
Ignore all non-transactional elements such as:
- Headers, footers, disclaimers, summaries
- Opening/closing balances (unless part of a transaction)
- Account information or metadata
 
REQUIRED OUTPUT FORMAT
Output strictly as a CSV with these exact columns in this order:
 
Date,Transaction Details,Cheque No.,Amount Debit,Amount Credit,Balance
 
Rules:
- Each row = one transaction.
- Keep column count fixed (6 columns per row).
- Leave missing fields blank ("").
- Do not merge or duplicate columns.
- Preserve original order of transactions.
 
DETECTING TABLE STRUCTURE
Identify recurring tabular patterns containing:
- Date or Txn Date
- Narration or Description
- Reference / Cheque No. / UTR / Transaction ID
- Debit (Withdrawal) and Credit (Deposit)
- Balance
 
If headers are not repeated on every page, infer structure from previous pages.  
Maintain the same column order throughout the entire output.
 
DATA HANDLING RULES
1. Multi-line narrations:
   Merge continuation lines (e.g., beneficiary names or descriptions) into the same "Transaction Details" cell.  
   Do not create extra rows.
 
2. Unaligned or wrapped data:
   Use recurring numeric and spacing patterns to determine column separation (e.g., debit/credit/balance alignment).
 
3. Integrity checks:
   - Never fabricate or infer missing values.
   - Never merge different transactions into one.
   - Do not reorder, skip, or summarize transactions.
   - Preserve numbers exactly (no rounding or formatting).
   - Keep all text exactly as in the input (spacing and case-sensitive).
 
EXCLUDE
- Account summaries, totals, or closing balance sections
- Page numbers, headers, footers, and disclaimers
- Any explanatory or non-transaction text
 
OUTPUT RULES (STRICT CSV)
- Output only CSV data — nothing else.
- No markdown formatting, no ```csv or ``` delimiters.
- The first line must be the header:
  Date,Transaction Details,Cheque No.,Amount Debit,Amount Credit,Balance
- Each subsequent line must represent one transaction.
- Use commas (,) as separators.
- Wrap any text containing commas inside double quotes (" ").
- Do not add spaces before or after commas.
 
EXAMPLE FORMAT
Date,Transaction Details,Cheque No.,Amount Debit,Amount Credit,Balance
01/01/25,"ATM WITHDRAWAL - LUDHIANA","100589",21808.31,,47438.87
02/01/25,"NEFT CR - DCM NOUVELLE LTD","PUNBZ25006921805",,50000,97438.87
03/01/25,"SERVICE CHARGE","",222.33,,97216.54
 
VALIDATION REQUIREMENTS
- Each row has exactly 6 columns.
- Each row represents one transaction.
- Numeric fields are accurate and not merged.
-make sure there is no blank balance column
-there should be atleast one of debit or credit column should have value
-if credit debit are merging..then according to balance and prev balance difference ..predict whether it is credit or debit and fill accordingly
- Multi-line narrations stay within the same row.
- Blank cells remain empty — do not fill or infer.
- No additional commentary or metadata outside the CSV.
 
Note: Balance column should not be blank for any transaction row.
Now extract only the transaction-level table data from the text below and output it in strict CSV format:
 
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
                col1, col2, col3 = st.columns(3)

            # ===== SMART EXTRACTION STRATEGY =====
            extraction_result = extract_page_with_strategy(tmp_path, i + 1, primary_flavor=flavor, use_ai_fallback=True)
            extracted_df = extraction_result["extracted_df"]
            method_used = extraction_result["method_used"]
            quality_score = extraction_result["extraction_quality"]
            notes = extraction_result["notes"]
            validation = extraction_result["validation"]

            with col1:
                st.markdown(f"**� Method:** `{method_used}`")
                st.markdown(f"**📊 Quality:** `{quality_score}%`")
                if quality_score >= 95:
                    st.success("✅ Excellent extraction")
                elif quality_score >= 80:
                    st.info("ℹ️ Good extraction")
                else:
                    st.warning(f"⚠️ Fair extraction ({quality_score}%)")

            with col2:
                st.markdown(f"**📝 Rows:** `{len(extracted_df)}`")
                st.markdown(f"**🏛️ Columns:** `{len(extracted_df.columns)}`")

            with col3:
                st.markdown("**📋 Extraction Notes:**")
                for note in notes:
                    st.caption(f"• {note}")

            # Show extracted data
            if not extracted_df.empty:
                st.markdown("**📊 Extracted Data:**")
                st.dataframe(extracted_df, width='stretch', height=300)
            else:
                st.info("ℹ️ No data extracted from this page")

            # Show validation results if available
            if validation:
                st.markdown("**✓ Validation Results:**")
                val_col1, val_col2, val_col3 = st.columns(3)
                with val_col1:
                    st.metric("Valid Rows", validation["valid_rows"])
                with val_col2:
                    st.metric("Invalid Rows", validation["invalid_rows"])
                with val_col3:
                    st.metric("Total Rows", validation["total_rows"])
                
                # Show error summary
                if validation["invalid_rows"] > 0:
                    st.markdown("**Error Summary:**")
                    error_col1, error_col2, error_col3 = st.columns(3)
                    with error_col1:
                        st.metric("Date Errors", validation["summary"]["date_errors"])
                    with error_col2:
                        st.metric("Balance Errors", validation["summary"]["balance_errors"])
                    with error_col3:
                        st.metric("Debit/Credit Errors", validation["summary"]["debit_credit_errors"])

            # Collect extracted data
            if not extracted_df.empty:
                extracted_df["Source"] = method_used
                extracted_df["Bank"] = bank_name
                extracted_df["Account"] = account_name
                extracted_df["Account No"] = account_number
                extracted_df["File"] = file.name
                extracted_df["Page"] = i + 1
                extracted_df["Quality"] = quality_score
                all_extracted_data.append(extracted_df)

            results.append({
                "File": file.name,
                "Page": i + 1,
                "Method": method_used,
                "Rows": len(extracted_df),
                "Quality": f"{quality_score}%",
                "Valid": validation["valid_rows"] if validation else 0,
                "Invalid": validation["invalid_rows"] if validation else 0,
            })

        summary_df = pd.DataFrame(results)
        st.markdown(f"### 📊 Summary — {file.name}")
        st.dataframe(summary_df, width='stretch')
        
        # Calculate file-level quality
        total_valid = summary_df["Valid"].sum()
        total_rows = summary_df["Rows"].sum()
        file_quality = round((total_valid / total_rows * 100) if total_rows > 0 else 0, 1)
        
        st.markdown(f"**File Quality Score: `{file_quality}%`**")
        
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
    st.markdown("## 📊 Smart Extraction Metrics Dashboard")
    
    total_files = len(uploaded_files)
    total_pages = sum(len(summarize_df) for summarize_df in overall_summary)
    combined_all = pd.concat(overall_summary, ignore_index=True)
    
    # Calculate accuracy
    total_valid = combined_all["Valid"].sum() if "Valid" in combined_all.columns else 0
    total_rows = combined_all["Rows"].sum() if "Rows" in combined_all.columns else 0
    overall_accuracy = round((total_valid / total_rows * 100) if total_rows > 0 else 0, 1)
    
    # Method usage stats
    method_counts = combined_all["Method"].value_counts() if "Method" in combined_all.columns else {}
    camelot_count = method_counts.get("Camelot", 0)
    ai_count = method_counts.get("AI", 0)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        render_metric_card(total_files, "Files Processed")
    with col2:
        render_metric_card(total_pages, "Total Pages")
    with col3:
        render_metric_card(f"{overall_accuracy}%", "Accuracy")
    with col4:
        render_metric_card(camelot_count, "Camelot Extractions")
    with col5:
        render_metric_card(ai_count, "AI Enhancements")

    st.success(f"✅ Overall Extraction Accuracy: **{overall_accuracy}%**") if overall_accuracy >= 95 else st.info(f"📊 Overall Extraction Accuracy: **{overall_accuracy}%**")

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
