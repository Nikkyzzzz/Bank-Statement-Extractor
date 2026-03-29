import streamlit as st
import pandas as pd
import re
import base64
import json
import numpy as np
import fitz  # PyMuPDF
from datetime import datetime
from collections import defaultdict
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# -------------------------------------------------------------------
# Column mappings for file merging
# -------------------------------------------------------------------
COLUMN_MAPPINGS = {
    "Txn Date": [
        "date", "txn date", "transaction date", "post date", "trans date", "value date"
    ],
    "Description": [
        "description", "details", "narration", "particulars", "transaction details",
        "remarks", "transaction remarks", "desc", "narration/description"
    ],
    "Cheque No": [
        "cheque no", "chq./ref.no", "chq no", "ref no./cheque no.", "instrument id",
        "reference no", "cheque no /  ref no", "ref no", "cheque number"
    ],
    "Withdrawal (in Rs.)": [
        "withdrawal", "debit", "dr amount", "dr", "debit amount", "withdrawal amount",
        "withdra wal", "dr amt", "dramt", "dramount", "amount debit", "withdrawal amt"
    ],
    "Deposits (in Rs.)": [
        "credit", "deposit", "cr amount", "cr", "credit amount", "cr amt", "cramt",
        "cramount", "amount credit", "credits", "deposit amt"
    ],
    "Balance (in Rs.)": [
        "balance", "bal", "closing balance", "closing", "running balance",
        "available balance", "closing balance (in rs.)"
    ]
}

# Standard column order
STANDARD_COLUMNS = [
    "Sl No", "Txn Date", "Description", "Cheque No",
    "Withdrawal (in Rs.)", "Deposits (in Rs.)", "Balance (in Rs.)", "File Name"
]

# -------------------------------------------------------------------
# UI CSS
# -------------------------------------------------------------------
MODERN_CSS = """
<style>
    :root {
        --primary-color: #1e3c72;
        --primary-light: #2a5298;
        --success-color: #28a745;
        --success-light: #20c997;
        --text-light: #333;
        --text-muted: #555;
        --bg-light: #f8f9fa;
        --border-light: #e0e0e0;
        --shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Dark mode variables */
    @media (prefers-color-scheme: dark) {
        :root {
            --primary-color: #1e3c72;
            --primary-light: #2a5298;
            --success-color: #28a745;
            --success-light: #20c997;
            --text-light: #e0e0e0;
            --text-muted: #b0b0b0;
            --bg-light: #1e1e1e;
            --border-light: #404040;
            --shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }
    }
    
    /* Main styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1e3c72, #2a5298);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--primary-color);
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--border-light);
    }
    
    .card {
        background: var(--bg-light);
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: var(--shadow);
        margin: 1rem 0;
        border-left: 4px solid #2a5298;
        color: var(--text-light);
    }
    
    .card h2, .card h3 {
        color: var(--primary-color);
        margin-top: 0;
    }
    
    .card p {
        color: var(--text-muted);
    }
    
    .feature-card {
        background: linear-gradient(135deg, rgba(245, 247, 250, 0.9) 0%, rgba(195, 207, 226, 0.9) 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        transition: transform 0.3s ease;
        color: var(--text-light);
        border: 1px solid var(--border-light);
    }
    
    @media (prefers-color-scheme: dark) {
        .feature-card {
            background: linear-gradient(135deg, rgba(60, 70, 90, 0.8) 0%, rgba(80, 100, 140, 0.8) 100%);
        }
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.2);
    }
    
    .feature-card h3, .feature-card p {
        color: var(--text-light);
    }
    
    .success-box {
        background: linear-gradient(135deg, rgba(212, 237, 218, 0.9) 0%, rgba(195, 230, 203, 0.9) 100%);
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: #155724;
    }
    
    @media (prefers-color-scheme: dark) {
        .success-box {
            background: linear-gradient(135deg, rgba(40, 100, 60, 0.9) 0%, rgba(32, 160, 120, 0.8) 100%);
            color: #90ee90;
            border-color: #28a745;
        }
    }
    
    .info-box {
        background: linear-gradient(135deg, rgba(209, 236, 241, 0.9) 0%, rgba(190, 229, 235, 0.9) 100%);
        border: 1px solid #bee5eb;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: #004085;
    }
    
    @media (prefers-color-scheme: dark) {
        .info-box {
            background: linear-gradient(135deg, rgba(30, 60, 100, 0.9) 0%, rgba(50, 120, 180, 0.8) 100%);
            color: #87ceeb;
            border-color: #0d6efd;
        }
    }
    
    .warning-box {
        background: linear-gradient(135deg, rgba(255, 243, 205, 0.9) 0%, rgba(255, 234, 167, 0.9) 100%);
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: #856404;
    }
    
    @media (prefers-color-scheme: dark) {
        .warning-box {
            background: linear-gradient(135deg, rgba(100, 80, 0, 0.9) 0%, rgba(180, 140, 0, 0.8) 100%);
            color: #ffd700;
            border-color: #ffc107;
        }
    }
    
    /* Button styling */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
        transition: all 0.3s ease;
        border: none;
    }
    
    .primary-button {
        background: linear-gradient(135deg, #1e3c72, #2a5298) !important;
        color: white !important;
    }
    
    .secondary-button {
        background: linear-gradient(135deg, #6c757d, #5a6268) !important;
        color: white !important;
    }
    
    .success-button {
        background: linear-gradient(135deg, #28a745, #20c997) !important;
        color: white !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #1e3c72, #2a5298);
    }
    
    /* Metric cards */
    .metric-card {
        background: var(--bg-light);
        padding: 1rem;
        border-radius: 8px;
        box-shadow: var(--shadow);
        text-align: center;
        border-top: 4px solid #2a5298;
        color: var(--text-light);
    }
    
    .metric-card h3, .metric-card p {
        color: var(--text-light);
        margin: 0.5rem 0;
    }
    
    /* File uploader styling */
    .uploadedFile {
        background: var(--bg-light);
        border: 1px dashed var(--border-light);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        color: var(--text-muted);
    }
    
    /* Dataframe styling */
    .stDataFrame {
        border: 1px solid var(--border-light);
        border-radius: 8px;
    }
    
    /* Step indicator */
    .step-indicator {
        display: flex;
        justify-content: space-between;
        margin: 2rem 0;
        position: relative;
        padding: 1.5rem;
        background: var(--bg-light);
        border-radius: 10px;
        box-shadow: var(--shadow);
    }
    
    .step-indicator::before {
        content: '';
        position: absolute;
        top: 35px;
        left: 5%;
        right: 5%;
        height: 2px;
        background: var(--border-light);
        z-index: 1;
    }
    
    .step {
        display: flex;
        flex-direction: column;
        align-items: center;
        position: relative;
        z-index: 2;
        flex: 1;
    }
    
    .step-circle {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background: var(--border-light);
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        margin-bottom: 0.5rem;
        border: 3px solid var(--bg-light);
        color: var(--text-muted);
        transition: all 0.3s ease;
    }
    
    .step.active .step-circle {
        background: linear-gradient(135deg, #1e3c72, #2a5298);
        color: white;
        box-shadow: 0 0 15px rgba(42, 82, 152, 0.4);
    }
    
    .step.completed .step-circle {
        background: linear-gradient(135deg, #28a745, #20c997);
        color: white;
        box-shadow: 0 0 15px rgba(40, 167, 69, 0.4);
    }
    
    .step-label {
        font-size: 0.9rem;
        font-weight: 500;
        color: var(--text-muted);
        text-align: center;
        transition: color 0.3s ease;
    }
    
    .step.active .step-label {
        color: var(--primary-color);
        font-weight: 600;
    }
    
    .step.completed .step-label {
        color: var(--success-color);
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background: var(--bg-light) !important;
        color: var(--text-light) !important;
    }
    
    /* Checkboxes and inputs */
    .stCheckbox label {
        color: var(--text-light) !important;
    }
    
    /* Ensure text in all boxes is readable */
    .stMarkdown {
        color: var(--text-light);
    }
</style>
"""

# -------------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------------
def normalize_column_name(col):
    col = str(col).strip().lower().replace("\n", " ").replace("  ", " ")
    col = re.sub(r"[^a-z0-9\s./-]", "", col)
    return col

def map_column(col):
    normalized = normalize_column_name(col)
    for standard, variations in COLUMN_MAPPINGS.items():
        for variant in variations:
            if normalized == variant.lower():
                return standard
            if variant.lower() in normalized:
                return standard
    if "sl" in normalized and "no" in normalized:
        return "Sl No"
    return None

def split_merged_decimals(df: pd.DataFrame) -> pd.DataFrame:
    """Split merged decimal values in Balance column."""
    df_split = df.copy()
    
    balance_col = None
    for idx, col_name in enumerate(df_split.columns):
        col_lower = str(col_name).lower()
        if 'balance' in col_lower:
            balance_col = idx
            break
    
    if balance_col is None:
        return df_split
    
    pattern = r'([-\d,]+\.\d{2})\s+([-\d,]+\.\d+)'
    
    for row_idx in range(len(df_split)):
        balance_val = str(df_split.iloc[row_idx, balance_col]).strip()
        matches = re.findall(pattern, balance_val)
        
        if matches and len(matches) > 0:
            first_num, second_num = matches[0]
            if balance_col > 0:
                prev_col_val = str(df_split.iloc[row_idx, balance_col - 1]).strip()
                if prev_col_val == '' or prev_col_val.lower() == 'nan':
                    df_split.iloc[row_idx, balance_col - 1] = first_num
            df_split.iloc[row_idx, balance_col] = second_num
    
    return df_split

def remove_duplicate_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate header rows from extracted dataframe."""
    if df.empty:
        return df
    
    df_clean = df.copy()
    headers = set(str(h).lower().strip() for h in df_clean.columns)
    
    rows_to_drop = []
    for idx, row in df_clean.iterrows():
        matching_cells = 0
        for val in row:
            if str(val).lower().strip() in headers:
                matching_cells += 1
        
        if matching_cells >= len(df_clean.columns) * 0.6:
            rows_to_drop.append(idx)
        
        row_str = " ".join(str(v).lower() for v in row)
        if re.search(r"sl\s*no|txn\s*date|description|cheque|withdrawal|deposit|balance", row_str):
            if any(keyword in row_str for keyword in ["txn date", "description", "balance"]):
                rows_to_drop.append(idx)
    
    df_clean = df_clean.drop(rows_to_drop).reset_index(drop=True)
    return df_clean

def pdf_page_to_image_base64(pdf_path, page_num):
    """Convert a PDF page to base64 encoded image."""
    try:
        doc = fitz.open(pdf_path)
        page = doc[page_num - 1]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        image_bytes = pix.tobytes("png")
        base64_image = base64.standard_b64encode(image_bytes).decode("utf-8")
        doc.close()
        return base64_image
    except Exception as e:
        st.error(f"Error converting PDF page to image: {e}")
        return None

def refine_and_validate_data(df):
    """Refine and validate extracted data."""
    if df.empty:
        return {
            "refined_df": df,
            "validation_summary": {"valid_rows": 0, "invalid_rows": 0},
            "refinement_count": 0,
            "fixable_issues": [],
        }
    
    valid_rows = len(df[df.astype(str).apply(lambda x: x.str.strip() != '').any(axis=1)])
    invalid_rows = len(df) - valid_rows
    
    return {
        "refined_df": df,
        "validation_summary": {
            "valid_rows": valid_rows,
            "invalid_rows": invalid_rows,
        },
        "refinement_count": 0,
        "fixable_issues": [],
    }

def parse_amount(s: str):
    if pd.isna(s):
        return 0.0
    s = str(s).strip().replace('"', '')
    neg = False
    if s.startswith('(') and s.endswith(')'):
        neg = True
        s = s[1:-1]
    if s.startswith('-'):
        neg = True
        s = s[1:]
    s2 = re.sub(r"[^0-9.]", "", s)
    if s2 == '':
        return 0.0
    val = float(s2)
    return -val if neg else val

def normalize_text(s: str):
    if pd.isna(s):
        return ''
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def extract_references(s: str):
    if pd.isna(s):
        return []
    s = str(s)
    refs = re.findall(r"\[([^\]]+)\]", s)
    refs += re.findall(r"\b([A-Z0-9]{6,})\b", s)
    refs = list({r.strip().lower() for r in refs if r.strip()})
    return refs

def classify_txn(row):
    w = row.get('Withdrawal (in Rs.)', '')
    d = row.get('Deposits (in Rs.)', '')
    wv = parse_amount(w)
    dv = parse_amount(d)
    if dv and (not wv or abs(dv) > abs(wv)):
        return 'credit', dv
    if wv:
        return 'debit', abs(wv)
    return 'unknown', 0.0

def is_promotional(narr):
    EXCLUDE_PROMO = ["promo", "offer"]
    n = normalize_text(narr)
    for kw in EXCLUDE_PROMO:
        if kw in n and not any(rfw in n for rfw in ["refund", "returned", "order cancel", "cancelled", "ref"]):
            return True
    return False

def standardize_columns(df):
    """Standardize column names to consistent format"""
    df_clean = df.copy()
    
    column_mapping = {
        'transaction date': 'txn_date', 'date': 'txn_date', 'txn date': 'txn_date',
        'value date': 'txn_date', 'posting date': 'txn_date', 'transaction_date': 'txn_date',
        'description': 'description', 'narration': 'description', 'details': 'description',
        'transaction details': 'description', 'particulars': 'description',
        'type': 'type', 'transaction type': 'type', 'txn type': 'type', 'dr/cr': 'type',
        'amount': 'amount', 'transaction amount': 'amount', 'txn amount': 'amount',
        'balance': 'balance', 'running balance': 'balance', 'available balance': 'balance'
    }
    
    new_columns = []
    for col in df_clean.columns:
        col_clean = str(col).lower().strip().replace(' ', '_')
        new_col = column_mapping.get(col_clean, col_clean)
        new_columns.append(new_col)
    
    df_clean.columns = new_columns
    
    essential_columns = ['txn_date', 'description', 'amount']
    for col in essential_columns:
        if col not in df_clean.columns:
            df_clean[col] = None
    
    return df_clean

def validate_dataframe(df):
    """Validate DataFrame structure and provide debugging info"""
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    required = ['txn_date', 'description', 'amount']
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"❌ Missing columns: {missing}")
        return False
    else:
        print("✅ All required columns present")
        return True

# -------------------------------------------------------------------
# UI Helper Functions
# -------------------------------------------------------------------
def create_step_indicator(current_step):
    """Create a step indicator showing progress through the workflow."""
    steps = [
        {"label": "Upload Files", "key": "upload"},
        {"label": "Preview Data", "key": "preview"},
        {"label": "Merge Files", "key": "merge"},
        {"label": "Analysis", "key": "analysis"}
    ]
    
    current_index = next((i for i, step in enumerate(steps) if step["key"] == current_step), 0)
    
    html = "<div class='step-indicator'>"
    for i, step in enumerate(steps):
        status_class = "active" if step["key"] == current_step else ("completed" if i < current_index else "")
        html += f"<div class='step {status_class}'><div class='step-circle'>{i+1}</div><div class='step-label'>{step['label']}</div></div>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

def create_metric_card(value, label, icon):
    """Create a metric card for displaying statistics."""
    st.markdown(f"""
    <div class="metric-card">
        <h3>{icon} {value}</h3>
        <p>{label}</p>
    </div>
    """, unsafe_allow_html=True)

def styled_button(label, key=None, type="primary"):
    """Create a styled button with custom CSS classes."""
    if type == "primary":
        st.markdown(f'<div class="primary-button">', unsafe_allow_html=True)
        clicked = st.button(label, key=key, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    elif type == "secondary":
        st.markdown(f'<div class="secondary-button">', unsafe_allow_html=True)
        clicked = st.button(label, key=key, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    elif type == "success":
        st.markdown(f'<div class="success-button">', unsafe_allow_html=True)
        clicked = st.button(label, key=key, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        clicked = st.button(label, key=key, use_container_width=True)
    return clicked

def display_analysis_details(result, dataset_name):
    """Display detailed analysis results with relevant entries shown below each category."""
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        refund_count = result['is_refund'].sum()
        create_metric_card(refund_count, "Refunds", "🔄")
    with col2:
        return_count = result['is_returned'].sum()
        create_metric_card(return_count, "Returns", "📤")
    with col3:
        cash_count = result['is_cash_withdrawal'].sum()
        create_metric_card(cash_count, "Cash Withdrawals", "💵")
    with col4:
        charge_count = result['is_bank_charge'].sum()
        create_metric_card(charge_count, "Bank Charges", "💸")
    
    col5, col6 = st.columns(2)
    with col5:
        commission_count = result['is_commission_paid'].sum()
        create_metric_card(commission_count, "Commissions", "💰")
    with col6:
        statutory_count = result['is_statutory'].sum()
        create_metric_card(statutory_count, "Statutory Payments", "🏛️")
    
    st.markdown("### 📋 Detailed Analysis Results")
    
    # Refunds
    if refund_count > 0:
        with st.expander(f"🔄 Refunds ({refund_count} entries)", expanded=True):
            refund_entries = result[result['is_refund'] == True][['Txn Date', 'Description', 'amount']]
            st.dataframe(refund_entries, use_container_width=True)
            st.write(f"**Total Refund Amount:** ₹{refund_entries['amount'].sum():,.2f}")
    
    # Returns
    if return_count > 0:
        with st.expander(f"📤 Returns ({return_count} entries)", expanded=True):
            return_entries = result[result['is_returned'] == True][['Txn Date', 'Description', 'amount']]
            st.dataframe(return_entries, use_container_width=True)
            st.write(f"**Total Return Amount:** ₹{return_entries['amount'].sum():,.2f}")
    
    # Cash Withdrawals
    if cash_count > 0:
        with st.expander(f"💵 Cash Withdrawals ({cash_count} entries)", expanded=True):
            cash_entries = result[result['is_cash_withdrawal'] == True][['Txn Date', 'Description', 'amount']]
            st.dataframe(cash_entries, use_container_width=True)
            st.write(f"**Total Cash Withdrawn:** ₹{cash_entries['amount'].sum():,.2f}")
    
    # Bank Charges
    if charge_count > 0:
        with st.expander(f"💸 Bank Charges ({charge_count} entries)", expanded=True):
            charge_entries = result[result['is_bank_charge'] == True][['Txn Date', 'Description', 'amount']]
            st.dataframe(charge_entries, use_container_width=True)
            st.write(f"**Total Bank Charges:** ₹{charge_entries['amount'].sum():,.2f}")
    
    # Commissions
    if commission_count > 0:
        with st.expander(f"💰 Commissions ({commission_count} entries)", expanded=True):
            commission_entries = result[result['is_commission_paid'] == True][['Txn Date', 'Description', 'amount']]
            st.dataframe(commission_entries, use_container_width=True)
            st.write(f"**Total Commission Paid:** ₹{commission_entries['amount'].sum():,.2f}")
    
    # Statutory Payments
    if statutory_count > 0:
        with st.expander(f"🏛️ Statutory Payments ({statutory_count} entries)", expanded=True):
            statutory_entries = result[result['is_statutory'] == True][['Txn Date', 'Description', 'amount']]
            st.dataframe(statutory_entries, use_container_width=True)
            st.write(f"**Total Statutory Payments:** ₹{statutory_entries['amount'].sum():,.2f}")
    
    # Full results table
    with st.expander("📊 View Full Analysis Table"):
        st.dataframe(result, use_container_width=True)