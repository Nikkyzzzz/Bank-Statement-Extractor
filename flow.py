import streamlit as st
import pandas as pd
import os
import re
import camelot.io as camelot
import pdfplumber
import contextlib
import tempfile
import io
import base64
import json
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
from dotenv import load_dotenv
import fitz  # PyMuPDF
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize OpenAI client
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key:
    client = OpenAI(api_key=openai_api_key)
else:
    client = None

# Configure Streamlit with improved settings
st.set_page_config(
    page_title="Bank Statement Processor", 
    layout="wide", 
    page_icon="🏦",
    initial_sidebar_state="collapsed"
)

# Enhanced CSS for modern, attractive UI with Dark Mode Support
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

# Apply custom CSS
st.markdown(MODERN_CSS, unsafe_allow_html=True)

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
# PDF Extraction Functions (from check5.py)
# -------------------------------------------------------------------
def extract_pdfplumber_pagewise(path):
    """Extract tables from PDFs using pdfplumber - excellent for borderless tables."""
    all_pages = []
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            with pdfplumber.open(path) as pdf:
                for page_idx, page in enumerate(pdf.pages, start=1):
                    try:
                        # Use pdfplumber's table extraction (layout-based)
                        tables = page.extract_tables()
                        
                        if tables:
                            for table in tables:
                                # Convert table list-of-lists to DataFrame
                                if table and len(table) > 0:
                                    df = pd.DataFrame(table[1:], columns=table[0])
                                    df = df.astype(str)
                                    # Clean up column names
                                    df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]
                                    all_pages.append(df)
                    except Exception:
                        pass
    except Exception:
        pass
    
    return all_pages

def extract_camelot_pagewise(path, flavor):
    """Extract tables page-wise using Camelot."""
    all_pages = []
    extracted_headers = None  # Will store headers from page 1
    
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
            
            # PAGE 1: Extract headers
            if i == 1 and not df.empty:
                # Use first row as headers ONLY on page 1
                extracted_headers = [re.sub(r"\s+", " ", str(c)).strip() for c in df.iloc[0].values]
                df = df.iloc[1:].reset_index(drop=True)
                df.columns = extracted_headers
            # PAGE 2+: Apply same headers from page 1
            elif i > 1 and not df.empty and extracted_headers is not None:
                # All data rows - no header extraction
                df = df.astype(str)
                # Apply the headers from page 1, match by column count
                if len(df.columns) == len(extracted_headers):
                    df.columns = extracted_headers
                else:
                    # If column count doesn't match, try to align
                    df.columns = extracted_headers[:len(df.columns)]
            elif not df.empty:
                df = df.astype(str)
            
            # Apply split merged decimals function
            if not df.empty:
                df = split_merged_decimals(df)
            
            all_pages.append(df)
    except Exception as e:
        pass
    
    return all_pages

def split_merged_decimals(df: pd.DataFrame) -> pd.DataFrame:
    """Split merged decimal values in Balance column."""
    df_split = df.copy()
    
    # Find column indices (case-insensitive)
    balance_col = None
    
    for idx, col_name in enumerate(df_split.columns):
        col_lower = str(col_name).lower()
        if 'balance' in col_lower:
            balance_col = idx
            break
    
    if balance_col is None:
        return df_split
    
    # Pattern to detect TWO decimal numbers in one cell
    pattern = r'([-\d,]+\.\d{2})\s+([-\d,]+\.\d+)'
    
    # Process each row in the Balance column
    for row_idx in range(len(df_split)):
        balance_val = str(df_split.iloc[row_idx, balance_col]).strip()
        
        # Check if Balance has two decimal numbers merged
        matches = re.findall(pattern, balance_val)
        
        if matches and len(matches) > 0:
            first_num, second_num = matches[0]
            
            # First number → moves to previous column (Deposits or Withdrawal)
            if balance_col > 0:
                prev_col_val = str(df_split.iloc[row_idx, balance_col - 1]).strip()
                # Only put first number in previous column if it's empty
                if prev_col_val == '' or prev_col_val.lower() == 'nan':
                    df_split.iloc[row_idx, balance_col - 1] = first_num
            
            # Second number → Stays in Balance column
            df_split.iloc[row_idx, balance_col] = second_num
    
    return df_split

def remove_duplicate_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate header rows from extracted dataframe."""
    if df.empty:
        return df
    
    df_clean = df.copy()
    headers = set(str(h).lower().strip() for h in df_clean.columns)
    
    # Find rows that are likely duplicate headers
    rows_to_drop = []
    for idx, row in df_clean.iterrows():
        # Count how many cells in this row match header values
        matching_cells = 0
        for val in row:
            if str(val).lower().strip() in headers:
                matching_cells += 1
        
        # If most cells match headers, it's likely a header row
        if matching_cells >= len(df_clean.columns) * 0.6:  # 60% threshold
            rows_to_drop.append(idx)
        
        # Also check for common header patterns
        row_str = " ".join(str(v).lower() for v in row)
        if re.search(r"sl\s*no|txn\s*date|description|cheque|withdrawal|deposit|balance", row_str):
            if any(keyword in row_str for keyword in ["txn date", "description", "balance"]):
                rows_to_drop.append(idx)
    
    # Remove duplicate header rows
    df_clean = df_clean.drop(rows_to_drop).reset_index(drop=True)
    
    return df_clean

def pdf_page_to_image_base64(pdf_path, page_num):
    """Convert a PDF page to base64 encoded image."""
    try:
        doc = fitz.open(pdf_path)
        page = doc[page_num - 1]
        
        # Render page to image with high DPI for better OCR
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for clarity
        image_bytes = pix.tobytes("png")
        base64_image = base64.standard_b64encode(image_bytes).decode("utf-8")
        
        doc.close()
        return base64_image
    except Exception as e:
        st.error(f"Error converting PDF page to image: {e}")
        return None

def extract_openai_pagewise(pdf_path):
    """Extract tables from PDFs using OpenAI's GPT-4 Vision API."""
    if not client:
        st.error("❌ OpenAI API key not configured. Set OPENAI_API_KEY environment variable.")
        return []
    
    all_pages = []
    extracted_headers = None
    
    try:
        # Get total page count
        with contextlib.redirect_stderr(io.StringIO()):
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
        
        for page_num in range(1, total_pages + 1):
            try:
                # Convert page to base64 image
                base64_image = pdf_page_to_image_base64(pdf_path, page_num)
                if not base64_image:
                    all_pages.append(pd.DataFrame())
                    continue
                
                # Send to OpenAI GPT-4 Vision API
                response = client.chat.completions.create(
                    model="gpt-4o",
                    max_tokens=4096,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{base64_image}",
                                        "detail": "high"
                                    },
                                },
                                {
                                    "type": "text",
                                    "text": """Extract all transaction data from this bank statement page into a structured table format.
                                    
Return the data as a JSON object with:
- "headers": array of column names
- "rows": array of row data (each row is an array of values)

Important:
1. Include ALL visible columns in the table
2. Each row should have the same number of values as there are headers
3. Extract exact values as shown
4. For Date columns, preserve the format shown
5. For Amount columns, preserve the format shown (including commas, decimals, currency symbols)
6. Use empty string "" for missing values
7. Do NOT include summary rows or page totals

Return ONLY valid JSON, no additional text."""
                                }
                            ],
                        }
                    ],
                )
                
                # Parse response
                try:
                    response_text = response.choices[0].message.content
                    
                    # Extract JSON from response
                    json_match = re.search(r'\{[\s\S]*\}', response_text)
                    if json_match:
                        table_data = json.loads(json_match.group())
                        
                        headers = table_data.get("headers", [])
                        rows = table_data.get("rows", [])
                        
                        if headers and rows:
                            # Create DataFrame
                            df = pd.DataFrame(rows, columns=headers)
                            df = df.astype(str)
                            
                            # PAGE 1: Extract and store headers
                            if page_num == 1:
                                extracted_headers = [re.sub(r"\s+", " ", str(c)).strip() for c in headers]
                            
                            # Apply same headers for consistency across pages
                            if page_num > 1 and extracted_headers and len(df.columns) == len(extracted_headers):
                                df.columns = extracted_headers
                            
                            # Clean up data
                            df = split_merged_decimals(df)
                            all_pages.append(df)
                        else:
                            all_pages.append(pd.DataFrame())
                    else:
                        all_pages.append(pd.DataFrame())
                        
                except json.JSONDecodeError:
                    st.warning(f"⚠️ Could not parse OpenAI response for page {page_num}")
                    all_pages.append(pd.DataFrame())
                    
            except Exception as e:
                st.warning(f"⚠️ Error processing page {page_num}: {str(e)}")
                all_pages.append(pd.DataFrame())
        
        return all_pages
        
    except Exception as e:
        st.error(f"Error during OpenAI extraction: {e}")
        return []

def refine_and_validate_data(df):
    """Refine and validate extracted data."""
    if df.empty:
        return {
            "refined_df": df,
            "validation_summary": {"valid_rows": 0, "invalid_rows": 0},
            "refinement_count": 0,
            "fixable_issues": [],
        }
    
    # Count valid rows (non-empty rows)
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

# -------------------------------------------------------------------
# File Merging Functions (from joinfile.py) - UPDATED
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
            # Allow partial match (robust matching)
            if variant.lower() in normalized:
                return standard
    if "sl" in normalized and "no" in normalized:
        return "Sl No"
    return None

def read_file(file):
    """Robust file reader with multiple fallback methods."""
    try:
        # Reset file pointer to beginning
        if hasattr(file, 'seek'):
            file.seek(0)
        
        ext = os.path.splitext(file.name)[1].lower()
        
        if ext in [".xlsx", ".xls"]:
            # Try multiple engines for Excel files
            engines_to_try = ['openpyxl', 'xlrd']
            
            for engine in engines_to_try:
                try:
                    if hasattr(file, 'seek'):
                        file.seek(0)
                    
                    if ext == ".xlsx":
                        df = pd.read_excel(file, engine=engine)
                    else:  # .xls
                        df = pd.read_excel(file, engine=engine)
                    
                    if df is not None and not df.empty:
                        st.success(f"✅ Successfully read {file.name} using {engine} engine")
                        return df
                        
                except Exception as e:
                    if hasattr(file, 'seek'):
                        file.seek(0)
                    continue
            
            # If specific engines fail, try without specifying engine
            try:
                if hasattr(file, 'seek'):
                    file.seek(0)
                df = pd.read_excel(file)
                if df is not None and not df.empty:
                    st.success(f"✅ Successfully read {file.name} using default engine")
                    return df
            except Exception as e:
                st.warning(f"⚠️ Standard Excel reading failed for {file.name}")
            
            # Final attempt: try to read as CSV
            st.info(f"🔄 Attempting to read {file.name} as CSV as final fallback...")
            try:
                if hasattr(file, 'seek'):
                    file.seek(0)
                
                # Read file content
                content = file.read()
                if isinstance(content, bytes):
                    content = content.decode('utf-8', errors='ignore')
                
                # Try different delimiters
                for delimiter in [',', '\t', ';', '|']:
                    try:
                        if hasattr(file, 'seek'):
                            file.seek(0)
                        
                        if delimiter == ',':
                            df = pd.read_csv(file)
                        else:
                            df = pd.read_csv(file, delimiter=delimiter)
                        
                        if df is not None and not df.empty:
                            st.success(f"✅ Successfully read {file.name} as CSV with '{delimiter}' delimiter")
                            return df
                    except:
                        if hasattr(file, 'seek'):
                            file.seek(0)
                        continue
                        
            except Exception as e:
                st.error(f"❌ All reading methods failed for {file.name}")
                return None
                    
        elif ext == ".csv":
            # Try different encodings for CSV
            encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-8-sig']
            
            for encoding in encodings_to_try:
                try:
                    if hasattr(file, 'seek'):
                        file.seek(0)
                    df = pd.read_csv(file, encoding=encoding)
                    if df is not None and not df.empty:
                        st.success(f"✅ Successfully read {file.name} with {encoding} encoding")
                        return df
                except Exception as e:
                    if hasattr(file, 'seek'):
                        file.seek(0)
                    continue
            
            # Final attempt with error handling
            try:
                if hasattr(file, 'seek'):
                    file.seek(0)
                df = pd.read_csv(file, encoding='utf-8', on_bad_lines='skip')
                if df is not None and not df.empty:
                    st.warning(f"⚠️ Read {file.name} with some lines skipped")
                    return df
            except Exception as e:
                st.error(f"❌ Could not read CSV file {file.name}")
                return None
                
        else:
            st.warning(f"Unsupported file format: {file.name}")
            return None
            
    except Exception as e:
        st.error(f"❌ Unexpected error reading file {file.name}: {str(e)}")
        return None

def convert_problematic_excel(file):
    """Convert problematic Excel file to a simpler format."""
    try:
        # Read the file content
        if hasattr(file, 'seek'):
            file.seek(0)
        content = file.read()
        
        # Create a simple DataFrame with the file info
        df = pd.DataFrame({
            'File_Name': [file.name],
            'File_Size': [len(content)],
            'Status': ['Problematic_File_Needs_Manual_Processing']
        })
        
        st.warning(f"⚠️ Created placeholder for problematic file: {file.name}")
        st.info(f"💡 Please manually process this file and re-upload as CSV")
        
        return df
        
    except Exception as e:
        st.error(f"❌ Could not create placeholder for {file.name}: {str(e)}")
        return None

def merge_files(uploaded_files):
    """Merge multiple uploaded files."""
    all_data = []

    for file in uploaded_files:
        df = read_file(file)
        if df is None or df.empty:
            st.warning(f"⚠️ Could not read {file.name}, attempting conversion...")
            df = convert_problematic_excel(file)
            
        if df is None or df.empty:
            continue

        # Normalize columns
        mapped_cols = {}
        for col in df.columns:
            mapped = map_column(col)
            if mapped:
                mapped_cols[col] = mapped

        df = df.rename(columns=mapped_cols)

        # If the DataFrame has duplicate column names, make them unique by suffixing
        if df.columns.duplicated().any():
            new_cols = []
            counts = {}
            for c in df.columns:
                if c in counts:
                    counts[c] += 1
                    new_cols.append(f"{c}__dup{counts[c]}")
                else:
                    counts[c] = 0
                    new_cols.append(c)
            df.columns = new_cols

        # Keep only required columns, fill missing
        for col in STANDARD_COLUMNS:
            if col not in df.columns and col != "File Name":
                df[col] = ""

        # Reindex safely: pick only columns that exist and preserve order
        cols_to_use = [col for col in STANDARD_COLUMNS if col in df.columns]
        
        # Ensure File Name column is included
        if 'File Name' in df.columns and 'File Name' not in cols_to_use:
            cols_to_use.append('File Name')
        
        df = df[cols_to_use]
        
        # Add File Name if not present
        if 'File Name' not in df.columns:
            df["File Name"] = file.name

        all_data.append(df)

    if all_data:
        try:
            merged_df = pd.concat(all_data, ignore_index=True)
            return merged_df
        except Exception as e:
            st.error("Failed to concatenate DataFrames. See details below.")
            st.exception(e)
            # Show per-file column lists
            for i, df in enumerate(all_data):
                st.write(f"File {i+1} columns:", list(df.columns))
            return None
    else:
        st.warning("No valid data found in the uploaded files.")
        return None

def merge_dataframes(dataframes):
    """Merge multiple dataframes using the same logic as merge_files but for already loaded dataframes."""
    all_data = []

    for df in dataframes:
        if df is None or df.empty:
            continue

        # Normalize columns
        mapped_cols = {}
        for col in df.columns:
            mapped = map_column(col)
            if mapped:
                mapped_cols[col] = mapped

        df = df.rename(columns=mapped_cols)

        # If the DataFrame has duplicate column names, make them unique by suffixing
        if df.columns.duplicated().any():
            new_cols = []
            counts = {}
            for c in df.columns:
                if c in counts:
                    counts[c] += 1
                    new_cols.append(f"{c}__dup{counts[c]}")
                else:
                    counts[c] = 0
                    new_cols.append(c)
            df.columns = new_cols

        # Keep only required columns, fill missing
        for col in STANDARD_COLUMNS:
            if col not in df.columns and col != "File Name":
                df[col] = ""

        # Reindex safely: pick only columns that exist and preserve order
        cols_to_use = [col for col in STANDARD_COLUMNS if col in df.columns]
        
        # Ensure File Name column is included
        if 'File Name' in df.columns and 'File Name' not in cols_to_use:
            cols_to_use.append('File Name')
        
        df = df[cols_to_use]

        all_data.append(df)

    if all_data:
        try:
            merged_df = pd.concat(all_data, ignore_index=True)
            
            # Ensure File Name column exists in the final merged dataframe
            if 'File Name' not in merged_df.columns:
                st.warning("⚠️ File Name column was lost during merging. This may affect file tracking.")
            
            return merged_df
        except Exception as e:
            st.error("Failed to concatenate DataFrames. See details below.")
            st.exception(e)
            # Show per-file column lists
            for i, df in enumerate(all_data):
                st.write(f"File {i+1} columns:", list(df.columns))
            return None
    else:
        st.warning("No valid data found in the files.")
        return None

# -------------------------------------------------------------------
# Analysis Functions (from new1.py)
# -------------------------------------------------------------------
REFUND_KEYWORDS = ["refund", "refd", "returned", "chargeback", "chargeback credit", "pg refund", "cashback for refund"]
EXCLUDE_PROMO = ["promo", "offer"]

def parse_amount(s: str):
    if pd.isna(s):
        return 0.0
    # remove quotes/spaces
    s = str(s).strip().replace('"', '')
    # handle negative sign placed before number or with leading '-'
    neg = False
    if s.startswith('(') and s.endswith(')'):
        neg = True
        s = s[1:-1]
    if s.startswith('-'):
        neg = True
        s = s[1:]
    # remove commas and any non-digit characters except dot
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
    # common bracketed reference patterns like [BI1234], [6059IRT25000392], order ids etc
    refs = re.findall(r"\[([^\]]+)\]", s)
    # also look for token-like references (ARN/RRN/UPI) - alphanumeric blocks with numbers
    refs += re.findall(r"\b([A-Z0-9]{6,})\b", s)
    # normalize
    refs = list({r.strip().lower() for r in refs if r.strip()})
    return refs

def load_table(path_or_buffer):
    """Load a table from a path or file-like buffer. Supports CSV and Excel (xls/xlsx)."""
    # If it's a path string, decide by suffix
    try:
        # path-like string
        if isinstance(path_or_buffer, str):
            lower = path_or_buffer.lower()
            if lower.endswith('.csv'):
                return pd.read_csv(path_or_buffer, skipinitialspace=True)
            if lower.endswith('.xls') or lower.endswith('.xlsx'):
                return pd.read_excel(path_or_buffer)
    except Exception:
        # fall back to trying as buffer
        pass

    # If it's a file-like buffer (UploadedFile from Streamlit or io.BytesIO)
    # Try CSV first, then Excel
    try:
        # pandas can read file-like objects directly
        return pd.read_csv(path_or_buffer, skipinitialspace=True)
    except Exception:
        try:
            return pd.read_excel(path_or_buffer)
        except Exception as e:
            raise ValueError(f"Could not read input as CSV or Excel: {e}")

def classify_txn(row):
    # If Withdrawal column has value -> Debit, if Deposits has value -> Credit
    w = row.get('Withdrawal (in Rs.)', '')
    d = row.get('Deposits (in Rs.)', '')
    wv = parse_amount(w)
    dv = parse_amount(d)
    if dv and (not wv or abs(dv) > abs(wv)):
        return 'credit', dv
    if wv:
        return 'debit', abs(wv)
    # fallback: if Balance changed up -> credit
    return 'unknown', 0.0

def is_promotional(narr):
    n = normalize_text(narr)
    for kw in EXCLUDE_PROMO:
        if kw in n and not any(rfw in n for rfw in ["refund", "returned", "order cancel", "cancelled", "ref" ]):
            return True
    return False

# =============================================================================
# STANDARDIZATION & VALIDATION UTILITIES
# =============================================================================

def standardize_columns(df):
    """
    Standardize column names to consistent format across different bank statement formats
    """
    # Create a copy to avoid modifying original
    df_clean = df.copy()
    
    # Define comprehensive column mapping
    column_mapping = {
        # Date columns
        'transaction date': 'txn_date',
        'date': 'txn_date',
        'txn date': 'txn_date',
        'value date': 'txn_date',
        'posting date': 'txn_date',
        'transaction_date': 'txn_date',
        'txn_date': 'txn_date',
        
        # Description columns
        'description': 'description',
        'narration': 'description',
        'details': 'description',
        'transaction details': 'description',
        'particulars': 'description',
        'transaction_description': 'description',
        'narration/description': 'description',
        
        # Type columns
        'type': 'type',
        'transaction type': 'type',
        'transaction_type': 'type',
        'txn type': 'type',
        'txn_type': 'type',
        'dr/cr': 'type',
        'debit/credit': 'type',
        'credit/debit': 'type',
        
        # Amount columns
        'amount': 'amount',
        'transaction amount': 'amount',
        'transaction_amount': 'amount',
        'txn amount': 'amount',
        'txn_amount': 'amount',
        'value': 'amount',
        'transaction_amt': 'amount',
        'amt': 'amount',
        'debit': 'amount',
        'credit': 'amount',
        
        # Balance columns
        'balance': 'balance',
        'running balance': 'balance',
        'available balance': 'balance',
        'current balance': 'balance'
    }
    
    # Clean and map column names
    new_columns = []
    for col in df_clean.columns:
        col_clean = str(col).lower().strip().replace(' ', '_')
        new_col = column_mapping.get(col_clean, col_clean)
        new_columns.append(new_col)
    
    df_clean.columns = new_columns
    
    # Ensure essential columns exist, create if missing
    essential_columns = ['txn_date', 'description', 'amount']
    for col in essential_columns:
        if col not in df_clean.columns:
            print(f"Warning: Essential column '{col}' not found. Creating empty column.")
            df_clean[col] = None
    
    return df_clean

def validate_dataframe(df):
    """
    Validate DataFrame structure and provide debugging info
    """
    print("=== DATAFRAME VALIDATION ===")
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Data types:\n{df.dtypes}")
    print(f"First 3 rows:")
    print(df.head(3))
    print("============================\n")
    
    # Check for required columns
    required = ['txn_date', 'description', 'amount']
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"❌ Missing columns: {missing}")
        return False
    else:
        print("✅ All required columns present")
        return True

def create_sample_data():
    """
    Create sample bank statement data for testing
    """
    sample_data = {
        'Txn Date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'],
        'Description': [
            'Salary Credit',
            'ATM Withdrawal',
            'Payment Return - Transaction Failed', 
            'Groceries Payment',
            'Chargeback Adjustment'
        ],
        'Transaction Type': ['Credit', 'Debit', 'Credit', 'Debit', 'Credit'],
        'Amount': [5000.00, 200.00, 150.00, 85.50, 299.99],
        'Balance': [5000.00, 4800.00, 4950.00, 4864.50, 5164.49]
    }
    
    return pd.DataFrame(sample_data)

def load_bank_data(file_path, file_type='csv'):
    """
    Load bank data from various file formats with proper error handling
    """
    try:
        if file_type.lower() == 'csv':
            df = pd.read_csv(file_path)
        elif file_type.lower() == 'excel':
            df = pd.read_excel(file_path)
        else:
            print(f"Unsupported file type: {file_type}")
            return None
        
        print(f"✅ Successfully loaded data from {file_path}")
        print(f"Original columns: {df.columns.tolist()}")
        
        # Standardize columns
        df_std = standardize_columns(df)
        
        return df_std
        
    except Exception as e:
        print(f"❌ Error loading file: {str(e)}")
        return None

def detect_refunds(df: pd.DataFrame):
    # Normalize columns
    df = df.copy()
    # Ensure Txn Date column
    df['Txn Date'] = pd.to_datetime(df['Txn Date'], dayfirst=True, errors='coerce')
    df['narration'] = df['Description'].fillna('').astype(str)
    df['narr_norm'] = df['narration'].apply(normalize_text)
    df['refs'] = df['narration'].apply(extract_references)
    df['type'], df['amount'] = zip(*df.apply(classify_txn, axis=1))
    df['is_promo'] = df['narration'].apply(is_promotional)

    # Build lists of debits and credits
    debits = []
    credits = []
    for i, r in df.iterrows():
        rec = r.to_dict()
        rec['index'] = i
        if rec['type'] == 'debit':
            debits.append(rec)
        elif rec['type'] == 'credit':
            credits.append(rec)

    # index debits by reference and merchant string
    deb_by_ref = defaultdict(list)
    deb_by_merchant = defaultdict(list)
    for d in debits:
        for ref in d.get('refs', []):
            deb_by_ref[ref].append(d)
        # use normalized narration as merchant proxy
        deb_by_merchant[d['narr_norm']].append(d)

    # Track matched pairs and remaining refundable amount per debit
    matches = []
    remaining = {d['index']: d['amount'] for d in debits}

    # Helper to attempt match for a credit
    def try_match_credit(c):
        c_amount = c['amount']
        c_date = c['Txn Date']
        narr = c['narr_norm']
        refs = c.get('refs', [])
        matched = []
        # require narration to explicitly mention refund-like keywords
        if not any(k in narr for k in REFUND_KEYWORDS):
            # don't consider this credit a refund
            return []

        # skip purely promotional entries that aren't refunds
        if c['is_promo'] and not any(k in narr for k in REFUND_KEYWORDS):
            return []

        # helper to check date and amount constraints for a candidate debit
        def candidate_ok(d):
            # date window: debit must be on or before credit and within 180 days
            if d['Txn Date'] is pd.NaT or c_date is pd.NaT:
                date_ok = True
            else:
                date_ok = (d['Txn Date'] <= c_date) and ((c_date - d['Txn Date']).days <= 180)
            if not date_ok:
                return False
            # amount: credit amount must be <= remaining debit amount (+ small tolerance)
            amt_allowed = remaining.get(d['index'], 0)
            if amt_allowed <= 0:
                return False
            if c_amount > amt_allowed * 1.01:
                return False
            return True

        # 1) exact reference match (highest priority)
        for ref in refs:
            candidates = deb_by_ref.get(ref, [])
            for d in candidates:
                if not candidate_ok(d):
                    continue
                # allow small tolerance relative to debit amount
                take = min(c_amount, remaining[d['index']])
                remaining[d['index']] = round(remaining[d['index']] - take, 2)
                matches.append({'credit_idx': c['index'], 'debit_idx': d['index'], 'matched_amount': take, 'match_type': 'ref'})
                c_amount -= take
                if c_amount <= 0.005:
                    return matches

        # 2) merchant string match (priority after reference) + same/less amount + within window
        for m, candidates in deb_by_merchant.items():
            if not m:
                continue
            # require some token overlap between credit narration and merchant string
            if not any(token in m for token in narr.split()[:6]):
                continue
            for d in candidates:
                if not candidate_ok(d):
                    continue
                # record match
                take = min(c_amount, remaining[d['index']])
                remaining[d['index']] = round(remaining[d['index']] - take, 2)
                matches.append({'credit_idx': c['index'], 'debit_idx': d['index'], 'matched_amount': take, 'match_type': 'merchant'})
                c_amount -= take
                if c_amount <= 0.005:
                    return matches

        # 3) category match (basic: if narration contains 'upi' or 'pg' or 'pos')
        categories = ['upi', 'pg', 'pos']
        for cat in categories:
            if cat in narr:
                for d in debits:
                    if d['index'] == c['index']:
                        continue
                    if cat not in d['narr_norm']:
                        continue
                    if not candidate_ok(d):
                        continue
                    # record match
                    take = min(c_amount, remaining[d['index']])
                    remaining[d['index']] = round(remaining[d['index']] - take, 2)
                    matches.append({'credit_idx': c['index'], 'debit_idx': d['index'], 'matched_amount': take, 'match_type': 'category'})
                    c_amount -= take
                    if c_amount <= 0.005:
                        return matches

        return matches

    # process credits in chronological order
    credits_sorted = sorted(credits, key=lambda r: r['Txn Date'] or datetime.min)
    for c in credits_sorted:
        try_match_credit(c)

    # attach match info to dataframe
    df['matched'] = [[] for _ in range(len(df))]
    for m in matches:
        cidx = m['credit_idx']
        didx = m['debit_idx']
        df.at[cidx, 'matched'] = df.at[cidx, 'matched'] + [m]
        df.at[didx, 'matched'] = df.at[didx, 'matched'] + [m]

    # Tag refunds
    df['is_refund'] = df['matched'].apply(lambda x: len(x) > 0 and df.at[x[0]['credit_idx'], 'type'] == 'credit' if isinstance(x, list) and x else False)

    # For debits, compute remaining refundable balance
    df['remaining_refundable'] = None
    for d in debits:
        df.at[d['index'], 'remaining_refundable'] = remaining.get(d['index'], d['amount'])

    # Prepare output columns
    out_cols = ['Txn Date', 'Description', 'type', 'amount', 'matched', 'is_refund', 'remaining_refundable']
    return df[out_cols]

def detect_returns(df: pd.DataFrame):
    """Detect rail-driven returns (NEFT/RTGS/IMPS/UPI/ECS/NACH/Cheque returns)."""
    df = df.copy()
    # Ensure date column exists
    df['Txn Date'] = pd.to_datetime(df.get('Txn Date'), dayfirst=True, errors='coerce')
    # Ensure type and amount columns exist
    if 'type' not in df.columns or 'amount' not in df.columns:
        df['type'], df['amount'] = zip(*df.apply(classify_txn, axis=1))
    df['narr_norm'] = df['Description'].fillna('').astype(str).apply(normalize_text)
    df['refs'] = df['Description'].fillna('').astype(str).apply(extract_references)

    RETURN_KEYWORDS = [
        'neft return', 'rtgs return', 'imps return', 'upi reversal', 'upi reversal by psp',
        'ecs return', 'nach dr rtn', 'nach return', 'ach return',
        'cheque return', 'cheque bounce', 'chq ret', 'unpaid', 'return to originator', 'rtn'
    ]

    # instrument-specific windows (days)
    WINDOWS = {'neft': 3, 'rtgs': 3, 'imps': 3, 'upi': 3, 'ecs': 7, 'nach': 7, 'cheque': 14}

    # helper to test if narration indicates return and instrument
    def narr_indicates_return(narr):
        n = narr
        return any(k in n for k in RETURN_KEYWORDS)

    # matches list
    matches = []

    # precompute by amount mapping for quick lookup
    by_amount = defaultdict(list)
    for i, r in df.iterrows():
        rec = r.to_dict()
        rec['index'] = i
        by_amount[round(rec.get('amount', 0), 2)].append(rec)

    # tolerance function
    def amount_matches(a, b):
        # ±1% or ±10 rupees
        tol = max(abs(b) * 0.01, 10.0)
        return abs(a - b) <= tol

    for i, r in df.iterrows():
        n = r.get('narr_norm', '')
        val = r.get('amount', 0)
        if not narr_indicates_return(n):
            continue
        # look for potential originals of opposite direction
        cand_list = by_amount.get(round(val, 2), [])
        best = None
        best_score = None
        for cand in cand_list:
            if cand['index'] == i:
                continue
            # ensure direction matches rules: returned can be credit matching prior debit or debit matching prior credit
            dir_match = (r['type'] == 'credit' and cand['type'] == 'debit') or (r['type'] == 'debit' and cand['type'] == 'credit')
            if not dir_match:
                continue
            # date delta
            if pd.isna(cand['Txn Date']) or pd.isna(r['Txn Date']):
                days = 0
            else:
                days = abs((r['Txn Date'] - cand['Txn Date']).days)
            # approximate instrument window: use max window for now
            if days > max(WINDOWS.values()):
                continue
            # prefer same reference if present
            common_ref = set(cand.get('refs', [])) & set(r.get('refs', []))
            score = 0
            if common_ref:
                score -= 100
            # smaller date delta preferred
            score += days
            if best_score is None or score < best_score:
                best = cand
                best_score = score

        if best is not None:
            matches.append({'returned_idx': i, 'orig_idx': best['index'], 'amount': val})

    # attach flag
    df['is_returned'] = False
    df['returned_matches'] = [[] for _ in range(len(df))]
    for m in matches:
        df.at[m['returned_idx'], 'is_returned'] = True
        df.at[m['returned_idx'], 'returned_matches'] = df.at[m['returned_idx'], 'returned_matches'] + [m]
        df.at[m['orig_idx'], 'returned_matches'] = df.at[m['orig_idx'], 'returned_matches'] + [m]

    return df[['Txn Date', 'Description', 'type', 'amount', 'is_returned', 'returned_matches']]

def detect_cash_withdrawals(df: pd.DataFrame):
    """Detect cash withdrawals based on narration keywords and debit direction."""
    df = df.copy()
    # ensure date/type/amount columns exist
    df['Txn Date'] = pd.to_datetime(df.get('Txn Date'), dayfirst=True, errors='coerce')
    # Ensure type and amount columns exist FIRST
    if 'type' not in df.columns or 'amount' not in df.columns:
        df['type'], df['amount'] = zip(*df.apply(classify_txn, axis=1))
    df['narration'] = df['Description'].fillna('').astype(str)
    df['narr_norm'] = df['narration'].apply(normalize_text)

    CASH_KEYWORDS = [
        'atm wdl', 'atm withdrawal', 'cash wdl', 'cash withdrawal', 'cash paid', 'self chq', 'self cheque',
        'chq encash', 'teller wdl', 'cash wd branch', 'cardless cash', 'imt cash', 'aeps cash', 'microatm cash',
        'upi cash wd', 'upi atm', 'cash at pos', 'cashback pos', 'cash advance'
    ]

    df['is_cash_withdrawal'] = False
    df['cash_matches'] = [[] for _ in range(len(df))]

    for i, r in df.iterrows():
        if r.get('type') != 'debit':
            continue
        n = r.get('narr_norm', '')
        # must contain at least one of the cash keywords
        if not any(k in n for k in CASH_KEYWORDS):
            continue
        # exclude obvious payments
        if any(x in n for x in ['pos purchase', 'online', 'upi to', 'neft', 'merchant', 'ecom']):
            continue
        # check for reversal on same/next day credit
        dt = r.get('Txn Date')
        found_reversal = False
        for j, r2 in df.iterrows():
            if r2.get('type') == 'credit' and abs((r2.get('Txn Date') - dt).days) <= 1 if pd.notna(dt) and pd.notna(r2.get('Txn Date')) else False:
                if any(x in normalize_text(r2.get('Description','')) for x in ['reversal', 'failed']):
                    found_reversal = True
                    break
        if found_reversal:
            # don't mark as withdrawal
            continue
        # mark
        df.at[i, 'is_cash_withdrawal'] = True
        df.at[i, 'cash_matches'] = df.at[i, 'cash_matches'] + [{'index': i}]

    return df[['Txn Date', 'Description', 'type', 'amount', 'is_cash_withdrawal', 'cash_matches']]

def detect_bank_charges(df: pd.DataFrame, max_amount=1000.0):
    """Detect bank charges based on narration keywords and small debit amounts."""
    df = df.copy()
    # ensure basic columns
    df['Txn Date'] = pd.to_datetime(df.get('Txn Date'), dayfirst=True, errors='coerce')
    # Ensure type and amount columns exist FIRST
    if 'type' not in df.columns or 'amount' not in df.columns:
        df['type'], df['amount'] = zip(*df.apply(classify_txn, axis=1))
    df['narration'] = df['Description'].fillna('').astype(str)
    df['narr_norm'] = df['narration'].apply(normalize_text)

    CHARGE_KEYWORDS = [
        'chg', 'charges', 'fee', 'fees', 'penalty', 'service chg', 'service fee', 'processing chg',
        'bank chg', 'sms chg', 'atm chg', 'imps chg', 'neft chg', 'rtgs chg', 'gst chg', 'interest chg',
        'convenience fee', 'pos chg', 'cheque book chg', 'statement chg', 'bal enq chg', 'min bal penalty',
        'penal chg', 'debit card annual fee', 'locker rent', 'late fee', 'demand draft chg', 'forex markup',
        'cash handling chg', 'return chg', 'int chg'
    ]

    df['is_bank_charge'] = False
    df['bank_charge_matches'] = [[] for _ in range(len(df))]

    for i, r in df.iterrows():
        if r.get('type') != 'debit':
            continue
        n = r.get('narr_norm','')
        if any(x in n for x in ['charge back', 'refund', 'reversal']):
            continue
        if not any(k in n for k in CHARGE_KEYWORDS):
            continue
        # amount filter
        amt = abs(r.get('amount', 0))
        if amt > max_amount and 'debit card annual fee' not in n and 'locker rent' not in n:
            # allow large known annual fees
            continue
        # look for nearby transaction to link (same or previous day)
        linked = None
        dt = r.get('Txn Date')
        if pd.notna(dt):
            for j, r2 in df.iterrows():
                if j == i:
                    continue
                if abs((r2.get('Txn Date') - dt).days) <= 1 if pd.notna(r2.get('Txn Date')) else False:
                    linked = j
                    break
        # mark
        df.at[i, 'is_bank_charge'] = True
        df.at[i, 'bank_charge_matches'] = df.at[i, 'bank_charge_matches'] + ([{'linked_idx': linked}] if linked is not None else [])

    return df[['Txn Date', 'Description', 'type', 'amount', 'is_bank_charge', 'bank_charge_matches']]

def detect_commissions(df: pd.DataFrame, max_amount=5000.0):
    """Detect commission/brokerage debits and return serializable matches."""
    df = df.copy()
    # ensure columns
    df['Txn Date'] = pd.to_datetime(df.get('Txn Date'), dayfirst=True, errors='coerce')
    # Ensure type and amount columns exist FIRST
    if 'type' not in df.columns or 'amount' not in df.columns:
        df['type'], df['amount'] = zip(*df.apply(classify_txn, axis=1))
    df['narration'] = df['Description'].fillna('').astype(str)
    df['narr_norm'] = df['narration'].apply(normalize_text)

    COMM_KW = [
        'commission', 'brokerage', 'agent comm', 'bank comm', 'collection comm', 'transfer comm',
        'processing comm', 'service comm', 'cash handling comm', 'exchange comm', 'remuneration', 'upfront comm',
        'referral comm ded', 'amc comm', 'sub-broker comm'
    ]

    df['is_commission_paid'] = False
    df['commission_matches'] = [[] for _ in range(len(df))]

    for i, r in df.iterrows():
        if r.get('type') != 'debit':
            continue
        n = r.get('narr_norm', '')
        if any(x in n for x in ['commission received', 'incentive', 'refund', 'reversal']):
            continue
        if not any(k in n for k in COMM_KW):
            continue
        amt = abs(r.get('amount', 0))
        if amt > max_amount and 'annual' not in n:
            continue
        # try link GST/TDS on same date -> return row indices
        linked = []
        dt = r.get('Txn Date')
        if pd.notna(dt):
            for j, r2 in df.iterrows():
                if j == i:
                    continue
                if pd.notna(r2.get('Txn Date')) and (r2.get('Txn Date') - dt).days == 0:
                    n2 = r2.get('narr_norm', '')
                    if any(x in n2 for x in ['gst', 'tds']):
                        linked.append(int(j))
        # mark
        df.at[i, 'is_commission_paid'] = True
        df.at[i, 'commission_matches'] = df.at[i, 'commission_matches'] + (linked if linked else [])

    return df[['Txn Date', 'Description', 'type', 'amount', 'is_commission_paid', 'commission_matches']]

def detect_statutory_payments(df: pd.DataFrame):
    """Detect statutory payments (GST, TDS, PF, ESI, Income Tax, etc.)."""
    df = df.copy()
    df['Txn Date'] = pd.to_datetime(df.get('Txn Date'), dayfirst=True, errors='coerce')
    # Ensure type and amount columns exist FIRST
    if 'type' not in df.columns or 'amount' not in df.columns:
        df['type'], df['amount'] = zip(*df.apply(classify_txn, axis=1))
    df['narration'] = df['Description'].fillna('').astype(str)
    df['narr_norm'] = df['narration'].apply(normalize_text)

    STAT_KW = [
        'gst', 'igst', 'cgst', 'sgst', 'tds', 'tcs', 'income tax', 'advance tax', 'corporate tax',
        'tax payment', 'tax paid', 'cbdt', 'cbic', 'epf', 'pf payment', 'esi', 'epfo', 'esic',
        'profession tax', 'mca fee', 'roc fee', 'stamp duty', 'challan', 'traces', 'gst pmt', 'payment to govt', 'govt tax', 'gstin'
    ]

    df['is_statutory'] = False
    df['statutory_matches'] = [[] for _ in range(len(df))]

    for i, r in df.iterrows():
        if r.get('type') != 'debit':
            continue
        n = r.get('narr_norm', '')
        # ignore bank charges mentioning GST
        if 'gst on' in n or 'gst on atm' in n:
            continue
        if not any(k in n for k in STAT_KW):
            continue
        # exclude reversals/credits
        if any(x in n for x in ['refund', 'reversal', 'credited']):
            continue
        df.at[i, 'is_statutory'] = True
        df.at[i, 'statutory_matches'] = df.at[i, 'statutory_matches'] + [int(i)]

    return df[['Txn Date', 'Description', 'type', 'amount', 'is_statutory', 'statutory_matches']]

def detect_recurring(df: pd.DataFrame):
    """
    Detect recurring transactions with improved error handling
    """
    print("=== STARTING DETECT_RECURRING ===")
    
    try:
        # Standardize columns first
        if 'txn_date' not in df.columns and 'Txn Date' in df.columns:
            df_std = df.copy()
        else:
            df_std = standardize_columns(df)
        
        # Ensure date column is datetime
        if 'Txn Date' in df_std.columns:
            df_std['Txn Date'] = pd.to_datetime(df_std['Txn Date'], dayfirst=True, errors='coerce')
        elif 'txn_date' in df_std.columns:
            df_std['txn_date'] = pd.to_datetime(df_std['txn_date'], dayfirst=True, errors='coerce')
        
        validate_dataframe(df_std)
        
        df_std['is_recurrent'] = False
        df_std['recurrence_pattern'] = None
        df_std['recurrence_frequency'] = None
        
        # Get description column
        desc_col = 'Description' if 'Description' in df_std.columns else 'description' if 'description' in df_std.columns else None
        
        if desc_col:
            # Group by normalized description to find recurring patterns
            df_std['desc_norm'] = df_std[desc_col].fillna('').astype(str).apply(normalize_text)
            
            # Count occurrences of each normalized description
            desc_counts = df_std['desc_norm'].value_counts()
            
            # Mark transactions with description appearing 3+ times as recurring
            recurring_descs = desc_counts[desc_counts >= 3].index.tolist()
            
            for idx, row in df_std.iterrows():
                if row['desc_norm'] in recurring_descs:
                    df_std.at[idx, 'is_recurrent'] = True
                    # Count frequency
                    count = desc_counts[row['desc_norm']]
                    if count >= 12:
                        df_std.at[idx, 'recurrence_frequency'] = 'Monthly or higher'
                    elif count >= 4:
                        df_std.at[idx, 'recurrence_frequency'] = 'Quarterly or higher'
                    else:
                        df_std.at[idx, 'recurrence_frequency'] = 'Occasional'
                    df_std.at[idx, 'recurrence_pattern'] = f'Occurs {count} times'
        
        result_columns = ['Txn Date', 'Description', 'type', 'amount', 'is_recurrent', 'recurrence_pattern', 'recurrence_frequency']
        available_columns = [col for col in result_columns if col in df_std.columns]
        
        print(f"✅ Recurring detection complete. Returning columns: {available_columns}")
        return df_std[available_columns]
        
    except Exception as e:
        print(f"❌ Error in detect_recurring: {str(e)}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def run_analysis(df: pd.DataFrame):
    """Run all analysis functions on the dataframe."""
    # run all detectors and merge columns
    out_refunds = detect_refunds(df)
    out_returns = detect_returns(df)
    out_cash = detect_cash_withdrawals(df)
    out_charges = detect_bank_charges(df)
    out_comm = detect_commissions(df)
    out_stat = detect_statutory_payments(df)

    # start from refunds output (keeps columns used earlier)
    out = out_refunds.copy()
    # merge columns from returns and cash outputs if present
    # detect_returns uses 'matched_returns' column name; map it to 'returned_matches'
    if 'is_returned' in out_returns.columns:
        out['is_returned'] = out_returns['is_returned']
    else:
        out['is_returned'] = False

    if 'matched_returns' in out_returns.columns:
        out['returned_matches'] = out_returns['matched_returns']
    else:
        out['returned_matches'] = [[] for _ in range(len(out))]
    
    for col in ['is_cash_withdrawal', 'cash_matches']:
        if col in out_cash.columns:
            out[col] = out_cash[col]
        else:
            out[col] = False if col == 'is_cash_withdrawal' else [[] for _ in range(len(out))]
    
    # bank charges
    for col in ['is_bank_charge', 'bank_charge_matches']:
        if col in out_charges.columns:
            out[col] = out_charges[col]
        else:
            out[col] = False if col == 'is_bank_charge' else [[] for _ in range(len(out))]
    
    # commissions (final columns: is_commission_paid, commission_matches)
    for col in ['is_commission_paid', 'commission_matches']:
        if col in out_comm.columns:
            out[col] = out_comm[col]
        else:
            out[col] = False if col == 'is_commission_paid' else [[] for _ in range(len(out))]
    
    # statutory payments
    for col in ['is_statutory', 'statutory_matches']:
        if col in out_stat.columns:
            out[col] = out_stat[col]
        else:
            out[col] = False if col == 'is_statutory' else [[] for _ in range(len(out))]

    # stringify complex/list columns to make Streamlit/Arrow happy
    complex_cols = ['matched', 'returned_matches', 'cash_matches', 'bank_charge_matches', 
                   'commission_matches', 'statutory_matches']

    def _safe_serialize(v):
        # None or NaN -> empty string
        try:
            if v is None:
                return ''
            if isinstance(v, str):
                return v
            # pandas NA
            if isinstance(v, float) and pd.isna(v):
                return ''
            # lists and dicts are JSON serializable
            if isinstance(v, (list, dict)):
                return json.dumps(v)
            # numpy arrays or pandas Series
            if isinstance(v, (np.ndarray,)) or hasattr(v, 'tolist'):
                try:
                    return json.dumps(v.tolist())
                except Exception:
                    return json.dumps(list(v))
            # fallback: try json dump, else str
            try:
                return json.dumps(v)
            except Exception:
                return str(v)
        except Exception:
            return str(v)

    for c in complex_cols:
        if c in out.columns:
            out[c] = out[c].apply(_safe_serialize)

    return out

# -------------------------------------------------------------------
# Enhanced UI Components
# -------------------------------------------------------------------
def create_step_indicator(current_step):
    """Create a step indicator showing progress through the workflow."""
    steps = [
        {"label": "Upload Files", "key": "upload"},
        {"label": "Preview Data", "key": "preview"},
        {"label": "Merge Files", "key": "merge"},
        {"label": "Analysis", "key": "analysis"}
    ]
    
    # Find current step index
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
    
    # Summary statistics
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
    
    # Display relevant entries for each category
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
    

    
    # Full results table (collapsed by default)
    with st.expander("📊 View Full Analysis Table"):
        st.dataframe(result, use_container_width=True)

# -------------------------------------------------------------------
# Main Application with Linear Flow
# -------------------------------------------------------------------
def main():
    # Initialize session state for workflow
    if 'current_step' not in st.session_state:
        st.session_state.current_step = "upload"
    
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = {}
    
    if 'files_to_merge' not in st.session_state:
        st.session_state.files_to_merge = []
    
    if 'separate_files' not in st.session_state:
        st.session_state.separate_files = []
    
    if 'merged_data' not in st.session_state:
        st.session_state.merged_data = None
    
    # Show step indicator
    create_step_indicator(st.session_state.current_step)
    
    # Route to appropriate step
    if st.session_state.current_step == "upload":
        show_upload_step()
    elif st.session_state.current_step == "preview":
        show_preview_step()
    elif st.session_state.current_step == "merge":
        show_merge_step()
    elif st.session_state.current_step == "analysis":
        show_analysis_step()

def show_upload_step():
    st.markdown("<h1 class='main-header'>📤 Upload Bank Statements</h1>", unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    <div class='card'>
        <h2>Upload Your Bank Statement Files</h2>
        <p>Upload PDF, Excel, or CSV files containing your bank statements. 
        The system will automatically process PDF files and prepare all files for merging.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # File upload section
    uploaded_files = st.file_uploader(
        "Choose bank statement files", 
        type=["pdf", "xlsx", "xls", "csv"], 
        accept_multiple_files=True,
        help="Upload PDF, Excel, or CSV files"
    )
    
    if uploaded_files:
        # Store uploaded files
        st.session_state.uploaded_files = uploaded_files
        
        # Show uploaded files
        st.markdown("<h2 class='sub-header'>📁 Uploaded Files</h2>", unsafe_allow_html=True)
        
        pdf_files = [f for f in uploaded_files if f.name.lower().endswith('.pdf')]
        excel_files = [f for f in uploaded_files if f.name.lower().endswith(('.xlsx', '.xls'))]
        csv_files = [f for f in uploaded_files if f.name.lower().endswith('.csv')]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            create_metric_card(len(pdf_files), "PDF Files", "📄")
        with col2:
            create_metric_card(len(excel_files), "Excel Files", "📊")
        with col3:
            create_metric_card(len(csv_files), "CSV Files", "📋")
        
        # File list
        with st.expander("View File Details"):
            for file in uploaded_files:
                file_type = "PDF" if file.name.lower().endswith('.pdf') else "Excel" if file.name.lower().endswith(('.xlsx', '.xls')) else "CSV"
                st.write(f"• **{file.name}** ({file_type}, {file.size // 1024} KB)")
        
        # Process button
        if styled_button("🚀 Process Files & Continue", "process_files", "primary"):
            process_uploaded_files(uploaded_files)

def process_uploaded_files(uploaded_files):
    """Process all uploaded files and extract data from PDFs."""
    st.session_state.processed_files = {}
    
    # Process PDF files
    pdf_files = [f for f in uploaded_files if f.name.lower().endswith('.pdf')]
    
    if pdf_files:
        with st.spinner("📄 Extracting data from PDF files..."):
            for file in pdf_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(file.read())
                    tmp_path = tmp.name
                
                # Extract from PDF
                camelot_pages = extract_camelot_pagewise(tmp_path, "lattice")
                
                if not camelot_pages or all(df.empty for df in camelot_pages):
                    camelot_pages = extract_pdfplumber_pagewise(tmp_path)
                
                # Try AI extraction if still no data
                if (not camelot_pages or all(df.empty for df in camelot_pages)) and client:
                    st.info(f"🤖 Trying AI extraction for {file.name}...")
                    camelot_pages = extract_openai_pagewise(tmp_path)
                
                # Combine all pages
                if camelot_pages and any(not df.empty for df in camelot_pages):
                    combined_df = pd.concat([df for df in camelot_pages if not df.empty], ignore_index=True)
                    combined_df = remove_duplicate_headers(combined_df)
                    st.session_state.processed_files[file.name] = combined_df
                else:
                    st.warning(f"⚠️ Could not extract data from {file.name}")
    
    # Process Excel/CSV files
    other_files = [f for f in uploaded_files if not f.name.lower().endswith('.pdf')]
    
    for file in other_files:
        df = read_file(file)
        if df is not None and not df.empty:
            st.session_state.processed_files[file.name] = df
        else:
            st.warning(f"⚠️ Could not read {file.name}")
    
    # Move to preview step
    st.session_state.current_step = "preview"
    st.rerun()

def show_preview_step():
    st.markdown("<h1 class='main-header'>👀 Data Preview</h1>", unsafe_allow_html=True)
    
    if not st.session_state.processed_files:
        st.error("No files processed. Please go back and upload files.")
        if st.button("⬅️ Back to Upload"):
            st.session_state.current_step = "upload"
            st.rerun()
        return
    
    # Show file statistics
    st.markdown("<h2 class='sub-header'>📊 File Overview</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        create_metric_card(len(st.session_state.processed_files), "Processed Files", "📁")
    with col2:
        total_rows = sum(len(df) for df in st.session_state.processed_files.values())
        create_metric_card(total_rows, "Total Rows", "📊")
    with col3:
        avg_rows = total_rows // len(st.session_state.processed_files) if st.session_state.processed_files else 0
        create_metric_card(avg_rows, "Avg Rows/File", "📈")
    
    # Show preview for each file
    st.markdown("<h2 class='sub-header'>📋 File Previews</h2>", unsafe_allow_html=True)
    
    for file_name, df in st.session_state.processed_files.items():
        with st.expander(f"📄 {file_name} ({len(df)} rows)"):
            st.dataframe(df.head(10), use_container_width=True)
            st.write(f"**Columns:** {', '.join(df.columns.tolist())}")
    
    # File selection for merging
    st.markdown("<h2 class='sub-header'>🔗 Select Files to Merge</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
        <h3>💡 Merge Selection</h3>
        <p>Select which files you want to merge together. Unselected files will be analyzed separately.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # File selection checkboxes
    selected_files = []
    file_options = list(st.session_state.processed_files.keys())
    
    cols = st.columns(2)
    for i, file_name in enumerate(file_options):
        with cols[i % 2]:
            if st.checkbox(f"**{file_name}**", value=True, key=f"merge_{file_name}"):
                selected_files.append(file_name)
    
    # Store selection and proceed
    if styled_button("🔄 Configure Merging & Continue", "configure_merge", "primary"):
        st.session_state.files_to_merge = selected_files
        st.session_state.separate_files = [f for f in file_options if f not in selected_files]
        st.session_state.current_step = "merge"
        st.rerun()
    
    # Back button
    if st.button("⬅️ Back to Upload"):
        st.session_state.current_step = "upload"
        st.rerun()

def show_merge_step():
    st.markdown("<h1 class='main-header'>🔗 File Merging Configuration</h1>", unsafe_allow_html=True)
    
    # Show merge configuration
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h3>📂 Files to Merge</h3>", unsafe_allow_html=True)
        if st.session_state.files_to_merge:
            for file_name in st.session_state.files_to_merge:
                df = st.session_state.processed_files[file_name]
                st.write(f"• **{file_name}** ({len(df)} rows)")
            
            if len(st.session_state.files_to_merge) > 1:
                if styled_button("🔄 Merge Selected Files", "merge_selected", "primary"):
                    merge_selected_files()
            else:
                st.info("Select at least 2 files to merge")
        else:
            st.info("No files selected for merging")
    
    with col2:
        st.markdown("<h3>📄 Separate Files</h3>", unsafe_allow_html=True)
        if st.session_state.separate_files:
            for file_name in st.session_state.separate_files:
                df = st.session_state.processed_files[file_name]
                st.write(f"• **{file_name}** ({len(df)} rows)")
        else:
            st.info("All files will be merged")
    
    # Show previews if available
    if st.session_state.merged_data is not None or st.session_state.separate_files:
        st.markdown("<h2 class='sub-header'>👀 Data Preview</h2>", unsafe_allow_html=True)
        
        # Show merged data preview
        if st.session_state.merged_data is not None:
            st.markdown("#### 📊 Merged Data")
            st.dataframe(st.session_state.merged_data.head(10), use_container_width=True)
            st.write(f"**Total Rows:** {len(st.session_state.merged_data)}")
        
        # Show separate files preview
        if st.session_state.separate_files:
            st.markdown("#### 📄 Separate Files")
            for file_name in st.session_state.separate_files:
                with st.expander(f"📄 {file_name}"):
                    df = st.session_state.processed_files[file_name]
                    st.dataframe(df.head(10), use_container_width=True)
    
    # Proceed to analysis
    if st.session_state.merged_data is not None or st.session_state.separate_files:
        if styled_button("🔍 Run Analysis & Continue", "run_analysis", "success"):
            st.session_state.current_step = "analysis"
            st.rerun()
    
    # Back button
    if st.button("⬅️ Back to Preview"):
        st.session_state.current_step = "preview"
        st.rerun()

def merge_selected_files():
    """Merge the selected files."""
    if len(st.session_state.files_to_merge) < 2:
        st.warning("Please select at least 2 files to merge")
        return
    
    with st.spinner("🔄 Merging files..."):
        dfs_to_merge = []
        for file_name in st.session_state.files_to_merge:
            df = st.session_state.processed_files[file_name].copy()
            if 'File Name' not in df.columns:
                df['File Name'] = file_name
            dfs_to_merge.append(df)
        
        merged_df = merge_dataframes(dfs_to_merge)
        
        if merged_df is not None:
            st.session_state.merged_data = merged_df
            st.success(f"✅ Successfully merged {len(st.session_state.files_to_merge)} files")
        else:
            st.error("❌ Failed to merge files")

def show_analysis_step():
    st.markdown("<h1 class='main-header'>🔍 Transaction Analysis</h1>", unsafe_allow_html=True)
    
    # Determine what to analyze
    datasets_to_analyze = {}
    
    if st.session_state.merged_data is not None:
        datasets_to_analyze["Merged Data"] = st.session_state.merged_data
    
    for file_name in st.session_state.separate_files:
        datasets_to_analyze[file_name] = st.session_state.processed_files[file_name]
    
    if not datasets_to_analyze:
        st.error("No data available for analysis")
        if st.button("⬅️ Back to Merge"):
            st.session_state.current_step = "merge"
            st.rerun()
        return
    
    # Show analysis targets
    st.markdown("<h2 class='sub-header'>📊 Analysis Targets</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        create_metric_card(len(datasets_to_analyze), "Datasets to Analyze", "📁")
    with col2:
        total_rows = sum(len(df) for df in datasets_to_analyze.values())
        create_metric_card(total_rows, "Total Transactions", "📊")
    
    # Run analysis
    if styled_button("🚀 Run Comprehensive Analysis", "run_comprehensive_analysis", "primary"):
        run_comprehensive_analysis_flow(datasets_to_analyze)
    
    # Back button
    if st.button("⬅️ Back to Merge"):
        st.session_state.current_step = "merge"
        st.rerun()

def run_comprehensive_analysis_flow(datasets_to_analyze):
    """Run analysis on all datasets with progress tracking."""
    analysis_results = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, (dataset_name, df) in enumerate(datasets_to_analyze.items()):
        progress = i / len(datasets_to_analyze)
        progress_bar.progress(progress)
        status_text.text(f"🔍 Analyzing {dataset_name}...")
        
        with st.spinner(f"Analyzing {dataset_name}..."):
            try:
                analysis_result = run_analysis(df)
                analysis_results[dataset_name] = analysis_result
            except Exception as e:
                st.error(f"❌ Error analyzing {dataset_name}: {str(e)}")
                analysis_results[dataset_name] = None
    
    progress_bar.progress(1.0)
    status_text.text("✅ Analysis complete!")
    
    # Display results
    st.markdown("<h2 class='sub-header'>📋 Analysis Results</h2>", unsafe_allow_html=True)
    
    for dataset_name, result in analysis_results.items():
        if result is not None:
            with st.expander(f"📊 {dataset_name} Analysis Results", expanded=True):
                display_analysis_details(result, dataset_name)
                
                # Download options
                st.markdown("#### 📥 Download Results")
                col1, col2 = st.columns(2)
                with col1:
                    csv_data = result.to_csv(index=False)
                    st.download_button(
                        label="💾 Download as CSV",
                        data=csv_data,
                        file_name=f"analysis_{dataset_name.replace(' ', '_')}.csv",
                        mime="text/csv",
                        key=f"csv_{dataset_name}"
                    )
                with col2:
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                        result.to_excel(writer, index=False, sheet_name="Analysis")
                    excel_buffer.seek(0)
                    st.download_button(
                        label="💾 Download as Excel",
                        data=excel_buffer,
                        file_name=f"analysis_{dataset_name.replace(' ', '_')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key=f"excel_{dataset_name}"
                    )
    
    # Restart workflow
    st.markdown("---")
    if styled_button("🔄 Start New Analysis", "restart_workflow", "secondary"):
        # Reset session state
        for key in ['current_step', 'uploaded_files', 'processed_files', 'files_to_merge', 'separate_files', 'merged_data']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

if __name__ == "__main__":
    main()