import streamlit as st
import camelot
import pdfplumber
import contextlib
import pandas as pd
import tempfile
import re, os, io, collections
from dotenv import load_dotenv
import fitz  # PyMuPDF
import json

# Optional OpenAI support (used as a last-resort extraction method)
try:
    import openai
except Exception:
    openai = None

# Load environment variables
load_dotenv()

# Configure Streamlit
st.set_page_config(page_title="Bank Statement Extractor", layout="wide")

# CSS for dark-friendly tables
_DARK_FRIENDLY_CSS = """
<style>
.stApp table td, .stApp table th,
div[data-testid="stDataFrame"] table td, div[data-testid="stDataFrame"] table th,
.stDataFrame table td, .stDataFrame table th {
    color: #0b1220 !important;
}
button[data-baseweb="button"], .stButton>button, .stDownloadButton>button {
    color: inherit !important;
}
.stDataFrame table, table.dataframe {
    border-color: rgba(15,18,32,0.06) !important;
}
</style>
"""

try:
    st.markdown(_DARK_FRIENDLY_CSS, unsafe_allow_html=True)
except Exception:
    pass


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
    """Extract tables page-wise using Camelot.
    
    IMPORTANT: Headers are only on PAGE 1. All subsequent pages use the same headers.
    This ensures exact column alignment throughout the document.
    """
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


def validate_transaction_row(row_dict):
    """Validate a single transaction row."""
    errors = []
    
    # Simple validation - just check if row has some data
    has_data = any(str(v).strip() not in ('', 'nan', 'None') for v in row_dict.values())
    
    return {
        "valid": has_data,
        "errors": errors,
    }


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


def extract_with_openai(path, model="gpt-3.5-turbo-16k"):
    """Try to extract tabular transaction data from a PDF using OpenAI as a last resort.

    This function will extract page text (pdfplumber -> fitz fallback), then ask OpenAI
    to parse the textual content into a JSON array of transaction rows. Each row should
    contain keys such as Date, Description, Withdrawal, Deposit, Balance (best effort).

    Returns a list of DataFrames, one per page (empty DataFrame if nothing parsed).
    """
    dfs = []

    # Ensure openai is available and API key is set
    if openai is None:
        return dfs

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        return dfs

    openai.api_key = openai_api_key

    # Extract text per page
    page_texts = []
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    txt = page.extract_text() or ""
                    page_texts.append(txt)
    except Exception:
        # fallback to fitz
        try:
            doc = fitz.open(path)
            for p in doc:
                page_texts.append(p.get_text("text") or "")
        except Exception:
            return dfs

    # For each page, ask the model to extract transactions
    for page_idx, text in enumerate(page_texts, start=1):
        text = (text or "").strip()
        if not text:
            dfs.append(pd.DataFrame())
            continue

        system_msg = (
            "You are a helpful assistant that extracts bank transaction tables from raw page text.\n"
            "Return only a JSON array of objects (no surrounding explanation). Each object should have keys:"
            " Date, Description, Withdrawal, Deposit, Balance. If a field is not present, use empty string."
        )

        user_msg = (
            "Here is the raw text from one PDF page (do not invent values).\n"
            "Extract each transaction you can find and return a JSON array of objects.\n\n" +
            "Page text:\n" + text[:30000]
        )

        try:
            resp = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=6000,
            )

            content = resp.choices[0].message.content.strip()

            # Try to extract JSON array from the content
            json_text = None
            # Find the first '[' and last ']' to attempt to slice out JSON array
            start = content.find('[')
            end = content.rfind(']')
            if start != -1 and end != -1 and end > start:
                json_text = content[start:end+1]

            parsed = []
            if json_text:
                try:
                    parsed = json.loads(json_text)
                except Exception:
                    parsed = []

            # If parsed is not a list/dict array, try to parse as line-delimited CSV-like text
            if not isinstance(parsed, list):
                parsed = []

            if parsed:
                # Normalize list of dicts to DataFrame
                try:
                    df = pd.DataFrame(parsed)
                except Exception:
                    df = pd.DataFrame()
            else:
                df = pd.DataFrame()

            dfs.append(df)
        except Exception:
            dfs.append(pd.DataFrame())

    return dfs


# -------------------- Main --------------------
st.title("📊 Bank Statement Extractor")

uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    overall_summary = []
    all_extracted_data = []
    file_data_map = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()

    for file_idx, file in enumerate(uploaded_files):
        progress = (file_idx + 0.1) / len(uploaded_files)
        progress_bar.progress(min(progress, 0.99))
        status_text.text(f"Processing file {file_idx + 1} of {len(uploaded_files)}: {file.name}")
        
        st.markdown(f"## 📘 File: `{file.name}`")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        with st.spinner(f"Extracting data from {file.name} ..."):
            # Try multiple extraction strategies for robustness
            # Strategy 1: Camelot (Lattice flavor) - grid-based, works with bordered tables
            camelot_pages = extract_camelot_pagewise(tmp_path, "lattice")
            
            # If lattice returns 0 rows, it means NO HORIZONTAL LINES (borderless)
            # Go directly to PdfPlumber (best for borderless tables)
            if not camelot_pages or all(df.empty for df in camelot_pages):
                # Skip Stream, go straight to PdfPlumber for borderless tables
                camelot_pages = extract_pdfplumber_pagewise(tmp_path)
                used_method = "PdfPlumber"

                # If PdfPlumber also returns nothing, try using OpenAI as a last resort
                if not camelot_pages or all(df.empty for df in camelot_pages):
                    # Only attempt if openai module is available and API key set
                    if openai is not None and os.environ.get("OPENAI_API_KEY"):
                        with st.spinner("Falling back to OpenAI for extraction..."):
                            openai_pages = extract_with_openai(tmp_path)
                        # If OpenAI returned anything useful, use it
                        if openai_pages and not all(df.empty for df in openai_pages):
                            camelot_pages = openai_pages
                            used_method = "OpenAI"
                            st.info("Using OpenAI-powered extraction as a fallback. Please verify results.")
                        else:
                            # Leave used_method as PdfPlumber but inform user
                            st.warning("Could not extract table data automatically (Camelot + PdfPlumber + OpenAI returned no rows).\n"
                                        "Consider reviewing the PDF or enabling a different extraction strategy.")
                    else:
                        st.info("PdfPlumber produced no rows and OpenAI fallback is not available (no API key or missing package).\n"
                                "Set OPENAI_API_KEY environment variable to enable OpenAI fallback.")
            else:
                used_method = "Camelot"

        total_pages = len(camelot_pages)
        results = []

        file_extracted_data = []
        
        st.markdown("### 📊 Extracted Data")
        
        for page_num, extracted_df in enumerate(camelot_pages, start=1):
            # Remove duplicate header rows
            if not extracted_df.empty:
                extracted_df = remove_duplicate_headers(extracted_df)
            
            # Validate and refine
            if not extracted_df.empty:
                refinement_result = refine_and_validate_data(extracted_df)
                extracted_df = refinement_result["refined_df"]
                validation = refinement_result["validation_summary"]
            else:
                validation = None

            # Show extracted data
            if not extracted_df.empty:
                st.markdown(f"**Page {page_num}** ({used_method} - {len(extracted_df)} rows)")
                st.dataframe(extracted_df, use_container_width=True)
            else:
                st.info("ℹ️ No data extracted from this page")

            # Collect extracted data
            if not extracted_df.empty:
                all_extracted_data.append(extracted_df)
                file_extracted_data.append(extracted_df)

            results.append({
                "File": file.name,
                "Page": page_num,
                "Method": used_method,
                "Rows": len(extracted_df),
                "Valid": validation["valid_rows"] if validation else 0,
                "Invalid": validation["invalid_rows"] if validation else 0,
            })
        
        # Store file-wise extracted data
        if file_extracted_data:
            file_data_map[file.name] = file_extracted_data

        overall_summary.append(results)

    # Update progress to complete
    progress_bar.progress(1.0)
    status_text.text("✅ Processing complete!")
    
    # Combine all results
    if overall_summary:
        final_summary = pd.DataFrame([item for sublist in overall_summary for item in sublist])
    else:
        final_summary = pd.DataFrame()
    
    if not final_summary.empty:
        st.markdown("## 🧾 Combined Summary — All Files")
        st.dataframe(final_summary, use_container_width=True)
        
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            csv = final_summary.to_csv(index=False)
            st.download_button(label="📥 Download Summary (CSV)", data=csv, file_name="extraction_summary.csv", mime="text/csv")
        with col_dl2:
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                final_summary.to_excel(writer, index=False)
            excel_buffer.seek(0)
            st.download_button(label="📥 Download Summary (Excel)", data=excel_buffer, file_name="extraction_summary.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # Full Extracted Data Download (Per-File)
    if file_data_map:
        st.markdown("---")
        st.markdown("## 🧩 Full Extracted Data — File Wise")

        for file_name, file_dfs in file_data_map.items():
            st.markdown(f"### {file_name}")
            combined_file_df = pd.concat(file_dfs, ignore_index=True)
            st.dataframe(combined_file_df, use_container_width=True)
            
            # Download buttons
            col1, col2 = st.columns(2)
            with col1:
                csv = combined_file_df.to_csv(index=False)
                st.download_button(
                    label=f"📥 CSV - {file_name}",
                    data=csv,
                    file_name=f"{file_name.replace('.pdf', '')}_extracted.csv",
                    mime="text/csv",
                    key=f"csv_{file_name}"
                )
            with col2:
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    combined_file_df.to_excel(writer, index=False)
                excel_buffer.seek(0)
                st.download_button(
                    label=f"📥 Excel - {file_name}",
                    data=excel_buffer,
                    file_name=f"{file_name.replace('.pdf', '')}_extracted.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"excel_{file_name}"
                )

else:
    st.markdown("---")
    st.markdown("""
    ### 👋 Welcome to Bank Statement Extractor
    
    **Quick Start:**
    1. 📤 Upload one or more bank statement PDF files using the uploader above
    2. ⏳ Wait for automatic extraction
    3. 📊 Review extracted tables
    4. 📥 Download results in CSV or Excel format
    
    **Features:**
    - 🐫 **Camelot (Lattice)**: Grid-based table extraction
    - 🌊 **Camelot (Stream)**: Text-line based extraction
    - 🧾 **PdfPlumber**: Layout-based extraction (excellent for borderless tables)
    - 💾 **Multiple Formats**: Export as CSV or Excel
    - 🔄 **Automatic Fallback**: Tries multiple methods automatically
    
    **Best For:**
    - ✅ Bank statements with bordered tables
    - ✅ Borderless/text-based statement layouts
    - ✅ Multi-page statements
    - ✅ Mixed table formats
    """)
    
    st.info("📤 Upload PDF files to begin extraction →")
