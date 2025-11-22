import streamlit as st
import camelot
import pdfplumber
import contextlib
import pandas as pd
import tempfile
import re, os, io, collections, base64
from dotenv import load_dotenv
import fitz  # PyMuPDF
import json
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize OpenAI client
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key:
    client = OpenAI(api_key=openai_api_key)
else:
    client = None

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
    """Extract tables from PDFs using OpenAI's GPT-4 Vision API.
    
    This method works even with missing borders and complex layouts.
    """
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
            # Prefer traditional methods first (faster). Only use OpenAI Vision as a fallback
            # when Camelot/PdfPlumber produce no table output.
            st.info("📊 Using traditional extraction methods (Camelot -> PdfPlumber) first...")
            camelot_pages = extract_camelot_pagewise(tmp_path, "lattice")

            # If Camelot didn't find anything, try PdfPlumber
            if not camelot_pages or all(df.empty for df in camelot_pages):
                camelot_pages = extract_pdfplumber_pagewise(tmp_path)
                if camelot_pages and any(not df.empty for df in camelot_pages):
                    used_method = "PdfPlumber"
                else:
                    used_method = None
            else:
                used_method = "Camelot"

            # Compute total extracted rows from the chosen traditional methods
            total_rows = 0
            try:
                if camelot_pages:
                    for df in camelot_pages:
                        # Count non-empty rows conservatively
                        try:
                            if not df.empty:
                                total_rows += len(df)
                        except Exception:
                            pass
            except Exception:
                total_rows = 0

            # If traditional methods produced no rows, and OpenAI client is configured,
            # use OpenAI Vision as a fallback (slower but can handle complex layouts).
            if (total_rows < 3):
                if not client:
                    st.warning("⚠️ No rows found and OpenAI API not configured; skipping AI fallback.")
                else:
                    st.info("🤖 Traditional extractors produced 0 rows; trying OpenAI Vision API as fallback...")
                    try:
                        openai_pages = extract_openai_pagewise(tmp_path)
                    except Exception as e:
                        st.error(f"Error while running OpenAI fallback: {e}")
                        openai_pages = []

                    if openai_pages and any(not df.empty for df in openai_pages):
                        camelot_pages = openai_pages
                        used_method = "OpenAI Vision"
                        # recompute total_rows
                        total_rows = sum(len(df) for df in camelot_pages if not df.empty)
                    else:
                        st.info("ℹ️ OpenAI fallback returned no tables or empty results.")

            # If a method produced rows, ensure used_method reflects that
            if total_rows > 0:
                # If used_method isn't already set (e.g., pdfplumber succeeded), determine it
                if used_method in (None, "None"):
                    # prefer Camelot label if Camelot produced rows
                    if camelot_pages and any((not df.empty) for df in camelot_pages):
                        # If we previously set used_method to PdfPlumber because camelot was empty, keep it
                        used_method = used_method or "Camelot"
                    else:
                        used_method = used_method or "PdfPlumber"
            else:
                # No rows found at all
                used_method = used_method or "None"

        # Show final per-file extraction summary so it's clear which method was used
        try:
            display_rows = total_rows
        except Exception:
            # fallback: count rows directly from camelot_pages
            display_rows = sum(len(df) for df in camelot_pages if not df.empty) if camelot_pages else 0

        st.markdown(f"**Extraction summary for `{file.name}` — Method: `{used_method}` | Total rows: `{display_rows}`**")

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
                st.dataframe(extracted_df, width='stretch')
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
        st.dataframe(final_summary, width='stretch')
        
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
            st.dataframe(combined_file_df, width='stretch')
            
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
