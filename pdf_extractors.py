import streamlit as st
import pandas as pd
import os
import re
import io
import contextlib
import tempfile
import json
import camelot.io as camelot
import pdfplumber
from openai import OpenAI
from utils import split_merged_decimals, remove_duplicate_headers, pdf_page_to_image_base64, refine_and_validate_data

# Initialize OpenAI client
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key:
    client = OpenAI(api_key=openai_api_key)
else:
    client = None

# -------------------------------------------------------------------
# PDF Extraction Functions
# -------------------------------------------------------------------
def extract_pdfplumber_pagewise(path):
    """Extract tables from PDFs using pdfplumber - excellent for borderless tables."""
    all_pages = []
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            with pdfplumber.open(path) as pdf:
                for page_idx, page in enumerate(pdf.pages, start=1):
                    try:
                        tables = page.extract_tables()
                        
                        if tables:
                            for table in tables:
                                if table and len(table) > 0:
                                    df = pd.DataFrame(table[1:], columns=table[0])
                                    df = df.astype(str)
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
    extracted_headers = None
    
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            with pdfplumber.open(path) as pdf:
                total_pages = len(pdf.pages)
        
        for i in range(1, total_pages + 1):
            try:
                with contextlib.redirect_stderr(io.StringIO()):
                    tables = camelot.read_pdf(path, pages=str(i), flavor=flavor)
                if tables and len(tables) > 0:
                    df = pd.concat([t.df for t in tables], ignore_index=True)
                else:
                    df = pd.DataFrame()
            except Exception:
                df = pd.DataFrame()
            
            if i == 1 and not df.empty:
                extracted_headers = [re.sub(r"\s+", " ", str(c)).strip() for c in df.iloc[0].values]
                df = df.iloc[1:].reset_index(drop=True)
                df.columns = extracted_headers
            elif i > 1 and not df.empty and extracted_headers is not None:
                df = df.astype(str)
                if len(df.columns) == len(extracted_headers):
                    df.columns = extracted_headers
                else:
                    df.columns = extracted_headers[:len(df.columns)]
            elif not df.empty:
                df = df.astype(str)
            
            if not df.empty:
                df = split_merged_decimals(df)
            
            all_pages.append(df)
    except Exception as e:
        pass
    
    return all_pages

def extract_openai_pagewise(pdf_path):
    """Extract tables from PDFs using OpenAI's GPT-4 Vision API."""
    if not client:
        st.error("❌ OpenAI API key not configured. Set OPENAI_API_KEY environment variable.")
        return []
    
    all_pages = []
    extracted_headers = None
    
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
        
        for page_num in range(1, total_pages + 1):
            try:
                base64_image = pdf_page_to_image_base64(pdf_path, page_num)
                if not base64_image:
                    all_pages.append(pd.DataFrame())
                    continue
                
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
                
                try:
                    response_text = response.choices[0].message.content
                    json_match = re.search(r'\{[\s\S]*\}', response_text)
                    if json_match:
                        table_data = json.loads(json_match.group())
                        headers = table_data.get("headers", [])
                        rows = table_data.get("rows", [])
                        
                        if headers and rows:
                            df = pd.DataFrame(rows, columns=headers)
                            df = df.astype(str)
                            
                            if page_num == 1:
                                extracted_headers = [re.sub(r"\s+", " ", str(c)).strip() for c in headers]
                            
                            if page_num > 1 and extracted_headers and len(df.columns) == len(extracted_headers):
                                df.columns = extracted_headers
                            
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

def extract_pdf_data(file_path):
    """Extract data from PDF using multiple methods."""
    camelot_pages = extract_camelot_pagewise(file_path, "lattice")
    
    if not camelot_pages or all(df.empty for df in camelot_pages):
        camelot_pages = extract_pdfplumber_pagewise(file_path)
    
    if (not camelot_pages or all(df.empty for df in camelot_pages)) and client:
        st.info(f"🤖 Trying AI extraction...")
        camelot_pages = extract_openai_pagewise(file_path)
    
    if camelot_pages and any(not df.empty for df in camelot_pages):
        combined_df = pd.concat([df for df in camelot_pages if not df.empty], ignore_index=True)
        combined_df = remove_duplicate_headers(combined_df)
        return combined_df
    
    return None