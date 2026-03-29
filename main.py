import streamlit as st
import pandas as pd
import os
import tempfile
import io
import numpy as np
from datetime import datetime
from utils import (
    MODERN_CSS, STANDARD_COLUMNS, map_column, normalize_column_name,
    create_step_indicator, create_metric_card, styled_button,
    display_analysis_details, refine_and_validate_data, split_merged_decimals,
    remove_duplicate_headers, parse_amount, normalize_text, extract_references,
    classify_txn, is_promotional, standardize_columns, validate_dataframe
)
from pdf_extractors import extract_pdf_data

# Configure Streamlit
st.set_page_config(
    page_title="Bank Statement Processor", 
    layout="wide", 
    page_icon="🏦",
    initial_sidebar_state="collapsed"
)

# Apply custom CSS
st.markdown(MODERN_CSS, unsafe_allow_html=True)

# -------------------------------------------------------------------
# File Merging Functions
# -------------------------------------------------------------------
def read_file(file):
    """Robust file reader with multiple fallback methods."""
    try:
        if hasattr(file, 'seek'):
            file.seek(0)
        
        ext = os.path.splitext(file.name)[1].lower()
        
        if ext in [".xlsx", ".xls"]:
            engines_to_try = ['openpyxl', 'xlrd']
            
            for engine in engines_to_try:
                try:
                    if hasattr(file, 'seek'):
                        file.seek(0)
                    
                    if ext == ".xlsx":
                        df = pd.read_excel(file, engine=engine)
                    else:
                        df = pd.read_excel(file, engine=engine)
                    
                    if df is not None and not df.empty:
                        st.success(f"✅ Successfully read {file.name} using {engine} engine")
                        return df
                        
                except Exception as e:
                    if hasattr(file, 'seek'):
                        file.seek(0)
                    continue
            
            try:
                if hasattr(file, 'seek'):
                    file.seek(0)
                df = pd.read_excel(file)
                if df is not None and not df.empty:
                    st.success(f"✅ Successfully read {file.name} using default engine")
                    return df
            except Exception as e:
                st.warning(f"⚠️ Standard Excel reading failed for {file.name}")
            
            st.info(f"🔄 Attempting to read {file.name} as CSV as final fallback...")
            try:
                if hasattr(file, 'seek'):
                    file.seek(0)
                
                content = file.read()
                if isinstance(content, bytes):
                    content = content.decode('utf-8', errors='ignore')
                
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
        if hasattr(file, 'seek'):
            file.seek(0)
        content = file.read()
        
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

def merge_dataframes(dataframes):
    """Merge multiple dataframes."""
    all_data = []

    for df in dataframes:
        if df is None or df.empty:
            continue

        mapped_cols = {}
        for col in df.columns:
            mapped = map_column(col)
            if mapped:
                mapped_cols[col] = mapped

        df = df.rename(columns=mapped_cols)

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

        for col in STANDARD_COLUMNS:
            if col not in df.columns and col != "File Name":
                df[col] = ""

        cols_to_use = [col for col in STANDARD_COLUMNS if col in df.columns]
        
        if 'File Name' in df.columns and 'File Name' not in cols_to_use:
            cols_to_use.append('File Name')
        
        df = df[cols_to_use]
        
        if 'File Name' not in df.columns:
            df["File Name"] = "Unknown"

        all_data.append(df)

    if all_data:
        try:
            merged_df = pd.concat(all_data, ignore_index=True)
            if 'File Name' not in merged_df.columns:
                st.warning("⚠️ File Name column was lost during merging.")
            return merged_df
        except Exception as e:
            st.error("Failed to concatenate DataFrames.")
            st.exception(e)
            return None
    else:
        st.warning("No valid data found in the files.")
        return None

# -------------------------------------------------------------------
# Analysis Functions
# -------------------------------------------------------------------
REFUND_KEYWORDS = ["refund", "refd", "returned", "chargeback", "chargeback credit", "pg refund", "cashback for refund"]

def detect_refunds(df: pd.DataFrame):
    df = df.copy()
    df['Txn Date'] = pd.to_datetime(df['Txn Date'], dayfirst=True, errors='coerce')
    df['narration'] = df['Description'].fillna('').astype(str)
    df['narr_norm'] = df['narration'].apply(normalize_text)
    df['refs'] = df['narration'].apply(extract_references)
    df['type'], df['amount'] = zip(*df.apply(classify_txn, axis=1))
    df['is_promo'] = df['narration'].apply(is_promotional)

    debits = []
    credits = []
    for i, r in df.iterrows():
        rec = r.to_dict()
        rec['index'] = i
        if rec['type'] == 'debit':
            debits.append(rec)
        elif rec['type'] == 'credit':
            credits.append(rec)

    deb_by_ref = {}
    deb_by_merchant = {}
    for d in debits:
        for ref in d.get('refs', []):
            if ref not in deb_by_ref:
                deb_by_ref[ref] = []
            deb_by_ref[ref].append(d)
        deb_by_merchant[d['narr_norm']] = deb_by_merchant.get(d['narr_norm'], []) + [d]

    matches = []
    remaining = {d['index']: d['amount'] for d in debits}

    def try_match_credit(c):
        c_amount = c['amount']
        c_date = c['Txn Date']
        narr = c['narr_norm']
        refs = c.get('refs', [])
        matched = []
        
        if not any(k in narr for k in REFUND_KEYWORDS):
            return []
        if c['is_promo'] and not any(k in narr for k in REFUND_KEYWORDS):
            return []
        
        def candidate_ok(d):
            if d['Txn Date'] is pd.NaT or c_date is pd.NaT:
                date_ok = True
            else:
                date_ok = (d['Txn Date'] <= c_date) and ((c_date - d['Txn Date']).days <= 180)
            if not date_ok:
                return False
            amt_allowed = remaining.get(d['index'], 0)
            if amt_allowed <= 0:
                return False
            if c_amount > amt_allowed * 1.01:
                return False
            return True
        
        for ref in refs:
            candidates = deb_by_ref.get(ref, [])
            for d in candidates:
                if not candidate_ok(d):
                    continue
                take = min(c_amount, remaining[d['index']])
                remaining[d['index']] = round(remaining[d['index']] - take, 2)
                matches.append({'credit_idx': c['index'], 'debit_idx': d['index'], 'matched_amount': take, 'match_type': 'ref'})
                c_amount -= take
                if c_amount <= 0.005:
                    return matches
        
        for m, candidates in deb_by_merchant.items():
            if not m:
                continue
            if not any(token in m for token in narr.split()[:6]):
                continue
            for d in candidates:
                if not candidate_ok(d):
                    continue
                take = min(c_amount, remaining[d['index']])
                remaining[d['index']] = round(remaining[d['index']] - take, 2)
                matches.append({'credit_idx': c['index'], 'debit_idx': d['index'], 'matched_amount': take, 'match_type': 'merchant'})
                c_amount -= take
                if c_amount <= 0.005:
                    return matches
        
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
                    take = min(c_amount, remaining[d['index']])
                    remaining[d['index']] = round(remaining[d['index']] - take, 2)
                    matches.append({'credit_idx': c['index'], 'debit_idx': d['index'], 'matched_amount': take, 'match_type': 'category'})
                    c_amount -= take
                    if c_amount <= 0.005:
                        return matches
        
        return matches

    credits_sorted = sorted(credits, key=lambda r: r['Txn Date'] or datetime.min)
    for c in credits_sorted:
        try_match_credit(c)

    df['matched'] = [[] for _ in range(len(df))]
    for m in matches:
        cidx = m['credit_idx']
        didx = m['debit_idx']
        df.at[cidx, 'matched'] = df.at[cidx, 'matched'] + [m]
        df.at[didx, 'matched'] = df.at[didx, 'matched'] + [m]

    df['is_refund'] = df['matched'].apply(lambda x: len(x) > 0 and df.at[x[0]['credit_idx'], 'type'] == 'credit' if isinstance(x, list) and x else False)
    df['remaining_refundable'] = None
    for d in debits:
        df.at[d['index'], 'remaining_refundable'] = remaining.get(d['index'], d['amount'])

    out_cols = ['Txn Date', 'Description', 'type', 'amount', 'matched', 'is_refund', 'remaining_refundable']
    return df[out_cols]

def detect_returns(df: pd.DataFrame):
    """Detect rail-driven returns (NEFT/RTGS/IMPS/UPI/ECS/NACH/Cheque returns)."""
    df = df.copy()
    df['Txn Date'] = pd.to_datetime(df.get('Txn Date'), dayfirst=True, errors='coerce')
    if 'type' not in df.columns or 'amount' not in df.columns:
        df['type'], df['amount'] = zip(*df.apply(classify_txn, axis=1))
    df['narr_norm'] = df['Description'].fillna('').astype(str).apply(normalize_text)
    df['refs'] = df['Description'].fillna('').astype(str).apply(extract_references)

    RETURN_KEYWORDS = [
        'neft return', 'rtgs return', 'imps return', 'upi reversal', 'upi reversal by psp',
        'ecs return', 'nach dr rtn', 'nach return', 'ach return',
        'cheque return', 'cheque bounce', 'chq ret', 'unpaid', 'return to originator', 'rtn'
    ]
    WINDOWS = {'neft': 3, 'rtgs': 3, 'imps': 3, 'upi': 3, 'ecs': 7, 'nach': 7, 'cheque': 14}

    def narr_indicates_return(narr):
        return any(k in narr for k in RETURN_KEYWORDS)

    matches = []
    by_amount = {}
    for i, r in df.iterrows():
        rec = r.to_dict()
        rec['index'] = i
        amt = round(rec.get('amount', 0), 2)
        if amt not in by_amount:
            by_amount[amt] = []
        by_amount[amt].append(rec)

    def amount_matches(a, b):
        tol = max(abs(b) * 0.01, 10.0)
        return abs(a - b) <= tol

    for i, r in df.iterrows():
        n = r.get('narr_norm', '')
        val = r.get('amount', 0)
        if not narr_indicates_return(n):
            continue
        cand_list = by_amount.get(round(val, 2), [])
        best = None
        best_score = None
        for cand in cand_list:
            if cand['index'] == i:
                continue
            dir_match = (r['type'] == 'credit' and cand['type'] == 'debit') or (r['type'] == 'debit' and cand['type'] == 'credit')
            if not dir_match:
                continue
            if pd.isna(cand['Txn Date']) or pd.isna(r['Txn Date']):
                days = 0
            else:
                days = abs((r['Txn Date'] - cand['Txn Date']).days)
            if days > max(WINDOWS.values()):
                continue
            common_ref = set(cand.get('refs', [])) & set(r.get('refs', []))
            score = 0
            if common_ref:
                score -= 100
            score += days
            if best_score is None or score < best_score:
                best = cand
                best_score = score

        if best is not None:
            matches.append({'returned_idx': i, 'orig_idx': best['index'], 'amount': val})

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
    df['Txn Date'] = pd.to_datetime(df.get('Txn Date'), dayfirst=True, errors='coerce')
    if 'type' not in df.columns or 'amount' not in df.columns:
        df['type'], df['amount'] = zip(*df.apply(classify_txn, axis=1))
    df['narration'] = df['Description'].fillna('').astype(str)
    df['narr_norm'] = df['narration'].apply(normalize_text)

    CASH_KEYWORDS = [
        'atm wdl', 'atm withdrawal', 'cash wdl', 'cash withdrawal', 'cash paid', 'self chq', 'self cheque',
        'chq encash', 'teller wdl', 'cardless cash', 'imt cash', 'aeps cash', 'microatm cash',
        'upi cash wd', 'upi atm', 'cash at pos', 'cashback pos', 'cash advance'
    ]

    df['is_cash_withdrawal'] = False
    df['cash_matches'] = [[] for _ in range(len(df))]

    for i, r in df.iterrows():
        if r.get('type') != 'debit':
            continue
        n = r.get('narr_norm', '')
        if not any(k in n for k in CASH_KEYWORDS):
            continue
        if any(x in n for x in ['pos purchase', 'online', 'upi to', 'neft', 'merchant', 'ecom']):
            continue
        dt = r.get('Txn Date')
        found_reversal = False
        for j, r2 in df.iterrows():
            if r2.get('type') == 'credit' and abs((r2.get('Txn Date') - dt).days) <= 1 if pd.notna(dt) and pd.notna(r2.get('Txn Date')) else False:
                if any(x in normalize_text(r2.get('Description','')) for x in ['reversal', 'failed']):
                    found_reversal = True
                    break
        if found_reversal:
            continue
        df.at[i, 'is_cash_withdrawal'] = True
        df.at[i, 'cash_matches'] = df.at[i, 'cash_matches'] + [{'index': i}]

    return df[['Txn Date', 'Description', 'type', 'amount', 'is_cash_withdrawal', 'cash_matches']]

def detect_bank_charges(df: pd.DataFrame, max_amount=1000.0):
    """Detect bank charges based on narration keywords and small debit amounts."""
    df = df.copy()
    df['Txn Date'] = pd.to_datetime(df.get('Txn Date'), dayfirst=True, errors='coerce')
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
        amt = abs(r.get('amount', 0))
        if amt > max_amount and 'debit card annual fee' not in n and 'locker rent' not in n:
            continue
        linked = None
        dt = r.get('Txn Date')
        if pd.notna(dt):
            for j, r2 in df.iterrows():
                if j == i:
                    continue
                if abs((r2.get('Txn Date') - dt).days) <= 1 if pd.notna(r2.get('Txn Date')) else False:
                    linked = j
                    break
        df.at[i, 'is_bank_charge'] = True
        df.at[i, 'bank_charge_matches'] = df.at[i, 'bank_charge_matches'] + ([{'linked_idx': linked}] if linked is not None else [])

    return df[['Txn Date', 'Description', 'type', 'amount', 'is_bank_charge', 'bank_charge_matches']]

def detect_commissions(df: pd.DataFrame, max_amount=5000.0):
    """Detect commission/brokerage debits and return serializable matches."""
    df = df.copy()
    df['Txn Date'] = pd.to_datetime(df.get('Txn Date'), dayfirst=True, errors='coerce')
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
        df.at[i, 'is_commission_paid'] = True
        df.at[i, 'commission_matches'] = df.at[i, 'commission_matches'] + (linked if linked else [])

    return df[['Txn Date', 'Description', 'type', 'amount', 'is_commission_paid', 'commission_matches']]

def detect_statutory_payments(df: pd.DataFrame):
    """Detect statutory payments (GST, TDS, PF, ESI, Income Tax, etc.)."""
    df = df.copy()
    df['Txn Date'] = pd.to_datetime(df.get('Txn Date'), dayfirst=True, errors='coerce')
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
        if 'gst on' in n or 'gst on atm' in n:
            continue
        if not any(k in n for k in STAT_KW):
            continue
        if any(x in n for x in ['refund', 'reversal', 'credited']):
            continue
        df.at[i, 'is_statutory'] = True
        df.at[i, 'statutory_matches'] = df.at[i, 'statutory_matches'] + [int(i)]

    return df[['Txn Date', 'Description', 'type', 'amount', 'is_statutory', 'statutory_matches']]

def detect_recurring(df: pd.DataFrame):
    """Detect recurring transactions with improved error handling"""
    try:
        if 'txn_date' not in df.columns and 'Txn Date' in df.columns:
            df_std = df.copy()
        else:
            df_std = standardize_columns(df)
        
        if 'Txn Date' in df_std.columns:
            df_std['Txn Date'] = pd.to_datetime(df_std['Txn Date'], dayfirst=True, errors='coerce')
        elif 'txn_date' in df_std.columns:
            df_std['txn_date'] = pd.to_datetime(df_std['txn_date'], dayfirst=True, errors='coerce')
        
        validate_dataframe(df_std)
        
        df_std['is_recurrent'] = False
        df_std['recurrence_pattern'] = None
        df_std['recurrence_frequency'] = None
        
        desc_col = 'Description' if 'Description' in df_std.columns else 'description' if 'description' in df_std.columns else None
        
        if desc_col:
            df_std['desc_norm'] = df_std[desc_col].fillna('').astype(str).apply(normalize_text)
            desc_counts = df_std['desc_norm'].value_counts()
            recurring_descs = desc_counts[desc_counts >= 3].index.tolist()
            
            for idx, row in df_std.iterrows():
                if row['desc_norm'] in recurring_descs:
                    df_std.at[idx, 'is_recurrent'] = True
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
        
        return df_std[available_columns]
        
    except Exception as e:
        print(f"Error in detect_recurring: {str(e)}")
        return pd.DataFrame()

def run_analysis(df: pd.DataFrame):
    """Run all analysis functions on the dataframe."""
    out_refunds = detect_refunds(df)
    out_returns = detect_returns(df)
    out_cash = detect_cash_withdrawals(df)
    out_charges = detect_bank_charges(df)
    out_comm = detect_commissions(df)
    out_stat = detect_statutory_payments(df)

    out = out_refunds.copy()
    
    if 'is_returned' in out_returns.columns:
        out['is_returned'] = out_returns['is_returned']
    else:
        out['is_returned'] = False

    if 'returned_matches' in out_returns.columns:
        out['returned_matches'] = out_returns['returned_matches']
    else:
        out['returned_matches'] = [[] for _ in range(len(out))]
    
    for col in ['is_cash_withdrawal', 'cash_matches']:
        if col in out_cash.columns:
            out[col] = out_cash[col]
        else:
            out[col] = False if col == 'is_cash_withdrawal' else [[] for _ in range(len(out))]
    
    for col in ['is_bank_charge', 'bank_charge_matches']:
        if col in out_charges.columns:
            out[col] = out_charges[col]
        else:
            out[col] = False if col == 'is_bank_charge' else [[] for _ in range(len(out))]
    
    for col in ['is_commission_paid', 'commission_matches']:
        if col in out_comm.columns:
            out[col] = out_comm[col]
        else:
            out[col] = False if col == 'is_commission_paid' else [[] for _ in range(len(out))]
    
    for col in ['is_statutory', 'statutory_matches']:
        if col in out_stat.columns:
            out[col] = out_stat[col]
        else:
            out[col] = False if col == 'is_statutory' else [[] for _ in range(len(out))]

    complex_cols = ['matched', 'returned_matches', 'cash_matches', 'bank_charge_matches', 
                   'commission_matches', 'statutory_matches']

    def _safe_serialize(v):
        try:
            if v is None:
                return ''
            if isinstance(v, str):
                return v
            if isinstance(v, float) and pd.isna(v):
                return ''
            if isinstance(v, (list, dict)):
                import json
                return json.dumps(v)
            if isinstance(v, (np.ndarray,)) or hasattr(v, 'tolist'):
                try:
                    import json
                    return json.dumps(v.tolist())
                except Exception:
                    import json
                    return json.dumps(list(v))
            try:
                import json
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
    
    st.markdown("""
    <div class='card'>
        <h2>Upload Your Bank Statement Files</h2>
        <p>Upload PDF, Excel, or CSV files containing your bank statements. 
        The system will automatically process PDF files and prepare all files for merging.</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Choose bank statement files", 
        type=["pdf", "xlsx", "xls", "csv"], 
        accept_multiple_files=True,
        help="Upload PDF, Excel, or CSV files"
    )
    
    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        
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
        
        with st.expander("View File Details"):
            for file in uploaded_files:
                file_type = "PDF" if file.name.lower().endswith('.pdf') else "Excel" if file.name.lower().endswith(('.xlsx', '.xls')) else "CSV"
                st.write(f"• **{file.name}** ({file_type}, {file.size // 1024} KB)")
        
        if styled_button("🚀 Process Files & Continue", "process_files", "primary"):
            process_uploaded_files(uploaded_files)

def process_uploaded_files(uploaded_files):
    """Process all uploaded files and extract data from PDFs."""
    st.session_state.processed_files = {}
    
    pdf_files = [f for f in uploaded_files if f.name.lower().endswith('.pdf')]
    
    if pdf_files:
        with st.spinner("📄 Extracting data from PDF files..."):
            for file in pdf_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(file.read())
                    tmp_path = tmp.name
                
                combined_df = extract_pdf_data(tmp_path)
                
                if combined_df is not None:
                    st.session_state.processed_files[file.name] = combined_df
                else:
                    st.warning(f"⚠️ Could not extract data from {file.name}")
    
    other_files = [f for f in uploaded_files if not f.name.lower().endswith('.pdf')]
    
    for file in other_files:
        df = read_file(file)
        if df is not None and not df.empty:
            st.session_state.processed_files[file.name] = df
        else:
            st.warning(f"⚠️ Could not read {file.name}")
    
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
    
    st.markdown("<h2 class='sub-header'>📋 File Previews</h2>", unsafe_allow_html=True)
    
    for file_name, df in st.session_state.processed_files.items():
        with st.expander(f"📄 {file_name} ({len(df)} rows)"):
            st.dataframe(df.head(10), use_container_width=True)
            st.write(f"**Columns:** {', '.join(df.columns.tolist())}")
    
    st.markdown("<h2 class='sub-header'>🔗 Select Files to Merge</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
        <h3>💡 Merge Selection</h3>
        <p>Select which files you want to merge together. Unselected files will be analyzed separately.</p>
    </div>
    """, unsafe_allow_html=True)
    
    selected_files = []
    file_options = list(st.session_state.processed_files.keys())
    
    cols = st.columns(2)
    for i, file_name in enumerate(file_options):
        with cols[i % 2]:
            if st.checkbox(f"**{file_name}**", value=True, key=f"merge_{file_name}"):
                selected_files.append(file_name)
    
    if styled_button("🔄 Configure Merging & Continue", "configure_merge", "primary"):
        st.session_state.files_to_merge = selected_files
        st.session_state.separate_files = [f for f in file_options if f not in selected_files]
        st.session_state.current_step = "merge"
        st.rerun()
    
    if st.button("⬅️ Back to Upload"):
        st.session_state.current_step = "upload"
        st.rerun()

def show_merge_step():
    st.markdown("<h1 class='main-header'>🔗 File Merging Configuration</h1>", unsafe_allow_html=True)
    
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
    
    if st.session_state.merged_data is not None or st.session_state.separate_files:
        st.markdown("<h2 class='sub-header'>👀 Data Preview</h2>", unsafe_allow_html=True)
        
        if st.session_state.merged_data is not None:
            st.markdown("#### 📊 Merged Data")
            st.dataframe(st.session_state.merged_data.head(10), use_container_width=True)
            st.write(f"**Total Rows:** {len(st.session_state.merged_data)}")
        
        if st.session_state.separate_files:
            st.markdown("#### 📄 Separate Files")
            for file_name in st.session_state.separate_files:
                with st.expander(f"📄 {file_name}"):
                    df = st.session_state.processed_files[file_name]
                    st.dataframe(df.head(10), use_container_width=True)
    
    if st.session_state.merged_data is not None or st.session_state.separate_files:
        if styled_button("🔍 Run Analysis & Continue", "run_analysis", "success"):
            st.session_state.current_step = "analysis"
            st.rerun()
    
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
    
    st.markdown("<h2 class='sub-header'>📊 Analysis Targets</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        create_metric_card(len(datasets_to_analyze), "Datasets to Analyze", "📁")
    with col2:
        total_rows = sum(len(df) for df in datasets_to_analyze.values())
        create_metric_card(total_rows, "Total Transactions", "📊")
    
    if styled_button("🚀 Run Comprehensive Analysis", "run_comprehensive_analysis", "primary"):
        run_comprehensive_analysis_flow(datasets_to_analyze)
    
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
    
    st.markdown("<h2 class='sub-header'>📋 Analysis Results</h2>", unsafe_allow_html=True)
    
    for dataset_name, result in analysis_results.items():
        if result is not None:
            with st.expander(f"📊 {dataset_name} Analysis Results", expanded=True):
                display_analysis_details(result, dataset_name)
                
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
    
    st.markdown("---")
    if styled_button("🔄 Start New Analysis", "restart_workflow", "secondary"):
        for key in ['current_step', 'uploaded_files', 'processed_files', 'files_to_merge', 'separate_files', 'merged_data']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

if __name__ == "__main__":
    main()