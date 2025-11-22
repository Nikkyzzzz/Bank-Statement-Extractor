import streamlit as st
import pandas as pd
import io

# -------------------------
# 1. Target schema
# -------------------------
TARGET_COLUMNS = [
    "Transaction Date",
    "Narration/Description",
    "Cheque Number",
    "Credit (Deposit)",
    "Debit (Withdrawal)",
    "Balance",
    "File Name"
]

# -------------------------
# 2. Keyword mappings
# -------------------------
COLUMN_MAPPINGS = {
    "Transaction Date": ["date", "txn date", "transaction date", "value date", "trans date"],
    "Narration/Description": ["description", "details", "narration", "particulars", "transaction details"],
    "Cheque Number": ["cheque", "chq", "cheque no", "cheque number"],
    "Credit (Deposit)": ["credit", "deposit", "cr amount", "amount credit", "amt credit", "credit amount"],
    "Debit (Withdrawal)": ["debit", "withdrawal", "dr amount", "amount debit", "amt debit", "withdrawal amount"],
    "Balance": ["balance", "bal", "closing balance", "available balance"]
}

# -------------------------
# 3. Column normalization
# -------------------------
def normalize_columns(df, file_name):
    df_cols = [c.strip().lower() for c in df.columns]
    new_cols = {}

    for target, keywords in COLUMN_MAPPINGS.items():
        for i, col in enumerate(df_cols):
            if any(k in col for k in keywords):
                new_cols[df.columns[i]] = target
                break

    df = df.rename(columns=new_cols)
    for col in TARGET_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df = df[TARGET_COLUMNS[:-1]]  # exclude File Name for now
    df["File Name"] = file_name
    return df

# -------------------------
# 4. Streamlit App UI
# -------------------------
st.set_page_config(page_title="Bank CSV Merger", layout="wide")
st.title("🏦 Bank Statement CSV Merger")

uploaded_files = st.file_uploader(
    "Upload multiple extracted CSV files",
    type=["csv"],
    accept_multiple_files=True
)

if uploaded_files:
    merged_df = pd.DataFrame(columns=TARGET_COLUMNS)

    for file in uploaded_files:
        try:
            df = pd.read_csv(file)
            normalized = normalize_columns(df, file.name)
            merged_df = pd.concat([merged_df, normalized], ignore_index=True)
        except Exception as e:
            st.error(f"❌ Error reading {file.name}: {e}")

    st.success(f"✅ Successfully merged {len(uploaded_files)} files.")
    st.dataframe(merged_df.head(10))

    # Download buttons
    csv_bytes = merged_df.to_csv(index=False).encode('utf-8')
    excel_bytes = io.BytesIO()
    merged_df.to_excel(excel_bytes, index=False)
    excel_bytes.seek(0)

    st.download_button("📥 Download as CSV", csv_bytes, "merged_bank_statements.csv", "text/csv")
    st.download_button("📘 Download as Excel", excel_bytes, "merged_bank_statements.xlsx", "application/vnd.ms-excel")
else:
    st.info("Upload 2 or more CSVs to begin merging.")
