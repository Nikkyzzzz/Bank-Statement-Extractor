import streamlit as st
import pandas as pd
import io
import re

# --------------------------------------------------------
# 1. TARGET SCHEMA
# --------------------------------------------------------
TARGET_COLUMNS = [
    "Transaction Date",
    "Narration/Description",
    "Cheque Number",
    "Credit (Deposit)",
    "Debit (Withdrawal)",
    "Balance",
    "File Name"
]


# --------------------------------------------------------
# 2. EXTENDED COLUMN MAPPINGS (Covers ALL 12 formats)
# --------------------------------------------------------
COLUMN_MAPPINGS = {
    "Transaction Date": [
        "txn date", "transaction date", "post date", "date", "value date",
        "transaction posted date", "txn dt", "trans date"
    ],

    "Narration/Description": [
        "description", "narration", "details", "remarks", "transaction remarks",
        "transaction description", "desc", "particulars"
    ],

    "Cheque Number": [
        "cheque", "chq", "cheque no", "cheque number", "ref no", "reference no",
        "instrument id", "chq./ref.no", "cheque no.", "ref no.", "instrument"
    ],

    "Credit (Deposit)": [
        "credit", "cr amount", "deposit", "deposit amt", "credit amount",
        "amount credit", "cr", "credit amt", "cr."
    ],

    "Debit (Withdrawal)": [
        "debit", "dr amount", "withdrawal", "withdrawal amt",
        "amount debit", "dr", "withdrawal (in rs.)", "withdra wal"
    ],

    "Balance": [
        "balance", "closing balance", "running balance", "balance (in rs.)",
        "available balance"
    ]
}


# --------------------------------------------------------
# 3. COLUMN NORMALIZATION + AMOUNT TYPE HANDLING
# --------------------------------------------------------
def is_number(value):
    if value is None:
        return False
    value = str(value).replace(",", "").strip()
    return bool(re.match(r"^-?\d+(\.\d+)?$", value))


def extract_number(text):
    if text is None:
        return None
    text = str(text)
    matches = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return matches[0] if matches else None


def normalize_columns(df, file_name):
    df_cols_lower = [c.strip().lower() for c in df.columns]
    new_col_map = {}

    # MAP BASED ON COLUMN NAME
    for target, keywords in COLUMN_MAPPINGS.items():
        for i, col in enumerate(df_cols_lower):
            if any(k in col for k in keywords):
                new_col_map[df.columns[i]] = target
                break

    df = df.rename(columns=new_col_map)

    # Ensure essential columns exist
    for col in TARGET_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    # -------------------------------------------------------
    # 1️⃣ HANDLE AMOUNT TYPE (CR/DR logic)
    # -------------------------------------------------------
    amount_type = None
    amount_col = None

    for c in df.columns:
        if "amount type" in c.lower():
            amount_type = c
        if "amount" in c.lower() and c != amount_type:
            amount_col = c

    if amount_type and amount_col:
        df["Credit (Deposit)"] = df.apply(lambda x: x[amount_col] if str(x[amount_type]).lower() in ["cr", "credit"] else "", axis=1)
        df["Debit (Withdrawal)"] = df.apply(lambda x: x[amount_col] if str(x[amount_type]).lower() in ["dr", "debit"] else "", axis=1)

    # -------------------------------------------------------
    # 2️⃣ CLEAN CREDIT/DEBIT COLUMNS (REMOVE NARRATION)
    # -------------------------------------------------------
    for col in ["Credit (Deposit)", "Debit (Withdrawal)"]:
        cleaned_values = []
        narration_extra = []

        for val in df[col]:
            number = extract_number(val)
            cleaned_values.append(number if is_number(number) else "")

        df[col] = cleaned_values

    # -------------------------------------------------------
    # 3️⃣ BUILD PROPER NARRATION COLUMN
    # -------------------------------------------------------
    narration_final = []

    for i, row in df.iterrows():
        n = ""
        possible_cols = ["Narration/Description", "description", "details", "remarks"]
        for pc in possible_cols:
            for col in df.columns:
                if pc in col.lower():
                    n = row[col]
                    break

        # Add leftover narration (non-numeric parts of credit/debit)
        for col in ["Credit (Deposit)", "Debit (Withdrawal)"]:
            original = row.get(col, "")
            if original and not is_number(original):
                n = f"{n} {original}"

        narration_final.append(n.strip())

    df["Narration/Description"] = narration_final

    # -------------------------------------------------------
    # 4️⃣ FINAL OUTPUT
    # -------------------------------------------------------
    df = df[TARGET_COLUMNS[:-1]]
    df["File Name"] = file_name

    return df



# --------------------------------------------------------
# 4. STREAMLIT UI
# --------------------------------------------------------
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
    st.dataframe(merged_df.head(20))

    # DOWNLOAD CSV
    csv_bytes = merged_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download as CSV",
        csv_bytes,
        "merged_bank_statements.csv",
        "text/csv"
    )

    # DOWNLOAD EXCEL
    excel_bytes = io.BytesIO()
    merged_df.to_excel(excel_bytes, index=False)
    excel_bytes.seek(0)

    st.download_button(
        "📘 Download as Excel",
        excel_bytes,
        "merged_bank_statements.xlsx",
        "application/vnd.ms-excel"
    )

else:
    st.info("Upload 2 or more CSVs to begin merging.")
