import streamlit as st
import pandas as pd
import io

# -------------------------
# 1. Target schema
# -------------------------
TARGET_COLUMNS = [
    "Transaction Date",
    "Narration/Description",
    "Credit (Deposit)",
    "Debit (Withdrawal)",
    "Balance",
    "File Name"
]

# -------------------------
# 2. Keyword mappings
# -------------------------
COLUMN_MAPPINGS = {
    # Transaction date variations
    "Transaction Date": [
        "date",
        "txn date",
        "transaction date",
        "value date",
        "value dt",
        "post date",
        "transaction posted date",
        "trans date",
        "txn dt",
        "txn date"
    ],
    # Narration / description variations
    "Narration/Description": [
        "description",
        "details",
        "details",
        "narration",
        "particulars",
        "transaction details",
        "transaction description",
        "transaction desc",
        "remarks",
        "transaction remarks",
        "kims remarks",
        "desc",
        "narration/description",
        "remarks/description"
    ],
    # Credit values (some files have separate CR column)
    "Credit (Deposit)": [
        "credit",
        "deposit",
        "deposit (in rs.)",
        "deposits (in rs.)",
        "cr amount",
        "cr",
        "cr.",
        "credit amount",
        "cr amt",
        "cramt",
        "cramount",
        "amount credit",
        "credits",
        "credit amount",
        "cr amount",
        "credit amount (in rs.)"
    ],
    # Debit values (some files have separate DR column)
    "Debit (Withdrawal)": [
        "debit",
        "withdrawal",
        "withdrawal (in rs.)",
        "withdrawals (in rs.)",
        "dr amount",
        "dr",
        "dr.",
        "debit amount",
        "withdrawal amount",
        "withdra wal",
        "dr amt",
        "dramt",
        "dramount",
        "withdrawal amt",
        "withdrawal amt."
    ],
    # Balance variations
    "Balance": [
        "balance",
        "bal",
        "closing balance",
        "closing",
        "running balance",
        "available balance",
        "closing balance (in rs.)",
        "closing balance",
        "closing balance (in rs)",
        "running balance",
        "closing balance (in rs.)"
    ]
}

# Extra keywords to detect an 'Amount' column and an 'Amount Type' column
AMOUNT_KEYWORDS = [
    "amount",
    "amount (in rs.)",
    "amount in rs",
    "amount (rs)",
    "withdrawal amt",
    "deposit amt",
    "deposit amt.",
    "withdrawal amt.",
    "dr amount",
    "cr amount",
    "amt",
    "amount"
]

AMOUNT_TYPE_KEYWORDS = [
    "amount type",
    "type",
    "amounttype",
    "txn type",
    "dr/cr",
    "dr/cr type",
    "cr/dr",
    "cr/dr type",
    "dr/cr.",
    "cr.",
    "dr."
]

# -------------------------
# 3. Column normalization
# -------------------------
def normalize_columns(df, file_name):
    df_cols = [c.strip().lower() for c in df.columns]
    new_cols = {}
    amount_col = None
    amount_type_col = None

    for target, keywords in COLUMN_MAPPINGS.items():
        for i, col in enumerate(df_cols):
            if any(k in col for k in keywords):
                new_cols[df.columns[i]] = target
                break

    # detect amount and amount-type columns
    for i, col in enumerate(df_cols):
        if amount_col is None and any(k in col for k in AMOUNT_KEYWORDS):
            amount_col = df.columns[i]
        if amount_type_col is None and any(k in col for k in AMOUNT_TYPE_KEYWORDS):
            amount_type_col = df.columns[i]

    df = df.rename(columns=new_cols)

    # Ensure target columns exist
    for col in TARGET_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    # If there is an Amount Type + Amount column, split into credit/debit
    if amount_type_col and amount_col:
        def parse_amount(x):
            if pd.isna(x):
                return ""
            s = str(x).strip()
            if s == "":
                return ""
            s_clean = s.replace(',', '')
            if s_clean.startswith('(') and s_clean.endswith(')'):
                s_clean = '-' + s_clean[1:-1]
            try:
                val = float(s_clean)
                if val.is_integer():
                    return str(int(val))
                return str(val)
            except Exception:
                return s

        credits = []
        debits = []
        for _, row in df.iterrows():
            at = str(row.get(amount_type_col, '')).strip().lower()
            av = parse_amount(row.get(amount_col, ''))
            credit_val = ""
            debit_val = ""
            if at:
                if at.startswith('cr') or 'credit' in at or at.startswith('c'):
                    credit_val = av
                elif at.startswith('dr') or 'debit' in at or at.startswith('d'):
                    debit_val = av
                else:
                    try:
                        if float(av) < 0:
                            debit_val = str(abs(float(av)))
                        else:
                            credit_val = av
                    except Exception:
                        credit_val = av
            else:
                try:
                    if float(av) < 0:
                        debit_val = str(abs(float(av)))
                    else:
                        credit_val = av
                except Exception:
                    credit_val = av

            credits.append(credit_val)
            debits.append(debit_val)

        df['Credit (Deposit)'] = credits
        df['Debit (Withdrawal)'] = debits

    # Final column order: only the requested columns + File Name
    final_cols = [
        'Transaction Date',
        'Narration/Description',
        'Credit (Deposit)',
        'Debit (Withdrawal)',
        'Balance'
    ]

    for col in final_cols:
        if col not in df.columns:
            df[col] = ""

    df = df[final_cols]
    df['File Name'] = file_name
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
