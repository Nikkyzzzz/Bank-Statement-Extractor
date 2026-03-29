# 🏦 Bank Statement Extractor

A powerful Streamlit-based application for extracting, processing, merging, and analyzing bank statements from multiple formats (PDF, Excel, CSV). This tool automates the tedious process of consolidating bank statements and provides intelligent analysis of transactions.

## ✨ Features

### 📄 Multi-Format Support
- **PDF Extraction**: Extract tables from PDF bank statements using multiple engines:
  - Camelot (lattice & stream modes)
  - PDFPlumber
  - OpenAI GPT-4 Vision (AI-powered extraction)
- **Excel Files**: Support for `.xlsx` and `.xls` formats
- **CSV Files**: Import existing CSV bank statements

### 🔗 Intelligent File Merging
- Automatically maps column names from different banks to a standard format
- Handles variations in column naming conventions
- Preserves file source information for tracking
- Supports both merged and separate file analysis

### 🔍 Advanced Transaction Analysis
The application includes sophisticated detection algorithms for:

- **🔄 Refunds**: Identifies refund transactions with reference matching
- **📤 Returns**: Detects rail-driven returns (NEFT/RTGS/IMPS/UPI/Cheque bounces)
- **💵 Cash Withdrawals**: Identifies ATM withdrawals and cash transactions
- **💸 Bank Charges**: Detects bank fees, service charges, and penalties
- **💰 Commissions**: Identifies commission and brokerage payments
- **🏛️ Statutory Payments**: Recognizes GST, TDS, PF, ESI, and other tax payments
- **🔁 Recurring Transactions**: Identifies subscription and recurring payments

### 📊 Data Export
- Export processed data to Excel (`.xlsx`)
- Export processed data to CSV (`.csv`)
- Includes all original data plus analysis results

### 🎨 Modern UI
- Clean, intuitive interface with dark mode support
- Step-by-step workflow with progress tracking
- Interactive data preview and analysis results
- Responsive design for all screen sizes

## 🚀 Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/Nikkyzzzz/Bank-Statement-Extractor.git
   cd Bank-Statement-Extractor
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up OpenAI API (Optional)**
   
   For AI-powered PDF extraction, create a `.env` file in the project root:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## 📖 Usage

### Starting the Application

Run the Streamlit app using:

```bash
streamlit run flow.py
```

The application will open in your default web browser at `http://localhost:8501`

### Workflow

#### 1. 📤 Upload Files
- Upload your bank statement files (PDF, Excel, or CSV)
- Supported formats: `.pdf`, `.xlsx`, `.xls`, `.csv`
- Multiple files can be uploaded at once

#### 2. 🔍 Preview Data
- Review extracted data from each file
- For PDFs, select extraction method:
  - **Camelot Lattice**: Best for tables with visible borders
  - **Camelot Stream**: Best for borderless tables
  - **PDFPlumber**: Excellent for complex layouts
  - **OpenAI Vision**: AI-powered extraction (requires API key)
- Preview and validate data before proceeding

#### 3. 🔗 Merge Files
- Choose which files to merge into a consolidated dataset
- Select files to keep separate for individual analysis
- Column mapping is handled automatically
- Preview merged data before analysis

#### 4. 📊 Analysis
- Run intelligent transaction analysis
- View categorized results:
  - Refunds and returns
  - Cash withdrawals
  - Bank charges and fees
  - Commissions paid
  - Statutory payments
  - Recurring transactions
- Export results to Excel or CSV

## 📁 Project Structure

```
Bank-Statement-Extractor/
├── flow.py                 # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .gitignore             # Git ignore rules
├── README.md              # This file
├── Bank Statements/       # Sample bank statements (PDF)
└── Input files CSV and Excel/  # Extracted/processed files
```

## 🔧 Configuration

### Column Mapping

The application automatically maps various column names to standard formats:

- **Date**: `Txn Date`, `Transaction Date`, `Value Date`, etc.
- **Description**: `Description`, `Narration`, `Particulars`, etc.
- **Withdrawal**: `Withdrawal`, `Debit`, `DR Amount`, etc.
- **Deposit**: `Deposit`, `Credit`, `CR Amount`, etc.
- **Balance**: `Balance`, `Closing Balance`, `Running Balance`, etc.

### Extraction Methods

**Camelot Lattice**
- Best for: Tables with visible grid lines
- Use when: Bank statements have clear table borders

**Camelot Stream**
- Best for: Borderless tables
- Use when: Statements have no visible table lines

**PDFPlumber**
- Best for: Complex layouts, mixed content
- Use when: Other methods fail or for inconsistent formats

**OpenAI Vision**
- Best for: Scanned documents, poor quality PDFs
- Use when: Traditional methods produce poor results
- Requires: OpenAI API key in `.env` file

## 🛠️ Technical Details

### Key Dependencies

- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **Camelot**: PDF table extraction
- **PDFPlumber**: Alternative PDF extraction
- **OpenAI**: AI-powered document processing
- **PyMuPDF**: PDF rendering
- **OpenPyXL**: Excel file handling

### Analysis Algorithms

The application uses sophisticated pattern matching and heuristics to:

1. **Match refunds to original transactions** using reference numbers, merchant names, and amounts
2. **Detect returned payments** by analyzing transaction pairs and date windows
3. **Identify cash withdrawals** using keyword detection and transaction direction
4. **Classify bank charges** based on narration patterns and amount thresholds
5. **Recognize recurring payments** through frequency analysis

## 📊 Supported Banks

The application works with statements from any bank but has been tested with:

- State Bank of India (SBI)
- ICICI Bank
- HDFC Bank
- Punjab National Bank (PNB)
- Bank of India (BOI)
- Union Bank of India (UBI)
- YES Bank
- And many more...

## ⚠️ Important Notes

### Data Privacy
- **Keep your repository private** if using real bank statements
- Bank statements contain sensitive financial information
- Never commit real statements to public repositories

### File Size Limits
- Large PDF files may take time to process
- Consider splitting very large statements
- OpenAI Vision has API costs per page

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the MIT License.

## 🐛 Troubleshooting

### PDF Extraction Issues
- Try different extraction methods (Camelot, PDFPlumber, OpenAI)
- Ensure PDF is not password-protected
- Check if PDF contains actual text (not just images)

### Excel Reading Errors
- Verify file is not corrupted
- Try opening in Excel/LibreOffice first
- Convert to CSV if issues persist

### Memory Issues
- Process files in smaller batches
- Close other applications
- Increase available RAM

## 📧 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Contact the repository owner

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- PDF extraction powered by [Camelot](https://camelot-py.readthedocs.io/)
- AI extraction using [OpenAI GPT-4](https://openai.com/)

---

**Made with ❤️ for automating financial data processing**
