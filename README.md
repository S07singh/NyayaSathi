# ⚖️ NyayaSathi AI — Legal Document Simplifier & Risk Analyzer

> An AI-powered tool that helps citizens understand rental agreements, employment contracts, business contracts, and legal notices — in plain English.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | Streamlit |
| **LLM (Fast Mode)** | Groq API (Llama 3.1 70B) |
| **LLM (Private Mode)** | Ollama — Llama 3.1 (local) |
| **Embeddings** | SentenceTransformers (`all-MiniLM-L6-v2`) |
| **Vector Database** | FAISS (Facebook AI Similarity Search) |
| **Document Parsing** | pdfplumber, python-docx |
| **Visualisation** | Plotly |
| **Language** | Python 3.10+ |

---

## ✨ Features

- **📄 Document Upload** — Accepts PDF and DOCX files with automatic text extraction
- **🏷️ Auto-Classification** — Classifies documents into Rental Agreement, Employment Contract, Business Contract, or Legal Notice
- **✂️ Clause Segmentation** — Splits documents into logical clauses for granular analysis
- **📖 Legal Simplification** — Converts complex legal jargon into plain English using LLM
- **⚠️ Risk Analysis** — Scores clauses across 5 risk dimensions (penalty severity, clause imbalance, termination unfairness, liability excess, missing rights) with a weighted 0–100 score
- **🧠 RAG-Powered Context** — Grounds explanations in legal reference documents using FAISS vector search
- **💬 AI Chat** — Context-aware Q&A about the uploaded document with mode selection (Fast / Private)
- **📊 Report Generation** — Downloadable Markdown report with summary, obligations, deadlines, risk breakdown, and negotiation points

---

## 📂 Project Structure

```
nyayasathi/
│
├── app.py                          # Streamlit entry point (5-tab UI)
├── config.py                       # Centralised settings & constants
├── requirements.txt                # Python dependencies
│
├── services/
│   ├── document_parser.py          # PDF / DOCX text extraction
│   ├── classifier.py               # Hybrid keyword + embedding classifier
│   ├── clause_segmenter.py         # Regex + paragraph-based clause splitting
│   ├── risk_analyzer.py            # LLM-assisted 5-dimension risk scoring
│   ├── llm_router.py               # Groq / Ollama routing with error handling
│   └── rag_engine.py               # FAISS RAG pipeline (chunk → embed → index → query)
│
├── embeddings/
│   └── embedding_model.py          # SentenceTransformer wrapper with caching
│
├── data/
│   └── legal_references/           # Seed reference documents for RAG
│       ├── rental_agreement_reference.txt
│       ├── employment_contract_reference.txt
│       ├── business_contract_reference.txt
│       └── legal_notice_reference.txt
│
├── utils/
│   ├── logger.py                   # Structured logging factory
│   └── text_cleaner.py             # Unicode / whitespace normalisation
│
└── reports/
    └── report_generator.py         # Markdown report generator
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- (Optional) [Ollama](https://ollama.com/) installed for Private Mode
- (Optional) [Groq API key](https://console.groq.com/) for Fast Mode

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/S07singh/NyayaSathi.git
cd NyayaSathi

# 2. Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / macOS

# 3. Install dependencies
pip install -r requirements.txt
```

### Configuration

```bash
# Set Groq API key for Fast Mode (optional)
set GROQ_API_KEY=your_groq_api_key_here        # Windows
# export GROQ_API_KEY=your_groq_api_key_here   # Linux / macOS

# Start Ollama for Private Mode (optional)
ollama run llama3.1
```

### Run

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 🖥️ UI Overview

| Tab | Description |
|-----|-------------|
| **📋 Overview** | Document classification, clause count, risk score, obligations & deadlines |
| **📑 Clauses** | Expandable clause list with per-clause LLM simplification |
| **⚠️ Risk Analysis** | Gauge chart, colour-coded bar chart, detailed risk cards |
| **💬 AI Chat** | RAG-grounded Q&A about the uploaded document |
| **📊 Report** | Full analysis report with download button |

---

## ⚖️ Risk Scoring

Risk is evaluated across five weighted dimensions:

| Dimension | Weight |
|-----------|--------|
| Penalty Severity | 30% |
| Clause Imbalance | 25% |
| Termination Unfairness | 20% |
| Liability Excess | 15% |
| Missing Rights | 10% |

Each clause receives a score from 0 (no risk) to 100 (extreme risk), and the document-level score is the average across all clauses.

---

## 🤖 LLM Modes

| Mode | Provider | Use Case |
|------|----------|----------|
| **⚡ Fast** | Groq Cloud API | Low-latency responses, requires internet & API key |
| **🔒 Private** | Ollama (local) | Data never leaves your machine, requires Ollama running |

---

## 📄 License

This project is open source under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- [Streamlit](https://streamlit.io/) — UI framework
- [SentenceTransformers](https://www.sbert.net/) — Embedding models
- [FAISS](https://github.com/facebookresearch/faiss) — Vector similarity search
- [Groq](https://groq.com/) — Ultra-fast LLM inference
- [Ollama](https://ollama.com/) — Local LLM serving
- [pdfplumber](https://github.com/jsvine/pdfplumber) — PDF text extraction

---

<p align="center">
  Built with ❤️ for making legal documents accessible to everyone.
</p>
