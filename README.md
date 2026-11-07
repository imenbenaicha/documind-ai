# DocuMind AI

**DocuMind AI** is a multimodal Retrieval-Augmented Generation (RAG) application that lets users ask questions about PDF documents and receive answers grounded in document content.

The application combines PDF text extraction, visual analysis, hybrid retrieval, and LLM-based answer generation. It also provides page-level source references and a PDF viewer to help users inspect supporting content.

## Features

- **PDF upload:** Upload PDF documents and work with a collection of documents.
- **Text extraction and chunking:** Extract text page by page and split it into overlapping chunks while preserving document and page metadata.
- **Visual understanding in PDFs:** Extract embedded images and render image-based/scanned pages for vision analysis.
- **Vision-powered descriptions:** Use a Groq vision model to describe visual content such as charts, diagrams, tables, and objects. Visual descriptions are added to the searchable document collection.
- **Hybrid retrieval:** Combine semantic similarity search with BM25 keyword retrieval using Reciprocal Rank Fusion (RRF).
- **Grounded answers:** Generate answers from retrieved document evidence, with an answerability check to avoid unsupported responses.
- **Page-level sources:** Link answers to the source document and page.
- **PDF inspection and highlighting:** View cited PDF pages and highlight supporting text where available.
- **Conversation history:** Store document conversations using SQLite and continue previous conversations.
- **Follow-up questions:** Rewrite contextual follow-up questions into standalone questions before retrieval.

> **Current scope:** Visual understanding is integrated for visual content inside uploaded PDFs, including embedded images and rendered image-based pages. Direct standalone PNG/JPG upload is not documented as an available feature in this version.

## Architecture

```text
                 PDF upload
                     |
          +----------+-----------+
          |                      |
     PDF text extraction    Visual extraction
          |                (embedded images /
     Text chunking          rendered pages)
          |                      |
          |                 Groq Vision
          |                      |
          +----------+-----------+
                     |
          Unified searchable chunks
                     |
          +----------+-----------+
          |                      |
     Semantic retrieval      BM25 retrieval
          |                      |
          +----------+-----------+
                     |
          Reciprocal Rank Fusion
                     |
            Retrieved evidence
                     |
             Groq LLM answer
                     |
       Answer + page-level sources
                     |
       Streamlit conversation UI
```

## Technology stack

- **Python**
- **Streamlit** — user interface
- **PyMuPDF (fitz) / pypdf** — PDF rendering and text extraction
- **Pillow** — image preparation
- **Sentence Transformers** — semantic embeddings (`sentence-transformers/all-MiniLM-L6-v2`)
- **BM25** — keyword retrieval
- **scikit-learn** — cosine similarity
- **Groq API** — answer generation and vision analysis
- **SQLite** — persistent conversation storage

## Project structure

```text
documind-ai/
├── app.py
├── config.py
├── requirements.txt
├── .env.example
├── .gitignore
├── src/
│   └── documind/
│       ├── __init__.py
│       ├── conversation_store.py
│       ├── generation.py
│       ├── semantic_search.py
│       └── vision.py
└── evaluation/
    └── evaluate.py
```

The tree above shows the main application files. Keep it aligned with the files that are actually present in your repository.

## Getting started

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd documind-ai
```

### 2. Create and activate a virtual environment

**macOS / Linux**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

**Windows PowerShell**

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Use the dependency versions recorded in `requirements.txt`. The local development environment was tested with Python 3.11. If you change core numerical-library versions, verify compatibility across NumPy, PyTorch, Pandas, and PyArrow.

### 4. Configure the Groq API key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

Never commit `.env` or publish your API key. The repository should contain a `.env.example` with placeholder values only.

### 5. Run the application

```bash
streamlit run app.py
```

Streamlit will print a local URL in the terminal. Open that URL in your browser, upload a PDF, and ask a question about its text or visual content.

## How visual PDF processing works

1. The application extracts selectable text from PDF pages.
2. It extracts embedded images and renders image-based pages when needed.
3. Images are prepared and sent to the configured Groq vision model.
4. The vision model returns structured descriptions of the visual content.
5. Those descriptions are converted into searchable chunks with document and page metadata.
6. Text and visual chunks are indexed together for semantic and keyword retrieval.
7. Retrieved evidence is passed to the answer-generation model.

Vision API usage is subject to Groq model availability, account quotas, and rate limits. Processing image-heavy PDFs can require multiple vision requests.

## Example questions

- What are the main findings shown in this chart?
- Which category has the largest value in the diagram?
- How many storage racks are visible in the image?
- What sequence is shown in the process diagram?
- What does the table indicate about the reported values?

## Security and files

Before pushing the project to a public repository, make sure the following are excluded from Git:

- `.env` and any file containing API keys or credentials
- `.venv/`
- `__pycache__/` and compiled Python files
- Local SQLite databases (for example, `documind.db`)
- Uploaded PDFs and private documents
- Generated test files or other local data that should not be public

A `.gitignore` file should include at least:

```gitignore
# Secrets
.env
.env.*
!.env.example

# Virtual environments
.venv/
venv/

# Python
__pycache__/
*.py[cod]
.pytest_cache/

# Local databases
*.db
*.sqlite
*.sqlite3

# Uploaded documents and local data
data/
uploads/
*.pdf

# OS / editor
.DS_Store
.vscode/
```

Review this list against your actual project before committing. If you intentionally want to publish a sample PDF, add that specific file explicitly rather than committing all PDFs.

## Limitations

- Visual analysis depends on the quality and resolution of the source PDF images.
- Image-heavy documents may require several vision API calls and can encounter provider rate limits.
- Answers depend on the evidence successfully extracted and retrieved from the uploaded document.
- Direct standalone image upload (PNG/JPG) is not included in the documented feature set for this version.

## Future improvements

- Direct upload and question answering for standalone images.
- More comprehensive evaluation of retrieval quality, answer correctness, and visual-grounding accuracy.
- Stronger evidence-provenance display distinguishing text-derived and image-derived support.
- Additional handling for complex tables, low-resolution scans, and multi-page diagrams.

## License

Choose and add a license before distributing this project publicly. Until a license is added, do not assume others have permission to reuse the code.
