# 🧠 Adaptive RAG (Retrieval-Augmented Generation)

A clean, minimalist implementation of **Adaptive RAG** built with standard Python concepts.

---

## 🎯 What is Adaptive RAG?

In a standard RAG system, **every single question** is sent to a document search database. Even if you simply say *"Hello!"* or ask *"What is 2 + 2?"*, standard RAG searches through your PDFs and retrieves irrelevant chunks before answering.

**Adaptive RAG** is smarter:
Like a student who only opens their textbook when they actually need it:
1. **Direct Path (No Retrieval):** If you ask a greeting, general knowledge, or conversational question, it answers immediately using the LLM.
2. **Retrieval Path:** If you ask a question specifically related to your uploaded document, it finds relevant chunks using semantic search and answers using those facts.
3. **Low-Confidence / Fallback Path:** If it searches your document and finds that no chunks match your question closely, it warns you honestly instead of guessing or hallucinating facts.

---

## 📂 Project Structure & File Purpose

Every file is written with simple functions, lists, dicts, and loops—no complex abstractions or heavy frameworks.

| File | Purpose |
| :--- | :--- |
| [`app.py`](app.py) | **The User Interface:** Built with Streamlit. Handles file uploads, chat interface, toggles for Mock Mode, and visual badges showing which route Adaptive RAG took. |
| [`document_processor.py`](document_processor.py) | **Document Reading & Chunking:** Extracts text page-by-page from PDFs using PyMuPDF and splits the text into clean, overlapping word chunks using basic Python loops. |
| [`retriever.py`](retriever.py) | **Semantic Search:** Converts text chunks into numerical vectors using Sentence Transformers, and finds the most relevant chunks using simple NumPy cosine similarity math (no external vector database needed). |
| [`adaptive_rag.py`](adaptive_rag.py) | **The Decision Brain:** Analyzes the user's question, decides whether retrieval is needed, checks similarity confidence, and coordinates the final answer. |
| [`llm_service.py`](llm_service.py) | **AI & Mock Service:** Communicates with the Google Gemini API. Includes a built-in Mock Mode so you can test everything without needing an API key or internet connection. |
| [`requirements.txt`](requirements.txt) | **Dependencies List:** Contains only the essential, lightweight packages needed for this project. |
| [`.env.example`](.env.example) | **Configuration Template:** A sample configuration file where you can define your `GEMINI_API_KEY` and toggle `MOCK_MODE`. |
| [`tests/`](tests/) | **Test Folder:** Contains lightweight verification scripts (like `test_smoke.py`) to confirm imports and mock mode behavior. |

---

## 🛠️ Setup Instructions

### 1. Prerequisites
- Python 3.9+ installed on your system.

### 2. Create and Activate a Virtual Environment
Open your terminal (PowerShell or Bash) and navigate to the project directory:

```powershell
cd C:\Users\tanis\.gemini\antigravity\scratch\adaptive_rag
python -m venv venv
```

Activate the virtual environment:
- **Windows (PowerShell):**
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
- **macOS / Linux:**
  ```bash
  source venv/bin/activate
  ```

### 3. Install Requirements
```powershell
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to a new file named `.env`:

```powershell
copy .env.example .env
```

Open `.env` and configure your settings:
- **To test without an API key:**
  ```ini
  MOCK_MODE=True
  GEMINI_API_KEY=
  ```
- **To use real Gemini AI responses:**
  1. Get a free API key from [Google AI Studio](https://aistudio.google.com/).
  2. Put your key in `.env`:
     ```ini
     GEMINI_API_KEY=your_actual_key_here
     MOCK_MODE=False
     ```

### 5. Run the Smoke Test
Verify that your environment and files are ready:

```powershell
python -m tests.test_smoke
```

### 6. Launch the Streamlit App
```powershell
streamlit run app.py
```
This opens the web interface in your default browser at `http://localhost:8501`.
