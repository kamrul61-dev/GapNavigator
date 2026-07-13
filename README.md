# GapNavigator Lite – AI Career Transition Planner

GapNavigator Lite is a clean, modular, production-ready AI application designed to help students and career changers transition into new technical roles. By analyzing a candidate's resume (PDF) and comparing it against a target job description, the app computes a job readiness score, identifies key skill gaps, and generates a personalized, week-by-week 30-day learning roadmap complete with vetted study resources.

This application is built using modern AI engineering patterns, featuring structured Pydantic data validation, a modular LangGraph agent orchestrator, and an offline-cached FAISS vector store (RAG pipeline) to suggest actual learning materials without LLM hallucinations.

---

## Key Features

1. **Structured Resume Parsing**: Extracts name, contact info, education, and career experience from uploaded PDF files using **PyMuPDF** and structured Pydantic outputs.
2. **Skill Gap Analysis**: Compares candidate's skills against job requirements, classifying them into *Match*, *Gap*, and *Suggested (Nice-to-have)*.
3. **Job Readiness Score**: Calculates a readiness percentage (0-100%) and explains key strengths and weaknesses.
4. **30-Day Learning Roadmap**: Generates a structured 4-week study plan consisting of weekly goals, specific topics, and practical projects/tasks.
5. **RAG Recommendations**: Queries a local FAISS database of pre-populated learning materials (documentation, courses, playlists) to output real links for the missing skills, avoiding hallucinated recommendations.
6. **Execution Tracing**: Integrates **LangSmith** to trace agent execution graphs and LLM prompt/response performance.

---

## Project Structure

```
GapNavigator/
├── app/
│   ├── main.py                 # Streamlit UI & Orchestration Layer
│   ├── agents/
│   │   ├── state.py            # LangGraph shared State definitions
│   │   ├── resume_agent.py     # Resume extraction node
│   │   ├── gap_agent.py        # Skill gap analysis node
│   │   ├── roadmap_agent.py    # Learning roadmap & RAG recommendation node
│   │   └── workflow.py         # Graph compilation and orchestrator
│   ├── models/
│   │   └── schemas.py          # Pydantic validation schemas
│   ├── parsers/
│   │   └── pdf_parser.py       # PyMuPDF resume parsing engine
│   ├── prompts/
│   │   ├── resume_extract.txt  # System prompts (externalized)
│   │   ├── gap_analysis.txt
│   │   └── roadmap_gen.txt
│   ├── rag/
│   │   ├── knowledge_base/     # Text/Markdown files containing vetted courses
│   │   │   └── vetted_resources.md
│   │   └── retriever.py        # FAISS local indexing and retrieval
│   └── utils/
│       └── logger.py           # Logging setup
├── tests/                      # Pytest unit tests
│   ├── test_parser.py
│   ├── test_rag.py
│   └── test_workflow.py
├── requirements.txt            # Project dependencies
├── .env.example                # Sample environment variables
└── .gitignore
```

---

## Installation & Setup

### Prerequisites
- Python 3.10 or 3.11
- A Google Gemini API Key (obtained from [Google AI Studio](https://aistudio.google.com/))

### 1. Clone the repository and install dependencies
```bash
# Navigate to the project folder
cd GapNavigator

# Create a virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Create a `.env` file in the root directory by copying the sample:
```bash
cp .env.example .env
```
Open `.env` and fill in your details:
```ini
# Google Gemini API Key (Required)
GOOGLE_API_KEY=your-gemini-api-key-here

# Google Gemini Model Name (Optional, defaults to gemini-3.5-flash)
GEMINI_MODEL=gemini-3.5-flash

# Google Gemini Embedding Model Name (Optional, defaults to models/gemini-embedding-2)
GEMINI_EMBEDDING_MODEL=models/gemini-embedding-2

# LangSmith Tracing (Optional - set to true to monitor logs in LangSmith)
LANGCHAIN_TRACING_V2=false
LANGCHAIN_API_KEY=your-langsmith-api-key-here
LANGCHAIN_PROJECT="gapnavigator-lite"
```

---

## How to Run

### Run the Web Interface
Start the Streamlit application using:
```bash
streamlit run app/main.py
```
This will open the interface in your default browser at `http://localhost:8501`.

### Run Unit Tests
To run the test suite (fully offline-compatible, no API keys required):
```bash
pytest
```

---

## LangGraph Multi-Agent Architecture
The application runs a sequential processing workflow managed by LangGraph:
1. **Resume Agent**: Reads parsed PDF text and extracts candidate metadata, education, and skills.
2. **Gap Analysis Agent**: Compares extracted data against the raw job description to calculate readiness, strengths/weaknesses, and missing skills.
3. **Roadmap Agent**: Retrieves vetted learning documents from the FAISS database matching the missing skills and outputs a structured weekly learning plan with real hyperlinked resources.

---

## Future Improvements
- Support parsing Word documents (`.docx`).
- Add user authentication and history logging to save past transition roadmaps.
- Implement an interactive checklist for users to track their progress along the 30-day roadmap.
