# 📝 MeetNote — Video & Meeting Intelligence

An autonomous platform that converts meetings, video presentations, and voice recordings into structured executive notes, action deliverables, key decisions, and searchable knowledge.

---

## ✨ Capabilities

- 🎥 **Dual Input Sources**: Process audio/video directly from **YouTube URLs** or local media uploads (`.mp3`, `.wav`, `.m4a`, `.mp4`, `.mkv`, `.webm`).
- 🎙️ **Dual Transcription Engines**:
  - **OpenAI Whisper & Faster-Whisper (INT8)** for ultra-fast local decoding.
  - **Instant YouTube Caption Extraction (~2s)** for multi-hour recordings.
  - **Sarvam AI (`saaras:v2.5`)** with parallel workers for Hindi / Hinglish translation.
- 📋 **Executive Briefing**: Structured synthesis powered by Mistral AI (`open-mistral-7b`).
- 🎯 **Action Deliverables & Alignments**: Extract tasks, assigned owners, deadlines, and key decisions.
- 💬 **Semantic Q&A**: In-memory ephemeral ChromaDB vector search & LangChain LCEL RAG.
- 📄 **Export Reports**: Instant export to **PDF Reports** and **Markdown Notes**.
- 🚀 **Cloud Deployable**: 1-click deploy to Render (FastAPI or Streamlit) with zero storage leaks.

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Backend API**: FastAPI & Uvicorn
- **LLM Orchestration**: LangChain LCEL (`langchain-core`, `langchain-mistralai`)
- **LLM Provider**: Mistral AI (`open-mistral-7b`)
- **Vector Store & Embeddings**: ChromaDB (Ephemeral In-Memory) & HuggingFace (`all-MiniLM-L6-v2`)
- **STT Engines**: OpenAI Faster-Whisper, YouTube Transcript API, Sarvam AI
- **Media Engine**: `yt-dlp`, `pydub`, `static-ffmpeg`
- **PDF Generation**: ReportLab

---

## 🚀 Quickstart

### 1. Clone & Setup
```bash
git clone https://github.com/Raj-Rathod-Ai/MeetNotes.git
cd MeetNotes

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Create a `.env` file in the root directory:
```env
MISTRAL_API_KEY=your_mistral_api_key
SARVAM_API_KEY=your_sarvam_api_key
SARVAM_STT_MODEL=saaras:v2.5
MISTRAL_MODEL=open-mistral-7b
HF_HUB_DISABLE_SYMLINKS_WARNING=1
```

### 3. Run Web App or API
```bash
# Run Streamlit UI
streamlit run app.py

# Or Run FastAPI REST Backend
uvicorn api:app --reload --port 8000
```
Open **`http://localhost:8501`** in your browser.

---

## ☁️ Deployment on Render

This repository includes pre-configured [render.yaml](file:///c:/RAG/Video-Agent/render.yaml) blueprints:
1. Connect your GitHub repository to **Render**.
2. Set your environment variables (`MISTRAL_API_KEY`, `SARVAM_API_KEY`).
3. Deploy! Automatic ephemeral audio purging keeps disk usage at **0 MB**.
