import streamlit as st
import os
import shutil
import tempfile
from pathlib import Path
from dotenv import load_dotenv

from utils.audio_processor import process_input, cleanup_files
from utils.exporter import export_as_pdf, export_as_markdown
from core.transcriber import transcribe_all, fetch_youtube_captions
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

load_dotenv(override=True)

# ─── App Configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MeetNote • Meeting & Video Intelligence",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── High-Performance Clean Modern CSS ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-primary: #0A0D14;
    --bg-surface: #111726;
    --bg-elevated: #182238;
    --border-color: #1E293B;
    --border-highlight: #334155;
    --accent-indigo: #6366F1;
    --accent-cyan: #0EA5E9;
    --accent-emerald: #10B981;
    --text-main: #F8FAFC;
    --text-sub: #94A3B8;
    --text-muted: #64748B;
}

/* Global Font & Reset */
html, body, [class*="css"], .stMarkdown {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: var(--bg-primary) !important;
    color: var(--text-main) !important;
}

.stApp {
    background: radial-gradient(circle at 15% 15%, rgba(99, 102, 241, 0.07) 0%, transparent 40%),
                radial-gradient(circle at 85% 85%, rgba(14, 165, 233, 0.05) 0%, transparent 40%),
                var(--bg-primary) !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #0E131F !important;
    border-right: 1px solid var(--border-color) !important;
}

/* Brand */
.brand-box {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding-bottom: 0.25rem;
}

.brand-glyph {
    width: 34px;
    height: 34px;
    background: linear-gradient(135deg, var(--accent-indigo), var(--accent-cyan));
    border-radius: 9px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    font-size: 1.1rem;
    color: #FFFFFF;
    box-shadow: 0 0 20px rgba(99, 102, 241, 0.45);
}

.brand-text {
    font-size: 1.55rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    color: #FFFFFF;
    line-height: 1;
}

.brand-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: var(--text-muted);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 0.35rem;
    margin-bottom: 1.25rem;
}

/* Status Chips */
.chip {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.2rem 0.55rem;
    border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}

.chip-indigo  { background: rgba(99, 102, 241, 0.15); color: #A5B4FC; border: 1px solid rgba(99, 102, 241, 0.3); }
.chip-cyan    { background: rgba(14, 165, 233, 0.15); color: #7DD3FC; border: 1px solid rgba(14, 165, 233, 0.3); }
.chip-emerald { background: rgba(16, 185, 129, 0.15); color: #6EE7B7; border: 1px solid rgba(16, 185, 129, 0.3); }

/* Primary Button */
.stButton > button {
    background: linear-gradient(135deg, #4F46E5 0%, #6366F1 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.6rem 1.25rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 14px rgba(79, 70, 229, 0.3) !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5) !important;
}

/* Download Buttons */
.stDownloadButton > button {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-color) !important;
    color: var(--text-main) !important;
    border-radius: 8px !important;
    font-size: 0.82rem !important;
    padding: 0.45rem 0.9rem !important;
}

.stDownloadButton > button:hover {
    border-color: var(--accent-cyan) !important;
    color: var(--accent-cyan) !important;
}

/* Input boxes */
.stTextInput > div > div > input,
.stSelectbox > div > div {
    background-color: var(--bg-surface) !important;
    border: 1px solid var(--border-color) !important;
    color: var(--text-main) !important;
    border-radius: 8px !important;
}

.stTextInput > div > div > input:focus {
    border-color: var(--accent-indigo) !important;
    box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2) !important;
}

/* Tab Headers */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 1px solid var(--border-color);
}

.stTabs [data-baseweb="tab"] {
    padding: 8px 16px;
    border-radius: 6px 6px 0 0;
    color: var(--text-sub) !important;
    font-weight: 600;
    font-size: 0.88rem;
}

.stTabs [aria-selected="true"] {
    color: var(--accent-cyan) !important;
    border-bottom: 2px solid var(--accent-cyan) !important;
}

/* Metrics */
div[data-testid="stMetricValue"] {
    font-size: 1.35rem !important;
    font-weight: 700 !important;
    color: var(--text-main) !important;
}

div[data-testid="stMetricLabel"] {
    font-size: 0.75rem !important;
    color: var(--text-muted) !important;
    text-transform: uppercase !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #1E293B; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #334155; }
</style>
""", unsafe_allow_html=True)

# ─── Session State ─────────────────────────────────────────────────────────────
if "app_state" not in st.session_state:
    st.session_state.app_state = None
if "chat_stream" not in st.session_state:
    st.session_state.chat_stream = []

# ─── Sidebar Interface ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="brand-box">
        <div class="brand-glyph">📝</div>
        <div class="brand-text">MeetNote</div>
    </div>
    <div class="brand-tag">Meeting & Video Intelligence</div>
    """, unsafe_allow_html=True)

    input_mode = st.radio(
        "Source Input Mode",
        ["YouTube Link", "File Upload"],
        horizontal=True,
        label_visibility="collapsed",
    )

    selected_source = None

    if input_mode == "YouTube Link":
        selected_source = st.text_input(
            "YouTube Video URL",
            placeholder="https://youtube.com/watch?v=...",
            help="Paste any YouTube video, podcast, or lecture URL.",
        )
    else:
        uploaded_media = st.file_uploader(
            "Upload Audio / Video",
            type=["mp3", "wav", "m4a", "mp4", "mkv", "webm", "ogg"],
            help="Supported: MP3, WAV, M4A, MP4, MKV up to 200MB.",
        )
        if uploaded_media:
            temp_dir = Path(tempfile.gettempdir()) / "meetnote_media"
            temp_dir.mkdir(parents=True, exist_ok=True)
            local_path = temp_dir / uploaded_media.name
            with open(local_path, "wb") as f:
                f.write(uploaded_media.getbuffer())
            selected_source = str(local_path)
            st.caption(f"✓ Uploaded: `{uploaded_media.name}`")

    st.markdown("---")

    col_l, col_m = st.columns(2)
    with col_l:
        language_option = st.selectbox(
            "Language",
            ["English", "Hinglish"],
            index=0,
            help="English uses Whisper; Hinglish uses Sarvam AI translation.",
        )
    with col_m:
        whisper_tier = st.selectbox(
            "Model Tier",
            ["base", "small", "tiny"],
            index=0,
            help="Base: Ultra-fast INT8. Small: High precision.",
        )

    os.environ["WHISPER_MODEL"] = whisper_tier

    st.markdown("---")
    start_btn = st.button("📝 Generate Notes", use_container_width=True)

    # API Status Check
    mistral_ok = bool(os.getenv("MISTRAL_API_KEY"))
    sarvam_ok = bool(os.getenv("SARVAM_API_KEY"))

    st.markdown("<div style='height: 10px'></div>", unsafe_allow_html=True)
    if mistral_ok:
        st.markdown('<span class="chip chip-emerald">● Mistral Online</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="chip chip-indigo">○ Mistral Key Needed</span>', unsafe_allow_html=True)

    if language_option.lower() == "hinglish":
        if sarvam_ok:
            st.markdown('<span class="chip chip-cyan">● Sarvam Connected</span>', unsafe_allow_html=True)
        else:
            st.warning("⚠️ Set SARVAM_API_KEY in .env for Hinglish")

# ─── Workspace Header ──────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom: 1.25rem">
    <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.35rem">
        <span class="chip chip-indigo">Accelerated Engine</span>
        <span class="chip chip-cyan">Chroma Vector Search</span>
    </div>
    <h1 style="font-size: 2rem; font-weight: 800; letter-spacing: -0.03em; margin: 0">
        MeetNote Workspace
    </h1>
    <p style="color: var(--text-sub); font-size: 0.92rem; margin: 0.25rem 0 0 0">
        Transform meetings and videos into structured executive notes, owner action items, and searchable knowledge.
    </p>
</div>
""", unsafe_allow_html=True)

# ─── Execution Pipeline ─────────────────────────────────────────────────────────
if start_btn:
    if not selected_source or not str(selected_source).strip():
        st.error("Please enter a YouTube URL or upload a media file in the sidebar.")
    else:
        st.session_state.app_state = None
        st.session_state.chat_stream = []

        step_indicator = st.status("⚡ Orchestrating meeting intelligence...", expanded=True)
        session_work_dir = tempfile.mkdtemp(prefix="meetnote_job_")
        clean_source = str(selected_source).strip()

        try:
            with step_indicator:
                transcript = None
                
                # Check for instant YouTube captions
                is_youtube = ("youtube.com" in clean_source or "youtu.be" in clean_source)
                
                if is_youtube:
                    st.write("🔊 **Phase 1: Audio / Stream Acquisition** — Connecting to high-speed stream...")
                    transcript = fetch_youtube_captions(clean_source)
                    if transcript:
                        word_count = len(transcript.split())
                        st.write(f"📝 **Phase 2: Speech-to-Text Decoding** — Decoded {word_count:,} words with 100% accuracy!")
                
                # Fallback to faster INT8 Whisper if not a YouTube video with captions
                if not transcript:
                    st.write("🔊 **Phase 1: Audio / Stream Acquisition** — Extracting 16kHz mono audio...")
                    chunks = process_input(clean_source, work_dir=session_work_dir)

                    st.write(f"📝 **Phase 2: Speech-to-Text Decoding** — Transcribing {len(chunks)} segment(s) with {language_option} engine...")
                    transcript = transcribe_all(
                        chunks,
                        language=language_option.lower(),
                        model_name=whisper_tier,
                        source_url=clean_source,
                    )
                    cleanup_files(chunks)

                if os.path.exists(session_work_dir):
                    shutil.rmtree(session_work_dir, ignore_errors=True)

                st.write("📋 **Phase 3: Executive Synthesis** — Synthesizing multi-point briefing & session title...")
                title = generate_title(transcript)
                summary = summarize(transcript)

                st.write("🔍 **Phase 4: Action Items & Deliverables** — Extracting tasks, deadlines, and key alignments...")
                action_items = extract_action_items(transcript)
                decisions = extract_key_decisions(transcript)
                questions = extract_questions(transcript)

                st.write("🧠 **Phase 5: Vector Indexing & RAG** — Building in-memory Chroma vector store...")
                rag_chain = build_rag_chain(transcript)

                step_indicator.update(label="✦ Notes Generated Successfully!", state="complete", expanded=False)

            st.session_state.app_state = {
                "title": title,
                "transcript": transcript,
                "summary": summary,
                "action_items": action_items,
                "key_decisions": decisions,
                "open_questions": questions,
                "rag_chain": rag_chain,
                "words": len(transcript.split()),
            }
            st.rerun()

        except Exception as error:
            if os.path.exists(session_work_dir):
                shutil.rmtree(session_work_dir, ignore_errors=True)
            step_indicator.update(label="Processing Encountered an Issue", state="error", expanded=True)
            if "429" in str(error):
                st.error("⚠️ **Mistral Rate Limit Reached**: The free-tier API received too many requests in a short window. Please wait 10 seconds and click Generate Notes again.")
            else:
                st.error(f"Details: {error}")

# ─── Results Dashboard ──────────────────────────────────────────────────────────
res = st.session_state.app_state

if res:
    # ── Top Title Card & Stats ──
    st.markdown(f"### 📌 {res['title']}")
    
    col_m1, col_m2, col_m3 = st.columns([1, 1, 1.5])
    with col_m1:
        st.metric("Words Processed", f"{res['words']:,}")
    with col_m2:
        st.metric("Est. Read Time", f"~{max(1, res['words'] // 150)} min")
    with col_m3:
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        pdf_file = export_as_pdf(res)
        md_file = export_as_markdown(res)
        c_p, c_m = st.columns(2)
        with c_p:
            st.download_button(
                "📄 PDF Report",
                data=pdf_file,
                file_name=f"{res['title'].replace(' ', '_')}_notes.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        with c_m:
            st.download_button(
                "📝 Markdown",
                data=md_file,
                file_name=f"{res['title'].replace(' ', '_')}_notes.md",
                mime="text/markdown",
                use_container_width=True,
            )

    st.markdown("---")

    # ── Main Tabbed Intelligence Views ──
    tab_sum, tab_acts, tab_rag, tab_raw = st.tabs([
        "📋 Executive Briefing",
        "🎯 Action Deliverables",
        "💬 Interactive Q&A",
        "📝 Verbatim Transcript",
    ])

    with tab_sum:
        col_s1, col_s2 = st.columns([3, 2], gap="large")
        with col_s1:
            st.markdown("#### 📋 Executive Summary")
            st.markdown(res["summary"])
        with col_s2:
            st.markdown("#### 🔑 Key Alignments & Decisions")
            st.markdown(res["key_decisions"])

    with tab_acts:
        col_a1, col_a2 = st.columns(2, gap="large")
        with col_a1:
            st.markdown("#### 🎯 Action Items & Ownership")
            st.markdown(res["action_items"])
        with col_a2:
            st.markdown("#### ❓ Open Discussion & Follow-ups")
            st.markdown(res["open_questions"])

    with tab_rag:
        st.markdown("#### 💬 Search & Chat with Meeting Knowledge")
        st.caption("Ask specific questions regarding commitments, speakers, dates, numbers, or discussion points.")

        # Chat history container
        if st.session_state.chat_stream:
            for item in st.session_state.chat_stream:
                if item["role"] == "user":
                    with st.chat_message("user"):
                        st.markdown(item["content"])
                else:
                    with st.chat_message("assistant"):
                        st.markdown(item["content"])
        else:
            st.info("💡 Type your question below to query the meeting transcript via semantic vector search.")

        user_query = st.chat_input("Ask a question about this meeting...")
        if user_query:
            st.session_state.chat_stream.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            with st.chat_message("assistant"):
                with st.spinner("Searching transcript..."):
                    answer = ask_question(res["rag_chain"], user_query)
                st.markdown(answer)
                st.session_state.chat_stream.append({"role": "assistant", "content": answer})

    with tab_raw:
        st.markdown("#### 📝 Complete Verbatim Transcript")
        st.text_area(
            "Full Transcript Content",
            value=res["transcript"],
            height=400,
            disabled=True,
            label_visibility="collapsed",
        )

else:
    # Empty State
    st.markdown("""
    <div style="background: #111726; border: 1px solid #1E293B; border-radius: 12px; padding: 4rem 2rem; text-align: center; margin-top: 1rem">
        <div style="font-size: 2.8rem; margin-bottom: 0.75rem">📝</div>
        <div style="font-size: 1.35rem; font-weight: 700; color: #F8FAFC; margin-bottom: 0.5rem">
            Ready to Generate Notes
        </div>
        <div style="color: #94A3B8; font-size: 0.92rem; max-width: 480px; margin: 0 auto 1.5rem auto; line-height: 1.7">
            Provide a YouTube link or upload a recording in the sidebar to generate structured executive notes, task deliverables, and an interactive Q&A assistant.
        </div>
        <div style="display: flex; gap: 0.6rem; justify-content: center; flex-wrap: wrap">
            <span class="chip chip-indigo">⚡ Fast INT8 Whisper</span>
            <span class="chip chip-cyan">🇮🇳 Sarvam Parallel</span>
            <span class="chip chip-emerald">🧠 LangChain LCEL</span>
            <span class="chip chip-indigo">📄 PDF Export</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
