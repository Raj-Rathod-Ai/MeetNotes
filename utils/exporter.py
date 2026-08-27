import io
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


def export_as_markdown(data: dict) -> str:
    """Formats all meeting intelligence results into a clean markdown document."""
    md = []
    md.append(f"# 🎬 Meeting Notes: {data.get('title', 'Untitled Meeting')}\n")
    
    md.append("## 📋 Summary")
    md.append(data.get("summary", "No summary available.") + "\n")
    
    md.append("## ✅ Action Items")
    md.append(data.get("action_items", "No action items found.") + "\n")
    
    md.append("## 🔑 Key Decisions")
    md.append(data.get("key_decisions", "No key decisions found.") + "\n")
    
    md.append("## ❓ Open Questions & Follow-ups")
    md.append(data.get("open_questions", "No open questions found.") + "\n")
    
    md.append("## 📝 Full Transcript")
    md.append(data.get("transcript", "No transcript available.") + "\n")
    
    return "\n".join(md)


def export_as_pdf(data: dict) -> bytes:
    """Generates a professional PDF document of the meeting notes."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        "MeetingTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#7c3aed"),
        spaceAfter=12,
    )
    
    h2_style = ParagraphStyle(
        "SectionHeader",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#06b6d4"),
        spaceBefore=14,
        spaceAfter=6,
    )
    
    body_style = ParagraphStyle(
        "MeetingBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=8,
    )

    story = []

    # Title
    meeting_title = data.get("title", "Meeting Notes")
    story.append(Paragraph(meeting_title, title_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e8f0"), spaceAfter=12))

    # Sections helper
    def add_section(header_text, content_text):
        story.append(Paragraph(header_text, h2_style))
        paragraphs = content_text.split("\n")
        for p in paragraphs:
            p_clean = p.strip()
            if p_clean:
                # Escape XML sensitive chars for reportlab
                p_safe = p_clean.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                story.append(Paragraph(p_safe, body_style))
        story.append(Spacer(1, 8))

    add_section("📋 Summary", data.get("summary", "N/A"))
    add_section("✅ Action Items", data.get("action_items", "N/A"))
    add_section("🔑 Key Decisions", data.get("key_decisions", "N/A"))
    add_section("❓ Open Questions", data.get("open_questions", "N/A"))
    
    # Transcript snippet
    transcript = data.get("transcript", "")
    if transcript:
        add_section("📝 Full Transcript", transcript)

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
