"""
PancrAI — PDF Export Utility
Converts HTML diagnostic reports to downloadable PDF files.
Uses ReportLab as the primary engine with an html2pdf fallback.
"""

import io
import os
import re
from datetime import datetime
from typing import Optional


def html_to_pdf_bytes(html_content: str) -> bytes:
    """
    Convert an HTML report string to a PDF byte stream.

    Tries ReportLab for rich formatting. Falls back to a plain-text
    PDF if the HTML is too complex or ReportLab is unavailable.

    Args:
        html_content: Full HTML string (e.g., from gemini_report).

    Returns:
        Raw PDF bytes ready for st.download_button or HTTP response.
    """
    try:
        return _reportlab_pdf(html_content)
    except Exception as e:
        print(f"[PDF] ReportLab failed ({e}), using plain-text fallback")
        return _plaintext_pdf(html_content)


def _strip_html(html: str) -> str:
    """Remove HTML tags and decode common entities."""
    text = re.sub(r"<style[^>]*>.*?</style>", " ", html, flags=re.DOTALL)
    text = re.sub(r"<script[^>]*>.*?</script>", " ", text, flags=re.DOTALL)
    text = re.sub(r"<br\s*/?>", "\n", text)
    text = re.sub(r"<p[^>]*>", "\n", text)
    text = re.sub(r"</p>", "\n", text)
    text = re.sub(r"<li[^>]*>", "\n  • ", text)
    text = re.sub(r"<h[1-6][^>]*>", "\n### ", text)
    text = re.sub(r"</h[1-6]>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    # Entities
    entities = {
        "&amp;": "&", "&lt;": "<", "&gt;": ">",
        "&nbsp;": " ", "&#39;": "'", "&quot;": '"',
    }
    for ent, char in entities.items():
        text = text.replace(ent, char)
    # Collapse whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _reportlab_pdf(html_content: str) -> bytes:
    """
    Generate a styled PDF using ReportLab.
    Parses the HTML sections from the PancrAI report format.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.colors import HexColor, white, black
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, HRFlowable,
        Table, TableStyle, KeepTogether,
    )
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    style_title = ParagraphStyle(
        "Title",
        parent=styles["Normal"],
        fontSize=18,
        fontName="Helvetica-Bold",
        textColor=HexColor("#1565C0"),
        alignment=TA_CENTER,
        spaceAfter=4,
    )
    style_subtitle = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontSize=10,
        textColor=HexColor("#555555"),
        alignment=TA_CENTER,
        spaceAfter=12,
    )
    style_section = ParagraphStyle(
        "Section",
        parent=styles["Normal"],
        fontSize=12,
        fontName="Helvetica-Bold",
        textColor=HexColor("#1565C0"),
        spaceBefore=14,
        spaceAfter=6,
        borderPad=4,
        borderColor=HexColor("#1565C0"),
        borderWidth=0,
        leftIndent=0,
    )
    style_body = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=10,
        leading=15,
        textColor=HexColor("#333333"),
        spaceAfter=4,
        alignment=TA_JUSTIFY,
    )
    style_bullet = ParagraphStyle(
        "Bullet",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        textColor=HexColor("#333333"),
        leftIndent=16,
        bulletIndent=8,
        spaceAfter=2,
    )
    style_disclaimer = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontSize=8,
        textColor=HexColor("#888888"),
        alignment=TA_CENTER,
        spaceBefore=20,
    )

    story = []

    # ── Header ──
    story.append(Paragraph("PancrAI — AI-Assisted Radiology Report", style_title))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')} | For Research Use Only",
        style_subtitle,
    ))
    story.append(HRFlowable(width="100%", thickness=2,
                              color=HexColor("#1565C0"), spaceAfter=12))

    # ── Parse sections from stripped HTML ──
    plain_text = _strip_html(html_content)

    # Split on section headers (### Section Title)
    section_pattern = re.compile(r"###\s*(.+?)\n(.*?)(?=###|\Z)", re.DOTALL)
    matches = list(section_pattern.finditer(plain_text))

    if matches:
        for match in matches:
            heading = match.group(1).strip()
            content = match.group(2).strip()

            story.append(Paragraph(heading, style_section))
            story.append(HRFlowable(width="100%", thickness=0.5,
                                     color=HexColor("#CCCCCC"), spaceAfter=6))

            # Split into bullets and paragraphs
            for line in content.split("\n"):
                line = line.strip()
                if not line:
                    continue
                if line.startswith("•") or line.startswith("-"):
                    story.append(Paragraph(f"• {line.lstrip('•- ').strip()}", style_bullet))
                else:
                    story.append(Paragraph(line, style_body))

            story.append(Spacer(1, 6))
    else:
        # No sections found — just render plain text
        for line in plain_text.split("\n"):
            if line.strip():
                story.append(Paragraph(line.strip(), style_body))

    # ── Disclaimer ──
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=0.5, color=HexColor("#CCCCCC")))
    story.append(Paragraph(
        "⚠ DISCLAIMER: This report was generated by an AI system and is intended for "
        "informational and research purposes only. It must be reviewed and validated by a "
        "licensed radiologist or physician before any clinical decision-making. "
        "PancrAI is NOT FDA/CE approved and does NOT replace professional medical judgment.",
        style_disclaimer,
    ))

    doc.build(story)
    return buffer.getvalue()


def _plaintext_pdf(html_content: str) -> bytes:
    """
    Ultra-simple PDF fallback using only ReportLab basic canvas.
    Works even if Platypus has issues.
    """
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4

        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=A4)
        W, H = A4

        plain = _strip_html(html_content)
        lines = plain.split("\n")

        c.setFont("Helvetica-Bold", 14)
        c.drawString(40, H - 50, "PancrAI — Radiology Report")
        c.setFont("Helvetica", 8)
        c.drawString(40, H - 65, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

        c.setFont("Helvetica", 10)
        y = H - 90
        for line in lines:
            if not line.strip():
                y -= 6
                continue
            if y < 60:
                c.showPage()
                y = H - 50
                c.setFont("Helvetica", 10)
            c.drawString(40, y, line[:100])  # truncate long lines
            y -= 14

        c.setFont("Helvetica", 7)
        c.drawString(40, 40, "DISCLAIMER: AI-generated report — for research use only. Not for clinical decision-making.")

        c.save()
        return buffer.getvalue()

    except Exception as e:
        # Last resort: return a minimal valid PDF string
        raise RuntimeError(f"PDF generation completely failed: {e}")


def save_report_pdf(html_content: str, output_path: str) -> str:
    """
    Generate and save a PDF report to disk.

    Args:
        html_content: HTML report string.
        output_path: File path to write the PDF.

    Returns:
        Absolute path to the saved PDF.
    """
    pdf_bytes = html_to_pdf_bytes(html_content)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(pdf_bytes)
    return os.path.abspath(output_path)
