"""
Generates a professional-looking PDF version of the AI-drafted clinical
report — the X-ray thumbnail, a findings table, the report text, and a
disclaimer footer.
"""
import io
import re
from datetime import datetime
from typing import Dict

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

PRIMARY_COLOR = colors.HexColor("#1F4E79")
ACCENT_COLOR = colors.HexColor("#2E86AB")
LIGHT_GREY = colors.HexColor("#F2F4F7")


def _markdown_to_paragraphs(text: str, styles) -> list:
    """Very small markdown-ish converter: handles ## headers, **bold**,
    and bullet lines — enough for the Report Agent's output format."""
    story = []
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            story.append(Spacer(1, 6))
            continue

        if line.startswith("## "):
            story.append(Spacer(1, 10))
            story.append(Paragraph(line[3:].strip(), styles["SectionHeader"]))
            story.append(Spacer(1, 4))
            continue

        # bold: **text** -> <b>text</b>
        line = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", line)

        if line.startswith("- "):
            story.append(Paragraph(f"&bull;&nbsp;&nbsp;{line[2:].strip()}", styles["Bullet"]))
        else:
            story.append(Paragraph(line, styles["Body"]))

    return story


def generate_pdf_report(
    image_path: str,
    predictions: Dict[str, float],
    report_text: str,
    patient_label: str = "N/A",
) -> bytes:
    """Returns the PDF as raw bytes, ready for a Streamlit download button."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=letter,
        topMargin=0.6 * inch, bottomMargin=0.6 * inch,
        leftMargin=0.7 * inch, rightMargin=0.7 * inch,
    )

    base_styles = getSampleStyleSheet()
    styles = {
        "Title": ParagraphStyle(
            "TitleStyle", parent=base_styles["Title"],
            textColor=PRIMARY_COLOR, fontSize=20, spaceAfter=2,
        ),
        "Subtitle": ParagraphStyle(
            "SubtitleStyle", parent=base_styles["Normal"],
            textColor=colors.HexColor("#666666"), fontSize=10, spaceAfter=14,
        ),
        "SectionHeader": ParagraphStyle(
            "SectionHeaderStyle", parent=base_styles["Heading2"],
            textColor=PRIMARY_COLOR, fontSize=13, spaceBefore=6,
        ),
        "Body": ParagraphStyle(
            "BodyStyle", parent=base_styles["Normal"],
            fontSize=10, leading=15,
        ),
        "Bullet": ParagraphStyle(
            "BulletStyle", parent=base_styles["Normal"],
            fontSize=10, leading=15, leftIndent=10,
        ),
        "Disclaimer": ParagraphStyle(
            "DisclaimerStyle", parent=base_styles["Normal"],
            fontSize=8, textColor=colors.HexColor("#888888"),
            leading=11, spaceBefore=16,
        ),
        "MetaLabel": ParagraphStyle(
            "MetaLabelStyle", parent=base_styles["Normal"],
            fontSize=9, textColor=colors.HexColor("#555555"),
        ),
    }

    story = []

    # --- Header ---
    story.append(Paragraph("CheXpert AI Diagnostic Assistant", styles["Title"]))
    story.append(Paragraph(
        "AI-Generated Chest X-ray Analysis Report &mdash; Research/Educational Use Only",
        styles["Subtitle"],
    ))

    meta_table = Table(
        [[
            Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["MetaLabel"]),
            Paragraph(f"<b>Reference:</b> {patient_label}", styles["MetaLabel"]),
        ]],
        colWidths=[3.2 * inch, 3.2 * inch],
    )
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_GREY),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 14))

    # --- X-ray thumbnail + prediction table side by side ---
    try:
        thumb = RLImage(image_path, width=2.3 * inch, height=2.3 * inch)
    except Exception:
        thumb = Paragraph("(image preview unavailable)", styles["Body"])

    pred_rows = [["Condition", "Probability"]]
    for label, prob in sorted(predictions.items(), key=lambda x: -x[1]):
        pred_rows.append([label, f"{prob:.2f}"])

    pred_table = Table(pred_rows, colWidths=[2.4 * inch, 1.2 * inch])
    pred_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PRIMARY_COLOR),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GREY]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DDDDDD")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("ALIGN", (1, 0), (1, -1), "CENTER"),
    ]))

    top_row = Table(
        [[thumb, pred_table]],
        colWidths=[2.6 * inch, 3.8 * inch],
    )
    top_row.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.append(top_row)
    story.append(Spacer(1, 10))

    # --- Report body (parsed from the Report Agent's markdown-ish text) ---
    story.extend(_markdown_to_paragraphs(report_text, styles))

    # --- Footer disclaimer (in case the Report Agent's own disclaimer got trimmed) ---
    story.append(Paragraph(
        "This document was generated by an AI system and is intended as a draft "
        "for review by a qualified radiologist or physician. It is not a medical "
        "diagnosis and must not be used as the sole basis for clinical decisions.",
        styles["Disclaimer"],
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()
