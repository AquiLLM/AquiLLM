"""Rebuild the synthetic upload fixture (requires reportlab, used only here)."""

from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen.canvas import Canvas

OUTPUT = Path(__file__).with_name("synthetic-upload.pdf")


def main():
    canvas = Canvas(str(OUTPUT), pagesize=letter, invariant=1)
    canvas.setTitle("AquiLLM synthetic upload fixture")
    canvas.setAuthor("AquiLLM contributors")
    canvas.setFont("Helvetica-Bold", 18)
    canvas.drawString(54, 738, "AquiLLM upload test")
    canvas.setFont("Helvetica", 11)
    lines = [
        "This document contains original synthetic test content.",
        "It is used to exercise PDF upload, ingestion, and collection navigation.",
        "The fictional catalog has three items: red notebook, blue folder, green pen.",
        "The notebook has 120 pages. The folder holds 25 sheets. The pen uses blue ink.",
        "These facts are test data and do not describe any real person or organization.",
        "No third-party textbook, figures, or font files are embedded in this fixture.",
    ]
    for index, line in enumerate(lines):
        canvas.drawString(54, 700 - index * 24, line)
    canvas.showPage()
    canvas.save()


if __name__ == "__main__":
    main()
