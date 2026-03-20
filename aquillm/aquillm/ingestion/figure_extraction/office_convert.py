"""Office document to PDF conversion using LibreOffice headless."""

import logging
import os
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

# Supported formats for conversion
CONVERTIBLE_FORMATS = {
    "docx": "writer",
    "doc": "writer",
    "odt": "writer",
    "rtf": "writer",
    "pptx": "impress",
    "ppt": "impress",
    "odp": "impress",
    "xlsx": "calc",
    "xls": "calc",
    "ods": "calc",
}

# LibreOffice command (try soffice first, then libreoffice)
LIBREOFFICE_COMMANDS = ["soffice", "libreoffice"]


def _find_libreoffice() -> str | None:
    """Find the LibreOffice executable."""
    for cmd in LIBREOFFICE_COMMANDS:
        try:
            result = subprocess.run(
                [cmd, "--version"],
                capture_output=True,
                timeout=10,
            )
            if result.returncode == 0:
                return cmd
        except (subprocess.SubprocessError, FileNotFoundError):
            continue
    return None


def convert_to_pdf(data: bytes, source_format: str, filename: str = "") -> bytes | None:
    """
    Convert an Office document to PDF using LibreOffice headless.
    
    Args:
        data: Raw document bytes
        source_format: File extension (e.g., 'docx', 'pptx', 'xlsx')
        filename: Optional filename for logging
        
    Returns:
        PDF bytes if successful, None otherwise
    """
    source_format = source_format.lower().strip().lstrip(".")
    
    if source_format not in CONVERTIBLE_FORMATS:
        logger.debug("Format %s not supported for conversion", source_format)
        return None
    
    libreoffice_cmd = _find_libreoffice()
    if not libreoffice_cmd:
        logger.warning("LibreOffice not found; cannot convert %s to PDF", source_format)
        return None
    
    # Use a temporary directory for conversion
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write input file
        input_filename = f"input.{source_format}"
        input_path = Path(tmpdir) / input_filename
        input_path.write_bytes(data)
        
        # Run LibreOffice conversion
        try:
            result = subprocess.run(
                [
                    libreoffice_cmd,
                    "--headless",
                    "--convert-to", "pdf",
                    "--outdir", tmpdir,
                    str(input_path),
                ],
                capture_output=True,
                timeout=120,  # 2 minute timeout
                cwd=tmpdir,
            )
            
            if result.returncode != 0:
                logger.warning(
                    "LibreOffice conversion failed for %s: %s",
                    filename or source_format,
                    result.stderr.decode("utf-8", errors="replace")[:500],
                )
                return None
            
        except subprocess.TimeoutExpired:
            logger.warning("LibreOffice conversion timed out for %s", filename or source_format)
            return None
        except subprocess.SubprocessError as exc:
            logger.warning("LibreOffice conversion error for %s: %s", filename or source_format, exc)
            return None
        
        # Find the output PDF
        output_path = Path(tmpdir) / "input.pdf"
        if not output_path.exists():
            # Try finding any PDF in the directory
            pdf_files = list(Path(tmpdir).glob("*.pdf"))
            if pdf_files:
                output_path = pdf_files[0]
            else:
                logger.warning("No PDF output found after conversion of %s", filename or source_format)
                return None
        
        pdf_bytes = output_path.read_bytes()
        
        if len(pdf_bytes) < 100:
            logger.warning("PDF output too small for %s", filename or source_format)
            return None
        
        logger.debug(
            "Successfully converted %s to PDF (%d bytes)",
            filename or source_format,
            len(pdf_bytes),
        )
        return pdf_bytes


def is_libreoffice_available() -> bool:
    """Check if LibreOffice is available for document conversion."""
    return _find_libreoffice() is not None
