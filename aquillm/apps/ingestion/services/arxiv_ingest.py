"""ArXiv fetch + document creation for API ingestion."""
from __future__ import annotations

import gzip
import io
import structlog
import tarfile
from typing import Any
from xml.dom import minidom

import chardet
import requests
from django.core.files.base import ContentFile

from apps.collections.models import Collection
from apps.documents.models import PDFDocument, TeXDocument

logger = structlog.stdlib.get_logger(__name__)


def insert_one_from_arxiv(arxiv_id: str, collection: Collection, user: Any) -> dict:
    """Ingest a paper from arXiv; returns ``{"message": str, "errors": list[str]}``."""

    def save_pdf_doc(content: bytes, title: str) -> None:
        doc = PDFDocument(
            collection=collection,
            title=title,
            ingested_by=user,
        )
        doc.pdf_file.save(f"arxiv:{arxiv_id}.pdf", ContentFile(content), save=False)
        doc.save()

    status: dict[str, object] = {"message": "", "errors": []}
    tex_req = requests.get("https://arxiv.org/src/" + arxiv_id)
    pdf_req = requests.get("https://arxiv.org/pdf/" + arxiv_id)
    metadata_req = requests.get("http://export.arxiv.org/api/query?id_list=" + arxiv_id)

    if metadata_req.status_code == 404 or (tex_req.status_code == 404 and pdf_req.status_code == 404):
        status["errors"].append("ERROR: 404 from ArXiv, is the DOI correct?")  # type: ignore[union-attr]
    elif (
        tex_req.status_code not in [200, 404]
        or pdf_req.status_code not in [200, 404]
        or metadata_req.status_code not in [200, 404]
    ):
        error_str = (
            f"ERROR -- DOI {arxiv_id}: LaTeX status code {tex_req.status_code}, "
            f"PDF status code {pdf_req.status_code}, metadata status code {metadata_req.status_code}"
        )
        logger.error(error_str)
        status["errors"].append(error_str)  # type: ignore[union-attr]
    else:
        xmldoc = minidom.parseString(metadata_req.content)
        title = " ".join(
            xmldoc.getElementsByTagName("entry")[0]
            .getElementsByTagName("title")[0]
            .firstChild.data.split()  # type: ignore
        )

        if tex_req.status_code == 200:
            if tex_req.content.startswith(b"%PDF"):
                status["message"] += f"Got PDF for {arxiv_id}\n"  # type: ignore[operator]
                save_pdf_doc(tex_req.content, title)
            else:
                status["message"] += f"Got LaTeX source for {arxiv_id}\n"  # type: ignore[operator]
                tgz_io = io.BytesIO(tex_req.content)
                tex_str = ""
                with gzip.open(tgz_io, "rb") as gz:
                    with tarfile.open(fileobj=gz) as tar:  # type: ignore
                        for member in tar.getmembers():
                            if member.isfile() and member.name.endswith(".tex"):
                                f = tar.extractfile(member)
                                if f:
                                    tex_bytes = f.read()
                                    encoding = chardet.detect(tex_bytes)["encoding"]
                                    if not encoding:
                                        if not any(x > 127 for x in tex_bytes):
                                            encoding = "ascii"
                                        else:
                                            raise ValueError("Could not detect encoding of LaTeX source")
                                    content = tex_bytes.decode(encoding)
                                    tex_str += content + "\n\n"
                doc = TeXDocument(
                    collection=collection,
                    title=title,
                    full_text=tex_str,
                    ingested_by=user,
                )
                if pdf_req.status_code == 200:
                    status["message"] += f"Got PDF for {arxiv_id}\n"  # type: ignore[operator]
                    doc.pdf_file.save(f"arxiv:{arxiv_id}.pdf", ContentFile(pdf_req.content), save=False)
                doc.save()
        elif pdf_req.status_code == 200:
            status["message"] += f"Got PDF for {arxiv_id}\n"  # type: ignore[operator]
            save_pdf_doc(pdf_req.content, title)

    return status


__all__ = ["insert_one_from_arxiv"]
