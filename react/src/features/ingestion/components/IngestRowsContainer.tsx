import React, { useEffect, useRef, useState } from "react";
import { getCsrfCookie } from "../../../main";

import { 
  DocType, 
  IngestRowsContainerProps, 
  IngestRowData, 
  UploadSummary,
  SubmissionStatus 
} from '../types';
import { IngestRow } from './IngestRow';

export { DocType } from '../types';

const IngestRowsContainer: React.FC<IngestRowsContainerProps> = ({
  ingestUploadsUrl,
  ingestArxivUrl,
  ingestPdfUrl,
  ingestVttUrl,
  ingestWebpageUrl,
  ingestHandwrittenUrl,
  collectionId,
  onUploadSuccess,
  layout = "default",
}) => {
  const [rows, setRows] = useState<IngestRowData[]>([
    {
      id: 0,
      docType: DocType.UPLOADS,
      uploadFiles: [],
      pdfTitle: "",
      pdfFiles: [],
      arxivId: "",
      vttTitle: "",
      vttFile: null,
      webpageUrl: "",
      webpageCrawlDepth: 1,
      handwrittenTitle: "",
      handwrittenFile: null,
      convertToLatex: false,
    },
  ]);
  const [submissionStatus, setSubmissionStatus] = useState<{
    [key: number]: SubmissionStatus;
  }>({});
  const [errorMessages, setErrorMessages] = useState<{ [key: number]: string }>(
    {}
  );
  const [uploadBatchIds, setUploadBatchIds] = useState<{ [key: number]: number }>(
    {}
  );
  const [uploadSummaries, setUploadSummaries] = useState<{ [key: number]: UploadSummary }>(
    {}
  );
  const pollingHandlesRef = useRef<{ [key: number]: number }>({});

  const updateRow = (id: number, updates: Partial<IngestRowData>) => {
    setRows((prevRows) =>
      prevRows.map((row) => (row.id === id ? { ...row, ...updates } : row))
    );
  };

  const clearPollingForRow = (rowId: number) => {
    const handle = pollingHandlesRef.current[rowId];
    if (handle !== undefined) {
      window.clearInterval(handle);
      delete pollingHandlesRef.current[rowId];
    }
  };

  useEffect(() => {
    return () => {
      Object.values(pollingHandlesRef.current).forEach((handle) =>
        window.clearInterval(handle)
      );
      pollingHandlesRef.current = {};
    };
  }, []);

  useEffect(() => {
    Object.entries(uploadBatchIds).forEach(([rowIdRaw, batchId]) => {
      const rowId = Number(rowIdRaw);
      if (!Number.isFinite(rowId) || pollingHandlesRef.current[rowId] !== undefined) {
        return;
      }

      const pollBatch = async () => {
        try {
          const response = await fetch(`${ingestUploadsUrl}${batchId}/`, {
            method: "GET",
            credentials: "same-origin",
          });
          if (!response.ok) {
            throw new Error(`Upload status check failed (${response.status}).`);
          }
          const payload = await response.json();
          const items = Array.isArray(payload?.items) ? payload.items : [];
          const modalities = Array.from(
            new Set(
              items.flatMap((batchItem: any) =>
                Array.isArray(batchItem?.modalities) ? batchItem.modalities : []
              )
            )
          ).map((value) => String(value));
          const providers = Array.from(
            new Set(
              items.flatMap((batchItem: any) =>
                Array.isArray(batchItem?.providers) ? batchItem.providers : []
              )
            )
          ).map((value) => String(value));
          const rawMediaSaved = items.some((batchItem: any) => Boolean(batchItem?.raw_media_saved));
          const textExtracted = items.some((batchItem: any) => Boolean(batchItem?.text_extracted));
          setUploadSummaries((prev) => ({
            ...prev,
            [rowId]: { modalities, providers, rawMediaSaved, textExtracted },
          }));

          const counts = payload?.counts || {};
          const queued = Number(counts.queued || 0);
          const processing = Number(counts.processing || 0);
          const success = Number(counts.success || 0);
          const failed = Number(counts.error || 0);
          if (queued + processing > 0) {
            return;
          }

          clearPollingForRow(rowId);
          setUploadBatchIds((prev) => {
            const next = { ...prev };
            delete next[rowId];
            return next;
          });

          if (success > 0) {
            onUploadSuccess?.();
            updateRow(rowId, { uploadFiles: [] });
          }

          if (failed > 0) {
            const failedItems = Array.isArray(payload?.items)
              ? payload.items.filter((item: any) => item?.status === "error")
              : [];
            const firstError = failedItems[0]?.error_message || "Unknown extraction error.";
            const prefix =
              success > 0
                ? `Uploaded ${success} file(s), but ${failed} failed. `
                : `Upload failed for ${failed} file(s). `;
            setErrorMessages((prev) => ({
              ...prev,
              [rowId]: `${prefix}${firstError}`,
            }));
            setSubmissionStatus((prev) => ({ ...prev, [rowId]: "error" }));
            return;
          }

          if (success > 0) {
            setSubmissionStatus((prev) => ({ ...prev, [rowId]: "success" }));
          } else {
            setErrorMessages((prev) => ({
              ...prev,
              [rowId]: "Upload finished with no ingested files.",
            }));
            setSubmissionStatus((prev) => ({ ...prev, [rowId]: "error" }));
          }
        } catch (error: any) {
          clearPollingForRow(rowId);
          setUploadBatchIds((prev) => {
            const next = { ...prev };
            delete next[rowId];
            return next;
          });
          setErrorMessages((prev) => ({
            ...prev,
            [rowId]: error?.message || "Failed to poll upload status.",
          }));
          setSubmissionStatus((prev) => ({ ...prev, [rowId]: "error" }));
        }
      };

      pollBatch();
      pollingHandlesRef.current[rowId] = window.setInterval(pollBatch, 2000);
    });
  }, [uploadBatchIds, ingestUploadsUrl, onUploadSuccess]);

  const addRow = () => {
      setRows((prevRows) => [
      ...prevRows,
      {
        id: prevRows.length > 0 ? prevRows[prevRows.length - 1].id + 1 : 0,
        docType: DocType.UPLOADS,
        uploadFiles: [],
        pdfTitle: "",
        pdfFiles: [],
        arxivId: "",
        vttTitle: "",
        vttFile: null,
        webpageUrl: "",
        webpageCrawlDepth: 1,
        handwrittenTitle: "",
        handwrittenFile: null,
        convertToLatex: false,
      },
      ]);
  };

  const updateRowDocType = (id: number, newDocType: DocType) => {
    setRows((prevRows) =>
      prevRows.map((row) => {
        if (row.id === id) {
          const newRow = { ...row, docType: newDocType };
          if (newDocType !== DocType.UPLOADS) {
            newRow.uploadFiles = [];
          }
          if (newDocType !== DocType.PDF) {
            newRow.pdfFiles = [];
            newRow.pdfTitle = "";
          }
          if (newDocType !== DocType.ARXIV) {
            newRow.arxivId = "";
          }
          if (newDocType !== DocType.VTT) {
            newRow.vttFile = null;
            newRow.vttTitle = "";
          }
          if (newDocType !== DocType.WEBPAGE) {
            newRow.webpageUrl = "";
            newRow.webpageCrawlDepth = 1;
          }
          if (newDocType !== DocType.HANDWRITTEN) {
            newRow.handwrittenFile = null;
            newRow.handwrittenTitle = "";
            newRow.convertToLatex = false;
          }
          return newRow;
        }
        return row;
      })
    );
  };

  const handleSubmit = async () => {
    const csrfToken = getCsrfCookie();
    setErrorMessages({});

    for (const row of rows) {
      setSubmissionStatus((prev) => ({ ...prev, [row.id]: "submitting" }));
      let url: string;
      let body: FormData | FormData[] | string;
      let headers: HeadersInit = {
        "X-CSRFToken": csrfToken,
        "X-Requested-With": "XMLHttpRequest",
      };

      try {
        switch (row.docType) {
          case DocType.UPLOADS:
            if (!row.uploadFiles?.length) {
              throw new Error("At least one file is required.");
            }
            setUploadSummaries((prev) => {
              const next = { ...prev };
              delete next[row.id];
              return next;
            });
            url = ingestUploadsUrl;
            const uploadsBody = new FormData();
            uploadsBody.append("collection", collectionId);
            row.uploadFiles.forEach((file) => uploadsBody.append("files", file));
            body = uploadsBody;
            break;
          case DocType.PDF:
            if (!row.pdfFiles?.length) {
              throw new Error("At least one PDF file is required.");
            }
            if (row.pdfFiles.length === 1 && !row.pdfTitle.trim()) {
              throw new Error("PDF title is required when uploading a single file.");
            }
            url = ingestPdfUrl;
            const pdfBodies: FormData[] = row.pdfFiles.map((file) => {
              const fd = new FormData();
              fd.append("pdf_file", file);
              fd.append(
                "title",
                row.pdfFiles.length === 1
                  ? row.pdfTitle.trim()
                  : file.name.replace(/\.pdf$/i, "")
              );
              fd.append("collection", collectionId);
              return fd;
            });
            body = pdfBodies;
            break;
          case DocType.ARXIV:
            if (!row.arxivId) {
              throw new Error("arXiv ID is required.");
            }
            url = ingestArxivUrl;
            body = new FormData();
            body.append("arxiv_id", row.arxivId);
            body.append("collection", collectionId);
            break;
          case DocType.VTT:
            if (!row.vttFile || !row.vttTitle) {
              throw new Error("VTT file and title are required.");
            }
            url = ingestVttUrl;
            body = new FormData();
            body.append("vtt_file", row.vttFile);
            body.append("title", row.vttTitle);
            body.append("collection", collectionId);
            break;
          case DocType.HANDWRITTEN:
            if (!row.handwrittenFile || !row.handwrittenTitle) {
              throw new Error("Image file and title are required.");
            }
            url = ingestHandwrittenUrl;
            body = new FormData();
            body.append("image_file", row.handwrittenFile);
            body.append("title", row.handwrittenTitle);
            body.append("collection", collectionId);
            body.append("convert_to_latex", row.convertToLatex ? "on" : "");
            break;
          case DocType.WEBPAGE:
            if (!row.webpageUrl) {
              throw new Error("Webpage URL is required.");
            }
            try {
              new URL(row.webpageUrl);
            } catch (e) {
              throw new Error("Invalid URL format.");
            }
            url = ingestWebpageUrl;
            body = JSON.stringify({
              url: row.webpageUrl,
              collection_id: collectionId,
              depth: row.webpageCrawlDepth,
            });
            headers["Content-Type"] = "application/json";
            break;
          default:
            throw new Error("Invalid document type selected.");
        }

        let response: Response;
        if (Array.isArray(body)) {
          for (const singleBody of body) {
            response = await fetch(url, {
              method: "POST",
              headers,
              body: singleBody,
            });
            if (!response.ok) {
              let errorData: { error?: string };
              try {
                errorData = await response.json();
              } catch {
                errorData = { error: `HTTP error! status: ${response.status}` };
              }
              throw new Error(
                errorData.error || `Request failed with status ${response.status}`
              );
            }
          }
          response = { ok: true, status: 200 } as Response;
        } else {
          response = await fetch(url, {
            method: "POST",
            headers: headers,
            body: body,
          });
        }

        if (response.ok) {
           if (row.docType === DocType.UPLOADS && response.status === 202) {
             const payload = await response.json();
             const batchId = Number(payload?.batch_id);
             if (!Number.isFinite(batchId)) {
               throw new Error("Upload batch was queued but no batch_id was returned.");
             }
             setUploadBatchIds((prev) => ({ ...prev, [row.id]: batchId }));
             setSubmissionStatus((prev) => ({ ...prev, [row.id]: "initiated" }));
           } else if (row.docType === DocType.WEBPAGE && response.status === 202) {
             setSubmissionStatus((prev) => ({ ...prev, [row.id]: "initiated" }));
           } else {
             setSubmissionStatus((prev) => ({ ...prev, [row.id]: "success" }));
             onUploadSuccess?.();
             updateRow(row.id, {
               uploadFiles: [],
               pdfTitle: "",
               pdfFiles: [],
               arxivId: "",
               vttTitle: "",
               vttFile: null,
               handwrittenTitle: "",
               handwrittenFile: null,
               convertToLatex: false,
               webpageUrl: row.docType !== DocType.WEBPAGE ? "" : row.webpageUrl,
             });
           }
        } else {
          let errorData: { error?: string };
          try {
            errorData = await response.json();
          } catch {
            errorData = { error: `HTTP error! status: ${response.status}` };
          }
          throw new Error(
            errorData.error || `Request failed with status ${response.status}`
          );
        }

      } catch (error: any) {
        console.error("Submission error for row", row.id, ":", error);
        setErrorMessages((prev) => ({ ...prev, [row.id]: error.message }));
        setSubmissionStatus((prev) => ({ ...prev, [row.id]: "error" }));
      }
    }
  };

  const actionButtons = (
    <>
      <button
        onClick={addRow}
        className="h-[40px] px-4 rounded-[20px] bg-scheme-shade_4 text-text-normal border border-border-high_contrast hover:bg-scheme-shade_5 transition-colors"
      >
        Add Another
      </button>
      <button
        onClick={handleSubmit}
        disabled={Object.values(submissionStatus).some(s => s === 'submitting')}
        className="h-[40px] px-4 rounded-[20px] bg-accent text-text-normal border border-border-high_contrast hover:bg-accent-dark transition-colors disabled:opacity-60 disabled:cursor-not-allowed"
      >
        Submit All
      </button>
    </>
  );

  return (
    <div className={layout === "compact" ? "space-y-3" : "space-y-4"}>
      {rows.map((row, index) => (
        <div
          key={row.id}
          className={`${
            layout === "compact"
              ? "bg-transparent p-0 rounded-[16px]"
              : "bg-scheme-shade_1 p-4 border border-border-mid_contrast rounded-lg shadow"
          }`}
        >
          <IngestRow
            row={row}
            onDocTypeChange={updateRowDocType}
            onRowChange={updateRow}
            layout={layout}
            actions={layout === "compact" && index === 0 ? actionButtons : undefined}
          />
          {submissionStatus[row.id] === "submitting" && (
            <p className="text-text-low_contrast mt-2">Submitting...</p>
          )}
          {submissionStatus[row.id] === "success" && (
            <p className="text-accent-light mt-2">Submission successful!</p>
          )}
          {submissionStatus[row.id] === "initiated" && (
             <p className="text-accent-light mt-2">
               {row.docType === DocType.UPLOADS ? "Batch ingestion queued..." : "Webpage crawl initiated..."}
             </p>
          )}
          {submissionStatus[row.id] === "error" && errorMessages[row.id] && (
            <p className="text-red-dark mt-2">Error: {errorMessages[row.id]}</p>
          )}
          {uploadSummaries[row.id] && (
            <div className="mt-2 flex flex-wrap items-center gap-2 text-xs">
              {uploadSummaries[row.id].modalities.map((modality) => (
                <span
                  key={`${row.id}-${modality}`}
                  className="px-2 py-1 rounded-full bg-scheme-shade_5 border border-border-mid_contrast text-text-normal"
                >
                  {modality}
                </span>
              ))}
              <span className="px-2 py-1 rounded-full bg-scheme-shade_5 border border-border-mid_contrast text-text-normal">
                raw media: {uploadSummaries[row.id].rawMediaSaved ? "saved" : "none"}
              </span>
              <span className="px-2 py-1 rounded-full bg-scheme-shade_5 border border-border-mid_contrast text-text-normal">
                text: {uploadSummaries[row.id].textExtracted ? "extracted" : "none"}
              </span>
              {uploadSummaries[row.id].providers.map((provider) => (
                <span
                  key={`${row.id}-provider-${provider}`}
                  className="px-2 py-1 rounded-full bg-scheme-shade_5 border border-border-mid_contrast text-text-normal"
                >
                  provider: {provider}
                </span>
              ))}
            </div>
          )}
        </div>
      ))}
      {layout !== "compact" && (
        <div className="flex items-center gap-3 pt-1">
          {actionButtons}
        </div>
      )}
    </div>
  );
};

export default IngestRowsContainer;
