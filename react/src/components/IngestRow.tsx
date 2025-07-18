import React, { useState } from "react";
import { FileText, Headphones, LinkIcon, PenLine } from "lucide-react";
import { getCsrfCookie } from "../main";

export enum DocType {
  PDF = "pdf",
  ARXIV = "arxiv",
  VTT = "vtt",
  WEBPAGE = "webpage",
  HANDWRITTEN = "handwritten",
}

interface IngestRowsContainerProps {
  ingestArxivUrl: string;
  ingestPdfUrl: string;
  ingestVttUrl: string;
  ingestWebpageUrl: string;
  ingestHandwrittenUrl: string;
  collectionId: string;
}

interface IngestRowData {
  id: number;
  docType: DocType;
  // For PDF rows
  pdfTitle: string;
  pdfFile: File | null;
  // For arXiv rows
  arxivId: string;
  // For VTT rows
  vttTitle: string;
  vttFile: File | null;
  // For Webpage rows
  webpageUrl: string;
  webpageCrawlDepth: number; // Add crawl depth state
  // For Handwritten Notes rows
  handwrittenTitle: string;
  handwrittenFile: File | null;
  convertToLatex: boolean;
}

const selectedClasses = "bg-accent text-slight_muted_white";
const unselectedClasses = "bg-scheme-shade_3 hover:bg-scheme-shade_4 text-text-normal";

interface DocTypeToggleProps {
  docType: DocType;
  setDocType: (docType: DocType) => void;
  arxivLogo: React.ReactNode;
}

const DocTypeToggle: React.FC<DocTypeToggleProps> = ({
  docType,
  setDocType,
  arxivLogo,
}) => {
  return (
    <div className="flex space-x-4">
      <button
        onClick={() => setDocType(DocType.PDF)}
        title="PDF"
        className={`flex items-center h-[40px] w-[40px] justify-center px-2 py-1 rounded-lg transition-colors element-border ${
          docType === DocType.PDF ? selectedClasses : unselectedClasses
        }`}
      >
        <FileText size={18} />
      </button>

      <button
        onClick={() => setDocType(DocType.ARXIV)}
        title="arXiv"
        className={`flex items-center h-[40px] w-[40px] justify-center px-2 py-1 rounded-lg transition-colors element-border ${
          docType === DocType.ARXIV ? selectedClasses : unselectedClasses
        }`}
      >
        {arxivLogo}
      </button>

      <button
        onClick={() => setDocType(DocType.VTT)}
        title="VTT"
        className={`flex items-center h-[40px] w-[40px] justify-center px-2 py-1 rounded-lg transition-colors element-border ${
          docType === DocType.VTT ? selectedClasses : unselectedClasses
        }`}
      >
        <Headphones size={18} />
      </button>

      <button
        onClick={() => setDocType(DocType.WEBPAGE)}
        title="Webpage"
        className={`flex items-center h-[40px] w-[40px] justify-center px-2 py-1 rounded-lg transition-colors element-border ${
          docType === DocType.WEBPAGE ? selectedClasses : unselectedClasses
        }`}
      >
        <LinkIcon size={18} />
      </button>

      <button
        onClick={() => setDocType(DocType.HANDWRITTEN)}
        title="Handwritten Notes"
        className={`flex items-center h-[40px] w-[40px] justify-center px-2 py-1 rounded-lg transition-colors element-border ${
          docType === DocType.HANDWRITTEN ? selectedClasses : unselectedClasses
        }`}
      >
        <PenLine size={18} />
      </button>
    </div>
  );
};

const ARXIVLogo = (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    viewBox="-12.5 0 100 100"
    className="w-[28px] h-[28px]"
  >
    <path
      d="M71.826 92.336 36.314 49.592l-8.977-10.558-4.977 4.656c-3.407 3.407-3.932 8.063-.437 11.558l44.495 42.379c1.035.942 2.36 1.384 3.733.965 1.485-.453 2.351-1.612 2.721-2.837.372-1.233-.174-2.326-1.047-3.419Z"
      fill="#b3aca5"
    />
    <g fill="#b32025">
      <path d="m36.314 49.592 9.557 11.503 5.761-5.356c3.513-3.513 3.465-8.318.04-11.743L7.971 2.072S6.3.044 4.534.001C2.768-.041 1.023.996.304 2.719-.384 4.37.11 5.528 1.619 7.689l29.307 35.565 5.388 6.338ZM27.61 60.663 10.494 81.687c-1.255 1.337-2.032 3.683-1.331 5.368.733 1.76 2.403 2.841 4.287 2.841 1.061 0 1.937-.373 3.082-1.523L36.86 69.474l-9.25-8.81Z" />
    </g>
    <path
      d="M64.663 12.878c-.491-1.208-2.5-2.651-3.581-2.791-1.217-.157-1.953.302-3.209 1.291L37.207 30.119l8.304 7.966 17.837-21.067c1.629-2.117 1.769-3.024 1.315-4.14Z"
      fill="#b3aca5"
    />
  </svg>
);

interface PDFFormProps {
  pdfTitle: string;
  pdfFile: File | null;
  onTitleChange: (value: string) => void;
  onFileChange: (file: File | null) => void;
}

const PDFForm: React.FC<PDFFormProps> = ({
  pdfTitle,
  pdfFile,
  onTitleChange,
  onFileChange,
}) => {
  return (
    <div className="flex gap-4">
      <label
        htmlFor="pdf-file-upload"
        className={`cursor-pointer flex items-center justify-center border border-border-mid_contrast p-2 rounded-lg transition-colors flex-grow h-[40px] ${
          pdfFile ? "bg-green-dark" : "hover:bg-scheme-shade_3"
        }`}
      >
        {pdfFile ? "File Selected" : "Select PDF File"}
      </label>
      <input
        id="pdf-file-upload"
        type="file"
        accept="application/pdf"
        onChange={(e) =>
          onFileChange(
            e.target.files && e.target.files.length ? e.target.files[0] : null
          )
        }
        className="hidden"
      />

      <input
        type="text"
        placeholder="Enter PDF title"
        value={pdfTitle}
        onChange={(e) => onTitleChange(e.target.value)}
        className="bg-scheme-shade_3 border border-border-mid_contrast p-2 rounded-lg h-[40px] placeholder:text-text-less_contrast flex-grow"
      />
    </div>
  );
};

interface VTTFormProps {
  vttTitle: string;
  vttFile: File | null;
  onTitleChange: (value: string) => void;
  onFileChange: (file: File | null) => void;
}

const VTTForm: React.FC<VTTFormProps> = ({
  vttTitle,
  vttFile,
  onTitleChange,
  onFileChange,
}) => {
  return (
    <div className="flex gap-4">
      <label
        htmlFor="vtt-file-upload"
        className={`cursor-pointer flex items-center justify-center border border-border-mid_contrast p-2 rounded-lg transition-colors flex-grow h-[40px] ${
          vttFile ? "bg-green-dark" : "hover:bg-scheme-shade_3"
        }`}
      >
        {vttFile ? "File Selected" : "Select VTT File"}
      </label>
      <input
        id="vtt-file-upload"
        type="file"
        accept=".vtt"
        onChange={(e) =>
          onFileChange(
            e.target.files && e.target.files.length ? e.target.files[0] : null
          )
        }
        className="hidden"
      />

      <input
        type="text"
        placeholder="Enter VTT title"
        value={vttTitle}
        onChange={(e) => onTitleChange(e.target.value)}
        className="bg-scheme-shade_3 border border-border-mid_contrast p-2 rounded-lg h-[40px] placeholder:text-text-less_contrast flex-grow"
      />
    </div>
  );
};

interface WebpageFormProps {
  urlValue: string;
  depthValue: number;
  onUrlChange: (value: string) => void;
  onDepthChange: (value: number) => void;
}

const WebpageForm: React.FC<WebpageFormProps> = ({
  urlValue,
  depthValue,
  onUrlChange,
  onDepthChange,
}) => {
  const handleDepthChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseInt(e.target.value, 10);
    // Allow 0, 1, 2, etc. Prevent negative numbers.
    onDepthChange(isNaN(value) || value < 0 ? 0 : value);
  };

  return (
    <div className="flex gap-4 items-center">
      <input
        type="url"
        placeholder="Enter Webpage URL"
        value={urlValue}
        onChange={(e) => onUrlChange(e.target.value)}
        className="bg-scheme-shade_3 border border-border-high_contrast p-2 rounded-lg h-[40px] placeholder:text-text-low_contrast flex-grow"
      />
      <div className="flex items-center gap-2">
         <label htmlFor="crawl-depth" className="text-sm text-text-high_contrast whitespace-nowrap">Crawl Depth:</label>
         <input
           id="crawl-depth"
           type="number"
           min="0" // 0 means only the initial page
           value={depthValue}
           onChange={handleDepthChange}
           className="bg-scheme-shade_3 border border-border-high_contrast p-2 rounded-lg h-[40px] w-20 text-center"
         />
      </div>
    </div>
  );
};

interface ArxivFormProps {
  value: string;
  onValueChange: (value: string) => void;
}

const ArxivForm: React.FC<ArxivFormProps> = ({ value, onValueChange }) => {
  return (
    <input
      type="text"
      placeholder="Enter arXiv ID"
      value={value}
      onChange={(e) => onValueChange(e.target.value)}
      className="bg-scheme-shade_3 border border-border-mid_contrast p-2 rounded-lg h-[40px] placeholder:text-text-less_contrast"
    />
  );
};

interface HandwrittenFormProps {
  handwrittenTitle: string;
  handwrittenFile: File | null;
  convertToLatex: boolean;
  onTitleChange: (value: string) => void;
  onFileChange: (file: File | null) => void;
  onConvertChange: (value: boolean) => void;
}

const HandwrittenForm: React.FC<HandwrittenFormProps> = ({
  handwrittenTitle,
  handwrittenFile,
  convertToLatex,
  onTitleChange,
  onFileChange,
  onConvertChange,
}) => {
  return (
    <div className="flex flex-col gap-4">
      <div className="flex gap-4">
        <label
          htmlFor="handwritten-file-upload"
          className={`cursor-pointer flex items-center justify-center border border-border-mid_contrast p-2 rounded-lg transition-colors flex-grow h-[40px] ${
            handwrittenFile ? "bg-green-dark" : "hover:bg-scheme-shade_3"
          }`}
        >
          {handwrittenFile ? "Image Selected" : "Select Image File"}
        </label>
        <input
          id="handwritten-file-upload"
          type="file"
          accept="image/*"
          onChange={(e) =>
            onFileChange(
              e.target.files && e.target.files.length ? e.target.files[0] : null
            )
          }
          className="hidden"
        />

        <input
          type="text"
          placeholder="Enter title for notes"
          value={handwrittenTitle}
          onChange={(e) => onTitleChange(e.target.value)}
          className="bg-scheme-shade_3 border border-border-mid_contrast p-2 rounded-lg h-[40px] placeholder:text-text-less_contrast flex-grow"
        />
      </div>
      
      <div className="flex items-center">
        <input
          id="convert-latex-checkbox"
          type="checkbox"
          checked={convertToLatex}
          onChange={(e) => onConvertChange(e.target.checked)}
          className="mr-2 h-4 w-4"
        />
        <label htmlFor="convert-latex-checkbox" className="text-text-normal">
          Convert to LaTeX (for math equations)
        </label>
      </div>
    </div>
  );
};

interface IngestRowProps {
  row: IngestRowData;
  onDocTypeChange: (id: number, newDocType: DocType) => void;
  onRowChange: (id: number, updates: Partial<IngestRowData>) => void;
}

const IngestRow: React.FC<IngestRowProps> = ({ row, onDocTypeChange, onRowChange }) => {
  return (
    <div className="pt-[1rem] flex items-start space-x-4">
      <DocTypeToggle
        docType={row.docType}
        setDocType={(newType) => onDocTypeChange(row.id, newType)}
        arxivLogo={ARXIVLogo}
      />
      <div className="flex-1">
        {row.docType === DocType.PDF ? (
          <PDFForm
            pdfTitle={row.pdfTitle}
            pdfFile={row.pdfFile}
            onTitleChange={(value) => onRowChange(row.id, { pdfTitle: value })}
            onFileChange={(file) => onRowChange(row.id, { pdfFile: file })}
          />
        ) : row.docType === DocType.ARXIV ? (
          <ArxivForm
            value={row.arxivId}
            onValueChange={(value) => onRowChange(row.id, { arxivId: value })}
          />
        ) : row.docType === DocType.VTT ? (
          <VTTForm
            vttTitle={row.vttTitle}
            vttFile={row.vttFile}
            onTitleChange={(value) => onRowChange(row.id, { vttTitle: value })}
            onFileChange={(file) => onRowChange(row.id, { vttFile: file })}
          />
        ) : row.docType === DocType.HANDWRITTEN ? (
          <HandwrittenForm
            handwrittenTitle={row.handwrittenTitle}
            handwrittenFile={row.handwrittenFile}
            convertToLatex={row.convertToLatex}
            onTitleChange={(value) => onRowChange(row.id, { handwrittenTitle: value })}
            onFileChange={(file) => onRowChange(row.id, { handwrittenFile: file })}
            onConvertChange={(value) => onRowChange(row.id, { convertToLatex: value })}
          />
        ) : (
          <WebpageForm
            urlValue={row.webpageUrl}
            depthValue={row.webpageCrawlDepth}
            onUrlChange={(value) => onRowChange(row.id, { webpageUrl: value })}
            onDepthChange={(value) => onRowChange(row.id, { webpageCrawlDepth: value })}
          />
        )}
      </div>
    </div>
  );
};

const IngestRowsContainer: React.FC<IngestRowsContainerProps> = ({
  ingestArxivUrl,
  ingestPdfUrl,
  ingestVttUrl,
  ingestWebpageUrl,
  ingestHandwrittenUrl,
  collectionId,
}) => {
  const [rows, setRows] = useState<IngestRowData[]>([
    {
      id: 0,
      docType: DocType.PDF,
      pdfTitle: "",
      pdfFile: null,
      arxivId: "",
      vttTitle: "",
      vttFile: null,
      webpageUrl: "",
      webpageCrawlDepth: 1, // Default crawl depth
      handwrittenTitle: "", 
      handwrittenFile: null,
      convertToLatex: false,
    },
  ]);
  const [submissionStatus, setSubmissionStatus] = useState<{
    [key: number]: "idle" | "submitting" | "success" | "error" | "initiated"; // Add 'initiated' state
  }>({});
  const [errorMessages, setErrorMessages] = useState<{ [key: number]: string }>(
    {}
  );

  const updateRow = (id: number, updates: Partial<IngestRowData>) => {
    setRows((prevRows) =>
      prevRows.map((row) => (row.id === id ? { ...row, ...updates } : row))
    );
  };

  const addRow = () => {
    setRows((prevRows) => [
      ...prevRows,
      {
        id: prevRows.length > 0 ? prevRows[prevRows.length - 1].id + 1 : 0,
        docType: DocType.PDF,
        pdfTitle: "",
        pdfFile: null,
        arxivId: "",
        vttTitle: "",
        vttFile: null,
        webpageUrl: "",
        webpageCrawlDepth: 1, // Default crawl depth for new rows
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
          if (newDocType !== DocType.PDF) {
            newRow.pdfFile = null;
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
            newRow.webpageCrawlDepth = 1; // Reset depth when switching away
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
      let body: FormData | string;
      let headers: HeadersInit = { "X-CSRFToken": csrfToken };

      try {
        switch (row.docType) {
          case DocType.PDF:
            if (!row.pdfFile || !row.pdfTitle) {
              throw new Error("PDF file and title are required.");
            }
            url = ingestPdfUrl;
            body = new FormData();
            body.append("pdf_file", row.pdfFile);
            body.append("title", row.pdfTitle);
            body.append("collection", collectionId);
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
              depth: row.webpageCrawlDepth, // Include depth in payload
            });
            headers["Content-Type"] = "application/json";
            break;
          default:
            throw new Error("Invalid document type selected.");
        }

        const response = await fetch(url, {
          method: "POST",
          headers: headers,
          body: body,
        });

        // Handle different success statuses
        if (response.ok) {
           if (row.docType === DocType.WEBPAGE && response.status === 202) {
             // Specific handling for async webpage crawl initiation
             setSubmissionStatus((prev) => ({ ...prev, [row.id]: "initiated" }));
             // Optionally keep the URL input field populated until final confirmation via WebSocket
             // updateRow(row.id, { webpageUrl: "" }); // Don't clear immediately
           } else {
             // Standard success handling for other types or synchronous responses
             setSubmissionStatus((prev) => ({ ...prev, [row.id]: "success" }));
             // Clear inputs on standard success
             updateRow(row.id, {
               pdfTitle: "",
               pdfFile: null,
               arxivId: "",
               vttTitle: "",
               vttFile: null,
               handwrittenTitle: "",
               handwrittenFile: null,
               convertToLatex: false,
               // Clear webpage URL only on standard success, not initiated
               webpageUrl: row.docType !== DocType.WEBPAGE ? "" : row.webpageUrl,
             });
           }
        } else {
          // Handle errors (status codes 4xx, 5xx)
          let errorData;
          try {
            errorData = await response.json();
          } catch (e) {
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

  return (
    <div className="space-y-4">
      {rows.map((row, index) => (
        <div key={row.id} className="bg-gray-shade_1 p-4 rounded-lg shadow">
          <IngestRow
            row={row}
            onDocTypeChange={updateRowDocType}
            onRowChange={updateRow}
          />
          {submissionStatus[row.id] === "submitting" && (
            <p className="text-yellow mt-2">Submitting...</p>
          )}
          {submissionStatus[row.id] === "success" && (
            <p className="text-green mt-2">Submission successful!</p>
          )}
          {submissionStatus[row.id] === "initiated" && (
             <p className="text-blue mt-2">Webpage crawl initiated...</p> // Message for initiated state
          )}
          {submissionStatus[row.id] === "error" && errorMessages[row.id] && (
            <p className="text-red mt-2">Error: {errorMessages[row.id]}</p>
          )}
        </div>
      ))}
      <button
        onClick={addRow}
        className="bg-scheme-shade_6 text-normal px-4 py-2 rounded-[20px] hover:bg-blue-dark transition-colors mr-[16px]"
      >
        Add Another
      </button>
      <button
        onClick={handleSubmit}
        disabled={Object.values(submissionStatus).some(s => s === 'submitting')}
        className="mt-4 mb-2 px-4 py-2 bg-accent rounded-[20px] text-slight_muted_white h-[40px]"

      >
        Submit All
      </button>
    </div>
  );
};

export default IngestRowsContainer;