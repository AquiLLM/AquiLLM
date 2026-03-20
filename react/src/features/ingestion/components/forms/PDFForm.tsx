import React from 'react';

interface PDFFormProps {
  pdfTitle: string;
  pdfFiles: File[];
  onTitleChange: (value: string) => void;
  onFileChange: (files: File[]) => void;
}

export const PDFForm: React.FC<PDFFormProps> = ({
  pdfTitle,
  pdfFiles,
  onTitleChange,
  onFileChange,
}) => {
  const fileLabel =
    pdfFiles.length === 0
      ? "Select PDF File(s)"
      : pdfFiles.length === 1
        ? "1 file selected"
        : `${pdfFiles.length} files selected`;
  return (
    <div className="flex gap-4">
      <label
        htmlFor="pdf-file-upload"
        className={`cursor-pointer flex items-center justify-center border border-border-mid_contrast p-2 rounded-lg transition-colors flex-grow h-[40px] ${
          pdfFiles.length > 0
            ? "bg-accent text-text-normal border-border-high_contrast"
            : "hover:bg-scheme-shade_3 text-text-normal"
        }`}
      >
        {fileLabel}
      </label>
      <input
        id="pdf-file-upload"
        type="file"
        accept="application/pdf"
        multiple
        onChange={(e) =>
          onFileChange(
            e.target.files ? Array.from(e.target.files) : []
          )
        }
        className="hidden"
      />

      {pdfFiles.length <= 1 && (
        <input
          type="text"
          placeholder="Enter PDF title"
          value={pdfTitle}
          onChange={(e) => onTitleChange(e.target.value)}
          className="bg-scheme-shade_3 border border-border-mid_contrast p-2 rounded-lg h-[40px] placeholder:text-text-less_contrast flex-grow"
        />
      )}
    </div>
  );
};
