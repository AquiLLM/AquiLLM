import React, { useRef, useState } from 'react';

interface UploadsFormProps {
  uploadFiles: File[];
  onFileChange: (files: File[]) => void;
}

export const UploadsForm: React.FC<UploadsFormProps> = ({ uploadFiles, onFileChange }) => {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  const fileLabel =
    uploadFiles.length === 0
      ? "Drag & Drop Files"
      : uploadFiles.length === 1
        ? uploadFiles[0].name
        : `${uploadFiles.length} files selected`;

  const openFilePicker = () => {
    inputRef.current?.click();
  };

  const handleDragOver = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
    if (!isDragging) {
      setIsDragging(true);
    }
  };

  const handleDragLeave = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setIsDragging(false);
    const files = event.dataTransfer?.files ? Array.from(event.dataTransfer.files) : [];
    if (files.length > 0) {
      onFileChange(files);
    }
  };

  return (
    <div className="w-full">
      <div
        role="button"
        tabIndex={0}
        onClick={openFilePicker}
        onKeyDown={(event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            openFilePicker();
          }
        }}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`cursor-pointer flex flex-col items-center justify-center border rounded-xl transition-colors w-full min-h-[88px] px-4 py-3 text-center ${
          isDragging
            ? "bg-accent bg-opacity-20 border-accent"
            : uploadFiles.length > 0
              ? "bg-accent bg-opacity-10 border-border-high_contrast"
              : "bg-scheme-shade_4 border-border-mid_contrast hover:bg-scheme-shade_3"
        }`}
      >
        <span className="text-base font-semibold text-text-normal">{fileLabel}</span>
        <span className="text-sm text-text-slightly_less_contrast">
          {uploadFiles.length > 0 ? "Click or drop to replace selection" : "Or click to select"}
        </span>
      </div>
      <input
        ref={inputRef}
        type="file"
        multiple
        onChange={(e) => onFileChange(e.target.files ? Array.from(e.target.files) : [])}
        className="hidden"
      />
    </div>
  );
};
