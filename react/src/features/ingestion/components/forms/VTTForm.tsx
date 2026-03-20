import React from 'react';

interface VTTFormProps {
  vttTitle: string;
  vttFile: File | null;
  onTitleChange: (value: string) => void;
  onFileChange: (file: File | null) => void;
}

export const VTTForm: React.FC<VTTFormProps> = ({
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
          vttFile
            ? "bg-accent text-text-normal border-border-high_contrast"
            : "hover:bg-scheme-shade_3 text-text-normal"
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
