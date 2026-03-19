import React from 'react';

interface HandwrittenFormProps {
  handwrittenTitle: string;
  handwrittenFile: File | null;
  convertToLatex: boolean;
  onTitleChange: (value: string) => void;
  onFileChange: (file: File | null) => void;
  onConvertChange: (value: boolean) => void;
}

export const HandwrittenForm: React.FC<HandwrittenFormProps> = ({
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
            handwrittenFile
              ? "bg-accent text-text-normal border-border-high_contrast"
              : "hover:bg-scheme-shade_3 text-text-normal"
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
