import React from 'react';

interface WebpageFormProps {
  urlValue: string;
  depthValue: number;
  onUrlChange: (value: string) => void;
  onDepthChange: (value: number) => void;
}

export const WebpageForm: React.FC<WebpageFormProps> = ({
  urlValue,
  depthValue,
  onUrlChange,
  onDepthChange,
}) => {
  const handleDepthChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseInt(e.target.value, 10);
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
           min="0"
           value={depthValue}
           onChange={handleDepthChange}
           className="bg-scheme-shade_3 border border-border-high_contrast p-2 rounded-lg h-[40px] w-20 text-center"
         />
      </div>
    </div>
  );
};
