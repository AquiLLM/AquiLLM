// ExportButton.tsx
// builds the export url from current filter query string and triggers a
// browser file download — does not load data into memory, the browser
// handles the streaming response directly

import React, { useState } from 'react';

interface ExportButtonProps {
  apiExport: string;
  exportQueryString: string;
  totalCount: number;
}

const ExportButton: React.FC<ExportButtonProps> = ({
  apiExport,
  exportQueryString,
  totalCount,
}) => {
  const [clicked, setClicked] = useState(false);

  const handleExport = () => {
    // build the full url with current filter params
    const url = exportQueryString
      ? `${apiExport}?${exportQueryString}`
      : apiExport;

    // create a hidden anchor and click it — this lets the browser handle the
    // streaming response including content-disposition attachment header
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = ''; // filename comes from content-disposition
    document.body.appendChild(anchor);
    anchor.click();
    document.body.removeChild(anchor);

    // brief visual feedback
    setClicked(true);
    setTimeout(() => setClicked(false), 1500);
  };

  const label = totalCount > 0
    ? `Export CSV (${totalCount.toLocaleString()} rows)`
    : 'Export CSV';

  return (
    <button
      onClick={handleExport}
      disabled={totalCount === 0}
      className={`px-4 py-[7px] rounded-[8px] text-sm border transition-colors font-medium
        ${clicked
          ? 'bg-green text-slight_muted_white border-green'
          : 'bg-scheme-shade_4 border-border-mid_contrast text-text-normal hover:bg-scheme-shade_5'
        }
        disabled:opacity-40 disabled:cursor-not-allowed`}
    >
      {clicked ? '✓ Downloading…' : label}
    </button>
  );
};

export default ExportButton;