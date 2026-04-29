import React, { useState } from 'react';

type Props = {
  value: string;
  onChange: (v: string) => void;
  onRun: () => void;
  onCopyLink: () => void;
  onClear: () => void;
};

const QueryEditor: React.FC<Props> = ({ value, onChange, onRun, onCopyLink, onClear }) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    onCopyLink();
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="mb-4">
      <div className="relative">
        <textarea
          rows={4}
          spellCheck={false}
          placeholder="messages | where rating < 3 | limit 20"
          className="w-full font-mono text-sm rounded-lg px-4 py-3 bg-scheme-shade_3 element-border text-text-normal placeholder-text-muted resize-y focus:outline-none focus:ring-2 focus:ring-blue-500/50"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={(e) => {
            if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
              e.preventDefault();
              onRun();
            }
          }}
        />
      </div>
      <div className="flex items-center gap-3 mt-2">
        <button
          onClick={onRun}
          className="px-5 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-700 text-white text-sm font-semibold active:scale-95 transition-all shadow-sm"
        >
          Run
        </button>
        <button
          onClick={handleCopy}
          className="px-4 py-1.5 rounded-lg bg-scheme-shade_5 hover:bg-scheme-shade_6 text-sm transition-all"
        >
          {copied ? 'Copied!' : 'Copy link'}
        </button>
        <button
          onClick={onClear}
          className="px-4 py-1.5 rounded-lg bg-scheme-shade_5 hover:bg-scheme-shade_6 text-sm transition-all"
        >
          Clear
        </button>
      </div>
    </div>
  );
};

export default QueryEditor;
