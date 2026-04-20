import React, { useState } from 'react';
import { Bug } from 'lucide-react';
import BugReportModal from './BugReportModal';

const BugReportButton: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <>
      <button
        onClick={() => setIsOpen(true)}
        className="h-[40px] w-[40px] rounded-[14px] hover:bg-scheme-shade_6 transition-all flex items-center justify-center cursor-pointer border border-scheme-contrast text-text-normal shrink-0"
        title="Report a bug"
      >
        <Bug size={18} />
      </button>
      <BugReportModal isOpen={isOpen} onClose={() => setIsOpen(false)} />
    </>
  );
};

export default BugReportButton;
