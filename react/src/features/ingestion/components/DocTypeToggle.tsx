import React from 'react';
import { FileText, FileUp, Headphones, LinkIcon, PenLine } from "lucide-react";
import { DocType } from '../types';
import { ArxivLogo } from '../../../shared/components/logos/ArxivLogo';

const selectedClasses = "bg-accent text-text-normal border border-border-high_contrast shadow-sm";
const unselectedClasses = "bg-scheme-shade_5 hover:bg-scheme-shade_6 text-text-normal border border-border-high_contrast";

interface DocTypeToggleProps {
  docType: DocType;
  setDocType: (docType: DocType) => void;
}

export const DocTypeToggle: React.FC<DocTypeToggleProps> = ({
  docType,
  setDocType,
}) => {
  return (
    <div className="flex space-x-4">
      <button
        onClick={() => setDocType(DocType.UPLOADS)}
        title="Upload Files"
        className={`flex items-center h-[40px] w-[40px] justify-center px-2 py-1 rounded-lg transition-colors element-border ${
          docType === DocType.UPLOADS ? selectedClasses : unselectedClasses
        }`}
      >
        <FileUp size={18} />
      </button>

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
        {ArxivLogo}
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
