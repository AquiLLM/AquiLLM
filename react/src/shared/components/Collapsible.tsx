import React from 'react';

interface CollapsibleProps {
  summary: string;
  summaryTextColor: string;
  content: React.ReactNode;
  isOpen?: boolean;
}

export const Collapsible: React.FC<CollapsibleProps> = ({ 
  summary, 
  summaryTextColor, 
  content, 
  isOpen = false 
}) => {
  return (
    <details className="mt-1" open={isOpen}>
      <summary className={`cursor-pointer ${summaryTextColor}`}>
        {summary}
      </summary>
      <div className="mt-1 pl-2 border-l border-border-mid_contrast">
        {content}
      </div>
    </details>
  );
};
