import React from 'react';

interface ArxivFormProps {
  value: string;
  onValueChange: (value: string) => void;
}

export const ArxivForm: React.FC<ArxivFormProps> = ({ value, onValueChange }) => {
  return (
    <input
      type="text"
      placeholder="Enter arXiv ID"
      value={value}
      onChange={(e) => onValueChange(e.target.value)}
      className="bg-scheme-shade_3 border border-border-mid_contrast p-2 rounded-lg h-[40px] placeholder:text-text-less_contrast"
    />
  );
};
