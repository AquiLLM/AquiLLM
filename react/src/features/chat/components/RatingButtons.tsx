import React, { useState } from 'react';
import { Star } from 'lucide-react';

interface RatingButtonsProps {
  rating?: number;
  feedback_text?: string;
  onRate: (rating: number) => void;
  onFeedback: (text: string) => void;
}

export const RatingButtons: React.FC<RatingButtonsProps> = ({ 
  rating, 
  feedback_text, 
  onRate, 
  onFeedback 
}) => {
  const [localFeedback, setLocalFeedback] = useState(feedback_text || '');
  const [submitted, setSubmitted] = useState(!!feedback_text);
  const [feedbackExpanded, setFeedbackExpanded] = useState(!!feedback_text);
  const [panelOpen, setPanelOpen] = useState(false);
  const hasExistingFeedback = !!feedback_text || typeof rating === 'number';

  const handleSubmit = () => {
    if (localFeedback.trim()) {
      onFeedback(localFeedback.trim());
      setSubmitted(true);
    }
  };
  
  return (
    <div className="flex flex-col gap-1 mt-1">
      {!panelOpen ? (
        <div className="transition-opacity opacity-0 group-hover:opacity-100 pointer-events-none group-hover:pointer-events-auto">
          <button
            onClick={() => setPanelOpen(true)}
            className="text-[11px] text-text-low_contrast hover:text-text-normal underline-offset-2 hover:underline"
          >
            {hasExistingFeedback ? 'View feedback' : 'Rate / feedback'}
          </button>
        </div>
      ) : (
        <>
          <div className="flex items-center gap-1">
            <span className="text-xs mr-1.5 text-text-low_contrast">Rate:</span>
            {[1, 2, 3, 4, 5].map((value) => (
              <button
                key={value}
                className="p-0.5 rounded hover:bg-scheme-shade_6 transition-colors"
                onClick={() => onRate(value)}
                aria-label={`Rate ${value} star${value > 1 ? 's' : ''}`}
                title={`Rate ${value} star${value > 1 ? 's' : ''}`}
              >
                <Star
                  size={16}
                  className={(rating ?? 0) >= value ? 'text-text-less_contrast' : 'text-text-lower_contrast'}
                  fill={(rating ?? 0) >= value ? 'currentColor' : 'none'}
                />
              </button>
            ))}
            {!feedbackExpanded && (
              <button
                onClick={() => setFeedbackExpanded(true)}
                className="ml-1.5 px-1.5 py-0.5 text-[11px] rounded bg-scheme-shade_4 text-text-normal border border-border-mid_contrast hover:bg-scheme-shade_5 transition-colors"
              >
                Add feedback
              </button>
            )}
            <button
              onClick={() => setPanelOpen(false)}
              className="ml-1 px-1.5 py-0.5 text-[11px] rounded text-text-low_contrast hover:text-text-normal transition-colors"
            >
              Hide
            </button>
          </div>
          {feedbackExpanded && (
            <div className="flex gap-1.5 items-center">
              <input
                type="text"
                value={localFeedback}
                onChange={(e) => {
                  setLocalFeedback(e.target.value);
                  setSubmitted(false);
                }}
                placeholder="Please enter feedback here..."
                className="flex-grow px-2 py-1 text-xs rounded bg-scheme-shade_3 text-text-normal border border-border-mid_contrast"
                onKeyDown={(e) => e.key === 'Enter' && handleSubmit()}
              />
              {submitted ? (
                <span className="text-xs text-green-500 whitespace-nowrap">Submitted!</span>
              ) : (
                <button
                  onClick={handleSubmit}
                  className="px-2 py-1 text-xs bg-scheme-shade_4 border border-border-mid_contrast rounded text-text-normal whitespace-nowrap hover:bg-scheme-shade_5 transition-colors"
                >
                  Submit
                </button>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
};
