import React from 'react';

interface ToolValueProps {
  value: string | number | boolean;
}

export const ToolValue: React.FC<ToolValueProps> = ({ value }) => {
  if (typeof value === 'string') {
    return <span className="font-mono text-text-non_user_text_bubble">{`"${value}"`}</span>;
  }
  return <span className="font-mono text-text-slightly_less_contrast">{String(value)}</span>;
};

interface ToolResultProps {
  result: any;
  level?: number;
}

export const ToolResult: React.FC<ToolResultProps> = ({ result, level = 0 }) => {
  if (typeof result === 'object' && result !== null) {
    if (Array.isArray(result)) {
      return (
        <details open={level < 1} className="mt-1">
          <summary className="cursor-pointer font-mono hover:text-accent">Array</summary>
          <div className="pl-4 border-l-2 border-border-mid_contrast mt-1 text-text-non_user_text_bubble">
            {result.map((item, index) => (
              <ToolResult key={index} result={item} level={level + 1} />
            ))}
          </div>
        </details>
      );
    } else {
      return (
        <div>
          {Object.entries(result).map(([key, value], index) =>
            typeof value === 'object' && value !== null ? (
              <details key={index} open={level < 1} className="mt-1">
                <summary className="cursor-pointer font-mono hover:text-accent">{key}</summary>
                <div className="pl-4 border-l-2 border-border-mid_contrast mt-1 text-text-non_user_text_bubble">
                  <ToolResult result={value} level={level + 1} />
                </div>
              </details>
            ) : (
              <div key={index} className="mt-1">
                <span className="font-mono">{key}: </span>
                <ToolValue value={value as string | number | boolean} />
              </div>
            )
          )}
        </div>
      );
    }
  }
  return <ToolValue value={result} />;
};
