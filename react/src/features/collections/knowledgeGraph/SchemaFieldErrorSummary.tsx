import React from 'react';

export interface SchemaFieldErrorSummaryProps {
  errors: Record<string, string>;
  onFocusField?: (field: string) => void;
  headingId?: string;
}

const SchemaFieldErrorSummary: React.FC<SchemaFieldErrorSummaryProps> = ({
  errors,
  onFocusField,
  headingId,
}) => {
  const entries = Object.entries(errors);
  if (entries.length === 0) return null;

  return (
    <div
      role="alert"
      aria-labelledby={headingId}
      className="mb-3 rounded-[16px] border border-red-400/40 bg-scheme-shade_5 p-3"
      data-testid="schema-field-error-summary"
    >
      <p id={headingId} className="text-sm font-medium text-red-200 mb-2">
        Fix the following fields:
      </p>
      <ul className="list-disc pl-5 m-0 space-y-1">
        {entries.map(([field, message]) => (
          <li key={field} className="text-sm text-red-100">
            {onFocusField ? (
              <button
                type="button"
                className="underline text-left cursor-pointer bg-transparent border-none p-0 text-red-100"
                onClick={() => onFocusField(field)}
              >
                {field}: {message}
              </button>
            ) : (
              <>
                {field}: {message}
              </>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
};

export default SchemaFieldErrorSummary;
