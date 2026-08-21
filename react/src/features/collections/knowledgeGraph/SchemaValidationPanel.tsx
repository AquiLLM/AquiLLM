import React, { useMemo } from 'react';
import type { ValidationIssue, ValidationResult } from './schemaTypes';
import { panelClass } from './schemaUiShared';

export interface SchemaValidationPanelProps {
  status: 'idle' | 'pending' | 'valid' | 'invalid';
  result: ValidationResult | null;
  activeIssueKey?: string | null;
  onIssueSelect?: (issue: ValidationIssue) => void;
}

function issueKey(issue: ValidationIssue): string {
  return `${issue.severity}:${issue.code}:${issue.location}`;
}

const SchemaValidationPanel: React.FC<SchemaValidationPanelProps> = ({
  status,
  result,
  activeIssueKey,
  onIssueSelect,
}) => {
  const grouped = useMemo(() => {
    const issues = result?.issues ?? [];
    return {
      errors: issues.filter((issue) => issue.severity === 'error'),
      warnings: issues.filter((issue) => issue.severity === 'warning'),
    };
  }, [result]);

  const renderGroup = (title: string, issues: ValidationIssue[]) => (
    <div>
      <h3 className="text-sm font-semibold mb-2">{title}</h3>
      {issues.length === 0 ? (
        <p className="text-sm text-text-lower_contrast m-0">None</p>
      ) : (
        <ul className="list-none p-0 m-0 space-y-1">
          {issues.map((issue) => {
            const key = issueKey(issue);
            const active = activeIssueKey === key;
            return (
              <li key={key}>
                <button
                  type="button"
                  className={`w-full text-left px-3 py-2 rounded-[12px] border text-sm cursor-pointer ${
                    active ? 'border-accent bg-accent/15' : 'border-border-mid_contrast bg-scheme-shade_5'
                  }`}
                  aria-pressed={active}
                  onClick={() => onIssueSelect?.(issue)}
                >
                  <span className="font-medium">{issue.code}</span>
                  <span className="text-text-lower_contrast ml-2">{issue.location}</span>
                  <div>{issue.message}</div>
                </button>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );

  return (
    <section className={`${panelClass} p-[14px]`} aria-label="Validation results">
      <header className="mb-3">
        <h2 className="text-base font-semibold">Validation</h2>
        <p className="text-sm text-text-lower_contrast m-0">Status: {status}</p>
      </header>
      {status === 'pending' ? <p className="text-sm">Validation in progress…</p> : null}
      {result ? (
        <div className="space-y-4">
          {renderGroup('Errors', grouped.errors)}
          {renderGroup('Warnings', grouped.warnings)}
        </div>
      ) : (
        <p className="text-sm text-text-lower_contrast">Run validation to inspect draft issues.</p>
      )}
    </section>
  );
};

export default SchemaValidationPanel;
