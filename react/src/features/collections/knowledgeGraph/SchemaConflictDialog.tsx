import React, { useMemo, useState } from 'react';
import type { SchemaConflictInfo } from './schemaTypes';
import type { ReviewedRebasePreview, ReviewedRebaseResolution } from './schemaFormBuffer';
import SchemaModalShell from './SchemaModalShell';
import { buttonDangerClass, buttonPrimaryClass, buttonSecondaryClass } from './schemaUiShared';

export interface SchemaConflictDialogProps {
  isOpen: boolean;
  conflict: SchemaConflictInfo | null;
  preview: ReviewedRebasePreview | null;
  pending?: boolean;
  onClose: () => void;
  onDiscardLocal: () => void;
  onReapply: (resolutions: ReviewedRebaseResolution[]) => void;
}

const SchemaConflictDialog: React.FC<SchemaConflictDialogProps> = ({
  isOpen,
  conflict,
  preview,
  pending = false,
  onClose,
  onDiscardLocal,
  onReapply,
}) => {
  const [choices, setChoices] = useState<Record<string, 'local' | 'server'>>({});

  const conflicts = preview?.conflicts ?? [];
  const allResolved = useMemo(
    () => conflicts.every((item) => choices[item.field] === 'local' || choices[item.field] === 'server'),
    [choices, conflicts],
  );

  const stagedFields = preview ? Object.keys(preview.autoStaged) : [];

  return (
    <SchemaModalShell
      isOpen={isOpen}
      title="Resolve schema conflict"
      onClose={onClose}
      allowEscape={!pending}
      footer={
        <>
          <button type="button" className={buttonSecondaryClass} disabled={pending} onClick={onClose}>
            Cancel
          </button>
          <button type="button" className={buttonDangerClass} disabled={pending} onClick={onDiscardLocal}>
            Discard local changes
          </button>
          <button
            type="button"
            className={buttonPrimaryClass}
            disabled={pending || (conflicts.length > 0 && !allResolved)}
            onClick={() =>
              onReapply(
                conflicts.map((item) => ({
                  field: item.field,
                  choice: choices[item.field] ?? 'server',
                })),
              )
            }
          >
            Reapply my changes
          </button>
        </>
      }
    >
      {conflict ? (
        <div className="space-y-3 text-sm">
          <p>
            Draft <strong>{conflict.draft_id}</strong> changed from revision {conflict.attempted_revision} to{' '}
            {conflict.current_revision}.
          </p>
          {stagedFields.length > 0 ? (
            <p>These non-conflicting fields will be kept automatically: {stagedFields.join(', ')}</p>
          ) : null}
          {conflicts.length > 0 ? (
            <ul className="list-none p-0 m-0 space-y-3">
              {conflicts.map((item) => (
                <li key={item.field} className="rounded-[12px] border border-border-mid_contrast p-3">
                  <p className="font-medium m-0 mb-2">{item.field}</p>
                  <label className="flex items-center gap-2 mb-1">
                    <input
                      type="radio"
                      name={`conflict-${item.field}`}
                      checked={choices[item.field] === 'local'}
                      onChange={() => setChoices((prev) => ({ ...prev, [item.field]: 'local' }))}
                    />
                    Keep my value: {String(item.localValue)}
                  </label>
                  <label className="flex items-center gap-2">
                    <input
                      type="radio"
                      name={`conflict-${item.field}`}
                      checked={choices[item.field] === 'server'}
                      onChange={() => setChoices((prev) => ({ ...prev, [item.field]: 'server' }))}
                    />
                    Use server value: {String(item.serverValue)}
                  </label>
                </li>
              ))}
            </ul>
          ) : (
            <p>No field-level conflicts remain. You can reapply staged changes against the latest revision.</p>
          )}
        </div>
      ) : (
        <p>No conflict details available.</p>
      )}
    </SchemaModalShell>
  );
};

export default SchemaConflictDialog;
