import React from 'react';
import type { SchemaDiffSummary } from './schemaTypes';
import SchemaModalShell from './SchemaModalShell';
import { buttonPrimaryClass, buttonSecondaryClass } from './schemaUiShared';

export interface SchemaDiffDialogProps {
  isOpen: boolean;
  diff: SchemaDiffSummary | null;
  impactSummary?: string | null;
  onClose: () => void;
  onContinue?: () => void;
}

function countBlock(label: string, counts: SchemaDiffSummary['entities']) {
  return (
    <p className="text-sm m-0">
      {label}: added {counts.added}, changed {counts.changed}, removed {counts.removed}
    </p>
  );
}

const SchemaDiffDialog: React.FC<SchemaDiffDialogProps> = ({
  isOpen,
  diff,
  impactSummary,
  onClose,
  onContinue,
}) => (
  <SchemaModalShell
    isOpen={isOpen}
    title="Schema diff review"
    onClose={onClose}
    footer={
      <>
        <button type="button" className={buttonSecondaryClass} onClick={onClose}>
          Close
        </button>
        {onContinue ? (
          <button type="button" className={buttonPrimaryClass} onClick={onContinue}>
            Continue
          </button>
        ) : null}
      </>
    }
  >
    {diff ? (
      <div className="space-y-3 text-sm">
        <p>
          Base v{diff.base_version} ({diff.base_checksum}) → candidate v{diff.candidate_version} (
          {diff.candidate_checksum})
        </p>
        {countBlock('Entity types', diff.entities)}
        {countBlock('Relation types', diff.relations)}
        {impactSummary ? (
          <p className="text-text-slightly_less_contrast">{impactSummary}</p>
        ) : null}
      </div>
    ) : (
      <p>No diff summary available.</p>
    )}
  </SchemaModalShell>
);

export default SchemaDiffDialog;
