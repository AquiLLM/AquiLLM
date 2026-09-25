import React from 'react';
import type { PublishStatus } from './schemaTypes';
import SchemaModalShell from './SchemaModalShell';
import { buttonPrimaryClass, buttonSecondaryClass } from './schemaUiShared';

export interface SchemaPublishDialogProps {
  isOpen: boolean;
  collectionName: string;
  draftRevision: number | null;
  candidateChecksum: string | null;
  publishStatus: PublishStatus;
  onClose: () => void;
  onConfirm: () => void;
}

const SchemaPublishDialog: React.FC<SchemaPublishDialogProps> = ({
  isOpen,
  collectionName,
  draftRevision,
  candidateChecksum,
  publishStatus,
  onClose,
  onConfirm,
}) => {
  const busy = publishStatus === 'pending' || publishStatus === 'polling';

  return (
    <SchemaModalShell
      isOpen={isOpen}
      title="Publish schema draft"
      onClose={onClose}
      allowEscape={!busy}
      footer={
        <>
          <button type="button" className={buttonSecondaryClass} disabled={busy} onClick={onClose}>
            Cancel
          </button>
          <button type="button" className={buttonPrimaryClass} disabled={busy} onClick={onConfirm}>
            {busy ? 'Publishing…' : 'Confirm publish'}
          </button>
        </>
      }
    >
      <p>
        Publish the validated draft for <strong>{collectionName}</strong> at revision{' '}
        <strong>{draftRevision ?? '—'}</strong> with checksum <strong>{candidateChecksum ?? '—'}</strong>?
      </p>
      {publishStatus === 'polling' ? (
        <p role="status">Publish in progress. The candidate is not published until completion.</p>
      ) : null}
      {publishStatus === 'failed' ? (
        <p role="alert" className="text-red-200">
          Publish failed. The draft remains editable.
        </p>
      ) : null}
    </SchemaModalShell>
  );
};

export default SchemaPublishDialog;
