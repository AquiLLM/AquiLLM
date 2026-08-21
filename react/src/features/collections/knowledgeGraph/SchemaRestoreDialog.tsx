import React from 'react';
import type { DraftSnapshot, SchemaHistoryVersion } from './schemaTypes';
import SchemaModalShell from './SchemaModalShell';
import { buttonDangerClass, buttonPrimaryClass, buttonSecondaryClass } from './schemaUiShared';

export interface SchemaRestoreDialogProps {
  isOpen: boolean;
  collectionName: string;
  version: SchemaHistoryVersion | null;
  activeDraft: DraftSnapshot | null;
  replacementChallengeToken?: string | null;
  pending?: boolean;
  onClose: () => void;
  onConfirm: () => void;
}

const SchemaRestoreDialog: React.FC<SchemaRestoreDialogProps> = ({
  isOpen,
  collectionName,
  version,
  activeDraft,
  replacementChallengeToken,
  pending = false,
  onClose,
  onConfirm,
}) => {
  const requiresReplacement = Boolean(activeDraft && replacementChallengeToken);

  return (
    <SchemaModalShell
      isOpen={isOpen}
      title="Restore published version"
      onClose={onClose}
      allowEscape={!pending}
      footer={
        <>
          <button type="button" className={buttonSecondaryClass} disabled={pending} onClick={onClose}>
            Cancel
          </button>
          <button
            type="button"
            className={requiresReplacement ? buttonDangerClass : buttonPrimaryClass}
            disabled={pending || !version}
            onClick={onConfirm}
          >
            {requiresReplacement ? 'Replace draft and restore' : 'Restore to new draft'}
          </button>
        </>
      }
    >
      {version ? (
        <div className="space-y-3 text-sm">
          <p>
            Restore version <strong>{version.version}</strong> ({version.checksum}) for{' '}
            <strong>{collectionName}</strong> into a new draft that requires validation and publish.
          </p>
          {activeDraft ? (
            <p role="alert" className="text-amber-100">
              A shared draft already exists ({activeDraft.draft_id}, revision {activeDraft.revision}, last editor{' '}
              {activeDraft.last_editor}). Restoring requires an atomic replace confirmed with challenge token{' '}
              <strong>{replacementChallengeToken}</strong>.
            </p>
          ) : null}
        </div>
      ) : (
        <p>Select a version to restore.</p>
      )}
    </SchemaModalShell>
  );
};

export default SchemaRestoreDialog;
