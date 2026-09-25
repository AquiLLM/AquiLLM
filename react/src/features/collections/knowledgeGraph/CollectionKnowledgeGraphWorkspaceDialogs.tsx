import React from 'react';
import type { CollectionSchemaEditorState } from './collectionSchemaReducer';
import SchemaConflictDialog from './SchemaConflictDialog';
import SchemaDiffDialog from './SchemaDiffDialog';
import SchemaPublishDialog from './SchemaPublishDialog';
import SchemaRestoreDialog from './SchemaRestoreDialog';
import type { ReviewedRebasePreview, ReviewedRebaseResolution } from './schemaFormBuffer';
import type { DraftSnapshot, SchemaConflictInfo, SchemaHistoryVersion } from './schemaTypes';

export interface CollectionKnowledgeGraphWorkspaceDialogsProps {
  collectionName: string;
  draft: DraftSnapshot | null;
  editorState: CollectionSchemaEditorState;
  diffOpen: boolean;
  publishOpen: boolean;
  restoreOpen: boolean;
  selectedHistory: SchemaHistoryVersion | null;
  conflictPreview: ReviewedRebasePreview | null;
  restoreChallengeToken?: string | null;
  impactSummary?: string | null;
  onCloseDiff: () => void;
  onClosePublish: () => void;
  onCloseRestore: () => void;
  onPublish?: () => void;
  onConfirmRestore?: () => void;
  onConflictDiscard?: () => void;
  onConflictReapply?: (resolutions: ReviewedRebaseResolution[]) => void;
}

const CollectionKnowledgeGraphWorkspaceDialogs: React.FC<
  CollectionKnowledgeGraphWorkspaceDialogsProps
> = ({
  collectionName,
  draft,
  editorState,
  diffOpen,
  publishOpen,
  restoreOpen,
  selectedHistory,
  conflictPreview,
  restoreChallengeToken,
  impactSummary,
  onCloseDiff,
  onClosePublish,
  onCloseRestore,
  onPublish,
  onConfirmRestore,
  onConflictDiscard,
  onConflictReapply,
}) => {
  const { validation, publish, conflict } = editorState;

  return (
    <>
      <SchemaDiffDialog
        isOpen={diffOpen}
        diff={validation.result?.diff_summary ?? null}
        impactSummary={impactSummary}
        onClose={onCloseDiff}
      />
      <SchemaPublishDialog
        isOpen={publishOpen}
        collectionName={collectionName}
        draftRevision={draft?.revision ?? null}
        candidateChecksum={validation.result?.identity.candidate_checksum ?? null}
        publishStatus={publish.status}
        onClose={onClosePublish}
        onConfirm={() => {
          onPublish?.();
          onClosePublish();
        }}
      />
      <SchemaRestoreDialog
        isOpen={restoreOpen}
        collectionName={collectionName}
        version={selectedHistory}
        activeDraft={draft}
        replacementChallengeToken={restoreChallengeToken}
        onClose={onCloseRestore}
        onConfirm={() => {
          onConfirmRestore?.();
          onCloseRestore();
        }}
      />
      <SchemaConflictDialog
        isOpen={Boolean(conflict)}
        conflict={conflict as SchemaConflictInfo | null}
        preview={conflictPreview}
        onClose={() => onConflictDiscard?.()}
        onDiscardLocal={() => onConflictDiscard?.()}
        onReapply={(resolutions) => onConflictReapply?.(resolutions)}
      />
    </>
  );
};

export default CollectionKnowledgeGraphWorkspaceDialogs;
