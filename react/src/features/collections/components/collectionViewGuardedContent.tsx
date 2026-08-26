import React from "react";
import type { Collection } from "../../../components/CollectionsTree";
import CollectionKnowledgeGraphWorkspace from "../knowledgeGraph/CollectionKnowledgeGraphWorkspace";
import { useCollectionSchemaEditor } from "../knowledgeGraph/useCollectionSchemaEditor";
import { useSchemaNavigationGuard } from "./CollectionSchemaNavigationGuard";
import type { CollectionContent } from "./collectionViewTypes";
import CollectionViewShell, {
  type CollectionViewShellProps,
} from "./CollectionViewShell";
import type { CollectionViewMode } from "./collectionViewTypes";
import type { FileSystemItem } from "../../../types/FileSystemItem";
import CollectionGraphVisualization from "../visualization/CollectionGraphVisualization";

function KnowledgeGraphWorkspaceSection({
  collectionId,
  collectionName,
  initialCanEdit,
  initialCanManage,
}: {
  collectionId: string;
  collectionName: string;
  initialCanEdit: boolean;
  initialCanManage: boolean;
}) {
  const { registerDirtyState, requestSelectionChange } =
    useSchemaNavigationGuard();
  const schemaEditor = useCollectionSchemaEditor({
    collectionId,
    collectionName,
    registerDirtyState,
    requestSelectionChange,
  });

  return (
    <CollectionKnowledgeGraphWorkspace
      collectionId={collectionId}
      collectionName={collectionName}
      initialCanEdit={initialCanEdit}
      initialCanManage={initialCanManage}
      {...schemaEditor}
    />
  );
}

export interface CollectionViewGuardedContentProps {
  collection: Collection;
  collectionId: string;
  breadcrumbs: Array<{
    name: string;
    id: number | null;
    path: string;
    fullPath: string;
  }>;
  contents: CollectionContent[];
  permissionSource: CollectionViewShellProps["permissionSource"];
  allCollections: Collection[];
  activeMode: CollectionViewMode;
  onActiveModeChange: (mode: CollectionViewMode) => void;
  initialCanEdit: boolean;
  initialCanManage: boolean;
  movingItem: FileSystemItem | Collection | null;
  isMoveModalOpen: boolean;
  batchMovingItems: FileSystemItem[];
  isBatchMoveModalOpen: boolean;
  isCreateSubcollectionOpen: boolean;
  successMessage: string | null;
  isBatchOperationLoading: boolean;
  isUserManagementModalOpen: boolean;
  onBack: () => void;
  onManageCollaborators: () => void;
  onDelete: () => void;
  onOpenCollectionSettingsMove: () => void;
  onOpenCreateSubcollection: () => void;
  onCloseCreateSubcollection: () => void;
  onSubmitCreateSubcollection: (collection: Collection) => void;
  onCloseMoveModal: () => void;
  onMoveSubmit: (itemId: number, newParentId: number | null) => void;
  onCloseBatchMoveModal: () => void;
  onBatchMoveSubmit: (newParentId: number | null) => void;
  fetchCollectionData: () => void;
  onOpenItem: (item: FileSystemItem) => void;
  onRemoveItem: (item: FileSystemItem) => void;
  onContextMove: (item: FileSystemItem) => void;
  onRenameItem: () => void;
  onBatchMove: (items: FileSystemItem[]) => void;
  onBatchRemove: (items: FileSystemItem[]) => void;
  onCloseUserManagement: () => void;
  onUserManagementSave: () => void;
}

const CollectionViewGuardedContent: React.FC<
  CollectionViewGuardedContentProps
> = (props) => {
  const { guardNavigation } = useSchemaNavigationGuard();
  const {
    collection,
    collectionId,
    activeMode,
    initialCanEdit,
    initialCanManage,
    ...shellProps
  } = props;

  return (
    <CollectionViewShell
      collection={collection}
      collectionId={collectionId}
      activeMode={activeMode}
      guardNavigation={guardNavigation}
      initialCanEdit={initialCanEdit}
      initialCanManage={initialCanManage}
      knowledgeGraphContent={
        activeMode === "knowledge-graph" ? (
          <KnowledgeGraphWorkspaceSection
            collectionId={collectionId}
            collectionName={collection.name}
            initialCanEdit={initialCanEdit}
            initialCanManage={initialCanManage}
          />
        ) : null
      }
      visualizationContent={
        activeMode === "visualization" ? (
          <CollectionGraphVisualization collectionId={collectionId} />
        ) : null
      }
      {...shellProps}
    />
  );
};

export default CollectionViewGuardedContent;
