// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { Collection } from "../../../components/CollectionsTree";
import CollectionViewShell from "./CollectionViewShell";
import type {
  CollectionContent,
  CollectionViewMode,
} from "./collectionViewTypes";

vi.mock("../../../components/CollectionSettingsMenu", () => ({
  default: () => <div data-testid="collection-settings-menu" />,
}));

vi.mock("../../platform_admin/components/UserManagementModal", () => ({
  default: () => null,
}));

vi.mock("./CollectionFilesWorkspace", () => ({
  default: () => (
    <div data-testid="collection-files-workspace">
      <div data-testid="collection-files-ingest">Ingest</div>
    </div>
  ),
}));

const collection: Collection = {
  id: 1,
  name: "Root",
  parent: null,
  collection: 1,
  path: "/root",
  children: [],
  document_count: 0,
  children_count: 0,
  created_at: "2026-01-01",
  updated_at: "2026-01-01",
};

const contents: CollectionContent[] = [];

function renderShell(activeMode: CollectionViewMode = "files") {
  const onActiveModeChange = vi.fn();
  render(
    <CollectionViewShell
      collection={collection}
      collectionId="1"
      breadcrumbs={[{ name: "Root", id: 1, path: "/root", fullPath: "/root" }]}
      contents={contents}
      permissionSource={null}
      allCollections={[collection]}
      activeMode={activeMode}
      onActiveModeChange={onActiveModeChange}
      initialCanEdit={true}
      initialCanManage={false}
      knowledgeGraphContent={
        <div data-testid="knowledge-graph-placeholder">Knowledge Graph</div>
      }
      visualizationContent={
        <div data-testid="visualization-placeholder">Visualization</div>
      }
      movingItem={null}
      isMoveModalOpen={false}
      batchMovingItems={[]}
      isBatchMoveModalOpen={false}
      isCreateSubcollectionOpen={false}
      successMessage={null}
      isBatchOperationLoading={false}
      isUserManagementModalOpen={false}
      onBack={vi.fn()}
      onManageCollaborators={vi.fn()}
      onDelete={vi.fn()}
      onOpenCollectionSettingsMove={vi.fn()}
      onOpenCreateSubcollection={vi.fn()}
      onCloseCreateSubcollection={vi.fn()}
      onSubmitCreateSubcollection={vi.fn()}
      onCloseMoveModal={vi.fn()}
      onMoveSubmit={vi.fn()}
      onCloseBatchMoveModal={vi.fn()}
      onBatchMoveSubmit={vi.fn()}
      fetchCollectionData={vi.fn()}
      onOpenItem={vi.fn()}
      onRemoveItem={vi.fn()}
      onContextMove={vi.fn()}
      onRenameItem={vi.fn()}
      onBatchMove={vi.fn()}
      onBatchRemove={vi.fn()}
      onCloseUserManagement={vi.fn()}
      onUserManagementSave={vi.fn()}
    />,
  );
  return { onActiveModeChange };
}

beforeEach(() => {
  window.history.replaceState(null, "", "/collections/1");
  window.apiUrls = {};
  window.pageUrls = {
    collection: "/collections/%(col_id)s/",
    user_collections: "/collections/",
  };
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("CollectionViewShell mode switching", () => {
  it("renders Files workspace by default", () => {
    renderShell("files");
    expect(screen.getByTestId("collection-view-panel-files")).toBeTruthy();
    expect(screen.getByTestId("collection-files-workspace")).toBeTruthy();
    expect(
      screen.queryByTestId("collection-view-panel-knowledge-graph"),
    ).toBeNull();
  });

  it("renders knowledge graph slot when mode is knowledge-graph", () => {
    renderShell("knowledge-graph");
    expect(
      screen.getByTestId("collection-view-panel-knowledge-graph"),
    ).toBeTruthy();
    expect(screen.getByTestId("knowledge-graph-placeholder")).toBeTruthy();
    expect(screen.queryByTestId("collection-files-workspace")).toBeNull();
  });

  it("does not render Files ingest UI in knowledge graph mode", () => {
    renderShell("knowledge-graph");
    expect(screen.queryByTestId("collection-files-ingest")).toBeNull();
  });

  it("renders the visualization slot without files or schema editor content", () => {
    renderShell("visualization");

    expect(
      screen.getByTestId("collection-view-panel-visualization"),
    ).toBeTruthy();
    expect(screen.getByTestId("visualization-placeholder")).toBeTruthy();
    expect(screen.queryByTestId("collection-files-workspace")).toBeNull();
    expect(screen.queryByTestId("knowledge-graph-placeholder")).toBeNull();
  });

  it("passes initial permission flags into the knowledge graph panel", () => {
    renderShell("knowledge-graph");
    const panel = screen.getByTestId("collection-view-panel-knowledge-graph");
    expect(panel.getAttribute("data-initial-can-edit")).toBe("true");
    expect(panel.getAttribute("data-initial-can-manage")).toBe("false");
  });

  it("requests mode change through CollectionModeNav", () => {
    const pushState = vi.spyOn(window.history, "pushState");
    const { onActiveModeChange } = renderShell("files");

    fireEvent.click(screen.getByRole("tab", { name: "Knowledge Graph" }));

    expect(pushState).toHaveBeenCalledWith(
      null,
      "",
      "/collections/1?view=knowledge-graph",
    );
    expect(onActiveModeChange).toHaveBeenCalledWith("knowledge-graph");
  });
});
