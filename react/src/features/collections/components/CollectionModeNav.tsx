import React, { useCallback, useEffect } from "react";
import {
  buildCollectionViewUrl,
  parseCollectionViewMode,
  type CollectionViewMode,
  type CollectionViewNavigationGuard,
} from "./collectionViewTypes";

export interface CollectionModeNavProps {
  activeMode: CollectionViewMode;
  onActiveModeChange: (mode: CollectionViewMode) => void;
  guardNavigation?: CollectionViewNavigationGuard;
}

const MODES: Array<{ mode: CollectionViewMode; label: string }> = [
  { mode: "files", label: "Files" },
  { mode: "knowledge-graph", label: "Knowledge Graph" },
  { mode: "visualization", label: "Visualization" },
];

const CollectionModeNav: React.FC<CollectionModeNavProps> = ({
  activeMode,
  onActiveModeChange,
  guardNavigation,
}) => {
  const currentUrlForMode = useCallback(
    (mode: CollectionViewMode) =>
      buildCollectionViewUrl(
        window.location.pathname,
        window.location.search,
        window.location.hash,
        mode,
      ),
    [],
  );

  const navigateToMode = useCallback(
    (mode: CollectionViewMode) => {
      if (mode === activeMode) return;
      if (guardNavigation?.({ type: "mode", mode }) === false) return;
      const nextUrl = currentUrlForMode(mode);
      window.history.pushState(null, "", nextUrl);
      onActiveModeChange(mode);
    },
    [activeMode, currentUrlForMode, guardNavigation, onActiveModeChange],
  );

  useEffect(() => {
    const handlePopState = () => {
      const nextMode = parseCollectionViewMode(window.location.search);
      if (nextMode === activeMode) return;
      if (guardNavigation?.({ type: "browser", mode: nextMode }) === false) {
        window.history.pushState(null, "", currentUrlForMode(activeMode));
        return;
      }
      onActiveModeChange(nextMode);
    };

    window.addEventListener("popstate", handlePopState);
    return () => window.removeEventListener("popstate", handlePopState);
  }, [activeMode, currentUrlForMode, guardNavigation, onActiveModeChange]);

  return (
    <nav
      aria-label="Collection workspace views"
      className="mb-[20px] bg-scheme-shade_4 border border-border-low_contrast rounded-[20px] p-[6px]"
    >
      <ul role="tablist" className="flex flex-wrap gap-[6px] list-none m-0 p-0">
        {MODES.map(({ mode, label }) => {
          const selected = activeMode === mode;
          return (
            <li key={mode} role="presentation">
              <button
                type="button"
                role="tab"
                id={`collection-view-tab-${mode}`}
                aria-selected={selected}
                aria-controls={`collection-view-panel-${mode}`}
                data-testid={`collection-mode-${mode}`}
                className={`h-[36px] px-[14px] rounded-[18px] text-sm font-medium transition-colors cursor-pointer border ${
                  selected
                    ? "bg-accent text-white border-accent"
                    : "bg-transparent text-text-slightly_less_contrast border-transparent hover:text-text-normal hover:bg-scheme-shade_5"
                }`}
                onClick={() => navigateToMode(mode)}
              >
                {label}
              </button>
            </li>
          );
        })}
      </ul>
    </nav>
  );
};

export default CollectionModeNav;
