import React from 'react';
import type { SchemaHistoryPage, SchemaHistoryVersion, SchemaPermissionsSnapshot } from './schemaTypes';
import { buttonPrimaryClass, buttonSecondaryClass, panelClass } from './schemaUiShared';

export interface SchemaHistoryPanelProps {
  permissions: SchemaPermissionsSnapshot;
  history: SchemaHistoryPage | null;
  loading: boolean;
  error: string | null;
  selectedVersion: SchemaHistoryVersion | null;
  onSelectVersion: (version: SchemaHistoryVersion) => void;
  onLoadMore?: () => void;
  onRestore?: (version: SchemaHistoryVersion) => void;
  onInspectDiff?: (version: SchemaHistoryVersion) => void;
}

const SchemaHistoryPanel: React.FC<SchemaHistoryPanelProps> = ({
  permissions,
  history,
  loading,
  error,
  selectedVersion,
  onSelectVersion,
  onLoadMore,
  onRestore,
  onInspectDiff,
}) => (
  <section className={`${panelClass} p-[14px]`} aria-label="Published schema history">
    <header className="mb-3 flex items-center justify-between gap-2">
      <h2 className="text-base font-semibold m-0">Version history</h2>
      {loading ? <span className="text-sm text-text-lower_contrast">Loading…</span> : null}
    </header>

    {error ? (
      <p role="alert" className="text-sm text-red-200">
        {error}
      </p>
    ) : null}

    <ul className="list-none p-0 m-0 space-y-2" role="list">
      {(history?.versions ?? []).map((version) => {
        const selected = selectedVersion?.version === version.version;
        return (
          <li key={version.version}>
            <button
              type="button"
              className={`w-full text-left px-3 py-2 rounded-[14px] border cursor-pointer ${
                selected ? 'border-accent bg-accent/15' : 'border-border-mid_contrast bg-scheme-shade_5'
              }`}
              aria-pressed={selected}
              onClick={() => onSelectVersion(version)}
            >
              <div className="font-medium text-sm">Version {version.version}</div>
              <div className="text-xs text-text-lower_contrast">{version.published_at}</div>
              <div className="text-xs">{version.checksum}</div>
              <div className="text-sm mt-1">{version.summary}</div>
            </button>
            {selected ? (
              <div className="mt-2 flex flex-wrap gap-2">
                {onInspectDiff ? (
                  <button type="button" className={buttonSecondaryClass} onClick={() => onInspectDiff(version)}>
                    Inspect diff
                  </button>
                ) : null}
                {permissions.can_restore && onRestore ? (
                  <button type="button" className={buttonPrimaryClass} onClick={() => onRestore(version)}>
                    Restore version
                  </button>
                ) : null}
              </div>
            ) : null}
          </li>
        );
      })}
    </ul>

    {history?.has_more && onLoadMore ? (
      <button type="button" className={`${buttonSecondaryClass} mt-3`} disabled={loading} onClick={onLoadMore}>
        Load more
      </button>
    ) : null}

    {!loading && !error && (history?.versions.length ?? 0) === 0 ? (
      <p className="text-sm text-text-lower_contrast">No published versions yet.</p>
    ) : null}
  </section>
);

export default SchemaHistoryPanel;
