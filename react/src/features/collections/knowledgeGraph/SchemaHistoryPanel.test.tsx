// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import SchemaHistoryPanel from './SchemaHistoryPanel';
import { historyPageFixture, manageDraftEnvelope, viewPublishedEnvelope } from './schemaTestFixtures';

afterEach(() => cleanup());

describe('SchemaHistoryPanel', () => {
  it('lists paginated versions for VIEW users without restore', () => {
    render(
      <SchemaHistoryPanel
        permissions={viewPublishedEnvelope.permissions}
        history={historyPageFixture}
        loading={false}
        error={null}
        selectedVersion={null}
        onSelectVersion={vi.fn()}
        onLoadMore={vi.fn()}
      />,
    );

    expect(screen.getByText('Version 4')).toBeTruthy();
    expect(screen.queryByRole('button', { name: 'Restore version' })).toBeNull();
    expect(screen.getByRole('button', { name: 'Load more' })).toBeTruthy();
  });

  it('offers restore for MANAGE when a version is selected', () => {
    const onRestore = vi.fn();
    render(
      <SchemaHistoryPanel
        permissions={manageDraftEnvelope.permissions}
        history={historyPageFixture}
        loading={false}
        error={null}
        selectedVersion={historyPageFixture.versions[0]}
        onSelectVersion={vi.fn()}
        onRestore={onRestore}
      />,
    );

    fireEvent.click(screen.getByRole('button', { name: 'Restore version' }));
    expect(onRestore).toHaveBeenCalledWith(historyPageFixture.versions[0]);
  });
});
