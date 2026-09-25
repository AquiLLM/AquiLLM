// @vitest-environment jsdom

import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import SchemaDraftToolbar from './SchemaDraftToolbar';
import { editDraftEnvelope, manageDraftEnvelope, viewPublishedEnvelope } from './schemaTestFixtures';

afterEach(() => cleanup());

describe('SchemaDraftToolbar permission matrix', () => {
  it('shows inspect-only controls for VIEW', () => {
    render(
      <SchemaDraftToolbar
        collectionName="View Collection"
        permissions={viewPublishedEnvelope.permissions}
        published={viewPublishedEnvelope.published}
        draft={null}
        dirty={false}
        pendingOperation={null}
        validationStatus="idle"
        validationResult={null}
        publishStatus="idle"
        canValidate={false}
        canPublish={false}
        onShowHistory={vi.fn()}
      />,
    );

    expect(screen.getByText(/View-only access/i)).toBeTruthy();
    expect(screen.queryByRole('button', { name: 'Create draft' })).toBeNull();
    expect(screen.queryByRole('button', { name: 'Publish' })).toBeNull();
  });

  it('shows create and validate for EDIT without publish or discard', () => {
    render(
      <SchemaDraftToolbar
        collectionName="Edit Collection"
        permissions={editDraftEnvelope.permissions}
        published={editDraftEnvelope.published}
        draft={editDraftEnvelope.draft}
        dirty={false}
        pendingOperation={null}
        validationStatus="idle"
        validationResult={null}
        publishStatus="idle"
        canValidate={true}
        canPublish={false}
        onValidate={vi.fn()}
      />,
    );

    expect(screen.getByRole('button', { name: 'Validate' })).toBeTruthy();
    expect(screen.queryByRole('button', { name: 'Publish' })).toBeNull();
    expect(screen.queryByRole('button', { name: 'Discard draft' })).toBeNull();
  });

  it('shows publish and discard for MANAGE when allowed', () => {
    render(
      <SchemaDraftToolbar
        collectionName="Manage Collection"
        permissions={manageDraftEnvelope.permissions}
        published={manageDraftEnvelope.published}
        draft={manageDraftEnvelope.draft}
        dirty={false}
        pendingOperation={null}
        validationStatus="valid"
        validationResult={null}
        publishStatus="idle"
        canValidate={true}
        canPublish={true}
        onPublish={vi.fn()}
        onDiscard={vi.fn()}
      />,
    );

    expect(screen.getByRole('button', { name: 'Publish' })).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Discard draft' })).toBeTruthy();
  });
});
