// @vitest-environment jsdom

import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import SchemaPublishDialog from './SchemaPublishDialog';

afterEach(() => cleanup());

describe('SchemaPublishDialog', () => {
  it('names collection, revision, and checksum in the confirmation', () => {
    render(
      <SchemaPublishDialog
        isOpen={true}
        collectionName="Manage Collection"
        draftRevision={5}
        candidateChecksum="candidate-checksum-v5"
        publishStatus="idle"
        onClose={vi.fn()}
        onConfirm={vi.fn()}
      />,
    );

    expect(screen.getByRole('dialog', { name: 'Publish schema draft' })).toBeTruthy();
    expect(screen.getByText(/Manage Collection/i)).toBeTruthy();
    expect(screen.getByText(/candidate-checksum-v5/i)).toBeTruthy();
  });

  it('shows polling status without claiming success and disables confirm', () => {
    const onConfirm = vi.fn();
    render(
      <SchemaPublishDialog
        isOpen={true}
        collectionName="Manage Collection"
        draftRevision={5}
        candidateChecksum="candidate-checksum-v5"
        publishStatus="polling"
        onClose={vi.fn()}
        onConfirm={onConfirm}
      />,
    );

    expect(screen.getByText(/Publish in progress/i)).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Publishing…' })).toHaveProperty('disabled', true);
    expect(onConfirm).not.toHaveBeenCalled();
  });
});
