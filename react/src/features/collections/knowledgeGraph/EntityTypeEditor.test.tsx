// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import EntityTypeEditor from './EntityTypeEditor';
import { editDraftEnvelope } from './schemaTestFixtures';

afterEach(() => cleanup());

describe('EntityTypeEditor', () => {
  const definition = editDraftEnvelope.draft!.entities[0];

  it('renders entity fields and respects editable capabilities', () => {
    render(
      <EntityTypeEditor
        collectionName="Demo"
        draftRevision={2}
        definition={definition}
        constraints={editDraftEnvelope.constraints}
        values={definition.values as unknown as Record<string, unknown>}
        dirty={false}
        pending={false}
        readOnly={false}
        onFieldChange={vi.fn()}
        onSave={vi.fn()}
        onRevert={vi.fn()}
      />,
    );

    expect((screen.getByLabelText('Description') as HTMLInputElement).disabled).toBe(false);
    expect((screen.getByLabelText('Name') as HTMLInputElement).disabled).toBe(true);
  });

  it('keeps inherited fields read-only for MANAGE when not in editable_fields', () => {
    render(
      <EntityTypeEditor
        collectionName="Demo"
        draftRevision={2}
        definition={definition}
        constraints={editDraftEnvelope.constraints}
        values={definition.values as unknown as Record<string, unknown>}
        dirty={false}
        pending={false}
        readOnly={true}
        onFieldChange={vi.fn()}
        onSave={vi.fn()}
        onRevert={vi.fn()}
      />,
    );

    expect(screen.getByText(/Read-only: server capabilities prevent editing/i)).toBeTruthy();
  });

  it('calls save when dirty and valid', () => {
    const onSave = vi.fn();
    render(
      <EntityTypeEditor
        collectionName="Demo"
        draftRevision={2}
        definition={definition}
        constraints={editDraftEnvelope.constraints}
        values={{ ...definition.values, description: 'Changed' }}
        dirty={true}
        pending={false}
        readOnly={false}
        onFieldChange={vi.fn()}
        onSave={onSave}
        onRevert={vi.fn()}
      />,
    );

    fireEvent.click(screen.getByRole('button', { name: 'Save' }));
    expect(onSave).toHaveBeenCalled();
  });
});
