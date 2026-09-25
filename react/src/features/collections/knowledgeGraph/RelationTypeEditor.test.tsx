// @vitest-environment jsdom

import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import RelationTypeEditor from './RelationTypeEditor';
import { editDraftEnvelope } from './schemaTestFixtures';

afterEach(() => cleanup());

describe('RelationTypeEditor', () => {
  const entityTypes = editDraftEnvelope.draft!.entities;
  const definition = editDraftEnvelope.draft!.relations[0];

  it('renders endpoint pickers from effective entity types', () => {
    render(
      <RelationTypeEditor
        collectionName="Demo"
        draftRevision={2}
        definition={definition}
        entityTypes={entityTypes}
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

    expect(screen.getByRole('listbox', { name: 'Allowed head entity types' })).toBeTruthy();
    expect(screen.getByRole('listbox', { name: 'Allowed tail entity types' })).toBeTruthy();
    expect(screen.getAllByText('person').length).toBeGreaterThan(0);
  });

  it('requires at least one head and tail selection before save', () => {
    const onSave = vi.fn();
    render(
      <RelationTypeEditor
        collectionName="Demo"
        draftRevision={2}
        definition={definition}
        entityTypes={entityTypes}
        constraints={editDraftEnvelope.constraints}
        values={{ ...definition.values, allowed_head_types: [], allowed_tail_types: [] }}
        dirty={true}
        pending={false}
        readOnly={false}
        onFieldChange={vi.fn()}
        onSave={onSave}
        onRevert={vi.fn()}
      />,
    );

    expect((screen.getByRole('button', { name: 'Save' }) as HTMLButtonElement).disabled).toBe(true);
  });
});
