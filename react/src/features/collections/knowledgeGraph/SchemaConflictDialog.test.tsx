// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import SchemaConflictDialog from './SchemaConflictDialog';
import { conflictInfoFixture } from './schemaTestFixtures';

afterEach(() => cleanup());

describe('SchemaConflictDialog', () => {
  it('requires explicit choices before reapply on conflicting fields', () => {
    const onReapply = vi.fn();
    render(
      <SchemaConflictDialog
        isOpen={true}
        conflict={conflictInfoFixture}
        preview={{
          autoStaged: {},
          conflicts: [
            {
              field: 'description',
              baseValue: 'Base',
              localValue: 'Local unsaved description',
              serverValue: 'Server accepted description',
            },
          ],
        }}
        onClose={vi.fn()}
        onDiscardLocal={vi.fn()}
        onReapply={onReapply}
      />,
    );

    const reapply = screen.getByRole('button', { name: 'Reapply my changes' });
    expect(reapply).toHaveProperty('disabled', true);
    fireEvent.click(screen.getByLabelText(/Keep my value/i));
    fireEvent.click(reapply);
    expect(onReapply).toHaveBeenCalledWith([{ field: 'description', choice: 'local' }]);
  });

  it('discards local changes when requested', () => {
    const onDiscardLocal = vi.fn();
    render(
      <SchemaConflictDialog
        isOpen={true}
        conflict={conflictInfoFixture}
        preview={{ autoStaged: {}, conflicts: [] }}
        onClose={vi.fn()}
        onDiscardLocal={onDiscardLocal}
        onReapply={vi.fn()}
      />,
    );

    fireEvent.click(screen.getByRole('button', { name: 'Discard local changes' }));
    expect(onDiscardLocal).toHaveBeenCalled();
  });
});
