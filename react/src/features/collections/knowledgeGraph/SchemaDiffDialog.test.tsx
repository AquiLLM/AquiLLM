// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import SchemaDiffDialog from './SchemaDiffDialog';
import { validationResultFixture } from './schemaTestFixtures';

afterEach(() => cleanup());

describe('SchemaDiffDialog', () => {
  it('shows bounded diff counts from the server summary', () => {
    render(
      <SchemaDiffDialog
        isOpen={true}
        diff={validationResultFixture.diff_summary}
        impactSummary="One entity description changed."
        onClose={vi.fn()}
      />,
    );

    expect(screen.getByRole('dialog', { name: 'Schema diff review' })).toBeTruthy();
    expect(screen.getByText(/added 0, changed 1, removed 0/i)).toBeTruthy();
    expect(screen.getByText(/One entity description changed/i)).toBeTruthy();
  });

  it('closes when Close is clicked', () => {
    const onClose = vi.fn();
    render(
      <SchemaDiffDialog isOpen={true} diff={validationResultFixture.diff_summary} onClose={onClose} />,
    );
    fireEvent.click(screen.getByRole('button', { name: 'Close' }));
    expect(onClose).toHaveBeenCalled();
  });
});
