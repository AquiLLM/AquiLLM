// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import SchemaValidationPanel from './SchemaValidationPanel';
import { validationResultFixture } from './schemaTestFixtures';

afterEach(() => cleanup());

describe('SchemaValidationPanel', () => {
  it('groups errors and warnings by issue code', () => {
    render(
      <SchemaValidationPanel status="invalid" result={validationResultFixture} onIssueSelect={vi.fn()} />,
    );

    expect(screen.getByText('alias_duplicate')).toBeTruthy();
    expect(screen.getByText(/Warnings/i)).toBeTruthy();
  });

  it('activates an issue when selected', () => {
    const onIssueSelect = vi.fn();
    render(
      <SchemaValidationPanel status="invalid" result={validationResultFixture} onIssueSelect={onIssueSelect} />,
    );

    fireEvent.click(screen.getByText('alias_duplicate'));
    expect(onIssueSelect).toHaveBeenCalledWith(validationResultFixture.issues[0]);
  });
});
