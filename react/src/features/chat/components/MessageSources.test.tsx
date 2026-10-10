// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import MessageSources from './MessageSources';

const openCitation = vi.hoisted(() => vi.fn());
vi.mock('./CitationModalProvider', () => ({ useCitationModal: () => ({ openCitation }) }));
afterEach(() => { cleanup(); openCitation.mockClear(); });

describe('MessageSources', () => {
  it('keeps passages available under numbered sources when titles cannot be loaded', () => {
    render(<MessageSources messageUuid="answer-1" citations={[
      { docId: 'doc-a', chunkId: '17', sourceNumber: 1, passageNumber: 1 },
      { docId: 'doc-a', chunkId: '23', sourceNumber: 1, passageNumber: 2 },
    ]} />);
    fireEvent.click(screen.getByRole('button', { name: 'Sources · 1 document · 2 passages' }));
    expect(screen.getByText('Source 1')).not.toBeNull();
    fireEvent.click(screen.getByRole('button', { name: 'Source 1 · Passage 2' }));
    expect(openCitation).toHaveBeenCalledWith({ docId: 'doc-a', chunkId: '23', messageUuid: 'answer-1' });
  });

  it('does not show an empty sources section for uncited messages', () => {
    render(<MessageSources citations={[]} />);
    expect(screen.queryByRole('button')).toBeNull();
  });
});
