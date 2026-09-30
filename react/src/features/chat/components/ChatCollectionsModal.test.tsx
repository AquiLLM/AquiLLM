// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import ChatCollectionsModal from './ChatCollectionsModal';

export const catalog = {
  skills_enabled: true,
  collections: [
    { id: 'parent', name: 'Research', parent: null, path: 'Research', is_skill_pack: false },
    { id: 'child', name: 'Figures', parent: 'parent', path: 'Research / Figures', is_skill_pack: false },
    { id: 'pack', name: 'skills', parent: 'child', path: 'Research / Figures / skills', is_skill_pack: true },
  ],
  skills: [
    { id: 'raw:1', name: 'Cite sources', description: 'Support factual claims.', instructions: '<b>Use evidence</b>\nBe precise.', collection_id: 'parent', collection_name: 'Research', collection_path: 'Research', source_path: 'Research / cite_skill.md', pack_id: null, pack_name: null, default_collection_ids: ['parent'] },
    { id: 'raw:2', name: 'Compare sources', description: 'Find differences.', instructions: 'Compare the evidence.', collection_id: 'pack', collection_name: 'skills', collection_path: 'Research / Figures / skills', source_path: 'Research / Figures / skills / comparison.md', pack_id: 'pack', pack_name: 'skills', default_collection_ids: ['child', 'pack'] },
  ],
};

afterEach(cleanup);
function setup(overrides = {}) {
  const onApply = vi.fn().mockResolvedValue(undefined);
  const onClose = vi.fn();
  const props = { open: true, catalog, selectedCollections: new Set(['parent']), skillOverrides: overrides, onApply, onClose,
    // The old component can render, so the red phase fails on missing behavior.
    filteredCollections: catalog.collections, searchTerm: '', onSearchTermChange: vi.fn(), onToggleCollection: vi.fn() };
  const view = render(<ChatCollectionsModal {...props} />);
  return { ...view, props, onApply, onClose };
}
const check = (name: string) => screen.getByRole('checkbox', { name }) as HTMLInputElement;

describe('chat context drafts', () => {
  it('discards collection edits on cancel and reopening', () => {
    const view = setup();
    expect(screen.getByRole('dialog')).toBeTruthy();
    fireEvent.click(check('Research'));
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }));
    expect(view.onApply).not.toHaveBeenCalled();
    view.rerender(<ChatCollectionsModal {...view.props} open={false} />);
    view.rerender(<ChatCollectionsModal {...view.props} />);
    expect(check('Research').checked).toBe(true);
  });
  it('selects all descendants and shows paths when searching', async () => {
    const view = setup();
    fireEvent.click(check('Research'));
    fireEvent.click(check('Research'));
    fireEvent.change(screen.getByRole('searchbox'), { target: { value: 'Figures' } });
    expect(check('Figures').checked).toBe(true);
    expect(check('skills').checked).toBe(true);
    expect(screen.getAllByText('Research / Figures / skills').length).toBeGreaterThan(0);
    await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Apply to chat' })));
    expect([...view.onApply.mock.calls[0][0].selectedCollections]).toEqual(['parent', 'child', 'pack']);
  });
  it('persists explicit false for an inherited skill and previews instructions as text', async () => {
    const view = setup();
    fireEvent.click(screen.getByRole('tab', { name: /Skills/ }));
    expect(check('Cite sources').checked).toBe(true);
    fireEvent.click(check('Cite sources'));
    fireEvent.click(screen.getByRole('button', { name: 'Read Cite sources instructions' }));
    const preview = screen.getByText('<b>Use evidence</b> Be precise.', { exact: false });
    expect(preview.querySelector('b')).toBeNull();
    await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Apply to chat' })));
    expect(view.onApply.mock.calls[0][0].skillOverrides['raw:1']).toBe(false);
  });
  it('searches source paths, enables a whole source and resets to inheritance', () => {
    setup({ 'raw:1': false });
    fireEvent.click(screen.getByRole('tab', { name: /Skills/ }));
    fireEvent.change(screen.getByRole('searchbox'), { target: { value: 'comparison.md' } });
    expect(screen.queryByRole('checkbox', { name: 'Cite sources' })).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: /Enable all/ }));
    expect(check('Compare sources').checked).toBe(true);
    fireEvent.change(screen.getByRole('searchbox'), { target: { value: '' } });
    fireEvent.click(screen.getByRole('button', { name: 'Read Cite sources instructions' }));
    fireEvent.click(screen.getByRole('button', { name: 'Follow collection selection' }));
    expect(check('Cite sources').checked).toBe(true);
  });
  it('clear all disables every available skill and keeps retrieval separate', async () => {
    const view = setup({ 'raw:2': true });
    fireEvent.click(screen.getByRole('button', { name: 'Clear all' }));
    await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Apply to chat' })));
    expect(view.onApply.mock.calls[0][0].selectedCollections.size).toBe(0);
    expect(view.onApply.mock.calls[0][0].skillOverrides).toEqual({ 'raw:1': false, 'raw:2': false });
  });
  it('keeps failed saves open with the draft intact, then allows retry', async () => {
    const view = setup();
    view.onApply.mockRejectedValueOnce(new Error('Access changed. Try again.'));
    await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Apply to chat' })));
    expect(within(screen.getByRole('dialog')).getByRole('alert').textContent).toContain('Access changed');
    expect(view.onClose).not.toHaveBeenCalled();
    expect(check('Research').checked).toBe(true);
    await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Apply to chat' })));
    expect(view.onClose).toHaveBeenCalledOnce();
  });
  it('restores opener focus and supports keyboard tab navigation and Escape', () => {
    const opener = document.createElement('button');
    document.body.append(opener); opener.focus();
    const view = setup();
    const collectionsTab = screen.getByRole('tab', { name: /Collections/ });
    fireEvent.keyDown(collectionsTab, { key: 'ArrowRight' });
    expect(document.activeElement).toBe(screen.getByRole('tab', { name: /Skills/ }));
    fireEvent.keyDown(screen.getByRole('dialog'), { key: 'Escape' });
    expect(view.onClose).toHaveBeenCalledOnce();
    view.unmount();
    expect(document.activeElement).toBe(opener);
    opener.remove();
  });
});
