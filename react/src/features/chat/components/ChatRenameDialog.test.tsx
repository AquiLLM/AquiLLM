// @vitest-environment jsdom
import { createElement } from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import ChatRenameDialog from './ChatRenameDialog';

beforeEach(() => {
  document.body.innerHTML = `<input name="csrfmiddlewaretoken" value="test-csrf" type="hidden">
    <a data-conversation-title="42">Original title</a>
    <span data-conversation-title="42">Original title</span>
    <button id="conversation-options-42" data-conversation-options="42">Chat options</button>
    <button data-rename-conversation="42" data-rename-url="/rename_ws_convo/42">Rename</button>`;
});
afterEach(() => { cleanup(); vi.unstubAllGlobals(); document.body.innerHTML = ''; });

function open() {
  render(createElement(ChatRenameDialog));
  fireEvent.click(screen.getByRole('button', { name: 'Rename' }));
  return screen.getByRole('textbox', { name: 'Chat name' }) as HTMLInputElement;
}

describe('renaming a past chat', () => {
  it('selects the current title, traps focus and cancels without saving', () => {
    const fetcher = vi.fn(); vi.stubGlobal('fetch', fetcher);
    const input = open();
    expect(input.value).toBe('Original title');
    expect(document.activeElement).toBe(input);
    expect([input.selectionStart, input.selectionEnd]).toEqual([0, 14]);
    fireEvent.change(input, { target: { value: 'Discarded draft' } });
    fireEvent.keyDown(input, { key: 'Tab', shiftKey: true });
    expect(document.activeElement).toBe(screen.getByRole('button', { name: 'Save' }));
    fireEvent.keyDown(input, { key: 'Escape' });
    expect(screen.queryByRole('dialog')).toBeNull();
    expect(document.querySelector('[data-conversation-title]')?.textContent).toBe('Original title');
    expect(document.activeElement?.id).toBe('conversation-options-42');
    expect(fetcher).not.toHaveBeenCalled();
  });

  it('saves the trimmed title with CSRF, updates every visible title and reopens with the saved name', async () => {
    const fetcher = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ id: 42, name: '<Project notes>' }) });
    vi.stubGlobal('fetch', fetcher);
    const input = open();
    fireEvent.change(input, { target: { value: '  <Project notes>  ' } });
    fireEvent.submit(input.closest('form')!);
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
    expect(fetcher).toHaveBeenCalledWith('/rename_ws_convo/42', expect.objectContaining({
      method: 'POST', credentials: 'same-origin', body: JSON.stringify({ name: '<Project notes>' }),
      headers: { 'Content-Type': 'application/json', 'X-CSRFToken': 'test-csrf' },
    }));
    expect([...document.querySelectorAll('[data-conversation-title]')].map(el => el.textContent)).toEqual(['<Project notes>', '<Project notes>']);
    expect(document.querySelector('project')).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: 'Rename' }));
    expect((screen.getByRole('textbox') as HTMLInputElement).value).toBe('<Project notes>');
  });

  it('rejects blank names without changing the saved title', () => {
    const fetcher = vi.fn(); vi.stubGlobal('fetch', fetcher);
    const input = open();
    fireEvent.change(input, { target: { value: '   ' } });
    fireEvent.submit(input.closest('form')!);
    expect(screen.getByRole('alert').textContent).toMatch(/name/i);
    expect(fetcher).not.toHaveBeenCalled();
    expect(document.querySelector('[data-conversation-title]')?.textContent).toBe('Original title');
  });

  it('keeps the draft and original title on failure, then allows retry', async () => {
    const fetcher = vi.fn().mockRejectedValueOnce(new TypeError('Network error'))
      .mockResolvedValueOnce({ ok: true, json: async () => ({ id: 42, name: 'Retry title' }) });
    vi.stubGlobal('fetch', fetcher);
    const input = open();
    fireEvent.change(input, { target: { value: 'Retry title' } });
    fireEvent.submit(input.closest('form')!);
    await screen.findByRole('alert');
    expect(document.activeElement).toBe(input);
    expect(input.value).toBe('Retry title');
    expect(document.querySelector('[data-conversation-title]')?.textContent).toBe('Original title');
    fireEvent.submit(input.closest('form')!);
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
    expect(document.querySelector('[data-conversation-title]')?.textContent).toBe('Retry title');
  });

  it('shows server rejection and does not publish an unsaved title', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: false, status: 404, json: async () => ({ error: 'Chat is no longer available.' }) }));
    const input = open();
    fireEvent.change(input, { target: { value: 'Unsaved' } });
    fireEvent.submit(input.closest('form')!);
    expect((await screen.findByRole('alert')).textContent).toMatch(/no longer available/);
    expect(document.querySelector('[data-conversation-title]')?.textContent).toBe('Original title');
  });

  it('prevents duplicate saves and dismissal until the server responds', async () => {
    let resolve!: (value: unknown) => void;
    const fetcher = vi.fn().mockReturnValue(new Promise(done => { resolve = done; }));
    vi.stubGlobal('fetch', fetcher);
    const input = open();
    fireEvent.submit(input.closest('form')!);
    fireEvent.submit(input.closest('form')!);
    expect(document.activeElement).toBe(screen.getByRole('dialog'));
    fireEvent.keyDown(document.activeElement!, { key: 'Tab' });
    expect(screen.getByRole('dialog').contains(document.activeElement)).toBe(true);
    fireEvent.keyDown(input, { key: 'Escape' });
    expect(screen.queryByRole('dialog')).not.toBeNull();
    expect(fetcher).toHaveBeenCalledTimes(1);
    resolve({ ok: true, json: async () => ({ id: 42, name: 'Original title' }) });
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
  });
});
