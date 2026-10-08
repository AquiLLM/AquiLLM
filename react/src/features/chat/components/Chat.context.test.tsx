// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import Chat from './Chat';
import type { WebSocketMessage } from '../types';

class Socket {
  static OPEN = 1; static CLOSED = 3; static latest: Socket;
  readyState = 1;
  onopen: ((event: Event) => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onclose: ((event: CloseEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  send = vi.fn(); close = vi.fn();
  constructor() { Socket.latest = this; }
  emit(data: WebSocketMessage) { this.onmessage?.({ data: JSON.stringify(data) } as MessageEvent); }
}
const catalog = {
  skills_enabled: true,
  collections: [{ id: 7, name: 'Research', parent: null, path: 'Research', is_skill_pack: false }],
  skills: [{ id: 'raw:1', name: 'Cite sources', description: 'Use evidence', instructions: 'Cite sources.', collection_id: '7', collection_name: 'Research', collection_path: 'Research', source_path: 'Research/cite_skill.md', pack_id: null, pack_name: null, default_collection_ids: ['7'] }],
};
beforeEach(() => {
  vi.stubGlobal('WebSocket', Socket);
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => catalog }));
  Element.prototype.scrollIntoView = vi.fn();
});
afterEach(() => { cleanup(); vi.unstubAllGlobals(); vi.restoreAllMocks(); });
async function ready() {
  render(<Chat convoId="1" />);
  await waitFor(() => expect(fetch).toHaveBeenCalled());
  expect(fetch).toHaveBeenCalledWith('/api/collections/chat-context/', expect.any(Object));
  act(() => Socket.latest.emit({ conversation: { messages: [], selected_collections: [7], skill_overrides: { 'raw:1': false, 'deleted:9': true } } }));
  fireEvent.click(screen.getByRole('button', { name: /Collections/ }));
  await screen.findByRole('checkbox', { name: 'Research' });
}
describe('Chat context integration', () => {
  it('waits for acknowledgment, prunes unavailable overrides, and appends applied choices', async () => {
    await ready();
    expect(fetch).toHaveBeenCalledWith('/api/collections/chat-context/', expect.any(Object));
    fireEvent.click(screen.getByRole('tab', { name: /Skills/ }));
    expect((screen.getByRole('checkbox', { name: 'Cite sources' }) as HTMLInputElement).checked).toBe(false);
    fireEvent.click(screen.getByRole('checkbox', { name: 'Cite sources' }));
    fireEvent.click(screen.getByRole('button', { name: 'Apply to chat' }));
    const sent = JSON.parse(Socket.latest.send.mock.calls[0][0]);
    expect(sent.skill_overrides).toEqual({ 'raw:1': true });
    expect(screen.getByRole('dialog')).toBeTruthy();
    expect((screen.getByTitle('Send Message') as HTMLButtonElement).disabled).toBe(true);
    await act(async () => Socket.latest.emit({ context_selection: { request_id: sent.request_id, selected_collections: [7], skill_overrides: { 'raw:1': true } } }));
    expect(screen.queryByRole('dialog')).toBeNull();
    expect(screen.getByRole('button', { name: /1 collection.*1 skill/i })).toBeTruthy();
    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'hello' } });
    fireEvent.click(screen.getByTitle('Send Message'));
    expect(JSON.parse(Socket.latest.send.mock.calls[1][0])).toMatchObject({ action: 'append', collections: [7], skill_overrides: { 'raw:1': true } });
  });
  it('preserves stored overrides by omitting the map when skills are disabled', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => ({ ...catalog, skills_enabled: false, skills: [] }) }));
    await ready();
    fireEvent.click(screen.getByRole('button', { name: 'Apply to chat' }));
    const sent = JSON.parse(Socket.latest.send.mock.calls[0][0]);
    expect(sent).not.toHaveProperty('skill_overrides');
    await act(async () => Socket.latest.emit({ context_selection: { request_id: sent.request_id, selected_collections: [7], skill_overrides: { 'raw:1': false, 'deleted:9': true } } }));
    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'hello' } });
    fireEvent.click(screen.getByTitle('Send Message'));
    expect(JSON.parse(Socket.latest.send.mock.calls[1][0])).not.toHaveProperty('skill_overrides');
    expect(screen.getByRole('button', { name: /0 skills/i })).toBeTruthy();
  });
  it('keeps rejected selection errors in the dialog and blocks Apply during a reply', async () => {
    await ready();
    fireEvent.click(screen.getByRole('button', { name: 'Apply to chat' }));
    const sent = JSON.parse(Socket.latest.send.mock.calls[0][0]);
    await act(async () => Socket.latest.emit({ context_selection_error: { request_id: sent.request_id, message: 'Collection access changed.' } }));
    expect(screen.getByRole('dialog').contains(screen.getByRole('alert'))).toBe(true);
    expect(screen.queryByRole('status', { name: 'AquiLLM is thinking' })).toBeNull();
    act(() => Socket.latest.emit({ delta: { messages: [{ role: 'assistant', content: '', tool_call_name: 'vector_search', message_uuid: 'tool' }] } }));
    expect((screen.getByRole('button', { name: 'Apply to chat' }) as HTMLButtonElement).disabled).toBe(true);
  });
});
