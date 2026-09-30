// @vitest-environment jsdom
import { act, cleanup, renderHook } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { useState } from 'react';
import { useChatWebSocket } from './useChatWebSocket';
import type { Conversation, SkillOverrides, WebSocketMessage } from '../types';

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
function setup() {
  const view = renderHook(() => {
    const [conversation, setConversation] = useState<Conversation>({ messages: [] });
    const [exception, setException] = useState('');
    const [, setDebugHtml] = useState<string | null>(null);
    const [, setInputDisabled] = useState(true);
    const [collections, setSelectedCollections] = useState(new Set<string>());
    const [overrides, setSkillOverrides] = useState<SkillOverrides>({});
    const socket = useChatWebSocket({ convoId: '1', setConversation, setException, setDebugHtml, setInputDisabled, setSelectedCollections, setSkillOverrides });
    return { ...socket, conversation, exception, collections, overrides };
  });
  act(() => Socket.latest.emit({ conversation: { messages: [], selected_collections: [7], skill_overrides: { 'raw:1': false } } }));
  return view;
}
beforeEach(() => { vi.useFakeTimers(); vi.stubGlobal('WebSocket', Socket); });
afterEach(() => { cleanup(); vi.useRealTimers(); vi.unstubAllGlobals(); });
describe('acknowledged context transport', () => {
  it('hydrates overrides and updates applied context only after matching acknowledgment', async () => {
    const view = setup();
    expect(view.result.current.overrides).toEqual({ 'raw:1': false });
    let saved!: Promise<unknown>;
    act(() => { saved = view.result.current.saveContext(new Set(['8']), { 'raw:1': true }); });
    const payload = JSON.parse(Socket.latest.send.mock.calls[0][0]);
    expect(payload).toEqual({ action: 'select_collections', collections: [8], skill_overrides: { 'raw:1': true }, request_id: expect.any(String) });
    expect(payload.request_id.length).toBeGreaterThan(0);
    expect(payload.request_id.length).toBeLessThanOrEqual(128);
    expect([...view.result.current.collections]).toEqual(['7']);
    act(() => Socket.latest.emit({ context_selection: { request_id: 'old', selected_collections: [99], skill_overrides: {} } }));
    expect([...view.result.current.collections]).toEqual(['7']);
    await act(async () => {
      Socket.latest.emit({ context_selection: { request_id: payload.request_id, selected_collections: [8], skill_overrides: { 'raw:1': true } } });
      await saved;
    });
    expect([...view.result.current.collections]).toEqual(['8']);
    expect(view.result.current.overrides).toEqual({ 'raw:1': true });
  });
  it('rejects correlated errors without setting terminal chat errors and ignores stale failures', async () => {
    const view = setup();
    let result!: Promise<unknown>;
    act(() => { result = view.result.current.saveContext(new Set(), {}).catch(error => error.message); });
    const payload = JSON.parse(Socket.latest.send.mock.calls[0][0]);
    act(() => Socket.latest.emit({ context_selection_error: { request_id: 'old', message: 'Old failure' } }));
    expect(view.result.current.contextSaving).toBe(true);
    await act(async () => {
      Socket.latest.emit({ context_selection_error: { request_id: payload.request_id, message: 'Access changed' } });
      expect(await result).toBe('Access changed');
    });
    expect(view.result.current.exception).toBe('');
    expect(view.result.current.terminalError).toBe('');
    expect([...view.result.current.collections]).toEqual(['7']);
  });
  it('times out, allows retry and never accepts the timed-out acknowledgment', async () => {
    const view = setup();
    let timedOut!: Promise<unknown>;
    act(() => { timedOut = view.result.current.saveContext(new Set(), {}).catch(error => error.message); });
    const old = JSON.parse(Socket.latest.send.mock.calls[0][0]);
    await act(async () => { vi.advanceTimersByTime(10000); expect(await timedOut).toMatch(/timed out/i); });
    let retry!: Promise<unknown>;
    act(() => { retry = view.result.current.saveContext(new Set(['9'])).catch(error => error.message); });
    const latest = JSON.parse(Socket.latest.send.mock.calls[1][0]);
    expect(latest.skill_overrides).toBeUndefined();
    act(() => Socket.latest.emit({ context_selection: { request_id: old.request_id, selected_collections: [], skill_overrides: {} } }));
    expect(view.result.current.contextSaving).toBe(true);
    await act(async () => { Socket.latest.emit({ context_selection: { request_id: latest.request_id, selected_collections: [9], skill_overrides: { 'raw:1': false } } }); await retry; });
    expect([...view.result.current.collections]).toEqual(['9']);
  });
  it('rejects disconnected and concurrent saves without sending extra requests', async () => {
    const view = setup();
    let pending!: Promise<unknown>;
    act(() => { pending = view.result.current.saveContext(new Set(), {}).catch(error => error.message); });
    await expect(view.result.current.saveContext(new Set(), {})).rejects.toThrow(/already/i);
    await act(async () => { Socket.latest.readyState = 3; Socket.latest.onclose?.({ code: 1006 } as CloseEvent); expect(await pending).toMatch(/connection/i); });
    await expect(view.result.current.saveContext(new Set(), {})).rejects.toThrow(/ready|connection/i);
    expect(Socket.latest.send).toHaveBeenCalledTimes(1);
  });
});
