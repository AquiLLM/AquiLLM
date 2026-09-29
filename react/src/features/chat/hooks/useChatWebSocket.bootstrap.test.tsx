// @vitest-environment jsdom

import bootstrapSource from '../../../../../aquillm/templates/aquillm/includes/chat_socket_bootstrap.js?raw';
import { useState } from 'react';
import { act, cleanup, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { Conversation, WebSocketMessage } from '../types';
import { useChatWebSocket } from './useChatWebSocket';

class Socket {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;
  static sockets: Socket[] = [];
  readyState = Socket.CONNECTING;
  onopen: ((event: Event) => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onclose: ((event: CloseEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  send = vi.fn();
  close = vi.fn(() => this.disconnect(1000));
  constructor(readonly url: string) { Socket.sockets.push(this); }
  open() { this.readyState = Socket.OPEN; this.onopen?.(new Event('open')); }
  emit(data: WebSocketMessage) { this.onmessage?.(new MessageEvent('message', { data: JSON.stringify(data) })); }
  disconnect(code = 1006) { this.readyState = Socket.CLOSED; this.onclose?.(new CloseEvent('close', { code })); }
}

const ignoreDebugHtml = () => {};
function Harness({ convoId = '314' }: { convoId?: string }) {
  const [conversation, setConversation] = useState<Conversation>({ messages: [] });
  const [error, setError] = useState('');
  const [disabled, setDisabled] = useState(true);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const { connectionStatus, wsRef } = useChatWebSocket({
    convoId, setConversation, setException: setError, setDebugHtml: ignoreDebugHtml,
    setInputDisabled: setDisabled, setSelectedCollections: setSelected,
  });
  return <>
    <output data-testid="state" data-disabled={disabled} data-status={connectionStatus}
      data-selected={[...selected].join(',')} data-messages={JSON.stringify(conversation.messages)}>{error}</output>
    <button onClick={() => wsRef.current?.send('hello')}>Send</button>
  </>;
}

const state = () => screen.getByTestId('state');
const hydrate = (socket: Socket, selected_collections: number[] = []) =>
  socket.emit({ conversation: { messages: [], selected_collections } });

function bootstrap(convoId = '314') {
  const script = document.createElement('script');
  script.dataset.convoId = convoId;
  // Execute the exact inline include in the page realm, without fetching assets.
  vi.spyOn(document, 'currentScript', 'get').mockReturnValue(script);
  window.eval(bootstrapSource);
  return Socket.sockets[Socket.sockets.length - 1];
}

describe('early chat WebSocket adoption', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    Socket.sockets = [];
    vi.stubGlobal('WebSocket', Socket);
    vi.spyOn(console, 'error').mockImplementation(() => {});
  });
  afterEach(() => {
    cleanup();
    window.dispatchEvent(new Event('pagehide'));
    delete (window as Window & { aquiChatSocketBootstrap?: unknown }).aquiChatSocketBootstrap;
    vi.clearAllTimers();
    vi.useRealTimers();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it('starts before React mounts and adopts the same connecting socket', () => {
    const early = bootstrap();
    expect(early?.url).toBe('ws://localhost:3000/ws/convo/314/');
    render(<Harness />);
    expect(Socket.sockets).toHaveLength(1);
    act(() => { early.open(); hydrate(early, [7]); });
    expect(state().dataset.status).toBe('ready');
    expect(state().dataset.selected).toBe('7');
    screen.getByRole('button', { name: 'Send' }).click();
    expect(early.send).toHaveBeenCalledWith('hello');
  });

  it('adopts when Django passes its numeric conversation ID at runtime', () => {
    const early = bootstrap();
    render(<Harness convoId={314 as unknown as string} />);
    expect(Socket.sockets).toHaveLength(1);
    act(() => { early.open(); hydrate(early, [8]); });
    expect(state().dataset.selected).toBe('8');
  });

  it('replays the snapshot, stream, and authoritative delta in original order', () => {
    const early = bootstrap();
    expect(early).toBeDefined();
    early.open();
    hydrate(early, [7, 9]);
    early.emit({ stream: { role: 'assistant', message_uuid: 'answer', content: 'partial' } });
    early.emit({ delta: { messages: [{ role: 'assistant', message_uuid: 'answer', content: 'Done' }] } });
    render(<Harness />);
    expect(Socket.sockets).toHaveLength(1);
    expect(state().dataset.status).toBe('ready');
    expect(state().dataset.selected).toBe('7,9');
    expect(state().dataset.messages).toBe('[{"role":"assistant","content":"Done","message_uuid":"answer"}]');
    expect(state().dataset.disabled).toBe('false');
  });

  it('adopts an open socket while waiting for a snapshot with saved collections', () => {
    const early = bootstrap();
    expect(early).toBeDefined();
    early.open();
    early.emit({ conversation: { messages: [] } });
    render(<Harness />);
    expect(state().dataset.status).toBe('hydrating');
    expect(state().dataset.disabled).toBe('true');
    act(() => hydrate(early, [5]));
    expect(state().dataset.status).toBe('ready');
    expect(state().dataset.selected).toBe('5');
  });

  it.each([4401, 4404, 4409])('preserves a buffered permanent close %i beyond expiry', (code) => {
    const early = bootstrap();
    expect(early).toBeDefined();
    early.open();
    early.emit({ exception: 'Server says refresh this conversation.', fatal: true });
    early.disconnect(code);
    vi.advanceTimersByTime(60000);
    render(<Harness />);
    act(() => vi.advanceTimersByTime(60000));
    expect(Socket.sockets).toHaveLength(1);
    expect(state().dataset.status).toBe('failed');
    expect(state().textContent).toBe('Server says refresh this conversation.');
    expect(state().dataset.disabled).toBe('true');
  });

  it('retains the fatal reason after expiry even when its debug payload is large', () => {
    const early = bootstrap();
    early.emit({ exception: 'Conversation changed elsewhere.', fatal: true, debug_html: 'x'.repeat(20000) });
    early.disconnect(4409);
    vi.advanceTimersByTime(60000);
    render(<Harness />);
    expect(state().textContent).toBe('Conversation changed elsewhere.');
    expect(state().dataset.status).toBe('failed');
  });

  it('retries a buffered transient close once after replaying its snapshot', () => {
    const early = bootstrap();
    expect(early).toBeDefined();
    early.open();
    hydrate(early, [7]);
    early.disconnect();
    render(<Harness />);
    expect(state().dataset.status).toBe('reconnecting');
    expect(state().dataset.selected).toBe('7');
    expect(state().dataset.disabled).toBe('true');
    act(() => vi.advanceTimersByTime(2000));
    expect(Socket.sockets).toHaveLength(2);
  });

  it('counts bootstrap time toward the existing readiness deadline', () => {
    const early = bootstrap();
    expect(early).toBeDefined();
    vi.advanceTimersByTime(4000);
    render(<Harness />);
    act(() => vi.advanceTimersByTime(999));
    expect(early.close).not.toHaveBeenCalled();
    act(() => vi.advanceTimersByTime(1));
    expect(early.close).toHaveBeenCalledTimes(1);
    act(() => vi.advanceTimersByTime(2000));
    expect(Socket.sockets).toHaveLength(2);
  });

  it('accepts an already hydrated socket after its original readiness deadline', () => {
    const early = bootstrap();
    expect(early).toBeDefined();
    early.open();
    hydrate(early);
    vi.advanceTimersByTime(6000);
    render(<Harness />);
    act(() => vi.advanceTimersByTime(60000));
    expect(Socket.sockets).toHaveLength(1);
    expect(early.close).not.toHaveBeenCalled();
    expect(state().dataset.status).toBe('ready');
  });

  it('expires an unclaimed live socket and falls back without partial replay', () => {
    const early = bootstrap();
    expect(early).toBeDefined();
    early.open();
    hydrate(early, [7]);
    vi.advanceTimersByTime(60000);
    expect(early.close).toHaveBeenCalledTimes(1);
    render(<Harness />);
    expect(Socket.sockets).toHaveLength(2);
    expect(state().dataset.status).toBe('connecting');
    expect(state().dataset.selected).toBe('');
    expect(state().dataset.disabled).toBe('true');
  });

  it('waits for an expired socket closing asynchronously so its permanent code is not lost', () => {
    const early = bootstrap();
    early.open();
    early.emit({ exception: 'Refresh required by the server.', fatal: true });
    early.close.mockImplementation(() => { early.readyState = Socket.CLOSING; });
    vi.advanceTimersByTime(15000);
    render(<Harness />);
    expect(Socket.sockets).toHaveLength(1);
    act(() => vi.advanceTimersByTime(100));
    act(() => early.disconnect(4409));
    act(() => vi.advanceTimersByTime(60000));
    expect(Socket.sockets).toHaveLength(1);
    expect(state().dataset.status).toBe('failed');
    expect(state().textContent).toBe('Refresh required by the server.');
  });

  it('bounds the wait when an expired socket never delivers its close event', () => {
    const early = bootstrap();
    early.close.mockImplementation(() => { early.readyState = Socket.CLOSING; });
    vi.advanceTimersByTime(15000);
    render(<Harness />);
    act(() => vi.advanceTimersByTime(7000));
    expect(Socket.sockets).toHaveLength(2);
    expect(state().dataset.status).toBe('reconnecting');
    expect(state().dataset.disabled).toBe('true');
  });

  it.each([4401, 4404, 4409])('preserves permanent close %i without an error payload', (code) => {
    const early = bootstrap();
    early.disconnect(code);
    vi.advanceTimersByTime(60000);
    render(<Harness />);
    act(() => vi.advanceTimersByTime(60000));
    expect(Socket.sockets).toHaveLength(1);
    expect(state().dataset.status).toBe('failed');
    expect(state().textContent).toContain('Please refresh the page.');
  });

  it('falls back when the early socket constructor fails', () => {
    vi.stubGlobal('WebSocket', class { constructor() { throw new Error('Unavailable'); } });
    bootstrap();
    vi.stubGlobal('WebSocket', Socket);
    render(<Harness />);
    expect(Socket.sockets).toHaveLength(1);
    act(() => { Socket.sockets[0].open(); hydrate(Socket.sockets[0]); });
    expect(state().dataset.status).toBe('ready');
  });

  it('does not create a second socket when bootstrap is evaluated twice', () => {
    bootstrap();
    bootstrap();
    render(<Harness />);
    expect(Socket.sockets).toHaveLength(1);
  });

  it('closes mismatched bootstrap ownership before connecting to another conversation', () => {
    const early = bootstrap();
    expect(early).toBeDefined();
    early.open();
    hydrate(early, [7]);
    render(<Harness convoId="315" />);
    expect(early.close).toHaveBeenCalledTimes(1);
    expect(Socket.sockets).toHaveLength(2);
    expect(Socket.sockets[1].url).toContain('/315/');
    expect(state().dataset.selected).toBe('');
  });

  it('releases the handoff once, closes on unmount, and ignores old socket events', () => {
    const early = bootstrap();
    expect(early).toBeDefined();
    early.open();
    hydrate(early);
    const { unmount } = render(<Harness />);
    unmount();
    expect(early.close).toHaveBeenCalledTimes(1);
    render(<Harness />);
    expect(Socket.sockets).toHaveLength(2);
    act(() => { early.emit({ exception: 'Stale event' }); early.disconnect(4409); });
    expect(state().textContent).toBe('');
    expect(state().dataset.status).toBe('connecting');
  });

  it('closes an unclaimed socket when leaving the destination page', () => {
    const early = bootstrap();
    expect(early).toBeDefined();
    window.dispatchEvent(new Event('pagehide'));
    expect(early.close).toHaveBeenCalledTimes(1);
  });

  it('bounds unclaimed event count and falls back without applying a partial snapshot', () => {
    const early = bootstrap();
    expect(early).toBeDefined();
    early.open();
    hydrate(early, [7]);
    for (let i = 0; i < 1000; i++) early.emit({ stream: { role: 'assistant', message_uuid: 'answer', content: `${i}` } });
    expect(early.close).toHaveBeenCalledTimes(1);
    render(<Harness />);
    expect(Socket.sockets).toHaveLength(2);
    expect(state().dataset.disabled).toBe('true');
    expect(state().dataset.selected).toBe('');
    expect(state().dataset.messages).toBe('[]');
  });

  it('bounds message bytes while allowing a large saved conversation', () => {
    const early = bootstrap();
    expect(early).toBeDefined();
    early.open();
    early.emit({ conversation: { selected_collections: [], messages: [{ role: 'assistant', content: 'x'.repeat(1000000) }] } });
    expect(early.close).not.toHaveBeenCalled();
    early.emit({ stream: { role: 'assistant', message_uuid: 'answer', content: 'x'.repeat(5000000) } });
    expect(early.close).toHaveBeenCalledTimes(1);
    render(<Harness />);
    expect(Socket.sockets).toHaveLength(2);
    expect(state().dataset.messages).toBe('[]');
  });

  it('counts a buffered pending snapshot and transient fatal error in the retry budget', () => {
    const early = bootstrap();
    expect(early).toBeDefined();
    const failPending = (socket: Socket) => {
      socket.open();
      socket.emit({ conversation: { selected_collections: [], messages: [{ role: 'user', content: 'Pending' }] } });
      socket.emit({ exception: 'Provider unavailable', fatal: true });
      socket.disconnect(1013);
    };
    failPending(early);
    render(<Harness />);
    for (let i = 1; i < 5; i++) {
      act(() => vi.advanceTimersByTime(2000));
      act(() => failPending(Socket.sockets[i]));
    }
    act(() => vi.advanceTimersByTime(60000));
    expect(Socket.sockets).toHaveLength(5);
    expect(state().dataset.status).toBe('failed');
  });
});
