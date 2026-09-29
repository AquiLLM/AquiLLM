// @vitest-environment jsdom

import { useState } from 'react';
import { act, cleanup, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { Conversation, WebSocketMessage } from '../types';
import { useChatWebSocket } from './useChatWebSocket';

class Socket {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSED = 3;
  static sockets: Socket[] = [];
  readyState = Socket.CONNECTING;
  onopen: ((event: Event) => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onclose: ((event: CloseEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  send = vi.fn();
  close = vi.fn(() => {
    this.readyState = Socket.CLOSED;
    this.onclose?.({ code: 1000, reason: '' } as CloseEvent);
  });

  constructor(readonly url: string) {
    Socket.sockets.push(this);
  }
  open() {
    this.readyState = Socket.OPEN;
    this.onopen?.(new Event('open'));
  }
  emit(data: WebSocketMessage) {
    this.onmessage?.({ data: JSON.stringify(data) } as MessageEvent);
  }
  disconnect(code = 1006) {
    this.readyState = Socket.CLOSED;
    this.onclose?.({ code, reason: '' } as CloseEvent);
  }
}

const ignoreDebugHtml = () => {};

function Harness({ convoId = '314' }: { convoId?: string }) {
  const [conversation, setConversation] = useState<Conversation>({ messages: [] });
  const [error, setError] = useState('');
  const [disabled, setDisabled] = useState(true);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const { connectionStatus } = useChatWebSocket({
    convoId, setConversation, setException: setError, setDebugHtml: ignoreDebugHtml,
    setInputDisabled: setDisabled, setSelectedCollections: setSelected,
  });
  return <output data-testid="state" data-disabled={disabled} data-status={connectionStatus}
    data-selected={[...selected].join(',')} data-messages={conversation.messages.length}>{error}</output>;
}

const state = () => screen.getByTestId('state');
const hydrate = (socket: Socket, selected_collections: number[] = []) =>
  socket.emit({ conversation: { messages: [], selected_collections } });

describe('chat WebSocket lifecycle', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    Socket.sockets = [];
    vi.stubGlobal('WebSocket', Socket);
  });
  afterEach(() => {
    cleanup();
    vi.clearAllTimers();
    vi.useRealTimers();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it('keeps input disabled until the authoritative snapshot and saved selection arrive', () => {
    render(<Harness />);
    const socket = Socket.sockets[0];
    expect(state().dataset.status).toBe('connecting');
    act(() => socket.open());
    expect(state().dataset.disabled).toBe('true');
    act(() => hydrate(socket, [7]));
    expect(state().dataset.selected).toBe('7');
    expect(state().dataset.disabled).toBe('false');
    expect(state().dataset.status).toBe('ready');
  });

  it('keeps a successful retry open and cancels old retry work', () => {
    render(<Harness />);
    const first = Socket.sockets[0];
    act(() => first.disconnect());
    act(() => vi.advanceTimersByTime(2000));
    const second = Socket.sockets[1];
    expect(second).toBeDefined();
    act(() => { second.open(); hydrate(second); });
    act(() => vi.advanceTimersByTime(15000));
    expect(second.close).not.toHaveBeenCalled();
    expect(Socket.sockets).toHaveLength(2);
    expect(state().dataset.status).toBe('ready');
  });

  it('starts only one retry after a timed out handshake', () => {
    render(<Harness />);
    act(() => vi.advanceTimersByTime(5000));
    expect(Socket.sockets[0].close).toHaveBeenCalledTimes(1);
    act(() => vi.advanceTimersByTime(2000));
    expect(Socket.sockets).toHaveLength(2);
    act(() => vi.advanceTimersByTime(1000));
    expect(Socket.sockets).toHaveLength(2);
  });

  it('bounds an open socket that never hydrates', () => {
    render(<Harness />);
    const first = Socket.sockets[0];
    act(() => first.open());
    act(() => vi.advanceTimersByTime(5000));
    expect(first.close).toHaveBeenCalledTimes(1);
    expect(state().dataset.disabled).toBe('true');
    act(() => vi.advanceTimersByTime(2000));
    expect(Socket.sockets).toHaveLength(2);
  });

  it('does not enable input from a snapshot missing saved selection', () => {
    render(<Harness />);
    const first = Socket.sockets[0];
    act(() => { first.open(); first.emit({ conversation: { messages: [] } }); });
    expect(state().dataset.disabled).toBe('true');
    act(() => vi.advanceTimersByTime(5000));
    expect(first.close).toHaveBeenCalledTimes(1);
  });

  it('stops after bounded failed attempts', () => {
    render(<Harness />);
    for (let i = 0; i < 5; i++) {
      act(() => Socket.sockets[i].disconnect());
      act(() => vi.advanceTimersByTime(2000));
    }
    expect(Socket.sockets).toHaveLength(5);
    expect(state().dataset.status).toBe('failed');
    expect(state().dataset.disabled).toBe('true');
  });

  it.each([1011, 1013])('bounds repeated fatal %i failures after pending snapshots', (code) => {
    vi.spyOn(console, 'error').mockImplementation(() => {});
    render(<Harness />);
    for (let i = 0; i < 5; i++) {
      const socket = Socket.sockets[i];
      act(() => {
        socket.open();
        socket.emit({
          conversation: {
            messages: [{ role: 'user', content: 'Please finish this saved turn.' }],
            selected_collections: [],
          },
        });
        socket.emit({ exception: 'Provider unavailable', fatal: true });
        socket.disconnect(code);
      });
      act(() => vi.advanceTimersByTime(2000));
    }
    expect(Socket.sockets).toHaveLength(5);
    expect(state().dataset.status).toBe('failed');
    expect(state().dataset.disabled).toBe('true');
    expect(state().textContent).toContain('Maximum reconnection attempts reached');
  });

  it('restores the retry budget after a pending turn settles on a recovered socket', () => {
    vi.spyOn(console, 'error').mockImplementation(() => {});
    render(<Harness />);
    for (let i = 0; i < 4; i++) {
      const socket = Socket.sockets[i];
      act(() => {
        socket.open();
        socket.emit({
          conversation: {
            messages: [{ role: 'user', content: 'Please finish this saved turn.' }],
            selected_collections: [],
          },
        });
        socket.emit({ exception: 'Provider unavailable', fatal: true });
        socket.disconnect(1013);
      });
      act(() => vi.advanceTimersByTime(2000));
    }

    const recovered = Socket.sockets[4];
    act(() => {
      recovered.open();
      recovered.emit({
        conversation: {
          messages: [{ role: 'user', content: 'Please finish this saved turn.' }],
          selected_collections: [],
        },
      });
      recovered.emit({
        delta: {
          messages: [{ role: 'assistant', content: 'Done.', message_uuid: 'settled-answer' }],
        },
      });
      recovered.disconnect(1013);
    });
    act(() => vi.advanceTimersByTime(2000));
    expect(Socket.sockets).toHaveLength(6);
    expect(state().dataset.status).toBe('reconnecting');
  });

  it('ignores retired socket events and cancels retries on cleanup', () => {
    const { unmount } = render(<Harness />);
    const first = Socket.sockets[0];
    act(() => first.disconnect());
    act(() => vi.advanceTimersByTime(2000));
    const second = Socket.sockets[1];
    act(() => { second.open(); hydrate(second); });
    act(() => { first.emit({ exception: 'stale' }); first.disconnect(); });
    expect(state().textContent).toBe('');
    expect(state().dataset.status).toBe('ready');
    unmount();
    act(() => vi.advanceTimersByTime(15000));
    expect(Socket.sockets).toHaveLength(2);
  });

  it('cancels a pending retry when unmounted', () => {
    const { unmount } = render(<Harness />);
    act(() => Socket.sockets[0].disconnect());
    unmount();
    act(() => vi.advanceTimersByTime(15000));
    expect(Socket.sockets).toHaveLength(1);
  });

  it('resets attempts and ignores old sockets on conversation change', () => {
    const { rerender } = render(<Harness convoId="314" />);
    const oldSocket = Socket.sockets[0];
    act(() => oldSocket.disconnect());
    rerender(<Harness convoId="315" />);
    expect(Socket.sockets).toHaveLength(2);
    expect(Socket.sockets[1].url).toContain('/315/');
    act(() => { oldSocket.emit({ exception: 'stale' }); vi.advanceTimersByTime(2000); });
    expect(Socket.sockets).toHaveLength(2);
    expect(state().textContent).toBe('');
  });

  it.each([4401, 4404, 4409])('does not retry permanent close %i', (code) => {
    render(<Harness />);
    act(() => Socket.sockets[0].disconnect(code));
    act(() => vi.advanceTimersByTime(15000));
    expect(Socket.sockets).toHaveLength(1);
    expect(state().dataset.status).toBe('failed');
  });

  it('honors a permanent close code following a socket error event', () => {
    render(<Harness />);
    const socket = Socket.sockets[0];
    act(() => {
      socket.onerror?.(new Event('error'));
      socket.disconnect(4404);
    });
    act(() => vi.advanceTimersByTime(15000));
    expect(Socket.sockets).toHaveLength(1);
    expect(state().dataset.status).toBe('failed');
  });

  it('preserves a fatal stale-writer error after permanent close', () => {
    render(<Harness />);
    const socket = Socket.sockets[0];
    act(() => {
      socket.open();
      socket.emit({ exception: 'Conversation changed elsewhere. Refresh and resend.', fatal: true });
      socket.disconnect(4409);
    });
    expect(state().textContent).toBe('Conversation changed elsewhere. Refresh and resend.');
    expect(state().dataset.disabled).toBe('true');
    act(() => vi.advanceTimersByTime(15000));
    expect(Socket.sockets).toHaveLength(1);
  });

  it('disables input immediately on fatal initialization errors', () => {
    render(<Harness />);
    const socket = Socket.sockets[0];
    act(() => { socket.open(); hydrate(socket); });
    expect(state().dataset.disabled).toBe('false');
    act(() => socket.emit({ exception: 'Unavailable', fatal: true }));
    expect(state().dataset.disabled).toBe('true');
    expect(state().textContent).toBe('Unavailable');
    act(() => socket.emit({ conversation: { messages: [], selected_collections: [9] } }));
    expect(state().dataset.disabled).toBe('true');
    expect(state().dataset.status).toBe('failed');
  });
});
