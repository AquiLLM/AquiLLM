// @vitest-environment jsdom

import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { WebSocketMessage } from '../types';
import Chat from './Chat';

class FakeWebSocket {
  static OPEN = 1;
  static CLOSED = 3;
  static latest: FakeWebSocket | null = null;

  readyState = FakeWebSocket.OPEN;
  onopen: ((event: Event) => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onclose: ((event: CloseEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;

  constructor(_url: string) {
    FakeWebSocket.latest = this;
  }

  close = vi.fn();
  send = vi.fn();

  emit(payload: WebSocketMessage) {
    this.onmessage?.({ data: JSON.stringify(payload) } as MessageEvent);
  }

  disconnect() {
    this.readyState = FakeWebSocket.CLOSED;
    this.onclose?.({ code: 1006, reason: '' } as CloseEvent);
  }
}

describe('Chat readiness and terminal errors', () => {
  beforeEach(() => {
    FakeWebSocket.latest = null;
    vi.stubGlobal('WebSocket', FakeWebSocket);
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ collections: [{ id: 7, name: 'Research', parent: null }] }),
    }));
    Element.prototype.scrollIntoView = vi.fn();
  });

  afterEach(() => {
    cleanup();
    vi.useRealTimers();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it('hides the spinner and leaves the retry input usable after a terminal tool error', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => {});
    render(<Chat convoId="314" />);
    await waitFor(() => expect(FakeWebSocket.latest).not.toBeNull());
    const socket = FakeWebSocket.latest!;

    act(() => {
      socket.onopen?.(new Event('open'));
      socket.emit({ conversation: { messages: [], selected_collections: [] } });
      socket.emit({
        delta: {
          messages: [{
            role: 'assistant',
            content: '',
            message_uuid: 'failed-tool-call',
            tool_call_name: 'vector_search',
            tool_call_input: { search_string: 'calibration' },
          }],
        },
      });
    });

    expect(screen.getByRole('status', { name: 'AquiLLM is thinking' })).toBeTruthy();
    expect((screen.getByRole('textbox') as HTMLTextAreaElement).disabled).toBe(true);

    act(() => {
      socket.emit({ exception: 'Tool result validation failed.' });
    });

    expect(screen.getByText('Tool result validation failed.')).toBeTruthy();
    expect(screen.queryByRole('status', { name: 'AquiLLM is thinking' })).toBeNull();
    expect((screen.getByRole('textbox') as HTMLTextAreaElement).disabled).toBe(false);

    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'retry' } });
    fireEvent.click(screen.getByTitle('Send Message'));

    expect(screen.queryByText('Tool result validation failed.')).toBeNull();
    expect(screen.getByRole('status', { name: 'AquiLLM is thinking' })).toBeTruthy();
    expect((screen.getByRole('textbox') as HTMLTextAreaElement).disabled).toBe(true);
    expect(socket.send).toHaveBeenCalledWith(JSON.stringify({
      action: 'append',
      message: { role: 'user', content: 'retry' },
      collections: [],
      files: [],
    }));
  });

  it('preserves a draft through startup and waits for saved selections before sending', async () => {
    render(<Chat convoId="314" />);
    await waitFor(() => expect(FakeWebSocket.latest).not.toBeNull());
    const socket = FakeWebSocket.latest!;
    const input = screen.getByRole('textbox') as HTMLTextAreaElement;
    const sendButton = screen.getByTitle('Send Message') as HTMLButtonElement;
    expect(input.disabled).toBe(false);
    expect(sendButton.disabled).toBe(true);
    fireEvent.change(input, { target: { value: 'hello' } });
    fireEvent.keyDown(input, { key: 'Enter' });
    expect(input.value).toBe('hello');
    expect(socket.send).not.toHaveBeenCalled();

    act(() => socket.onopen?.(new Event('open')));
    expect(input.disabled).toBe(false);
    expect(sendButton.disabled).toBe(true);
    fireEvent.keyDown(input, { key: 'Enter' });
    fireEvent.click(sendButton);
    expect(input.value).toBe('hello');
    expect(socket.send).not.toHaveBeenCalled();

    act(() => socket.emit({ conversation: { messages: [], selected_collections: [7] } }));
    expect(input.disabled).toBe(false);
    expect(input.value).toBe('hello');
    expect(sendButton.disabled).toBe(false);
    expect(socket.send).not.toHaveBeenCalled();
    fireEvent.keyDown(input, { key: 'Enter' });
    expect(socket.send).toHaveBeenCalledWith(JSON.stringify({
      action: 'append', message: { role: 'user', content: 'hello' }, collections: [7], files: [],
    }));
    expect(input.value).toBe('');
  });

  describe('quiet connection status', () => {
    beforeEach(() => vi.useFakeTimers());

    it('never shows a connection status for fast startup', async () => {
      await act(async () => { render(<Chat convoId="314" />); });
      const socket = FakeWebSocket.latest!;
      expect(screen.queryByRole('status')).toBeNull();

      act(() => {
        vi.advanceTimersByTime(400);
        socket.onopen?.(new Event('open'));
      });
      expect(screen.queryByRole('status')).toBeNull();
      act(() => {
        vi.advanceTimersByTime(400);
        socket.emit({ conversation: { messages: [], selected_collections: [] } });
      });
      act(() => vi.advanceTimersByTime(2000));
      expect(screen.queryByRole('status')).toBeNull();
    });

    it('shows a small neutral composer status only after startup remains slow', async () => {
      await act(async () => { render(<Chat convoId="314" />); });
      expect(screen.queryByRole('status')).toBeNull();
      act(() => vi.advanceTimersByTime(1199));
      expect(screen.queryByRole('status')).toBeNull();
      act(() => vi.advanceTimersByTime(1));
      const status = screen.getByRole('status');
      expect(status.textContent).toBe('Getting your chat ready…');
      expect(status.closest('.sticky')?.contains(screen.getByRole('textbox'))).toBe(true);
      expect(status.className).not.toContain('bg-red-dark');
    });

    it('keeps the total grace period when connecting changes to hydrating', async () => {
      await act(async () => { render(<Chat convoId="314" />); });
      const socket = FakeWebSocket.latest!;
      act(() => vi.advanceTimersByTime(900));
      act(() => socket.onopen?.(new Event('open')));
      act(() => vi.advanceTimersByTime(299));
      expect(screen.queryByRole('status')).toBeNull();
      act(() => vi.advanceTimersByTime(1));
      expect(screen.getByRole('status').textContent).toBe('Getting your chat ready…');
      act(() => socket.emit({ conversation: { messages: [], selected_collections: [] } }));
      expect(screen.queryByRole('status')).toBeNull();
    });

    it('gives a later reconnect its own grace period while retaining an editable draft', async () => {
      await act(async () => { render(<Chat convoId="314" />); });
      const socket = FakeWebSocket.latest!;
      act(() => vi.advanceTimersByTime(1200));
      expect(screen.getByRole('status').textContent).toBe('Getting your chat ready…');
      act(() => {
        socket.onopen?.(new Event('open'));
        socket.emit({ conversation: { messages: [], selected_collections: [7] } });
      });
      act(() => vi.advanceTimersByTime(2000));
      const input = screen.getByRole('textbox') as HTMLTextAreaElement;
      fireEvent.change(input, { target: { value: 'keep this draft' } });
      act(() => socket.disconnect());
      expect(screen.queryByRole('status')).toBeNull();
      expect(input.disabled).toBe(false);
      expect((screen.getByTitle('Send Message') as HTMLButtonElement).disabled).toBe(true);
      fireEvent.keyDown(input, { key: 'Enter' });
      expect(socket.send).not.toHaveBeenCalled();
      expect(input.value).toBe('keep this draft');
      act(() => vi.advanceTimersByTime(1199));
      expect(screen.queryByRole('status')).toBeNull();
      act(() => vi.advanceTimersByTime(1));
      expect(screen.getByRole('status').textContent).toBe('Reconnecting…');
      act(() => vi.advanceTimersByTime(800));
      const replacement = FakeWebSocket.latest!;
      act(() => {
        replacement.onopen?.(new Event('open'));
        replacement.emit({ conversation: { messages: [], selected_collections: [7] } });
      });
      expect(screen.queryByRole('status')).toBeNull();
      expect(input.value).toBe('keep this draft');
      expect(replacement.send).not.toHaveBeenCalled();
    });

    it('starts a fresh grace period when the conversation changes', async () => {
      let view: ReturnType<typeof render>;
      await act(async () => { view = render(<Chat convoId="314" />); });
      act(() => vi.advanceTimersByTime(900));
      view!.rerender(<Chat convoId="315" />);
      act(() => vi.advanceTimersByTime(300));
      expect(screen.queryByRole('status')).toBeNull();
      act(() => vi.advanceTimersByTime(899));
      expect(screen.queryByRole('status')).toBeNull();
      act(() => vi.advanceTimersByTime(1));
      expect(screen.getByRole('status').textContent).toBe('Getting your chat ready…');
    });

    it('cleans up pending connection timers on unmount', async () => {
      let view: ReturnType<typeof render>;
      await act(async () => { view = render(<Chat convoId="314" />); });
      act(() => vi.advanceTimersByTime(900));
      view!.unmount();
      expect(vi.getTimerCount()).toBe(0);
      act(() => vi.advanceTimersByTime(5000));
      expect(screen.queryByRole('status')).toBeNull();
    });

    it('shows fatal errors immediately without waiting for the neutral status grace period', async () => {
      vi.spyOn(console, 'error').mockImplementation(() => {});
      await act(async () => { render(<Chat convoId="314" />); });
      const socket = FakeWebSocket.latest!;
      act(() => socket.emit({ exception: 'Conversation unavailable.', fatal: true }));
      expect(screen.getByText('Conversation unavailable.')).toBeTruthy();
      expect((screen.getByTitle('Send Message') as HTMLButtonElement).disabled).toBe(true);
      expect(screen.queryByRole('status')).toBeNull();
      act(() => vi.advanceTimersByTime(1200));
      expect(screen.queryByRole('status')).toBeNull();
    });
  });

  it('blocks collection edits while an open socket is still hydrating', async () => {
    render(<Chat convoId="314" />);
    await waitFor(() => expect(FakeWebSocket.latest).not.toBeNull());
    const socket = FakeWebSocket.latest!;
    act(() => socket.onopen?.(new Event('open')));
    const collectionsButton = screen.getByRole('button', { name: 'Collections' }) as HTMLButtonElement;
    expect(collectionsButton.disabled).toBe(true);
    fireEvent.click(collectionsButton);
    expect(screen.queryByText('Select Collections')).toBeNull();
    expect(socket.send).not.toHaveBeenCalled();
  });

  it('closes an open collection editor on disconnect without persisting a stale selection', async () => {
    render(<Chat convoId="314" />);
    await waitFor(() => expect(FakeWebSocket.latest).not.toBeNull());
    const socket = FakeWebSocket.latest!;
    act(() => {
      socket.onopen?.(new Event('open'));
      socket.emit({ conversation: { messages: [], selected_collections: [7] } });
    });
    fireEvent.click(screen.getByRole('button', { name: /Collections/ }));
    expect(await screen.findByRole('checkbox', { name: 'Research' })).toBeTruthy();

    act(() => socket.disconnect());
    expect(screen.queryByRole('checkbox', { name: 'Research' })).toBeNull();
    expect((screen.getByRole('button', { name: /Collections/ }) as HTMLButtonElement).disabled).toBe(true);
    expect(socket.send).not.toHaveBeenCalled();
  });

  it('keeps collection fetch errors through hydration and retries without losing saved selection', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => {});
    const fetchMock = vi.fn().mockRejectedValueOnce(new Error('temporary outage'))
      .mockResolvedValue({ ok: true, json: async () => ({ collections: [{ id: 7, name: 'Research', parent: null }] }) });
    vi.stubGlobal('fetch', fetchMock);
    render(<Chat convoId="314" />);
    await screen.findByText(/Failed to load collections/);
    const socket = FakeWebSocket.latest!;
    act(() => socket.emit({ conversation: { messages: [], selected_collections: [7] } }));
    expect(screen.getByText(/Failed to load collections/)).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: /Collections/ }));
    expect(screen.queryAllByRole('checkbox')).toHaveLength(0);
    fireEvent.click(screen.getByRole('button', { name: 'Retry collections' }));
    await waitFor(() => expect(screen.queryByText(/Failed to load collections/)).toBeNull());
    fireEvent.click(screen.getByRole('button', { name: /Collections/ }));
    expect((await screen.findByRole('checkbox', { name: 'Research' }) as HTMLInputElement).checked).toBe(true);
    expect(socket.send).not.toHaveBeenCalled();
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });
});
