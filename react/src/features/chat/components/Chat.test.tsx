// @vitest-environment jsdom

import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { WebSocketMessage } from '../types';
import Chat from './Chat';

class FakeWebSocket {
  static OPEN = 1;
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
}

describe('Chat terminal tool errors', () => {
  beforeEach(() => {
    FakeWebSocket.latest = null;
    vi.stubGlobal('WebSocket', FakeWebSocket);
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ collections: [] }),
    }));
    Element.prototype.scrollIntoView = vi.fn();
  });

  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it('hides the spinner and leaves the retry input usable after a terminal tool error', async () => {
    render(<Chat convoId="314" />);
    await waitFor(() => expect(FakeWebSocket.latest).not.toBeNull());
    const socket = FakeWebSocket.latest!;

    act(() => {
      socket.onopen?.(new Event('open'));
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
});
