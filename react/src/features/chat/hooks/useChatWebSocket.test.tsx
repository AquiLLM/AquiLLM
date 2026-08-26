// @vitest-environment jsdom

import { useState } from 'react';
import { act, cleanup, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { Conversation, WebSocketMessage } from '../types';
import { shouldShowSpinner } from '../utils';
import { useChatWebSocket } from './useChatWebSocket';

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

const Harness = () => {
  const [conversation, setConversation] = useState<Conversation>({ messages: [] });

  useChatWebSocket({
    convoId: '314',
    setConversation,
    setException: vi.fn(),
    setDebugHtml: vi.fn(),
    setInputDisabled: vi.fn(),
  });

  return (
    <output data-testid="conversation" data-spinner={shouldShowSpinner(conversation.messages)}>
      {conversation.messages.map((message) => (
        <span key={message.message_uuid}>
          {message.role}:{message.tool_call_name || message.tool_name || message.content}
        </span>
      ))}
    </output>
  );
};

describe('useChatWebSocket', () => {
  beforeEach(() => {
    FakeWebSocket.latest = null;
    vi.stubGlobal('WebSocket', FakeWebSocket);
  });

  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it('places a streamed final answer after its authoritative tool transcript', async () => {
    render(<Harness />);
    await waitFor(() => expect(FakeWebSocket.latest).not.toBeNull());
    const socket = FakeWebSocket.latest!;

    act(() => {
      socket.emit({
        stream: {
          role: 'assistant',
          message_uuid: 'final-answer',
          content: 'Evidence-backed answer.',
        },
      });
      socket.emit({
        delta: {
          messages: [
            {
              role: 'assistant',
              content: '',
              message_uuid: 'tool-call',
              tool_call_name: 'vector_search',
              tool_call_input: { search_string: 'attensity' },
            },
            {
              role: 'tool',
              content: '{}',
              message_uuid: 'tool-result',
              tool_name: 'vector_search',
              for_whom: 'assistant',
            },
            {
              role: 'assistant',
              content: 'Evidence-backed answer.',
              message_uuid: 'final-answer',
              usage: 11787,
            },
          ],
        },
      });
    });

    await waitFor(() => {
      expect(screen.getByTestId('conversation').textContent).toBe(
        'assistant:vector_searchtool:vector_searchassistant:Evidence-backed answer.',
      );
    });
    expect(screen.getByTestId('conversation').getAttribute('data-spinner')).toBe('false');
  });
});
