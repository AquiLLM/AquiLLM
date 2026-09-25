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
  const [exception, setException] = useState('');
  const [, setDebugHtml] = useState<string | null>(null);
  const [inputDisabled, setInputDisabled] = useState(true);

  const socketState = useChatWebSocket({
    convoId: '314',
    setConversation,
    setException,
    setDebugHtml,
    setInputDisabled,
  });
  const terminalError = socketState.terminalError;

  return (
    <>
      <output
        data-testid="conversation"
        data-spinner={!terminalError && shouldShowSpinner(conversation.messages)}
        data-input-disabled={inputDisabled}
      >
        {conversation.messages.map((message, index) => (
          <span key={message.message_uuid || index}>
            {message.role}:{message.tool_call_name || message.tool_name || message.content}
          </span>
        ))}
      </output>
      <output data-testid="error">{terminalError || exception}</output>
      <button
        type="button"
        onClick={() => {
          socketState.beginTurn();
          setInputDisabled(true);
          setConversation((previous) => ({
            ...previous,
            messages: [...previous.messages, { role: 'user', content: 'retry' }],
          }));
        }}
      >
        Retry
      </button>
    </>
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

  it('settles an aborted tool turn and allows a later retry to run normally', async () => {
    render(<Harness />);
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

    expect(screen.getByTestId('conversation').getAttribute('data-spinner')).toBe('true');
    expect(screen.getByTestId('conversation').getAttribute('data-input-disabled')).toBe('true');

    act(() => {
      socket.emit({ exception: 'Tool result validation failed.' });
    });

    expect(screen.getByTestId('error').textContent).toBe('Tool result validation failed.');
    expect(screen.getByTestId('conversation').textContent).toBe('assistant:vector_search');
    expect(screen.getByTestId('conversation').getAttribute('data-spinner')).toBe('false');
    expect(screen.getByTestId('conversation').getAttribute('data-input-disabled')).toBe('false');

    act(() => {
      socket.onopen?.(new Event('open'));
    });

    expect(screen.getByTestId('error').textContent).toBe('Tool result validation failed.');
    expect(screen.getByTestId('conversation').getAttribute('data-spinner')).toBe('false');

    act(() => {
      screen.getByRole('button', { name: 'Retry' }).click();
    });

    expect(screen.getByTestId('error').textContent).toBe('');
    expect(screen.getByTestId('conversation').getAttribute('data-spinner')).toBe('true');
    expect(screen.getByTestId('conversation').getAttribute('data-input-disabled')).toBe('true');

    act(() => {
      socket.emit({
        delta: {
          messages: [{
            role: 'assistant',
            content: 'Retry completed.',
            message_uuid: 'retry-answer',
            usage: 42,
          }],
        },
      });
    });

    expect(screen.getByTestId('conversation').textContent).toContain('assistant:vector_search');
    expect(screen.getByTestId('conversation').textContent).toContain('user:retry');
    expect(screen.getByTestId('conversation').textContent).toContain('assistant:Retry completed.');
    expect(screen.getByTestId('conversation').getAttribute('data-spinner')).toBe('false');
    expect(screen.getByTestId('conversation').getAttribute('data-input-disabled')).toBe('false');
  });

  it('resumes a terminal tool turn after authoritative reconnect hydration', async () => {
    render(<Harness />);
    await waitFor(() => expect(FakeWebSocket.latest).not.toBeNull());
    const socket = FakeWebSocket.latest!;
    const toolCall = {
      role: 'assistant' as const,
      content: '',
      message_uuid: 'resumed-tool-call',
      tool_call_name: 'vector_search',
      tool_call_input: { search_string: 'calibration' },
    };

    act(() => {
      socket.onopen?.(new Event('open'));
      socket.emit({ delta: { messages: [toolCall] } });
      socket.emit({ exception: 'Tool result validation failed.' });
      socket.onopen?.(new Event('open'));
    });

    expect(screen.getByTestId('error').textContent).toBe('Tool result validation failed.');
    expect(screen.getByTestId('conversation').getAttribute('data-spinner')).toBe('false');

    act(() => {
      socket.emit({ conversation: { messages: [toolCall] } });
    });

    expect(screen.getByTestId('error').textContent).toBe('');
    expect(screen.getByTestId('conversation').getAttribute('data-spinner')).toBe('true');
    expect(screen.getByTestId('conversation').getAttribute('data-input-disabled')).toBe('true');

    act(() => {
      socket.emit({
        delta: {
          messages: [{
            role: 'tool',
            content: '{"result": []}',
            message_uuid: 'resumed-tool-result',
            tool_name: 'vector_search',
            for_whom: 'assistant',
          }],
        },
      });
    });

    expect(screen.getByTestId('error').textContent).toBe('');
    expect(screen.getByTestId('conversation').getAttribute('data-spinner')).toBe('true');
    expect(screen.getByTestId('conversation').getAttribute('data-input-disabled')).toBe('true');

    act(() => {
      socket.emit({
        delta: {
          messages: [{
            role: 'assistant',
            content: 'Recovered answer.',
            message_uuid: 'resumed-answer',
            usage: 84,
          }],
        },
      });
    });

    expect(screen.getByTestId('error').textContent).toBe('');
    expect(screen.getByTestId('conversation').textContent).toContain('tool:vector_search');
    expect(screen.getByTestId('conversation').textContent).toContain('assistant:Recovered answer.');
    expect(screen.getByTestId('conversation').getAttribute('data-spinner')).toBe('false');
    expect(screen.getByTestId('conversation').getAttribute('data-input-disabled')).toBe('false');
  });
});
