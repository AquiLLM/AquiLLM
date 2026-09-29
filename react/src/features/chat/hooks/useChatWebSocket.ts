import { useCallback, useRef, useState, useEffect, type Dispatch, type SetStateAction } from 'react';
import type { Message, Conversation, WebSocketMessage } from '../types';
import type { ChatSocketHandoff } from './chatSocketBootstrap';

function mergeMessages(existing: Message[], incoming: Message[]): Message[] {
  const existingByUuid = new Map<string, Message>();
  existing.forEach((msg) => {
    if (msg.message_uuid) existingByUuid.set(msg.message_uuid, msg);
  });
  const incomingUuids = new Set(
    incoming.flatMap((msg) => msg.message_uuid ? [msg.message_uuid] : []),
  );
  const preserved = existing.filter(
    (msg) => !msg.message_uuid || !incomingUuids.has(msg.message_uuid),
  );

  const authoritative = incoming.map((msg) => {
    const prior = msg.message_uuid ? existingByUuid.get(msg.message_uuid) : undefined;
    return prior ? { ...prior, ...msg } : msg;
  });

  return [...preserved, ...authoritative];
}

export interface UseChatWebSocketParams {
  convoId: string;
  setConversation: Dispatch<SetStateAction<Conversation>>;
  setException: (msg: string) => void;
  setDebugHtml: (html: string | null) => void;
  setInputDisabled: (disabled: boolean) => void;
  setSelectedCollections?: (collectionIds: Set<string>) => void;
}

const MAX_RECONNECTION_ATTEMPTS = 5;
const CONNECTION_TIMEOUT = 5000;
const RETRY_DELAY = 2000;

export type ChatConnectionStatus = 'connecting' | 'hydrating' | 'reconnecting' | 'ready' | 'failed';

export function useChatWebSocket({
  convoId,
  setConversation,
  setException,
  setDebugHtml,
  setInputDisabled,
  setSelectedCollections,
}: UseChatWebSocketParams) {
  const wsRef = useRef<WebSocket | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<ChatConnectionStatus>('connecting');
  const [terminalError, setTerminalError] = useState('');

  const beginTurn = useCallback(() => {
    setTerminalError('');
    setException('');
    setDebugHtml(null);
  }, [setDebugHtml, setException]);

  useEffect(() => {
    let active = true;
    let attempt = 0;
    let currentSocket: WebSocket | null = null;
    let timeoutId: ReturnType<typeof setTimeout> | undefined;
    let retryId: ReturnType<typeof setTimeout> | undefined;
    let fatalMessage = '';
    let connectionReady = false;
    let hadHydrated = false;
    // Claim once per effect; retries always create fresh sockets.
    let bootstrap: ChatSocketHandoff | null = window.aquiChatSocketBootstrap?.take(String(convoId)) ?? null;

    const clearTimers = () => {
      if (timeoutId) clearTimeout(timeoutId);
      if (retryId) clearTimeout(retryId);
      timeoutId = undefined;
      retryId = undefined;
    };

    const hasSettledTurn = (messages: Message[]) => {
      if (!messages.length) return true;
      const lastMessage = messages[messages.length - 1];
      return (
        (lastMessage.role === 'assistant' && !lastMessage.tool_call_name) ||
        (lastMessage.role === 'tool' && lastMessage.for_whom === 'user')
      );
    };

    const applyInputState = (messages: Message[]) => {
      setInputDisabled(!hasSettledTurn(messages));
    };

    const connect = () => {
      if (!active) return;
      attempt += 1;
      connectionReady = false;
      fatalMessage = '';
      setConnectionStatus(attempt === 1 && !hadHydrated ? 'connecting' : 'reconnecting');
      setInputDisabled(true);
      setTerminalError('');
      setException('');
      setDebugHtml(null);

      const fail = (socket: WebSocket | null, code?: number) => {
        if (!active || (socket && currentSocket !== socket)) return;
        clearTimers();
        currentSocket = null;
        wsRef.current = null;
        setInputDisabled(true);
        if (socket && socket.readyState !== WebSocket.CLOSED) socket.close();

        if (code === 4401 || code === 4404 || code === 4409) {
          setConnectionStatus('failed');
          const fallback = code === 4401 ? 'Authentication required.' :
            code === 4404 ? 'Conversation unavailable.' : 'Conversation changed elsewhere.';
          setException(fatalMessage || `${fallback} Please refresh the page.`);
          return;
        }
        if (attempt >= MAX_RECONNECTION_ATTEMPTS) {
          setConnectionStatus('failed');
          setException('Maximum reconnection attempts reached. Please refresh the page.');
          return;
        }
        setConnectionStatus('reconnecting');
        retryId = setTimeout(connect, RETRY_DELAY);
      };

      try {
        const protocol = window.location.protocol === 'https:' ? 'wss://' : 'ws://';
        const adopted = bootstrap;
        bootstrap = null;
        const ws = adopted?.socket ?? new WebSocket(`${protocol}${window.location.host}/ws/convo/${convoId}/`);
        currentSocket = ws;
        wsRef.current = ws;
        // An expired bootstrap can still be waiting for the browser's close event.
        // Give that event a bounded window to deliver its permanent/retryable code.
        const elapsed = adopted && ws.readyState !== WebSocket.CLOSING
          ? Math.max(0, Date.now() - adopted.startedAt) : 0;
        timeoutId = setTimeout(() => fail(ws), Math.max(0, CONNECTION_TIMEOUT - elapsed));

        const isCurrent = () => active && currentSocket === ws;

        ws.onopen = () => {
          if (!isCurrent() || fatalMessage) return;
          setConnectionStatus('hydrating');
        };

        ws.onmessage = (event) => {
          if (!isCurrent() || fatalMessage) return;
          try {
            const data: WebSocketMessage = JSON.parse(event.data);

            if (data.exception) {
              console.error('Server error:', data.exception);
              setTerminalError(data.exception);
              setException(data.exception);
              setDebugHtml(data.debug_html || null);
              if (data.fatal) {
                fatalMessage = data.exception;
                setInputDisabled(true);
                setConnectionStatus('failed');
              } else if (connectionReady) {
                setInputDisabled(false);
              }
              return;
            }

            setException('');

            if (data.conversation) {
              const updatedConversation = data.conversation;
              if (!Array.isArray(updatedConversation.selected_collections)) return;
              connectionReady = true;
              hadHydrated = true;
              if (timeoutId) clearTimeout(timeoutId);
              timeoutId = undefined;
              if (hasSettledTurn(updatedConversation.messages)) attempt = 0;
              setConnectionStatus('ready');
              setTerminalError('');
              if (setSelectedCollections) {
                setSelectedCollections(new Set(updatedConversation.selected_collections.map(String)));
              }
              const lastAssistantMessage = updatedConversation.messages
                .slice()
                .reverse()
                .find((msg) => msg.role === 'assistant' && msg.usage !== undefined);
              if (lastAssistantMessage && lastAssistantMessage.usage !== undefined) {
                updatedConversation.usage = lastAssistantMessage.usage;
              }
              setConversation(updatedConversation);
              applyInputState(updatedConversation.messages);
              return;
            }

            if (data.stream && data.stream.message_uuid) {
              const streamMsg: Message = {
                role: 'assistant',
                content: data.stream.content || '',
                message_uuid: data.stream.message_uuid,
                usage: data.stream.usage,
              };

              setConversation((prev) => {
                const mergedMessages = mergeMessages(prev.messages, [streamMsg]);
                return {
                  ...prev,
                  messages: mergedMessages,
                  usage: data.stream!.usage ?? prev.usage,
                };
              });
              return;
            }

            if (data.delta && data.delta.messages && data.delta.messages.length) {
              if (connectionReady && hasSettledTurn(data.delta.messages)) attempt = 0;
              setConversation((prev) => {
                const mergedMessages = mergeMessages(prev.messages, data.delta!.messages);
                const merged = {
                  ...prev,
                  messages: mergedMessages,
                  usage: data.delta!.usage ?? prev.usage,
                };
                if (connectionReady) applyInputState(merged.messages);
                return merged;
              });
            }
          } catch (error) {
            setException(`Error processing message: ${error instanceof Error ? error.message : 'Unknown error'}`);
          }
        };

        ws.onclose = (event) => {
          if (!isCurrent()) return;
          fail(ws, event.code);
        };

        ws.onerror = () => {
          if (!isCurrent()) return;
          setInputDisabled(true);
          if (!fatalMessage) setConnectionStatus('reconnecting');
          // The close event carries the server's permanent/retryable code.
          // Keep a deadline in case the browser never delivers that event.
          if (!timeoutId) timeoutId = setTimeout(() => fail(ws), CONNECTION_TIMEOUT);
        };

        // Claiming detaches the buffer synchronously. Install live handlers first,
        // then replay every recorded event in order, including a close before mount.
        for (const buffered of adopted?.events ?? []) {
          switch (buffered.type) {
            case 'open': ws.onopen?.(buffered.event); break;
            case 'message': ws.onmessage?.(buffered.event); break;
            case 'error': ws.onerror?.(buffered.event); break;
            case 'close': ws.onclose?.(buffered.event); break;
          }
        }
      } catch (error) {
        console.error('Error creating WebSocket:', error);
        fail(null);
      }
    };

    setTerminalError('');
    connect();

    return () => {
      active = false;
      clearTimers();
      if (currentSocket) currentSocket.close();
      wsRef.current = null;
    };
  }, [convoId, setConversation, setException, setDebugHtml, setInputDisabled, setSelectedCollections]);

  return { wsRef, terminalError, connectionStatus, beginTurn };
}
