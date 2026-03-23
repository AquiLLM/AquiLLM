import { useRef, useState, useEffect, type Dispatch, type SetStateAction } from 'react';
import type { Message, Conversation, WebSocketMessage } from '../types';

function mergeMessages(existing: Message[], incoming: Message[]): Message[] {
  const merged = [...existing];
  const indexByUuid = new Map<string, number>();
  merged.forEach((msg, idx) => {
    if (msg.message_uuid) indexByUuid.set(msg.message_uuid, idx);
  });

  incoming.forEach((msg) => {
    if (msg.message_uuid && indexByUuid.has(msg.message_uuid)) {
      const existingIdx = indexByUuid.get(msg.message_uuid)!;
      merged[existingIdx] = { ...merged[existingIdx], ...msg };
    } else {
      merged.push(msg);
      if (msg.message_uuid) indexByUuid.set(msg.message_uuid, merged.length - 1);
    }
  });

  return merged;
}

export interface UseChatWebSocketParams {
  convoId: string;
  setConversation: Dispatch<SetStateAction<Conversation>>;
  setException: (msg: string) => void;
  setDebugHtml: (html: string | null) => void;
  setInputDisabled: (disabled: boolean) => void;
}

const MAX_RECONNECTION_ATTEMPTS = 5;
const CONNECTION_TIMEOUT = 5000;

export function useChatWebSocket({
  convoId,
  setConversation,
  setException,
  setDebugHtml,
  setInputDisabled,
}: UseChatWebSocketParams) {
  const wsRef = useRef<WebSocket | null>(null);
  const [_isConnected, setIsConnected] = useState(false);
  const [connectionAttempts, setConnectionAttempts] = useState(0);

  useEffect(() => {
    let connectTimeoutId: ReturnType<typeof setTimeout> | undefined;

    if (connectionAttempts >= MAX_RECONNECTION_ATTEMPTS) {
      setException('Maximum reconnection attempts reached. Please refresh the page.');
      return undefined;
    }

    setException(`Attempting to connect... (Attempt ${connectionAttempts + 1} of ${MAX_RECONNECTION_ATTEMPTS})`);

    try {
      const protocol = window.location.protocol === 'https:' ? 'wss://' : 'ws://';
      const ws = new WebSocket(`${protocol}${window.location.host}/ws/convo/${convoId}/`);
      wsRef.current = ws;

      connectTimeoutId = setTimeout(() => {
        if (ws.readyState !== WebSocket.OPEN) {
          ws.close();
          setException('Connection timeout. Retrying...');
          setConnectionAttempts((prev) => prev + 1);
        }
      }, CONNECTION_TIMEOUT);

      const applyInputState = (messages: Message[]) => {
        if (!messages.length) {
          setInputDisabled(false);
          return;
        }
        const lastMessage = messages[messages.length - 1];
        const shouldEnableInput =
          (lastMessage.role === 'assistant' && !lastMessage.tool_call_input) ||
          (lastMessage.role === 'tool' && lastMessage.for_whom === 'user');
        setInputDisabled(!shouldEnableInput);
      };

      ws.onopen = () => {
        console.log('Connected to chat server');
        if (connectTimeoutId) clearTimeout(connectTimeoutId);
        setConnectionAttempts(0);
        setInputDisabled(false);
        setIsConnected(true);
        setException('');
      };

      ws.onmessage = (event) => {
        try {
          const data: WebSocketMessage = JSON.parse(event.data);

          if (data.exception) {
            console.error('Server error:', data.exception);
            setException(data.exception);
            setDebugHtml(data.debug_html || null);
            setInputDisabled(false);
            return;
          }

          setException('');

          if (data.conversation) {
            const updatedConversation = data.conversation;
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
            setConversation((prev) => {
              const mergedMessages = mergeMessages(prev.messages, data.delta!.messages);
              const merged = {
                ...prev,
                messages: mergedMessages,
                usage: data.delta!.usage ?? prev.usage,
              };
              applyInputState(merged.messages);
              return merged;
            });
          }
        } catch (error) {
          setException(`Error processing message: ${error instanceof Error ? error.message : 'Unknown error'}`);
        }
      };

      ws.onclose = (event) => {
        if (connectTimeoutId) clearTimeout(connectTimeoutId);
        console.log('Disconnected from chat server', event.code, event.reason);
        setInputDisabled(true);
        setIsConnected(false);

        let message = 'Disconnected from server. ';
        if (event.code === 1006) {
          message += 'Abnormal closure. ';
        } else if (event.code === 1015) {
          message += 'TLS handshake failed. ';
        }
        message += 'Attempting to reconnect...';

        setException(message);
        setTimeout(() => setConnectionAttempts((prev) => prev + 1), 2000);
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setException('Connection error occurred. Retrying...');
      };
    } catch (error) {
      console.error('Error creating WebSocket:', error);
      setException(`Failed to create connection: ${error instanceof Error ? error.message : 'Unknown error'}`);
      setTimeout(() => setConnectionAttempts((prev) => prev + 1), 2000);
    }

    return () => {
      if (connectTimeoutId) clearTimeout(connectTimeoutId);
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [connectionAttempts, convoId, setConversation, setException, setDebugHtml, setInputDisabled]);

  return { wsRef };
}
