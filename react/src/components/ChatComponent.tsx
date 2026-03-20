import React, { useState, useEffect, useRef } from 'react';
import { Send } from 'lucide-react';
import { CircularProgressbar } from 'react-circular-progressbar';

import type { 
  Message, 
  Collection, 
  Conversation, 
  WebSocketMessage, 
  ChatProps
} from '../features/chat/types';
import { MessageBubble, ToolCallGroup } from '../features/chat/components';
import { groupMessages, shouldShowSpinner } from '../features/chat/utils';

const Chat: React.FC<ChatProps> = ({ convoId, contextLimit }) => {
  const [conversation, setConversation] = useState<Conversation>({ messages: [] });
  const [_isConnected, setIsConnected] = useState(false);
  const [inputDisabled, setInputDisabled] = useState(true);
  const [messageInput, setMessageInput] = useState('');
  const [exception, setException] = useState('');
  const [debugHtml, setDebugHtml] = useState<string | null>(null);
  const [collections, setCollections] = useState<Collection[]>([]);
  const [selectedCollections, setSelectedCollections] = useState<Set<string>>(new Set());
  const [searchTerm, setSearchTerm] = useState('');
  const [connectionAttempts, setConnectionAttempts] = useState(0);
  const [showCollections, setShowCollections] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const conversationEndRef = useRef<HTMLDivElement>(null);
  const messageContainerRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const [textareaMinHeight, setTextareaMinHeight] = useState(0);
  const [contentOverflowing, setContentOverflowing] = useState(false);
  const isDragging = useRef(false);
  const dragStartY = useRef(0);
  const dragStartHeight = useRef(0);

  const MAX_RECONNECTION_ATTEMPTS = 5;
  const CONNECTION_TIMEOUT = 5000;
  const fallbackContextLimit = 200000;
  const contextLimitTokens = contextLimit && contextLimit > 0 ? contextLimit : fallbackContextLimit;

  useEffect(() => {
    if (conversationEndRef.current) {
      conversationEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [conversation]);

  useEffect(() => {
    fetchCollections();
  }, []);

  useEffect(() => {
    initWebSocket();
    
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [connectionAttempts]);

  const autoResizeTextarea = () => {
    const textarea = textareaRef.current;
    if (!textarea) return;
    textarea.style.height = 'auto';
    const contentHeight = textarea.scrollHeight;
    setContentOverflowing(contentHeight >= 450);
    textarea.style.height = `${Math.max(contentHeight, textareaMinHeight)}px`;
  };

  const handleDragStart = (e: React.MouseEvent) => {
    e.preventDefault();
    isDragging.current = true;
    dragStartY.current = e.clientY;
    const textarea = textareaRef.current;
    dragStartHeight.current = textarea ? textarea.offsetHeight : 0;

    const handleDragMove = (e: MouseEvent) => {
      if (!isDragging.current || !textareaRef.current) return;
      const delta = dragStartY.current - e.clientY;
      const newHeight = Math.max(0, Math.min(450, dragStartHeight.current + delta));
      setTextareaMinHeight(newHeight);
      textareaRef.current.style.height = 'auto';
      const contentHeight = textareaRef.current.scrollHeight;
      textareaRef.current.style.height = `${Math.max(contentHeight, newHeight)}px`;
    };

    const handleDragEnd = () => {
      isDragging.current = false;
      document.removeEventListener('mousemove', handleDragMove);
      document.removeEventListener('mouseup', handleDragEnd);
    };

    document.addEventListener('mousemove', handleDragMove);
    document.addEventListener('mouseup', handleDragEnd);
  };

  const fetchCollections = async () => {
    try {
      const response = await fetch("/api/collections/");
      if (!response.ok) {
        throw new Error(`HTTP error: ${response.status}`);
      }
      const data = await response.json();
      setCollections(data.collections);
    } catch (error) {
      console.error('Error fetching collections:', error);
      setException('Failed to load collections. Please refresh the page.');
    }
  };

  const initWebSocket = () => {
    if (connectionAttempts >= MAX_RECONNECTION_ATTEMPTS) {
      setException('Maximum reconnection attempts reached. Please refresh the page.');
      return;
    }

    setException(`Attempting to connect... (Attempt ${connectionAttempts + 1} of ${MAX_RECONNECTION_ATTEMPTS})`);
    
    try {
      const protocol = window.location.protocol === 'https:' ? 'wss://' : 'ws://';
      const ws = new WebSocket(`${protocol}${window.location.host}/ws/convo/${convoId}/`);
      wsRef.current = ws;
      
      const timeoutId = setTimeout(() => {
        if (ws.readyState !== WebSocket.OPEN) {
          ws.close();
          setException('Connection timeout. Retrying...');
          setConnectionAttempts(prev => prev + 1);
        }
      }, CONNECTION_TIMEOUT);
      
      ws.onopen = () => {
        console.log('Connected to chat server');
        clearTimeout(timeoutId);
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

          const mergeMessages = (existing: Message[], incoming: Message[]): Message[] => {
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
          };

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
              const mergedConversation = {
                ...prev,
                messages: mergedMessages,
                usage: data.stream!.usage ?? prev.usage,
              };
              return mergedConversation;
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
        clearTimeout(timeoutId);
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
        setTimeout(() => setConnectionAttempts(prev => prev + 1), 2000);
      };
      
      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setException('Connection error occurred. Retrying...');
      };
      
    } catch (error) {
      console.error('Error creating WebSocket:', error);
      setException(`Failed to create connection: ${error instanceof Error ? error.message : 'Unknown error'}`);
      setTimeout(() => setConnectionAttempts(prev => prev + 1), 2000);
    }
  };

  const sendMessage = () => {
    if (!messageInput.trim() || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    
    setInputDisabled(true);
    const newMessage: Message = { role: 'user', content: messageInput.trim()};
    
    const updatedConversation = { 
      ...conversation, 
      messages: [...conversation.messages, newMessage] 
    };
    
    setConversation(updatedConversation);
    
    const payload = {
      action: 'append',
      message: newMessage,
      collections: Array.from(selectedCollections),
      files: []
    };
    
    wsRef.current.send(JSON.stringify(payload));
    setMessageInput('');
    setTextareaMinHeight(0);
    setContentOverflowing(false);
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  };

  const rateMessage = (uuid: string | undefined, rating: number) => {
    if (!uuid || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    
    const payload = {
      action: 'rate',
      uuid: uuid,
      rating: rating
    };
    
    wsRef.current.send(JSON.stringify(payload));
    
    setConversation(prev => {
      const updatedMessages = prev.messages.map(msg => 
        msg.message_uuid === uuid ? { ...msg, rating } : msg
      );
      return { ...prev, messages: updatedMessages };
    });
  };

  const feedbackMessage = (uuid: string | undefined, feedback_text: string) => {
    if (!uuid || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    
    const payload = {
      action: 'feedback',
      uuid: uuid,
      feedback_text: feedback_text
    };
    
    wsRef.current.send(JSON.stringify(payload));
    
    setConversation(prev => {
      const updatedMessages = prev.messages.map(msg => 
        msg.message_uuid === uuid ? { ...msg, feedback_text } : msg
      );
      return { ...prev, messages: updatedMessages };
    });
  };

  const handleCollectionToggle = (collectionId: string) => {
    setSelectedCollections(prev => {
      const newSelected = new Set(prev);
      if (newSelected.has(collectionId)) {
        newSelected.delete(collectionId);
      } else {
        newSelected.add(collectionId);
      }
      return newSelected;
    });
  };

  const filteredCollections = collections.filter(collection =>
    collection.name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const getUsageColor = (ratio: number): string => {
    if (ratio >= 0.95) return 'var(--color-red-dark)';
    if (ratio >= 0.85) return 'var(--color-secondary_accent-dark)';
    if (ratio >= 0.7) return 'var(--color-secondary_accent-DEFAULT)';
    return 'var(--color-accent-dark)';
  };

  const usageValue = conversation.usage || 0;
  const clampedUsageValue = Math.min(usageValue, contextLimitTokens);
  const usageRatio = contextLimitTokens > 0 ? clampedUsageValue / contextLimitTokens : 0;

  return (
    <div className="flex flex-col h-full">
      {exception && (
        <div className="sticky top-0 z-50 font-mono text-text-normal p-4 mb-4 bg-red-dark rounded flex items-center justify-between">
          <span>{exception}</span>
          {debugHtml && (
            <button
              className="ml-4 px-3 py-1 bg-red-900 hover:bg-red-800 text-white rounded text-sm whitespace-nowrap"
              onClick={() => {
                const blob = new Blob([debugHtml], { type: 'text/html' });
                const url = URL.createObjectURL(blob);
                window.open(url, '_blank');
              }}
            >
              View Stack Trace
            </button>
          )}
        </div>
      )}

      <div
        ref={messageContainerRef}
        className="flex-grow overflow-y-auto w-full px-[20px] pt-[16px] md:px-[24px] md:pt-[20px]"
      >
        <div className="w-[98%] md:w-[96%] lg:w-[94%] xl:w-[92%] 2xl:max-w-[1800px] mx-auto gap-[12px] flex flex-col">
          {groupMessages(conversation.messages).map((item, index) => {
            if ('main' in item) {
              return (
                <div key={`group-${index}`} className="flex flex-col gap-1">
                  {item.main.content && (
                    <MessageBubble
                      message={item.main}
                      onRate={rateMessage}
                      onFeedback={feedbackMessage}
                    />
                  )}
                  <ToolCallGroup toolCalls={item.toolCalls} />
                </div>
              );
            }
            return (
              <MessageBubble
                key={`msg-${index}`}
                message={item}
                onRate={rateMessage}
                onFeedback={feedbackMessage}
              />
            );
          })}
          
          {shouldShowSpinner(conversation.messages) && (
            <div className="flex justify-center my-2">
              <div className="animate-spin rounded-full h-8 w-8 border-4 border-accent border-t-transparent"></div>
            </div>
          )}
          <div ref={conversationEndRef} />
        </div>
      </div>

      <div className="sticky bottom-0 w-full bg-scheme-shade_2 border-t border-border-mid_contrast mt-[8px]">
        <div className="w-[98%] md:w-[96%] lg:w-[94%] xl:w-[92%] 2xl:max-w-[1800px] mx-auto mb-[8px] mt-[8px]">
          <div className="flex items-center justify-center w-full gap-[12px]">
            <div className="flex h-[56px] min-w-[114px] shrink-0 flex-col items-center justify-center gap-[3px] rounded-[10px] border border-border-mid_contrast bg-scheme-shade_2 px-[8px] py-[6px]">
              <div className="h-[28px] w-[28px] rounded-full border border-border-mid_contrast bg-scheme-shade_2">
                <CircularProgressbar value={clampedUsageValue}
                                     maxValue={contextLimitTokens}
                                     strokeWidth={50}
                                     styles= {{
                                      path: {
                                        stroke: getUsageColor(usageRatio),
                                      },
                                      trail: {
                                        stroke: 'var(--color-border-low-contrast)',
                                      },
                                     }}
                                     text="" />
              </div>
              <div className="whitespace-nowrap text-center text-[11px] leading-[1.05] text-text-low_contrast">
                {`${usageValue.toLocaleString()} / ${contextLimitTokens.toLocaleString()}`}
              </div>
            </div>

            <div className="relative flex min-h-[56px] w-full flex-col justify-start gap-[8px] rounded-[10px] border border-border-mid_contrast bg-scheme-shade_2 px-4 py-[6px] transition-colors duration-200 has-[:focus]:border-transparent has-[:focus]:bg-scheme-shade_4">
              <div
                onMouseDown={contentOverflowing ? undefined : handleDragStart}
                className={`absolute left-1/2 -translate-x-1/2 top-0 -translate-y-1/2 z-50 flex justify-center px-2 py-1 group ${contentOverflowing ? 'pointer-events-none' : 'cursor-ns-resize'}`}
              >
                <div className={`w-12 h-1 rounded-full transition-colors ${contentOverflowing ? 'bg-transparent' : 'bg-border-mid_contrast group-hover:bg-text-low_contrast'}`} />
              </div>
              <div className="flex flex-grow items-center w-full">
                <textarea
                  id="message-input"
                  ref={textareaRef}
                  rows={1}
                  className="px-2 py-2 mr-[16px] flex-grow w-full rounded-lg bg-transparent border-none outline-none focus:outline-none focus:ring-0 disabled:cursor-not-allowed placeholder:text-text-lower_contrast text-text-normal resize-none overflow-y-auto max-h-[450px]"
                  placeholder={conversation.messages.length === 0 ? "How can I help you today?" : "Reply..."}
                  value={messageInput}
                  onChange={(e) => {
                    setMessageInput(e.target.value);
                    autoResizeTextarea();
                  }}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      sendMessage();
                    }
                  }}
                  disabled={inputDisabled}
                  autoComplete="off"
                />
                <button
                  onClick={sendMessage}
                  className="mr-[-4px] flex h-[44px] w-[44px] items-center justify-center rounded-[10px] border border-border-high_contrast bg-scheme-shade_4 p-0 text-text-normal transition-colors duration-200 hover:border-border-higher_contrast hover:bg-scheme-shade_5 disabled:cursor-not-allowed"
                  title="Send Message"
                  disabled={inputDisabled}
                >
                  <Send size={16} className="text-text-normal"/>
                </button>
              </div>
            </div>

            <div className="">
              <button
                onClick={() => setShowCollections(true)}
                className="flex h-[56px] w-[max-content] cursor-pointer items-center rounded-[10px] border border-border-high_contrast bg-scheme-shade_4 px-[16px] py-0 text-text-normal transition-colors duration-200 hover:border-border-higher_contrast hover:bg-scheme-shade_5"
              >
                <span className="text-text-normal">Collections</span>
                <span className="ml-2 text-sm text-text-normal">
                  {selectedCollections.size ? `(${selectedCollections.size} selected)` : ''}
                </span>
              </button>
            </div>
          </div>
        </div>
      </div>

      {showCollections && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50" onClick={() => setShowCollections(false)}>
          <div className="bg-scheme-shade_2 rounded-lg p-6 w-[90%] max-w-md max-h-[80vh] overflow-y-auto" onClick={(e) => e.stopPropagation()}>
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-semibold text-text-normal">Select Collections</h2>
              <button
                onClick={() => setShowCollections(false)}
                className="text-text-normal hover:text-text-slightly_less_contrast"
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            <div className="search-container mb-4">
              <input
                type="text"
                placeholder="Search..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="text-sm h-[36px] w-full p-2 border rounded-lg text-text-normal bg-scheme-shade_3 border-border-mid_contrast"
              />
            </div>

            <div className="p-2 border rounded-lg bg-scheme-shade_3 border-border-mid_contrast max-h-[400px] overflow-y-auto">
              {filteredCollections.map(collection => (
                <div key={collection.id} className="flex items-start gap-2 p-2 hover:bg-scheme-shade_4 rounded">
                  <input
                    type="checkbox"
                    id={`collection-${collection.id}`}
                    checked={selectedCollections.has(collection.id)}
                    onChange={() => handleCollectionToggle(collection.id)}
                    className={`w-4 h-4 mt-[3px] shrink-0 rounded cursor-pointer relative border ${
                      selectedCollections.has(collection.id)
                        ? "bg-accent border-accent after:content-['✓'] after:absolute after:text-white after:text-xs after:top-[-1px] after:left-[3px]"
                        : "bg-scheme-shade_5 border-border-mid_contrast"
                    }`}
                    style={{
                      appearance: 'none',
                      WebkitAppearance: 'none',
                      MozAppearance: 'none',
                    }}
                  />
                  <label htmlFor={`collection-${collection.id}`} className="text-sm leading-6 text-text-normal cursor-pointer">
                    {collection.name}
                  </label>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export const ChatComponent: React.FC<{ convoId: string; contextLimit?: number }> = ({ convoId, contextLimit }) => {
  return (
    <div className="h-full flex flex-col">
      <Chat convoId={convoId} contextLimit={contextLimit} />
    </div>
  );
};

export default ChatComponent;
