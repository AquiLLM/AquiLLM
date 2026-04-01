import React, { useState, useEffect, useMemo, useRef } from 'react';

import type { Message, Collection, Conversation, ChatProps } from '../types';
import { MessageBubble } from './MessageBubble';
import { ToolCallGroup } from './ToolCallGroup';
import ChatInputDock from './ChatInputDock';
import { groupMessages, shouldShowSpinner } from '../utils';
import { useChatWebSocket } from '../hooks/useChatWebSocket';
import ChatCollectionsModal from './ChatCollectionsModal';

const normalizeCollectionId = (collectionId: string | number): string => String(collectionId);

const toPayloadCollectionId = (collectionId: string): string | number => {
  const numericId = Number(collectionId);
  return Number.isInteger(numericId) && String(numericId) === collectionId ? numericId : collectionId;
};

const Chat: React.FC<ChatProps> = ({ convoId, contextLimit }) => {
  const [conversation, setConversation] = useState<Conversation>({ messages: [] });
  const [inputDisabled, setInputDisabled] = useState(true);
  const [messageInput, setMessageInput] = useState('');
  const [exception, setException] = useState('');
  const [debugHtml, setDebugHtml] = useState<string | null>(null);
  const [collections, setCollections] = useState<Collection[]>([]);
  const [selectedCollections, setSelectedCollections] = useState<Set<string>>(new Set());
  const [searchTerm, setSearchTerm] = useState('');
  const [showCollections, setShowCollections] = useState(false);
  const conversationEndRef = useRef<HTMLDivElement>(null);
  const messageContainerRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const [textareaMinHeight, setTextareaMinHeight] = useState(0);
  const [contentOverflowing, setContentOverflowing] = useState(false);
  const isDragging = useRef(false);
  const dragStartY = useRef(0);
  const dragStartHeight = useRef(0);

  const fallbackContextLimit = 200000;
  const contextLimitTokens = contextLimit && contextLimit > 0 ? contextLimit : fallbackContextLimit;

  const { wsRef } = useChatWebSocket({
    convoId,
    setConversation,
    setException,
    setDebugHtml,
    setInputDisabled,
  });

  useEffect(() => {
    if (conversationEndRef.current) {
      conversationEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [conversation]);

  useEffect(() => {
    const fetchCollections = async () => {
      try {
        const response = await fetch('/api/collections/');
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
    fetchCollections();
  }, []);

  const childrenByParentCollectionId = useMemo(() => {
    const map = new Map<string, string[]>();
    collections.forEach((collection) => {
      if (collection.parent == null) return;

      const parentId = normalizeCollectionId(collection.parent);
      const childId = normalizeCollectionId(collection.id);
      const siblings = map.get(parentId);

      if (siblings) {
        siblings.push(childId);
      } else {
        map.set(parentId, [childId]);
      }
    });
    return map;
  }, [collections]);

  const getDescendantCollectionIds = (collectionId: string): string[] => {
    const descendants: string[] = [];
    const visited = new Set<string>();
    const queue = [...(childrenByParentCollectionId.get(collectionId) ?? [])];

    while (queue.length > 0) {
      const currentId = queue.shift();
      if (!currentId || visited.has(currentId)) continue;

      visited.add(currentId);
      descendants.push(currentId);

      const children = childrenByParentCollectionId.get(currentId);
      if (children) {
        queue.push(...children);
      }
    }

    return descendants;
  };

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

  const sendMessage = () => {
    if (!messageInput.trim() || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;

    setInputDisabled(true);
    const newMessage: Message = { role: 'user', content: messageInput.trim() };

    const updatedConversation = {
      ...conversation,
      messages: [...conversation.messages, newMessage],
    };

    setConversation(updatedConversation);

    const payload = {
      action: 'append',
      message: newMessage,
      collections: Array.from(selectedCollections).map(toPayloadCollectionId),
      files: [],
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
      uuid,
      rating,
    };

    wsRef.current.send(JSON.stringify(payload));

    setConversation((prev) => {
      const updatedMessages = prev.messages.map((msg) =>
        msg.message_uuid === uuid ? { ...msg, rating } : msg
      );
      return { ...prev, messages: updatedMessages };
    });
  };

  const feedbackMessage = (uuid: string | undefined, feedback_text: string) => {
    if (!uuid || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;

    const payload = {
      action: 'feedback',
      uuid,
      feedback_text,
    };

    wsRef.current.send(JSON.stringify(payload));

    setConversation((prev) => {
      const updatedMessages = prev.messages.map((msg) =>
        msg.message_uuid === uuid ? { ...msg, feedback_text } : msg
      );
      return { ...prev, messages: updatedMessages };
    });
  };

  const handleCollectionToggle = (collectionId: string) => {
    const normalizedCollectionId = normalizeCollectionId(collectionId);
    setSelectedCollections((prev) => {
      const newSelected = new Set(prev);
      if (newSelected.has(normalizedCollectionId)) {
        newSelected.delete(normalizedCollectionId);
        getDescendantCollectionIds(normalizedCollectionId).forEach((descendantId) => {
          newSelected.delete(descendantId);
        });
      } else {
        newSelected.add(normalizedCollectionId);
        getDescendantCollectionIds(normalizedCollectionId).forEach((descendantId) => {
          newSelected.add(descendantId);
        });
      }
      return newSelected;
    });
  };

  const filteredCollections = collections.filter((collection) =>
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
              type="button"
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
              <div className="animate-spin rounded-full h-8 w-8 border-4 border-accent border-t-transparent" />
            </div>
          )}
          <div ref={conversationEndRef} />
        </div>
      </div>

      <ChatInputDock
        clampedUsageValue={clampedUsageValue}
        contextLimitTokens={contextLimitTokens}
        usageValue={usageValue}
        usageStrokeColor={getUsageColor(usageRatio)}
        contentOverflowing={contentOverflowing}
        onDragStart={handleDragStart}
        textareaRef={textareaRef}
        emptyThread={conversation.messages.length === 0}
        messageInput={messageInput}
        onMessageInputChange={setMessageInput}
        onAutoResize={autoResizeTextarea}
        onSend={sendMessage}
        inputDisabled={inputDisabled}
        onOpenCollections={() => setShowCollections(true)}
        selectedCount={selectedCollections.size}
      />

      <ChatCollectionsModal
        open={showCollections}
        onClose={() => setShowCollections(false)}
        searchTerm={searchTerm}
        onSearchTermChange={setSearchTerm}
        filteredCollections={filteredCollections}
        selectedCollections={selectedCollections}
        onToggleCollection={handleCollectionToggle}
      />
    </div>
  );
};

export default Chat;
