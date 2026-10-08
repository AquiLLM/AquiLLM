import React, { useState, useEffect, useRef, useCallback } from 'react';
import type { Message, Conversation, ChatProps, ChatContextCatalog, SkillOverrides, ContextSelection } from '../types';
import { MessageBubble } from './MessageBubble';
import { ToolCallGroup } from './ToolCallGroup';
import ChatInputDock from './ChatInputDock';
import { groupMessages, shouldShowSpinner } from '../utils';
import { useChatWebSocket } from '../hooks/useChatWebSocket';
import ChatCollectionsModal from './ChatCollectionsModal';
import { AquillmLogo } from '../../../shared/components';

import { payloadCollectionId, skillEnabled, validOverrides } from '../utils/contextSelection';

const Chat: React.FC<ChatProps> = ({ convoId, contextLimit }) => {
  const [conversation, setConversation] = useState<Conversation>({ messages: [] });
  const [inputDisabled, setInputDisabled] = useState(true);
  const [messageInput, setMessageInput] = useState('');
  const [exception, setException] = useState('');
  const [debugHtml, setDebugHtml] = useState<string | null>(null);
  const [catalog, setCatalog] = useState<ChatContextCatalog>({ collections: [], skills: [], skills_enabled: false });
  const [collectionsError, setCollectionsError] = useState('');
  const [collectionsLoading, setCollectionsLoading] = useState(true);
  const collectionsRequest = useRef<AbortController | null>(null);
  const [selectedCollections, setSelectedCollections] = useState<Set<string>>(new Set());
  const [skillOverrides, setSkillOverrides] = useState<SkillOverrides>({});
  const [showCollections, setShowCollections] = useState(false);
  const conversationEndRef = useRef<HTMLDivElement>(null);
  const messageContainerRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const [textareaMinHeight, setTextareaMinHeight] = useState(0);
  const [contentOverflowing, setContentOverflowing] = useState(false);
  const [delayedConnectionConvoId, setDelayedConnectionConvoId] = useState<string | null>(null);
  const isDragging = useRef(false);
  const dragStartY = useRef(0);
  const dragStartHeight = useRef(0);

  const fallbackContextLimit = 200000;
  const contextLimitTokens = contextLimit && contextLimit > 0 ? contextLimit : fallbackContextLimit;

  const { wsRef, terminalError, connectionStatus, beginTurn, saveContext, contextSaving } = useChatWebSocket({
    convoId,
    setConversation,
    setException,
    setDebugHtml,
    setInputDisabled,
    setSelectedCollections,
    setSkillOverrides,
  });
  const collectionsReady = connectionStatus === 'ready';
  const collectionsEditable = collectionsReady && !inputDisabled && !contextSaving;
  const waitingForConnection = connectionStatus === 'connecting' ||
    connectionStatus === 'hydrating' || connectionStatus === 'reconnecting';

  useEffect(() => {
    setDelayedConnectionConvoId(null);
    if (!waitingForConnection) return;

    const timer = setTimeout(() => setDelayedConnectionConvoId(convoId), 1200);
    return () => clearTimeout(timer);
  }, [convoId, waitingForConnection]);

  useEffect(() => { setShowCollections(false); }, [convoId]);
  useEffect(() => {
    if (conversationEndRef.current) {
      conversationEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [conversation]);

  const fetchCollections = useCallback(async () => {
    collectionsRequest.current?.abort();
    const controller = new AbortController();
    collectionsRequest.current = controller;
    setCollectionsLoading(true);
    try {
      const response = await fetch('/api/collections/chat-context/', { signal: controller.signal });
      if (!response.ok) {
        throw new Error(`HTTP error: ${response.status}`);
      }
      const data = await response.json();
      if (!Array.isArray(data.collections) || !Array.isArray(data.skills) || typeof data.skills_enabled !== 'boolean') throw new Error('Invalid context response');
      if (controller.signal.aborted) return;
      setCatalog(data);
      setCollectionsError('');
    } catch (error) {
      if (controller.signal.aborted) return;
      console.error('Error fetching collections:', error);
      setCollectionsError('Failed to load collections and skills. Please try again.');
    } finally {
      if (!controller.signal.aborted) setCollectionsLoading(false);
    }
  }, []);
  useEffect(() => {
    void fetchCollections();
    return () => collectionsRequest.current?.abort();
  }, [fetchCollections]);

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
    if (connectionStatus !== 'ready' || inputDisabled || contextSaving || !messageInput.trim() || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;

    beginTurn();
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
      collections: Array.from(selectedCollections).map(payloadCollectionId),
      ...(catalog.skills_enabled ? { skill_overrides: validOverrides(catalog.skills, skillOverrides) } : {}),
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

  const applyContext = async (selection: ContextSelection) => {
    if (!collectionsEditable) throw new Error('Wait for the chat to be ready before applying changes.');
    const available = new Set(catalog.collections.map(collection => String(collection.id)));
    const collections = new Set([...selection.selectedCollections].filter(id => available.has(id)));
    await saveContext(collections, catalog.skills_enabled ? validOverrides(catalog.skills, selection.skillOverrides) : undefined);
  };

  const getUsageColor = (ratio: number): string => {
    if (ratio >= 0.95) return 'var(--color-red-dark)';
    if (ratio >= 0.85) return 'var(--color-secondary_accent-dark)';
    if (ratio >= 0.7) return 'var(--color-secondary_accent-DEFAULT)';
    return 'var(--color-accent-dark)';
  };

  const usageValue = conversation.usage || 0;
  const clampedUsageValue = Math.min(usageValue, contextLimitTokens);
  const usageRatio = contextLimitTokens > 0 ? clampedUsageValue / contextLimitTokens : 0;
  const visibleException = terminalError || exception;
  const connectionMessage = !visibleException && waitingForConnection && delayedConnectionConvoId === convoId
    ? connectionStatus === 'reconnecting' ? 'Reconnecting…' : 'Getting your chat ready…'
    : null;

  return (
    <div className="flex flex-col h-full">
      {collectionsError && !showCollections && (
        <div role="alert" className="p-3 text-text-normal bg-scheme-shade_3 flex items-center justify-between gap-3">
          <span>{collectionsError}</span>
          <button type="button" onClick={() => void fetchCollections()} disabled={collectionsLoading} className="underline whitespace-nowrap">
            {collectionsLoading ? 'Loading collections…' : 'Retry collections'}
          </button>
        </div>
      )}
      {visibleException && (
        <div className="sticky top-0 z-50 font-mono text-text-normal p-4 mb-4 bg-red-dark rounded flex items-center justify-between">
          <span>{visibleException}</span>
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

          {!terminalError && shouldShowSpinner(conversation.messages) && (
            <div className="group flex justify-start" role="status" aria-label="AquiLLM is thinking">
              <div className="w-[88%] flex flex-col items-start">
                <div className="flex items-center gap-1.5 mb-1">
                  <AquillmLogo role="assistant" />
                  <span className="text-[11px] text-text-low_contrast">AquiLLM</span>
                </div>
                <div className="assistant-message chat-bubble-left-border-assistant element-border rounded-[10px] px-3 py-2 shadow-sm">
                  <div className="flex h-[18px] items-center gap-1.5" aria-hidden="true">
                    <span
                      className="h-2 w-2 rounded-full bg-accent opacity-50 animate-bounce"
                      style={{ animationDuration: '900ms' }}
                    />
                    <span
                      className="h-2 w-2 rounded-full bg-accent opacity-50 animate-bounce"
                      style={{ animationDelay: '120ms', animationDuration: '900ms' }}
                    />
                    <span
                      className="h-2 w-2 rounded-full bg-accent opacity-50 animate-bounce"
                      style={{ animationDelay: '240ms', animationDuration: '900ms' }}
                    />
                  </div>
                </div>
              </div>
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
        inputDisabled={connectionStatus === 'failed' || (connectionStatus === 'ready' && inputDisabled)}
        sendDisabled={inputDisabled || contextSaving || connectionStatus !== 'ready'}
        connectionMessage={connectionMessage}
        onOpenCollections={() => { if (collectionsEditable) setShowCollections(true); }}
        collectionsDisabled={!collectionsEditable}
        collectionsLoading={collectionsLoading}
        selectedCount={catalog.collections.filter(collection => selectedCollections.has(String(collection.id))).length}
        selectedSkillCount={catalog.skills_enabled ? catalog.skills.filter(skill => skillEnabled(skill, selectedCollections, skillOverrides)).length : 0}
      />

      <ChatCollectionsModal
        key={convoId}
        open={showCollections}
        onClose={() => setShowCollections(false)}
        catalog={catalog}
        selectedCollections={selectedCollections}
        skillOverrides={skillOverrides}
        onApply={applyContext}
        loading={collectionsLoading}
        error={collectionsError}
        onRetry={() => void fetchCollections()}
        applyDisabled={!collectionsEditable}
      />
    </div>
  );
};

export default Chat;
