import React, { useEffect, useRef, useState } from 'react';
import { fetchConversation } from './api';
import type { ThreadMessage } from './types';

type Props = {
  conversationId: string;
  focalMessageUuid: string;
  onClose: () => void;
};

const ThreadModal: React.FC<Props> = ({ conversationId, focalMessageUuid, onClose }) => {
  const [messages, setMessages] = useState<ThreadMessage[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const focalRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetchConversation(conversationId)
      .then((msgs) => { if (!cancelled) setMessages(msgs); })
      .catch((err) => { if (!cancelled) setError(err.message || 'Failed to load conversation'); });
    return () => { cancelled = true; };
  }, [conversationId]);

  useEffect(() => {
    if (focalRef.current) {
      focalRef.current.scrollIntoView({ block: 'center', behavior: 'smooth' });
    }
  }, [messages]);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60"
      onClick={onClose}
    >
      <div
        className="bg-scheme-shade_2 element-border rounded-lg flex flex-col max-w-3xl w-full mx-4"
        style={{ maxHeight: '88vh' }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between px-5 py-3 border-b border-scheme-contrast/20 flex-shrink-0">
          <div>
            <span className="font-semibold text-text-normal text-sm">Conversation thread</span>
            {messages && (
              <span className="ml-2 text-xs text-text-muted">
                {messages.length} message{messages.length !== 1 ? 's' : ''}
              </span>
            )}
          </div>
          <button
            onClick={onClose}
            className="text-text-low_contrast hover:text-text-normal transition-colors text-xl leading-none"
            title="Close"
          >
            &times;
          </button>
        </div>

        {messages === null && !error && (
          <div className="flex items-center justify-center py-16">
            <div className="animate-spin rounded-full h-8 w-8 border-4 border-blue-500 border-t-transparent" />
          </div>
        )}

        {error && (
          <div className="px-5 py-4 text-red-400 text-sm">{error}</div>
        )}

        {messages && (
          <div className="flex-1 overflow-y-auto px-5 py-4 space-y-3 min-h-0">
            {messages.length === 0 ? (
              <p className="text-text-muted text-sm">No messages found.</p>
            ) : (
              messages.map((m) => (
                <ThreadBubble
                  key={m.message_uuid}
                  msg={m}
                  isFocal={m.message_uuid === focalMessageUuid}
                  focalRef={m.message_uuid === focalMessageUuid ? focalRef : undefined}
                />
              ))
            )}
          </div>
        )}
      </div>
    </div>
  );
};

type BubbleProps = {
  msg: ThreadMessage;
  isFocal: boolean;
  focalRef?: React.RefObject<HTMLDivElement | null>;
};

const ThreadBubble: React.FC<BubbleProps> = ({ msg, isFocal, focalRef }) => {
  const dotClass =
    msg.role === 'user'
      ? 'w-3 h-3 rounded-full bg-scheme-shade_7 flex-shrink-0'
      : msg.role === 'assistant'
      ? 'w-3 h-3 rounded-full bg-blue-500 flex-shrink-0'
      : 'w-3 h-3 rounded-full flex-shrink-0';
  const dotStyle =
    msg.role !== 'user' && msg.role !== 'assistant'
      ? { backgroundColor: 'var(--color-secondary_accent-DEFAULT)' }
      : undefined;

  const label =
    msg.role === 'user'
      ? 'You'
      : msg.role === 'tool'
      ? 'Tool'
      : msg.model
      ? `AquiLLM (${msg.model})`
      : 'AquiLLM';

  let bubbleClass =
    'w-full p-2 rounded-[10px] shadow-sm whitespace-pre-wrap break-words element-border leading-[1.35] text-[14px] text-text-normal ';
  if (msg.role === 'tool') {
    bubbleClass += 'chat-bubble-left-border-tool text-text-non_user_text_bubble';
  } else {
    bubbleClass += 'chat-bubble-left-border-assistant';
  }
  if (isFocal) bubbleClass += ' ring-2 ring-blue-500/60';

  const starColor =
    msg.rating == null
      ? undefined
      : msg.rating <= 2
      ? 'var(--color-secondary_accent-DEFAULT)'
      : msg.rating === 3
      ? 'var(--color-text-low-contrast)'
      : 'var(--color-green-DEFAULT)';

  return (
    <div ref={focalRef} className="flex justify-start">
      <div className="w-[88%] flex flex-col items-start">
        <div className="flex items-center gap-1.5 mb-1">
          <span className={dotClass} style={dotStyle} />
          <span className="text-[11px] text-text-low_contrast">{label}</span>
          {msg.rating != null && (
            <span className="ml-1 text-[11px]" style={{ color: starColor }}>
              {'★'.repeat(msg.rating)}
              {'☆'.repeat(5 - msg.rating)}
            </span>
          )}
          {isFocal && (
            <span className="ml-1 text-[10px] px-1.5 py-0.5 rounded bg-blue-600/20 text-blue-400 font-semibold">
              result row
            </span>
          )}
        </div>

        <div className={bubbleClass}>
          {msg.role === 'tool' ? (
            <>
              <div className="mb-1 font-bold text-sm">Tool output: {msg.tool_name || ''}</div>
              {msg.result_dict && (
                <details className="text-xs" open={msg.for_whom === 'user'}>
                  <summary className="cursor-pointer text-text-low_contrast hover:text-text-normal">
                    {'exception' in (msg.result_dict || {}) ? 'View exception' : 'View result'}
                  </summary>
                  <pre
                    className="mt-1 whitespace-pre-wrap break-words rounded p-2 text-xs overflow-x-auto"
                    style={{ background: 'var(--color-tool-details-section-tool)' }}
                  >
                    {JSON.stringify(
                      msg.result_dict.exception ?? msg.result_dict.result ?? msg.result_dict,
                      null,
                      2,
                    )}
                  </pre>
                </details>
              )}
            </>
          ) : msg.role === 'assistant' && msg.tool_call_input ? (
            <>
              <div className="font-bold text-sm">Called tool: {msg.tool_call_name || ''}</div>
              <details className="text-xs mt-1">
                <summary className="cursor-pointer text-text-low_contrast hover:text-text-normal">
                  View arguments
                </summary>
                <pre
                  className="mt-1 whitespace-pre-wrap break-words rounded p-2 text-xs overflow-x-auto"
                  style={{ background: 'var(--color-tool-details-section-assistant)' }}
                >
                  {JSON.stringify(msg.tool_call_input, null, 2)}
                </pre>
              </details>
            </>
          ) : (
            msg.content || ''
          )}
        </div>

        {msg.feedback_text && (
          <div className="mt-1 text-xs text-text-low_contrast italic">
            Feedback: &ldquo;{msg.feedback_text}&rdquo;
          </div>
        )}
      </div>
    </div>
  );
};

export default ThreadModal;
