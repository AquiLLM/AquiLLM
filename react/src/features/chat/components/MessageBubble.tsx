import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';
import formatUrl from '../../../utils/formatUrl';
import { Collapsible, ToolResult, AquillmLogo, UserLogo } from '../../../shared/components';
import { RatingButtons } from './RatingButtons';
import type { Message } from '../types';

interface MessageBubbleProps {
  message: Message;
  onRate: (uuid: string | undefined, rating: number) => void;
  onFeedback: (uuid: string | undefined, feedback_text: string) => void;
}

export const MessageBubble: React.FC<MessageBubbleProps> = ({ message, onRate, onFeedback }) => {
  const displayTime = new Date().toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });

  const getMessageClasses = () => {
    let classes = "w-full p-2 rounded-[10px] shadow-sm whitespace-pre-wrap break-words element-border leading-[1.35] text-[14px]";
    
    if (message.role === 'user') {
      return `${classes} user-message chat-bubble-left-border-assistant text-text-normal`;
    } else if (message.role === 'assistant') {
      return `${classes} assistant-message chat-bubble-left-border-assistant text-text-normal`;
    } else if (message.role === 'tool') {
      return `${classes} border border-1 border-secondary_accent-light chat-bubble-left-border-tool text-text-non_user_text_bubble`;
    }
    
    return classes;
  };

  return (
    <div className="group flex justify-start">
      <div className="w-[88%] flex flex-col items-start">
        <div className="flex items-center gap-1.5 mb-1">
          {message.role === 'user' ? (
            <UserLogo />
          ) : (
            <AquillmLogo role={message.role} />
          )}
          <span className="text-[11px] text-text-low_contrast">
            {message.role === 'user' ? 'You' : message.role === 'tool' ? 'Tool' : 'AquiLLM'}
          </span>
        </div>

        <div className={getMessageClasses()}>
        {message.role === 'user' && (
          <div className="markdown-cell prose max-w-none whitespace-normal">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {message.content}
            </ReactMarkdown>
          </div>
        )}
        {message.role === 'assistant' && !message.tool_call_input && (
          <div className="markdown-cell prose prose-sm md:prose-base max-w-none whitespace-normal leading-relaxed">
            <ReactMarkdown 
              remarkPlugins={[remarkGfm]} 
              rehypePlugins={[rehypeRaw]}
              components={{
                h1: ({ children, ...props }) => (
                  <h1 {...props} className="mt-0 mb-3 text-[1.75rem] leading-tight font-semibold">
                    {children}
                  </h1>
                ),
                h2: ({ children, ...props }) => (
                  <h2 {...props} className="mt-4 mb-2 text-[1.35rem] leading-tight font-semibold">
                    {children}
                  </h2>
                ),
                h3: ({ children, ...props }) => (
                  <h3 {...props} className="mt-3 mb-2 text-[1.15rem] leading-tight font-semibold">
                    {children}
                  </h3>
                ),
                p: ({ children, ...props }) => (
                  <p {...props} className="my-2 leading-relaxed">
                    {children}
                  </p>
                ),
                img: ({ node, ...props }) => {
                  const altText = (props.alt ?? '').trim();
                  const showCaption = altText.length > 0 && !/^image$/i.test(altText);

                  return (
                    <figure className="my-3 w-fit max-w-full rounded-[10px] border border-border-mid_contrast bg-scheme-shade_3 p-2.5">
                      <img
                        {...props}
                        className="block max-w-full h-auto rounded-lg border border-border-mid_contrast cursor-pointer hover:opacity-90 transition-opacity"
                        style={{ maxHeight: '360px', objectFit: 'contain' }}
                        onClick={() => props.src && window.open(props.src, '_blank')}
                        title="Click to view full size"
                      />
                      {showCaption && (
                        <figcaption className="mt-2 text-xs font-medium text-text-low_contrast">
                          {altText}
                        </figcaption>
                      )}
                    </figure>
                  );
                },
              }}
            >
              {message.content}
            </ReactMarkdown>
          </div>
        )}
        
        {message.role === 'assistant' && message.tool_call_input && (
          <div className="mt-2.5 text-sm">
            <strong>Called Tool: {message.tool_call_name}</strong>
            <Collapsible 
              summary="View Tool Arguments" 
              summaryTextColor="text-text-non_user_text_bubble"
              content={
                <pre className="whitespace-pre-wrap break-words bg-tool_details-assistant p-2 rounded text-text-non_user_text_bubble">
                  {JSON.stringify(message.tool_call_input, null, 2)}
                </pre>
              }
            />
          </div>
        )}
        
        {message.role === 'tool' && (
          <>
            <div className="mb-2 font-bold">
              Tool Output: {message.tool_name}
            </div>
            <Collapsible 
              summary={'exception' in (message.result_dict || {}) ? 'View Exception' : 'View Results'}
              summaryTextColor="text-text-non_user_text_bubble"
              isOpen={message.for_whom === 'user'}
              content={
                <div className="bg-tool_details-tool p-2 rounded text-text-non_user_text_bubble">
                  <ToolResult result={'exception' in (message.result_dict || {}) ? 
                    message.result_dict?.exception : 
                    message.result_dict?.result} 
                  />
                </div>
              }
            />
          </>
        )}
        
        {(message.role === 'assistant' || message.role === 'tool') && (
          <RatingButtons 
            rating={message.rating}
            feedback_text={message.feedback_text}
            onRate={(rating) => onRate(message.message_uuid, rating)}
            onFeedback={(text) => onFeedback(message.message_uuid, text)}
          />
        )}
        {message.files && message.files.length > 0 && (
          <div className="mt-2">
            <ul className="list-disc list-inside">
              {message.files.map(([filename, id], index) => (
                <li key={index} className={`text-sm ${message.role === 'user' ? 'text-text-normal' : 'text-text-non_user_text_bubble'}`}>
                  <a href={formatUrl(window.apiUrls.api_conversation_file, {convo_file_id: id})} target="_blank" rel="noopener noreferrer" className="hover:underline">
                  {filename}
                  </a>
                </li>
              ))}
            </ul>
          </div>
        )}
        <p className="text-[11px] mt-1 text-text-low_contrast">
          {displayTime}
        </p>
        </div>
      </div>
    </div>
  );
};
