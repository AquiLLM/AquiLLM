import React, { useState, useEffect, useRef } from 'react';
import { Send } from 'lucide-react';
import { CircularProgressbar } from 'react-circular-progressbar';
import ReactMarkdown from 'react-markdown';
import formatUrl from '../utils/formatUrl';

// Define TypeScript interfaces
interface Message {
  role: 'user' | 'assistant' | 'tool';
  content: string;
  message_uuid?: string;
  rating?: number;
  feedback_text?: string;
  tool_call_name?: string;
  tool_call_input?: any;
  tool_name?: string;
  result_dict?: any;
  for_whom?: 'user' | 'assistant';
  usage?: number;
  files? : [string, number][];
}

interface Collection {
  id: string;
  name: string;
}

interface Conversation {
  messages: Message[];
  usage?: number;
}

interface WebSocketMessage {
  exception?: string;
  conversation?: Conversation;
  stream?: {
    message_uuid: string;
    delta: string;
  };
}

interface ChatProps {
  convoId: string;
}

const Chat: React.FC<ChatProps> = ({ convoId }) => {
  const [conversation, setConversation] = useState<Conversation>({ messages: [] });
  const [inputDisabled, setInputDisabled] = useState(true);
  const [messageInput, setMessageInput] = useState('');
  const [exception, setException] = useState('');
  const [collections, setCollections] = useState<Collection[]>([]);
  const [selectedCollections, setSelectedCollections] = useState<Set<string>>(new Set());
  const [searchTerm, setSearchTerm] = useState('');
  const [connectionAttempts, setConnectionAttempts] = useState(0);
  const [showCollections, setShowCollections] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const conversationEndRef = useRef<HTMLDivElement>(null);
  const messageContainerRef = useRef<HTMLDivElement>(null);
  
  const MAX_RECONNECTION_ATTEMPTS = 5;
  const CONNECTION_TIMEOUT = 5000;
  const MAX_USAGE = 200000;

  

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    if (conversationEndRef.current) {
      conversationEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [conversation]);

  // Fetch collections on component mount
  useEffect(() => {
    fetchCollections();
  }, []);

  // WebSocket connection handling
  useEffect(() => {
    initWebSocket();
    
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [connectionAttempts]);

  const fetchCollections = async () => {
    try {
      const response = await fetch("/api/collections/");
      if (!response.ok) {
        throw new Error(`HTTP error: ${response.status}`);
      }
      const data = await response.json();
      setCollections(data.collections);
      
      // Select all collections by default
      const allCollectionIds = new Set<string>(data.collections.map((c: Collection) => c.id));
      setSelectedCollections(allCollectionIds);
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
      
      // Set connection timeout
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
        setException('');
      };
      
      ws.onmessage = (event) => {
        try {
          const data: WebSocketMessage = JSON.parse(event.data);

          if (data.stream) {
            const { message_uuid, delta } = data.stream;
            if (delta) {
              setConversation((prev) => {
                const existingIndex = prev.messages.findIndex(
                  (msg) => msg.role === 'assistant' && msg.message_uuid === message_uuid
                );
                if (existingIndex >= 0) {
                  const updatedMessages = [...prev.messages];
                  const existing = updatedMessages[existingIndex];
                  updatedMessages[existingIndex] = {
                    ...existing,
                    content: `${existing.content || ''}${delta}`,
                  };
                  return { ...prev, messages: updatedMessages };
                }
                const streamedAssistant: Message = {
                  role: 'assistant',
                  content: delta,
                  message_uuid,
                };
                return { ...prev, messages: [...prev.messages, streamedAssistant] };
              });
            }
            return;
          }
          
          if (data.exception) {
            console.error('Server error:', data.exception);
            setException(data.exception);
            setInputDisabled(false);
            return;
          }
          
          setException('');
          
          if (data.conversation) {
            const updatedConversation = data.conversation;

            // Find the most recent assistant message
            const lastAssistantMessage = updatedConversation.messages
              .slice()
              .reverse()
              .find((msg) => msg.role === 'assistant');

            // Update conversation.usage if the last assistant message has usage
            if (lastAssistantMessage && lastAssistantMessage.usage !== undefined) {
              updatedConversation.usage = lastAssistantMessage.usage;
            }

            setConversation(updatedConversation);

            if (updatedConversation.messages.length) {
              const lastMessage = updatedConversation.messages[updatedConversation.messages.length - 1];
              const shouldEnableInput =
                (lastMessage.role === 'assistant' && !lastMessage.tool_call_input) ||
                (lastMessage.role === 'tool' && lastMessage.for_whom === 'user');

              setInputDisabled(!shouldEnableInput);
            }
          }
        } catch (error) {
          setException(`Error processing message: ${error instanceof Error ? error.message : 'Unknown error'}`);
        }
      };
      
      ws.onclose = (event) => {
        clearTimeout(timeoutId);
        console.log('Disconnected from chat server', event.code, event.reason);
        setInputDisabled(true);
        
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
  };

  const rateMessage = (uuid: string | undefined, rating: number) => {
    if (!uuid || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    
    const payload = {
      action: 'rate',
      uuid: uuid,
      rating: rating
    };
    
    wsRef.current.send(JSON.stringify(payload));
    
    // Update the rating in the local state
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
    
    // Update the feedback_text in the local state
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

  const getColor = (percent: number): string => {
    // Convert percentage to a position in the color spectrum
    // 100% -> green (hue: 120)
    // 50% -> yellow (hue: 60)
    // 0% -> red (hue: 0)
    percent = 100 - percent; // Reverse the percentage
    const hue: number  = (percent * 1.2); // Multiplier 1.2 makes green start around 85%
    return `hsl(${hue}, 75%, 35%)`;
  };

  return (
    <div className="flex flex-col h-full">
      {/* Exception Alert */}
      {exception && (
        <div className="sticky top-0 z-50 font-mono text-text-normal p-4 mb-4 bg-red-dark rounded">
          {exception}
        </div>
      )}

      {/* Conversation Container - Takes all available space and scrolls */}
      <div
        ref={messageContainerRef}
        className="flex-grow overflow-y-auto w-full px-[32px] pt-[32px]"
      >
        <div className="w-[98%] md:w-[96%] lg:w-[94%] xl:w-[92%] 2xl:max-w-[1800px] mx-auto gap-[32px] flex flex-col">
          {conversation.messages.map((message, index) => (
            <MessageBubble 
              key={`msg-${index}`} 
              message={message} 
              onRate={rateMessage}
              onFeedback={feedbackMessage}
            />
          ))}
          
          {shouldShowSpinner(conversation.messages) && (
            <div className="flex justify-center my-4">
              <div className="animate-spin rounded-full h-8 w-8 border-4 border-accent border-t-transparent"></div>
            </div>
          )}
          <div ref={conversationEndRef} />
        </div>
      </div>

      {/* Fixed bottom section for input */}
      <div className="sticky bottom-0 w-full bg-scheme-shade_2 border-t border-border-mid_contrast mt-[16px]">
        {/* Message Input Section */}
        <div className="w-[98%] md:w-[96%] lg:w-[94%] xl:w-[92%] 2xl:max-w-[1800px] mx-auto mb-[32px] mt-[16px]">
          <div className="flex items-center justify-center w-full gap-[32px]">
            {/* Token Usage */}
            <div className="flex flex-col items-center gap-1">
              <div style={{ width: '35px', height: '35px' }}>
                <CircularProgressbar value={conversation.usage || 0}
                                     maxValue={MAX_USAGE}
                                     strokeWidth={50}
                                     styles= {{
                                      path: {
                                        stroke: getColor(conversation.usage ? conversation.usage / MAX_USAGE : 0),
                                      }
                                     }}
                                     text="" />
              </div>
              <div className="text-center text-xs text-text-low_contrast whitespace-nowrap">
                {conversation.usage ? `${conversation.usage.toLocaleString()} / ${MAX_USAGE.toLocaleString()}` : `0 / ${MAX_USAGE.toLocaleString()}`}
              </div>
            </div>

            <div className="flex justify-start flex-col gap-[8px] bg-scheme-shade_3 py-2 px-4 rounded-[32px] w-full">
              <div className="flex flex-grow items-center w-full">
                <input
                  type="text"
                  id="message-input"
                  className="px-2 py-2 mr-[16px] flex-grow w-full rounded-lg bg-scheme-shade_3 focus:border-0 disabled:cursor-not-allowed placeholder:text-text-lower_contrast text-text-normal element-border"
                  placeholder="Type your message here..."
                  value={messageInput}
                  onChange={(e) => setMessageInput(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && sendMessage()}
                  disabled={inputDisabled}
                  autoComplete="off"
                />
                <button
                  onClick={sendMessage}
                  className="p-4 bg-accent text-text-normal rounded-[24px] disabled:cursor-not-allowed mr-[-8px]"
                  title="Send Message"
                  disabled={inputDisabled}
                >
                  <Send size={16} className="text-text-normal"/>
                </button>
              </div>
            </div>

            {/* Collections Section */}
            <div className="">
              <button
                onClick={() => setShowCollections(true)}
                className="cursor-pointer w-[max-content] px-[16px] py-[2px] text-text-normal bg-accent h-[36px] rounded-[18px] flex items-center"
              >
                <span className="text-text-normal">Select Collections</span>
                <span className="ml-2 text-sm text-text-normal">
                  {selectedCollections.size ? `(${selectedCollections.size} selected)` : ''}
                </span>
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Collections Modal */}
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
                <div key={collection.id} className="flex items-center p-2 hover:bg-scheme-shade_4 rounded">
                  <input
                    type="checkbox"
                    id={`collection-${collection.id}`}
                    checked={selectedCollections.has(collection.id)}
                    onChange={() => handleCollectionToggle(collection.id)}
                    className="h-4 w-4 text-accent-light rounded border-gray-300 focus:ring-blue-500"
                  />
                  <label htmlFor={`collection-${collection.id}`} className="ml-2 text-sm text-accent-light cursor-pointer">
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

// Helper function to determine if spinner should be shown
const shouldShowSpinner = (messages: Message[]) => {
  if (messages.length === 0) return false;
  const lastMessage = messages[messages.length - 1];
  return (
    lastMessage.role === 'user' ||
    (lastMessage.role === 'assistant' && lastMessage.tool_call_input) ||
    (lastMessage.role === 'tool' && lastMessage.for_whom === 'assistant')
  );
};

// MessageBubble component
const MessageBubble: React.FC<{ 
  message: Message, 
  onRate: (uuid: string | undefined, rating: number) => void,
  onFeedback: (uuid: string | undefined, feedback_text: string) => void
}> = ({ message, onRate, onFeedback }) => {
  const getMessageClasses = () => {
    let classes = "w-4/5 p-2.5 rounded-[12px] shadow-md whitespace-pre-wrap break-words element-border";
    
    if (message.role === 'user') {
      return `${classes} user-message self-end bg-scheme-shade_3 text-text-normal`;
    } else if (message.role === 'assistant') {
      return `${classes} assistant-message chat-bubble-left-border-assistant text-text-non_user_text_bubble`;
    } else if (message.role === 'tool') {
      return `${classes} border border-1 border-secondary_accent-light chat-bubble-left-border-tool text-text-non_user_text_bubble`;
    }
    
    return classes;
  };

  return (
    <div className={`flex items-center ${message.role === 'user' ? 'justify-end' : 'justify-start'} gap-2`}>
      {/* Avatar for assistant/tool */}
      {message.role !== 'user' && (
        <div className="mr-2">
          <AquillmLogo role={message.role} />
        </div>
      )}
      
      <div className={getMessageClasses()}>
        {/* Message content */}
        {message.role === 'user' && (
          <p className="whitespace-pre-wrap break-words">{message.content}</p>
        )}
        {message.role === 'assistant' && !message.tool_call_input && (
          <div className="prose max-w-none compact-prose leading-tight [&>*]:my-1 [&_ol>li>p]:inline">
            <ReactMarkdown>
              {message.content}
            </ReactMarkdown>
          </div>
        )}
        
        {/* Tool call details for assistant */}
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
        
        {/* Tool output */}
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
        
        {/* Rating buttons */}
        {(message.role === 'assistant' || message.role === 'tool') && (
          <RatingButtons 
            rating={message.rating}
            feedback_text={message.feedback_text}
            onRate={(rating) => onRate(message.message_uuid, rating)}
            onFeedback={(text) => onFeedback(message.message_uuid, text)}
          />
        )}
        {/* File attachments */}
        {message.files && message.files.length > 0 && (
          console.log(message.files),
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
        {/* Timestamp */}
        <p className={`text-xs mt-1.5 ${message.role === 'user' ? 'text-right' : ''}`}>
          {new Date().toLocaleString()}
        </p>
      </div>
      
      {/* Avatar for user */}
      {message.role === 'user' && (
        <div className="ml-2">
          <UserLogo />
        </div>
      )}
    </div>
  );
};

const Collapsible: React.FC<{
  summary: string,
  summaryTextColor: string,
  content: React.ReactNode,
  isOpen?: boolean
}> = ({ summary, summaryTextColor, content, isOpen = false }) => {
  return (
    <details className="mt-1.5" open={isOpen}>
      <summary className={`cursor-pointer ${summaryTextColor}`}>
        {summary}
      </summary>
      <div className="mt-1.5 pl-2.5 border-l-2 border-border-mid_contrast">
        {content}
      </div>
    </details>
  );
};

const ToolResult: React.FC<{ result: any, level?: number }> = ({ result, level = 0 }) => {
  if (typeof result === 'object' && result !== null) {
    if (Array.isArray(result)) {
      return (
        <details open={level < 1} className="mt-1">
          <summary className="cursor-pointer font-mono hover:text-accent">Array</summary>
          <div className="pl-4 border-l-2 border-border-mid_contrast mt-1 text-text-non_user_text_bubble">
            {result.map((item, index) => (
              <ToolResult key={index} result={item} level={level + 1} />
            ))}
          </div>
        </details>
      );
    } else {
      return (
        <div>
          {Object.entries(result).map(([key, value], index) =>
            typeof value === 'object' && value !== null ? (
              <details key={index} open={level < 1} className="mt-1">
                <summary className="cursor-pointer font-mono hover:text-accent">{key}</summary>
                <div className="pl-4 border-l-2 border-border-mid_contrast mt-1 text-text-non_user_text_bubble">
                  <ToolResult result={value} level={level + 1} />
                </div>
              </details>
            ) : (
              <div key={index} className="mt-1">
                <span className="font-mono">{key}: </span>
                <ToolValue value={value as string | number | boolean} />
              </div>
            )
          )}
        </div>
      );
    }
  }
  return <ToolValue value={result} />;
};

const ToolValue: React.FC<{ value: string | number | boolean }> = ({ value }) => {
  if (typeof value === 'string') {
    return <span className="font-mono text-text-non_user_text_bubble">{`"${value}"`}</span>;
  }
  return <span className="font-mono text-secondary_accent-light">{String(value)}</span>;
};

const RatingButtons: React.FC<{ 
  rating?: number, 
  feedback_text?: string,
  onRate: (rating: number) => void,
  onFeedback: (text: string) => void
}> = ({ rating, feedback_text, onRate, onFeedback }) => {
  const emojis = ['😞', '😕', '😐', '🙂', '😊'];
  const [localFeedback, setLocalFeedback] = useState(feedback_text || '');
  const [submitted, setSubmitted] = useState(!!feedback_text);

  const handleSubmit = () => {
    if (localFeedback.trim()) {
      onFeedback(localFeedback.trim());
      setSubmitted(true);
    }
  };
  
  return (
    <div className="flex flex-col gap-2 mt-2">
      <div className="flex items-center gap-1">
        <span className="text-sm mr-2">Rate:</span>
        {emojis.map((emoji, i) => (
          <button
            key={i}
            className={`p-1 rounded hover:bg-scheme-shade_6 ${rating === i + 1 ? 'bg-scheme-shade_6' : ''} transition-colors`}
            onClick={() => onRate(i + 1)}
          >
            {emoji}
          </button>
        ))}
      </div>
      <div className="flex gap-2 items-center">
        <input
          type="text"
          value={localFeedback}
          onChange={(e) => {
            setLocalFeedback(e.target.value);
            setSubmitted(false);
          }}
          placeholder="Please enter feedback here..."
          className="flex-grow px-2 py-1 text-sm rounded bg-scheme-shade_3 text-text-normal border border-border-mid_contrast"
          onKeyDown={(e) => e.key === 'Enter' && handleSubmit()}
        />
        {submitted ? (
          <span className="text-sm text-green-500 whitespace-nowrap">Submitted!</span>
        ) : (
          <button
            onClick={handleSubmit}
            className="px-3 py-1 text-sm bg-accent rounded text-text-normal whitespace-nowrap"
          >
            Submit
          </button>
        )}
      </div>
    </div>
  );
};

// Logo Components
const AquillmLogo: React.FC<{ role: string }> = ({ role }) => {
  const colorClass = role === 'assistant' ? 'text-accent' : 'text-secondary_accent';
  
  return (
    <svg 
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 32 26"
      className={`w-6 h-6 fill-current ${colorClass}`}
    >
      <path d="M24.8673 11.8191V11.4044L24.8558 10.9897L24.8329 10.598L24.81 10.2063L24.7299 9.44603L24.6383 8.73182L24.5009 8.04063L24.3521 7.38403L24.1803 6.76195L23.9743 6.17444L23.7567 5.62148L23.5048 5.09158L23.253 4.59624L22.9667 4.12393L22.6691 3.68617L22.3599 3.27148L22.0279 2.89131L21.6845 2.5342L21.341 2.20015L20.9746 1.90063L20.5968 1.62417L20.2075 1.37073L19.8183 1.14032L19.429 0.932974L19.0168 0.748654L18.6161 0.587361L18.2039 0.437599L17.7918 0.32243L17.3796 0.218757L16.9674 0.13811L16.5553 0.0804906L16.1431 0.034437L15.7424 0.011375L15.3417 -0.000120787L15.2845 1.26702C20.9288 1.26702 20.9288 8.78941 20.9288 13.2591C20.9288 15.3787 20.9288 18.7079 21.0433 19.6871C16.0286 24.0991 11.6322 24.7327 9.35383 24.7327C5.93057 24.7327 4.22469 22.1523 4.22469 18.4775C4.22469 15.6667 5.7016 9.46909 7.53344 6.54308C10.2125 2.3499 13.2923 1.26702 15.2845 1.26702L15.3417 -0.000120787C7.53345 -0.000120787 0 8.21344 0 16.3003C0 21.6339 3.42323 25.9999 9.23934 25.9999C12.8343 25.9999 16.9445 24.6751 21.2723 21.1731C22.0165 24.2143 23.8941 25.9999 26.4701 25.9999C29.4927 25.9999 31.2558 22.8435 31.2558 21.9219C31.2558 21.5187 30.9123 21.3459 30.5689 21.3459C30.1682 21.3459 29.9964 21.5187 29.8247 21.9219C28.8057 24.7327 26.7563 24.7327 26.6304 24.7327C24.8673 24.7327 24.8673 20.2631 24.8673 18.8807C24.8673 17.6711 24.8673 17.5675 25.4397 16.8763C30.7979 10.1027 32 3.44428 32 3.38668C32 3.27148 31.9428 2.8107 31.3131 2.8107C30.7406 2.8107 30.7406 2.98348 30.4544 4.02025C29.4354 7.63745 27.5464 11.9919 24.8673 15.3787V11.8191Z" />
    </svg>
  );
};

const UserLogo: React.FC = () => {
  return (
    <svg 
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 26 26"
      className="w-6 h-6 fill-current text-text-slightly_less_contrast"
    >
      <path d="M16.0352 0.932185C19.1653 1.30585 21.474 2.60517 22.9611 4.83017C24.4657 7.0551 25.2179 9.62823 25.2179 12.5496C25.2179 16.1333 24.1284 19.2159 21.9495 21.7975C19.7705 24.3791 17.1679 25.6699 14.1415 25.6699C12.4641 25.6699 10.9336 25.2283 9.55025 24.3451C8.16677 23.445 7.2416 22.358 6.77475 21.0842C6.30784 19.8104 6.07438 18.0185 6.07438 15.7086V7.2759C6.07438 5.74727 5.92724 4.70272 5.63296 4.14227C5.33924 3.58178 4.7945 3.30153 3.99874 3.30153C2.8401 3.30153 2.07916 4.43948 1.7159 6.71538H0.782104C1.0068 2.85988 2.69291 0.932129 5.84044 0.932129C6.91256 0.932129 7.79452 1.20388 8.4863 1.74739C9.19531 2.29089 9.67953 2.91931 9.93895 3.63266C10.2157 4.32902 10.3541 5.61134 10.3541 7.47963V16.0398C10.3541 18.044 10.4664 19.5557 10.6911 20.5747C10.9333 21.5768 11.4608 22.4006 12.2735 23.046C13.1036 23.6914 14.0115 24.0141 14.9972 24.0141C16.6573 24.0141 17.9976 23.0715 19.0179 21.1862C20.0382 19.2839 20.5484 16.422 20.5484 12.6005C20.5484 9.18668 20.1679 6.67295 19.407 5.05934C18.646 3.42887 17.4701 2.35884 15.8791 1.84927L16.0352 0.932185Z" />
    </svg>
  );
};

export const ChatComponent: React.FC<{ convoId: string }> = ({ convoId }) => {
  return (
    <div className="h-full flex flex-col">
      <Chat convoId={convoId} />
    </div>
  );
};

export default ChatComponent;