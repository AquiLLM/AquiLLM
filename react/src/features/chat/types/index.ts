export interface Message {
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
  files?: [string, number][];
}

export interface Collection {
  id: string | number;
  name: string;
  parent?: string | number | null;
}

export interface Conversation {
  messages: Message[];
  usage?: number;
}

export interface ConversationDelta {
  messages: Message[];
  usage?: number;
}

export interface StreamDelta {
  message_uuid: string;
  role: 'assistant';
  content?: string;
  done?: boolean;
  usage?: number;
}

export interface WebSocketMessage {
  exception?: string;
  debug_html?: string;
  conversation?: Conversation;
  delta?: ConversationDelta;
  stream?: StreamDelta;
}

export interface ChatProps {
  convoId: string;
  contextLimit?: number;
}

export interface MessageGroup {
  main: Message;
  toolCalls: Message[];
}
