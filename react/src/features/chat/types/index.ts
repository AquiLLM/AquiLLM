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

export type SkillOverrides = Record<string, boolean>;
export interface ContextCollection extends Collection {
  path: string;
  is_skill_pack: boolean;
}
export interface ContextSkill {
  id: string;
  name: string;
  description: string;
  instructions: string;
  collection_id: string;
  collection_name: string;
  collection_path: string;
  source_path: string;
  pack_id: string | null;
  pack_name: string | null;
  default_collection_ids: string[];
}
export interface ChatContextCatalog {
  collections: ContextCollection[];
  skills_enabled: boolean;
  skills: ContextSkill[];
}
export interface ContextSelection {
  selectedCollections: Set<string>;
  skillOverrides: SkillOverrides;
}
export interface ContextSelectionAcknowledgment {
  request_id?: string;
  selected_collections: (string | number)[];
  skill_overrides: SkillOverrides;
}

export interface Conversation {
  messages: Message[];
  usage?: number;
  selected_collections?: (string | number)[];
  skill_overrides?: SkillOverrides;
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
  context_selection?: ContextSelectionAcknowledgment;
  context_selection_error?: { request_id: string; message: string };
  exception?: string;
  fatal?: boolean;
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
