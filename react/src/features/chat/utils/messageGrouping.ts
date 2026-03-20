import type { Message, MessageGroup } from '../types';

/**
 * Group messages so tool-call sequences tuck under the preceding assistant message
 */
export const groupMessages = (messages: Message[]): (Message | MessageGroup)[] => {
  const result: (Message | MessageGroup)[] = [];
  let i = 0;
  while (i < messages.length) {
    const msg = messages[i];
    if (msg.role === 'assistant' && msg.tool_call_input) {
      const toolCalls: Message[] = [msg];
      let j = i + 1;
      while (j < messages.length) {
        const next = messages[j];
        if ((next.role === 'assistant' && next.tool_call_input) || next.role === 'tool') {
          toolCalls.push(next);
          j++;
        } else {
          break;
        }
      }
      const prev = result[result.length - 1];
      if (prev && !('main' in prev) && (prev as Message).role === 'assistant' && !(prev as Message).tool_call_input) {
        result[result.length - 1] = { main: prev as Message, toolCalls };
      } else {
        result.push({ main: { role: 'assistant', content: '' }, toolCalls });
      }
      i = j;
    } else {
      result.push(msg);
      i++;
    }
  }
  return result;
};

/**
 * Determine if spinner should be shown based on message state
 */
export const shouldShowSpinner = (messages: Message[]): boolean => {
  if (messages.length === 0) return false;
  const lastMessage = messages[messages.length - 1];
  return (
    lastMessage.role === 'user' ||
    (lastMessage.role === 'assistant' && !!lastMessage.tool_call_input) ||
    (lastMessage.role === 'tool' && lastMessage.for_whom === 'assistant')
  );
};
