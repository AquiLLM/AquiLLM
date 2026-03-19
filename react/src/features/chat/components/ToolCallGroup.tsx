import React, { useState } from 'react';
import { Collapsible, ToolResult } from '../../../shared/components';
import type { Message } from '../types';

interface ToolCallGroupProps {
  toolCalls: Message[];
}

export const ToolCallGroup: React.FC<ToolCallGroupProps> = ({ toolCalls }) => {
  const [expanded, setExpanded] = useState(false);

  const toolNames = toolCalls
    .filter(m => m.role === 'assistant' && m.tool_call_name)
    .map(m => m.tool_call_name!);

  const summary = toolNames.length === 1
    ? `Tool call: ${toolNames[0]}`
    : `${toolNames.length} tool calls: ${toolNames.join(', ')}`;

  return (
    <div className={`ml-0 -mt-1 ${expanded ? '' : '-mb-2'}`}>
      <div
        className={`text-[11px] text-text-low_contrast select-none transition-all duration-200 ${expanded ? '' : 'max-h-[14px] overflow-hidden'}`}
      >
        <span className="cursor-pointer hover:text-text-normal" onClick={() => setExpanded(!expanded)}>
          {expanded ? 'v' : '>'} {summary}
        </span>
        {expanded && (
          <div className="mt-1 pl-2 border-l border-border-mid_contrast space-y-0.5">
            {toolCalls.map((msg, i) => (
              <div key={i} className="text-xs text-text-low_contrast">
                {msg.role === 'assistant' && msg.tool_call_input && i > 0 && (
                  <hr className="border-border-mid_contrast my-1 w-1/3" />
                )}
                {msg.role === 'assistant' && msg.tool_call_input && (
                  <div>
                    <span className="font-semibold">{msg.tool_call_name}</span>
                    <Collapsible
                      summary="Arguments"
                      summaryTextColor="text-text-low_contrast"
                      content={
                        <pre className="whitespace-pre-wrap break-words text-text-low_contrast text-xs max-h-[450px] overflow-y-auto border border-border-mid_contrast rounded-[8px] p-2">
                          {JSON.stringify(msg.tool_call_input, null, 2)}
                        </pre>
                      }
                    />
                  </div>
                )}
                {msg.role === 'tool' && (
                  <div>
                    <Collapsible
                      summary={'exception' in (msg.result_dict || {}) ? 'Exception' : 'Output'}
                      summaryTextColor="text-text-low_contrast"
                      isOpen={msg.for_whom === 'user'}
                      content={
                        <div className="text-text-low_contrast text-xs max-h-[450px] overflow-y-auto border border-border-mid_contrast rounded-[8px] p-2">
                          <ToolResult result={'exception' in (msg.result_dict || {}) ?
                            msg.result_dict?.exception :
                            msg.result_dict?.result}
                          />
                        </div>
                      }
                    />
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
