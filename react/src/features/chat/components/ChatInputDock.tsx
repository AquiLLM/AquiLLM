import React from 'react';
import { Send } from 'lucide-react';
import { CircularProgressbar } from 'react-circular-progressbar';

export interface ChatInputDockProps {
  clampedUsageValue: number;
  contextLimitTokens: number;
  usageValue: number;
  usageStrokeColor: string;
  contentOverflowing: boolean;
  onDragStart: (e: React.MouseEvent) => void;
  textareaRef: React.RefObject<HTMLTextAreaElement | null>;
  emptyThread: boolean;
  messageInput: string;
  onMessageInputChange: (value: string) => void;
  onAutoResize: () => void;
  onSend: () => void;
  inputDisabled: boolean;
  onOpenCollections: () => void;
  selectedCount: number;
}

const ChatInputDock: React.FC<ChatInputDockProps> = ({
  clampedUsageValue,
  contextLimitTokens,
  usageValue,
  usageStrokeColor,
  contentOverflowing,
  onDragStart,
  textareaRef,
  emptyThread,
  messageInput,
  onMessageInputChange,
  onAutoResize,
  onSend,
  inputDisabled,
  onOpenCollections,
  selectedCount,
}) => (
  <div className="sticky bottom-0 w-full bg-scheme-shade_2 border-t border-border-mid_contrast mt-[8px]">
    <div className="w-[98%] md:w-[96%] lg:w-[94%] xl:w-[92%] 2xl:max-w-[1800px] mx-auto mb-[8px] mt-[8px]">
      <div className="flex items-center justify-center w-full gap-[12px]">
        <div className="flex h-[56px] min-w-[114px] shrink-0 flex-col items-center justify-center gap-[3px] rounded-[10px] border border-border-mid_contrast bg-scheme-shade_2 px-[8px] py-[6px]">
          <div className="h-[28px] w-[28px] rounded-full border border-border-mid_contrast bg-scheme-shade_2">
            <CircularProgressbar
              value={clampedUsageValue}
              maxValue={contextLimitTokens}
              strokeWidth={50}
              styles={{
                path: { stroke: usageStrokeColor },
                trail: { stroke: 'var(--color-border-low-contrast)' },
              }}
              text=""
            />
          </div>
          <div className="whitespace-nowrap text-center text-[11px] leading-[1.05] text-text-low_contrast">
            {`${usageValue.toLocaleString()} / ${contextLimitTokens.toLocaleString()}`}
          </div>
        </div>

        <div className="relative flex min-h-[56px] w-full flex-col justify-start gap-[8px] rounded-[10px] border border-border-mid_contrast bg-scheme-shade_2 px-4 py-[6px] transition-colors duration-200 has-[:focus]:border-transparent has-[:focus]:bg-scheme-shade_4">
          <div
            onMouseDown={contentOverflowing ? undefined : onDragStart}
            className={`absolute left-1/2 -translate-x-1/2 top-0 -translate-y-1/2 z-50 flex justify-center px-2 py-1 group ${contentOverflowing ? 'pointer-events-none' : 'cursor-ns-resize'}`}
          >
            <div
              className={`w-12 h-1 rounded-full transition-colors ${contentOverflowing ? 'bg-transparent' : 'bg-border-mid_contrast group-hover:bg-text-low_contrast'}`}
            />
          </div>
          <div className="flex flex-grow items-center w-full">
            <textarea
              id="message-input"
              ref={textareaRef}
              rows={1}
              className="px-2 py-2 mr-[16px] flex-grow w-full rounded-lg bg-transparent border-none outline-none focus:outline-none focus:ring-0 disabled:cursor-not-allowed placeholder:text-text-lower_contrast text-text-normal resize-none overflow-y-auto max-h-[450px]"
              placeholder={emptyThread ? 'How can I help you today?' : 'Reply...'}
              value={messageInput}
              onChange={(e) => {
                onMessageInputChange(e.target.value);
                onAutoResize();
              }}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  onSend();
                }
              }}
              disabled={inputDisabled}
              autoComplete="off"
            />
            <button
              onClick={onSend}
              className="mr-[-4px] flex h-[44px] w-[44px] items-center justify-center rounded-[10px] border border-border-high_contrast bg-scheme-shade_4 p-0 text-text-normal transition-colors duration-200 hover:border-border-higher_contrast hover:bg-scheme-shade_5 disabled:cursor-not-allowed"
              title="Send Message"
              type="button"
              disabled={inputDisabled}
            >
              <Send size={16} className="text-text-normal" />
            </button>
          </div>
        </div>

        <div className="">
          <button
            onClick={onOpenCollections}
            className="flex h-[56px] w-[max-content] cursor-pointer items-center rounded-[10px] border border-border-high_contrast bg-scheme-shade_4 px-[16px] py-0 text-text-normal transition-colors duration-200 hover:border-border-higher_contrast hover:bg-scheme-shade_5"
            type="button"
          >
            <span className="text-text-normal">Collections</span>
            <span className="ml-2 text-sm text-text-normal">
              {selectedCount ? `(${selectedCount} selected)` : ''}
            </span>
          </button>
        </div>
      </div>
    </div>
  </div>
);

export default ChatInputDock;
