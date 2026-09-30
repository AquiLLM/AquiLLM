import React from 'react';
import { Folder, Send, Sparkles } from 'lucide-react';
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
  sendDisabled: boolean;
  connectionMessage: string | null;
  onOpenCollections: () => void;
  collectionsDisabled: boolean;
  collectionsLoading?: boolean;
  selectedCount: number;
  selectedSkillCount: number;
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
  sendDisabled,
  connectionMessage,
  onOpenCollections,
  collectionsDisabled,
  collectionsLoading = false,
  selectedCount,
  selectedSkillCount,
}) => (
  <div className="sticky bottom-0 w-full bg-scheme-shade_2 border-t border-border-mid_contrast mt-[8px]">
    <div className="w-[98%] md:w-[96%] lg:w-[94%] xl:w-[92%] 2xl:max-w-[1800px] mx-auto mb-[8px] mt-[8px]">
      <div className="flex flex-wrap items-center justify-center w-full gap-2 sm:flex-nowrap sm:gap-3">
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

        <div className="relative order-first flex min-h-[56px] w-full min-w-0 flex-col justify-start gap-[8px] rounded-[10px] border border-border-mid_contrast bg-scheme-shade_2 px-4 py-[6px] transition-colors duration-200 has-[:focus]:border-transparent has-[:focus]:bg-scheme-shade_4 sm:order-none">
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
                  if (!sendDisabled) onSend();
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
              disabled={sendDisabled}
            >
              <Send size={16} className="text-text-normal" />
            </button>
          </div>
        </div>

        <div className="">
          <button
            onClick={onOpenCollections}
            disabled={collectionsDisabled}
            aria-busy={collectionsLoading}
            title={collectionsLoading ? 'Loading collections…' : undefined}
            aria-label={`Collections and skills: ${selectedCount} ${selectedCount === 1 ? 'collection' : 'collections'}, ${selectedSkillCount} ${selectedSkillCount === 1 ? 'skill' : 'skills'}`}
            className="flex h-[56px] w-[max-content] cursor-pointer flex-col items-start justify-center gap-1 rounded-[10px] border border-border-high_contrast bg-scheme-shade_4 px-3 py-0 text-xs text-text-normal transition-colors duration-200 hover:border-border-higher_contrast hover:bg-scheme-shade_5 disabled:cursor-not-allowed"
            type="button"
          >
            <span className="flex items-center gap-1.5"><Folder size={13} aria-hidden="true" />{selectedCount} {selectedCount === 1 ? 'collection' : 'collections'}</span>
            <span className="flex items-center gap-1.5"><Sparkles size={13} aria-hidden="true" />{selectedSkillCount} {selectedSkillCount === 1 ? 'skill' : 'skills'}</span>
          </button>
        </div>
      </div>
      <div className="mt-2 grid text-center text-xs text-text-low_contrast">
        <p
          className={`col-start-1 row-start-1 ${connectionMessage ? 'invisible' : ''}`}
          aria-hidden={connectionMessage ? true : undefined}
        >
          Search commands: <code>/search [document] question</code> or <code>/collection question</code>
        </p>
        {connectionMessage && (
          <p className="col-start-1 row-start-1" role="status">{connectionMessage}</p>
        )}
      </div>
    </div>
  </div>
);

export default ChatInputDock;
