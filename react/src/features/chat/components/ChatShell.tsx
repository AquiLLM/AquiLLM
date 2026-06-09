import React from 'react';
import Chat from './Chat';
import CitationModalProvider, { CitationPanelSlot } from './CitationModalProvider';

const ChatShell: React.FC<{ convoId: string; contextLimit?: number }> = (props) => (
  <CitationModalProvider>
    <div className="h-full flex">
      <div className="flex-1 min-w-0 flex flex-col">
        <Chat {...props} />
      </div>
      <CitationPanelSlot />
    </div>
  </CitationModalProvider>
);

export default ChatShell;
