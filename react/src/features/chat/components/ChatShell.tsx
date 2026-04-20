import React from 'react';
import Chat from './Chat';

const ChatShell: React.FC<{ convoId: string; contextLimit?: number }> = (props) => (
  <div className="h-full flex flex-col">
    <Chat {...props} />
  </div>
);

export default ChatShell;
