import React, { useEffect, useState, useRef } from 'react';
import { Terminal, X } from 'lucide-react';
import { ZoteroSyncMessage } from '../types';

const STORAGE_KEY = 'zotero_sync_messages';
const DAY_IN_MS = 24 * 60 * 60 * 1000;

// Helper to load messages from localStorage, filtering out expired ones
const loadMessages = (): ZoteroSyncMessage[] => {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (!stored) return [];

    const messages: ZoteroSyncMessage[] = JSON.parse(stored);
    const now = Date.now();
    const filtered = messages.filter(m => now - m.timestamp < DAY_IN_MS);

    // Save filtered list back if we removed any
    if (filtered.length !== messages.length) {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(filtered));
    }

    return filtered;
  } catch (e) {
    console.error('Failed to load Zotero sync messages from localStorage:', e);
    return [];
  }
};

// Helper to save messages to localStorage
const saveMessages = (messages: ZoteroSyncMessage[]) => {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(messages));
};

// Exported function to add a message - always uses frontend timestamp for consistency
export const addZoteroSyncMessage = (
  message: string,
  type: 'info' | 'error' | 'success' = 'info'
) => {
  const now = Date.now();
  const messages = loadMessages();
  const newMessage: ZoteroSyncMessage = {
    id: `${now}-${Math.random().toString(36).substr(2, 9)}`,
    timestamp: now,  // Always use frontend time to avoid timezone/format issues
    message,
    type,
  };
  messages.push(newMessage);
  saveMessages(messages);

  // Dispatch custom event so component can react
  window.dispatchEvent(new CustomEvent('zotero-sync-message'));
};

// Helper to format relative time
const formatRelativeTime = (timestamp: number): string => {
  const diff = Date.now() - timestamp;
  const minutes = Math.floor(diff / 60000);
  const hours = Math.floor(diff / 3600000);

  if (minutes < 1) return 'just now';
  if (minutes < 60) return `${minutes}m ago`;
  if (hours < 24) return `${hours}h ago`;
  return 'yesterday';
};

const ZoteroSyncStatus: React.FC = () => {
  const [messages, setMessages] = useState<ZoteroSyncMessage[]>(() => loadMessages());
  const [showModal, setShowModal] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WebSocket | null>(null);

  // Listen for new messages (from WebSocket or other sources)
  useEffect(() => {
    const handleNewMessage = () => {
      setMessages(loadMessages());
    };

    window.addEventListener('zotero-sync-message', handleNewMessage);
    return () => window.removeEventListener('zotero-sync-message', handleNewMessage);
  }, []);

  // Connect to WebSocket for real-time updates
  useEffect(() => {
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProtocol}//${window.location.host}/ws/zotero_sync/`;

    const connect = () => {
      wsRef.current = new WebSocket(wsUrl);

      wsRef.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.message) {
            addZoteroSyncMessage(data.message, data.type || 'info');
          }
        } catch (e) {
          console.error('Failed to parse Zotero sync WebSocket message:', e);
        }
      };

      wsRef.current.onclose = () => {
        // Reconnect after 5 seconds if connection is lost
        setTimeout(connect, 5000);
      };

      wsRef.current.onerror = (error) => {
        console.error('Zotero sync WebSocket error:', error);
      };
    };

    connect();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  // Auto-scroll to bottom when modal is open and messages change
  useEffect(() => {
    if (showModal) {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, showModal]);

  return (
    <div className={`w-full max-w-3xl mx-auto p-3 ${messages.length === 0 ? 'hidden' : ''}`}>
      {/* Header row */}
      <div className="flex items-center gap-2">
        {(() => {
          const hasError = messages.length > 0 && messages[messages.length - 1].type === 'error';
          return (
            <button
              onClick={() => setShowModal(true)}
              className={`flex items-center gap-2 ${hasError ? 'text-red-400' : 'hover:text-accent'}`}
            >
              <Terminal size={18} />
              <span className="font-medium">Zotero Sync</span>
              {hasError && <span className="font-medium">Error</span>}
            </button>
          );
        })()}
      </div>

      {/* Modal for Log */}
      {showModal && (
        <div className="fixed inset-0 flex items-center justify-center z-60">
          {/* Overlay */}
          <div
            className="fixed inset-0 bg-black opacity-50"
            onClick={() => setShowModal(false)}
          ></div>
          {/* Modal Content */}
          <div className="bg-scheme-shade_3 text-text-normal p-4 rounded-lg shadow-lg z-10 w-11/12 max-w-3xl h-[40vh] overflow-y-auto">
            <div className="flex justify-between mb-4">
              <div className="flex items-center font-mono">
                <Terminal size={18} className="mr-2" />
                Zotero Sync Messages
              </div>
              <button
                onClick={() => setShowModal(false)}
                className="text-text-lower_contrast hover:text-text-normal"
              >
                <X size={20} />
              </button>
            </div>
            <div className="font-mono text-sm space-y-2">
              {messages.map((msg) => (
                <div
                  key={msg.id}
                  className={`${
                    msg.type === 'error'
                      ? 'text-red-400'
                      : msg.type === 'success'
                      ? 'text-green-400'
                      : 'text-text-low_contrast'
                  }`}
                >
                  <span className="text-text-lower_contrast text-xs mr-2">
                    [{formatRelativeTime(msg.timestamp)}]
                  </span>
                  {msg.message}
                </div>
              ))}
              <div ref={messagesEndRef} />
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ZoteroSyncStatus;
