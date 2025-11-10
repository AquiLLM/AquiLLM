import React, { useEffect, useState, useRef } from 'react';
import { Terminal, X } from 'lucide-react';
import { PDFIngestionMonitorProps, IngestionMessage } from '../types';
import formatUrl from '../utils/formatUrl';
const PDFIngestionMonitor: React.FC<PDFIngestionMonitorProps> = ({ documentName, documentId }) => {
  const [progress, setProgress] = useState<number>(0);
  const [messages, setMessages] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [showModal, setShowModal] = useState<boolean>(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const ws = useRef<WebSocket | null>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    if (!ws.current) {
      const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${wsProtocol}//${window.location.host}/ingest/monitor/${documentId}/`;
      ws.current = new WebSocket(wsUrl);

      ws.current.onopen = () => {
        console.log('Connected to WebSocket');
      };

      ws.current.onmessage = (event) => {
        const data: IngestionMessage = JSON.parse(event.data);

        if (data.progress !== undefined) {
          if (data.progress > progress) {
            setProgress(data.progress);
          }
        }
        if (data.complete) {
          setProgress(100);
        }
        if (data.messages) {
          setMessages(prev => [...prev, ...(data.messages ?? [])]);
        }

        if (data.exception) {
          setError(data.exception);
        }
      };

      ws.current.onerror = () => {
        setError('WebSocket connection error');
      };

      return () => {
        if (ws.current) {
          ws.current.close();
        }
      };
    }
  }, [documentId]);

  return (
    <div className="w-full max-w-3xl mx-auto p-3 space-y-6">
      {/* Progress Section */}
      <div className="space-y-2">
        <a
          className="font-medium text-accent underline"
          href={formatUrl(window.pageUrls.document, { doc_id: documentId })}
        >
          {documentName}
        </a>
        <div className="flex items-center gap-2">
          {/* Terminal icon that triggers the modal */}
          <button onClick={() => setShowModal(true)} className="hover:text-accent">
            <Terminal size={18} />
          </button>
          <div className="flex-grow bg-scheme-shade_7 rounded-full h-5">
            <div
              className="bg-accent h-5 rounded-full transition-all duration-300 ease-in-out"
              style={{ width: `${progress}%` }}
            ></div>
          </div>
          <span className="font-medium min-w-[3rem] text-right">{progress}%</span>
        </div>
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
            <div className="flex justify-between">
              <div className="flex font-mono">
                <Terminal size={18} className="mr-6" />
                Ingestion Events for Document "{documentName}"
              </div>

              <button
                onClick={() => setShowModal(false)}
                className="text-text-lower_contrast hover:text-text-normal text-xl"
              >
                <X />
              </button>
            </div>
            <div className="font-mono text-sm space-y-1">
              {messages.map((message, index) => (
                <div key={index} className="text-text-low_contrast">
                  {message}
                </div>
              ))}
              <div ref={messagesEndRef} />
            </div>
          </div>
        </div>
      )}

      {/* Error Display */}
      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative">
          <strong className="font-bold">Error:</strong>
          <span className="block sm:inline"> {error}</span>
        </div>
      )}
    </div>
  );
};

export default PDFIngestionMonitor;
