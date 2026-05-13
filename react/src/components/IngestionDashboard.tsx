import React, { useEffect, useRef, useState } from 'react';
import { PDFIngestionMonitorProps, IngestionDashboardProps } from '../types';
import PDFIngestionMonitor from './PDFIngestionMonitor';



const IngestionDashboard: React.FC<IngestionDashboardProps> = ({ wsUrl, onNewDocument }) => {
  const [monitors, setMonitors] = useState<PDFIngestionMonitorProps[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  // Keep a ref to onNewDocument so we can call the latest version without
  // listing it as a useEffect dependency (which would cause a reconnect on
  // every parent render and re-trigger the server's on-connect replay of
  // in-progress documents, producing duplicates and an infinite open/close loop).
  const onNewDocumentRef = useRef(onNewDocument);
  useEffect(() => {
    onNewDocumentRef.current = onNewDocument;
  }, [onNewDocument]);

  // Tracks document IDs we've already accounted for via the on-connect replay.
  // Replay messages (isReplay: true) silently populate this set. Live messages
  // (no isReplay) only auto-open the modal if the ID is brand new — preventing
  // the modal from popping up on every page navigation for already-in-progress docs.
  const seenDocumentIdsRef = useRef<Set<string>>(new Set());

  useEffect(() => {
    if (!wsUrl) {
      setError('WebSocket URL not provided');
      setLoading(false);
      return;
    }

    seenDocumentIdsRef.current = new Set();
    const socket = new WebSocket(wsUrl);

    const handleMessage = (event: MessageEvent) => {
      try {
        const message = JSON.parse(event.data);
        if (message.type === 'dashboard.loaded') {
          setLoading(false);
          return;
        }
        if (message.type === 'document.ingestion.start') {
          const docId = String(message.documentId || "");
          const isReplay = Boolean(message.isReplay);
          const isNewDoc = !seenDocumentIdsRef.current.has(docId);
          seenDocumentIdsRef.current.add(docId);

          const newMonitor: PDFIngestionMonitorProps = {
            documentId: docId,
            documentName: String(message.documentName || "Untitled"),
            modality: message.modality ? String(message.modality) : undefined,
            rawMediaSaved: typeof message.rawMediaSaved === "boolean" ? message.rawMediaSaved : undefined,
            textExtracted: typeof message.textExtracted === "boolean" ? message.textExtracted : undefined,
            provider: message.provider ? String(message.provider) : undefined,
            providerModel: message.providerModel ? String(message.providerModel) : undefined,
          };
          setMonitors((prevMonitors) => {
            if (prevMonitors.some((m) => m.documentId === docId)) {
              return prevMonitors;
            }
            return [...prevMonitors, newMonitor];
          });
          setLoading(false);
          // Replays populate seenDocumentIds silently (no modal).
          // Live messages open the modal only for a brand-new document ID —
          // not for one we already knew about from the on-connect replay.
          if (!isReplay && isNewDoc) {
            onNewDocumentRef.current?.();
          }
        }
      } catch (err: any) {
        setError(err.message || 'An error occurred while processing the message.');
      }
    };

    const handleError = () => {
      setError('WebSocket error occurred');
    };

    socket.addEventListener('message', handleMessage);
    socket.addEventListener('error', handleError);

    // Cleanup: remove listeners and close socket on unmount
    return () => {
      socket.removeEventListener('message', handleMessage);
      socket.removeEventListener('error', handleError);
      socket.close();
    };
  }, [wsUrl]); // wsUrl only — onNewDocument is accessed via ref above

  if (loading) {
    return <div></div>;
  }

  if (error) {
    return <div className="text-red-600">Error: {error}</div>;
  }

  return (
    <div className="space-y-6">
      {monitors.length === 0 && <div>No documents being ingested</div>}
      {[...monitors].reverse().map((monitor) => (
        <PDFIngestionMonitor key={monitor.documentId} {...monitor} />
      ))}
    </div>
  );
};

export default IngestionDashboard;
