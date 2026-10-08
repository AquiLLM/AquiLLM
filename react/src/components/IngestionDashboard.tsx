import React, { useEffect, useRef, useState } from 'react';
import { PDFIngestionMonitorProps, IngestionDashboardProps } from '../types';
import PDFIngestionMonitor from './PDFIngestionMonitor';



const IngestionDashboard: React.FC<IngestionDashboardProps> = ({ wsUrl, onNewDocument }) => {
  const [monitors, setMonitors] = useState<PDFIngestionMonitorProps[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const seenDocuments = useRef(new Set<string>());
  const notifyNewDocument = useRef(onNewDocument);
  notifyNewDocument.current = onNewDocument;
  const [retryGeneration, setRetryGeneration] = useState(0);

  useEffect(() => {
    if (!wsUrl) {
      setError('WebSocket URL not provided');
      setLoading(false);
      return;
    }

    let active = true;
    let attempts = 0;
    let socket: WebSocket | null = null;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let detach = () => {};
    const handleMessage = (event: MessageEvent) => {
      try {
        const message = JSON.parse(event.data);
        if (message.type === 'document.ingestion.start') {
          const newMonitor: PDFIngestionMonitorProps = {
            documentId: String(message.documentId || ""),
            documentName: String(message.documentName || "Untitled"),
            modality: message.modality ? String(message.modality) : undefined,
            rawMediaSaved: typeof message.rawMediaSaved === "boolean" ? message.rawMediaSaved : undefined,
            textExtracted: typeof message.textExtracted === "boolean" ? message.textExtracted : undefined,
            provider: message.provider ? String(message.provider) : undefined,
            providerModel: message.providerModel ? String(message.providerModel) : undefined,
          };
          if (!newMonitor.documentId) return;
          const isNew = !seenDocuments.current.has(newMonitor.documentId);
          seenDocuments.current.add(newMonitor.documentId);
          setMonitors((prevMonitors) => isNew ? [...prevMonitors, newMonitor] :
            prevMonitors.map(monitor => monitor.documentId === newMonitor.documentId ? newMonitor : monitor));
          setLoading(false);
          if (isNew) notifyNewDocument.current?.();
        }
      } catch (err: any) {
        setError(err.message || 'An error occurred while processing the message.');
      }
    };

    const connect = () => {
      if (!active) return;
      attempts++;
      const fail = () => {
        clearTimeout(timer);
        detach();
        socket?.close();
        socket = null;
        setLoading(false);
        if (attempts >= 5) {
          setError('Could not reconnect to the ingestion monitor.');
          return;
        }
        setError('Reconnecting to the ingestion monitor…');
        timer = setTimeout(connect, 2000);
      };
      try {
        const next = new WebSocket(wsUrl);
        socket = next;
        const isCurrent = () => active && socket === next;
        const opened = () => {
          if (!isCurrent()) return;
          clearTimeout(timer);
          setLoading(false);
          setError(null);
          // Opening alone does not replenish the budget: flapping stays bounded.
        };
        const message = (event: MessageEvent) => {
          if (!isCurrent()) return;
          opened();
          handleMessage(event);
        };
        const closed = () => { if (isCurrent()) fail(); };
        next.addEventListener('open', opened);
        next.addEventListener('message', message);
        next.addEventListener('error', closed);
        next.addEventListener('close', closed);
        detach = () => {
          next.removeEventListener('open', opened);
          next.removeEventListener('message', message);
          next.removeEventListener('error', closed);
          next.removeEventListener('close', closed);
        };
        timer = setTimeout(() => { if (isCurrent()) fail(); }, 5000);
      } catch {
        fail();
      }
    };
    connect();
    return () => {
      active = false;
      clearTimeout(timer);
      detach();
      socket?.close();
    };
  }, [wsUrl, retryGeneration]);

  if (loading) {
    return <div></div>;
  }

  return (
    <div className="space-y-6">
      {error && <div role="alert" className="text-red-600">{error}
        {error.startsWith('Could not reconnect') && <button type="button" className="ml-2 underline" onClick={() => {
          setError(null);
          setRetryGeneration(previous => previous + 1);
        }}>Retry monitor</button>}
      </div>}
      {monitors.length === 0 && <div>No documents being ingested</div>}
      {[...monitors].reverse().map((monitor) => (
        <PDFIngestionMonitor key={monitor.documentId} {...monitor} />
      ))}
    </div>
  );
};

export default IngestionDashboard;
