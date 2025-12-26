import React, { useEffect, useState } from 'react';
import { PDFIngestionMonitorProps, IngestionDashboardProps } from '../types';
import PDFIngestionMonitor from './PDFIngestionMonitor';
import ZoteroSyncStatus from './ZoteroSyncStatus';



const IngestionDashboard: React.FC<IngestionDashboardProps> = ({ wsUrl, onNewDocument }) => {
  const [monitors, setMonitors] = useState<PDFIngestionMonitorProps[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!wsUrl) {
      setError('WebSocket URL not provided');
      setLoading(false);
      return;
    }

    const socket = new WebSocket(wsUrl);

    socket.addEventListener('open', () => {
      setLoading(false);
    });

    const handleMessage = (event: MessageEvent) => {
      try {
        const message = JSON.parse(event.data);
        if (message.type === 'document.ingestion.start') {
          const newMonitor: PDFIngestionMonitorProps = JSON.parse(event.data);
          setMonitors((prevMonitors) => [...prevMonitors, newMonitor]);
          setLoading(false);
          onNewDocument(); // Notify the parent component
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
  }, [wsUrl, onNewDocument]);

  if (loading) {
    return <div></div>;
  }

  if (error) {
    return <div className="text-red-600">Error: {error}</div>;
  }

  return (
    <div className="space-y-6">
      <ZoteroSyncStatus />
      {monitors.length === 0 && <div>No documents being ingested</div>}
      {[...monitors].reverse().map((monitor) => (
        <PDFIngestionMonitor key={monitor.documentId} {...monitor} />
      ))}
    </div>
  );
};

export default IngestionDashboard;