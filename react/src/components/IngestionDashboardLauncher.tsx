import React, { useState } from 'react';
import IngestionDashboard from './IngestionDashboard';
import { IngestionDashboardLauncherProps } from '../types';
import { X } from 'lucide-react';

interface IngestionDashboardModalProps {
  wsUrl: string;
  onClose: () => void;
  onNewDocument: () => void;
}

const IngestionDashboardModal: React.FC<IngestionDashboardModalProps> = ({ wsUrl, onClose, onNewDocument }) => {
  return (
    <div className="fixed bg-scheme-shade_3 rounded-t-lg shadow-lg bottom-0 left-1/2 max-w-2xl right-0 z-50 flex items-end justify-center pointer-events-auto">
      {/* Modal container styled as a bottom sheet */}
      <div
        className="relative w-full rounded-t-lg shadow-lg flex flex-col"
        style={{ height: '33vh' }}
      >
        {/* Header */}
        <div className="flex justify-between items-center border-b px-4 py-2 flex-shrink-0">
          <h2 className="text-lg font-bold">Ingestion Dashboard</h2>
          <button onClick={onClose} className="text-text-lower_contrast hover:text-text-normal">
            <X />
          </button>
        </div>
        {/* Content */}
        <div className="p-4 overflow-y-auto flex-1 min-h-0">
          <IngestionDashboard wsUrl={wsUrl} onNewDocument={onNewDocument} />
        </div>
      </div>
    </div>
  );
};

const IngestionDashboardLauncher: React.FC<IngestionDashboardLauncherProps> = ({ wsUrl }) => {
  const [modalOpen, setModalOpen] = useState(false);

  return (
    <>
      {/* Clickable div that toggles the modal's visibility */}
      <div onClick={() => setModalOpen(!modalOpen)} className="my-[5px] pl-[8px] py-[3px] font-sans text-lg w-full hover:bg-scheme-shade_5 rounded-[12px] transition-all">
        Ingestion Monitor
      </div>

      {/* The modal is always in the DOM; its visibility is controlled by the 'hidden' class */}
      <div className={`${modalOpen ? '' : 'hidden'}`}>
        <IngestionDashboardModal wsUrl={wsUrl} onClose={() => setModalOpen(false)} onNewDocument={() => setModalOpen(true)} />
      </div>
    </>
  );
};

export default IngestionDashboardLauncher;
