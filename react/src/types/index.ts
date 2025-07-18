export interface TestComponentProps {
  message?: string;
}

export interface IngestionDashboardProps {
  wsUrl: string;
  onNewDocument: () => void;
}

export interface IngestionDashboardLauncherProps {
  wsUrl: string;
}

export interface IngestionMessage {
  messages?: string[];
  progress?: number;
  exception?: string;
  complete?: boolean;
}

export interface IngestionDashboardProps {
  wsUrl: string;
}

export interface PDFIngestionMonitorProps {
  documentName: string;
  documentId: string;
}

declare global {
  interface Window {
    mountReactComponent: (
      elementId: string,
      componentName: string,
      props?: Record<string, unknown>
    ) => void;
    apiUrls: {
      [key: string]: string;
    }
    pageUrls: {
      [key: string]: string;
    }
  }
}