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
  modality?: string;
  rawMediaSaved?: boolean;
  textExtracted?: boolean;
  provider?: string;
  providerModel?: string;
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
    appFlags?: {
      /** When true, the chat eagerly LLM-narrows every citation in newly
       * arrived assistant messages so subsequent clicks open instantly. */
      eagerCitationNarrow?: boolean;
      [key: string]: unknown;
    }
  }
}
