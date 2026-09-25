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
      props?: Record<string, unknown>,
    ) => void;
    apiUrls: {
      api_collection_schema_workspace?: string;
      api_collection_schema_draft?: string;
      api_collection_schema_entity?: string;
      api_collection_schema_relation?: string;
      api_collection_schema_validate?: string;
      api_collection_schema_diff?: string;
      api_collection_schema_publish?: string;
      api_collection_schema_discard?: string;
      api_collection_schema_versions?: string;
      api_collection_schema_version_diff?: string;
      api_collection_schema_restore?: string;
      api_collection_schema_restore_replace?: string;
      api_collection_graph_visualization?: string;
      api_collection_graph_rebuild?: string;
    } & Record<string, string>;
    pageUrls: {
      [key: string]: string;
    };
    appFlags?: {
      /** When true, the chat eagerly LLM-narrows every citation in newly
       * arrived assistant messages so subsequent clicks open instantly. */
      eagerCitationNarrow?: boolean;
      [key: string]: unknown;
    };
  }
}
