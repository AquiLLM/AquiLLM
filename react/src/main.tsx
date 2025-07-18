import React from 'react';
import { createRoot } from 'react-dom/client';
import TestComponent from './components/TestComponent';
import PDFIngestionMonitor from './components/PDFIngestionMonitor';
import CollectionsPage from './components/CollectionsPage';
import CollectionView from './components/CollectionView';
import SearchPage from './components/SearchPage';

import IngestionDashboard from './components/IngestionDashboard';
import IngestionDashboardLauncher from './components/IngestionDashboardLauncher';
import IngestRowContainer from './components/IngestRow';
import WhitelistEmails from './components/WhitelistEmails';
import ChatComponent from './components/ChatComponent';
import UserSettings from './components/UserSettings';
import ChatFileUpload from './components/ChatFileUpload';
// Type for the components mapping
type ComponentsMap = {
  [key: string]: React.ComponentType<any>;
};

// Mount function with type safety
window.mountReactComponent = (
  elementId: string,
  componentName: string,
  props: Record<string, unknown> = {}
) => {
  const element = document.getElementById(elementId);
  if (!element) {
    console.error(`Element with id '${elementId}' not found`);
    return;
  }

  const root = createRoot(element);
  
  const components: ComponentsMap = {
    TestComponent: TestComponent,
    PDFIngestionMonitor: PDFIngestionMonitor,
    CollectionsPage: CollectionsPage,
    CollectionView: CollectionView,
    SearchPage: SearchPage,
    IngestionDashboard: IngestionDashboard,
    IngestionDashboardLauncher: IngestionDashboardLauncher,
    IngestRowContainer: IngestRowContainer,
    WhitelistEmails: WhitelistEmails,
    ChatComponent: ChatComponent,
    UserSettings: UserSettings,
    ChatFileUpload: ChatFileUpload,
    // Add other components here
  };
  
  const Component = components[componentName];
  if (Component) {
    root.render(<Component {...props} />);
  } else {
    console.error(`Component '${componentName}' not found`);
  }
};

export function getCsrfCookie(): string {
  const cookieName = "csrftoken";
  const cookies = document.cookie.split("; ");
  for (const cookie of cookies) {
    const [name, value] = cookie.split("=");
    if (name === cookieName) {
      return decodeURIComponent(value);
    }
  }
  throw new Error("CSRF token not found in cookies");
}