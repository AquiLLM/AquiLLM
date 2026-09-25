import React from 'react';
import { createRoot } from 'react-dom/client';
import '../../test.css';
import CollectionView from '../features/collections/components/CollectionView';

const root = document.getElementById('collection-view-root');
if (!root) {
  throw new Error('collection-view-root element not found');
}

createRoot(root).render(<CollectionView collectionId="42" />);
