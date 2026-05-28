// Separate Vite entry for the /aquillm/collections/<id>/notes/ page.
//
// See `vite.config.skills.ts` for why this lives in its own bundle (Monaco
// requires ES-module loading, which the main app's classic <script> tag can't
// handle). This bundle is loaded as `<script type="module">` only on the
// Collection Notes page; main.js stays untouched.
import React, { Suspense } from 'react';
import { createRoot } from 'react-dom/client';

const CollectionNotesPage = React.lazy(
  () => import('./features/skills/components/CollectionNotesPage'),
);

function mount() {
  const el = document.getElementById('collection-notes-page');
  if (!el) {
    console.error("Element 'collection-notes-page' not found");
    return;
  }
  const collectionId = Number(el.dataset.collectionId);
  const collectionName = el.dataset.collectionName ?? '';
  const collectionUrl = el.dataset.collectionUrl ?? '/aquillm/collections/';
  if (!Number.isFinite(collectionId) || collectionId <= 0) {
    console.error('Invalid collection-id data attribute on mount element');
    return;
  }
  createRoot(el).render(
    <Suspense fallback={null}>
      <CollectionNotesPage
        collectionId={collectionId}
        collectionName={collectionName}
        collectionUrl={collectionUrl}
      />
    </Suspense>,
  );
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', mount);
} else {
  mount();
}
