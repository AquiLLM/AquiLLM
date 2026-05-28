// Separate Vite entry for the /aquillm/skills/ page.
//
// Why not register on `window.mountReactComponent` like other pages?
// SkillsPage uses `React.lazy()` for the Monaco editor (3.8 MB minified).
// Putting that lazy import in `main.tsx` forces Rollup to emit the whole main
// bundle as an ES module (with top-level `export` statements), which the
// classic `<script src="main.js">` tag in base.html cannot parse — breaking
// every page. Keeping the lazy import in a dedicated entry lets `main.js` stay
// a single-file IIFE bundle while `skills.js` is loaded as a module only on the
// skills page.
import React, { Suspense } from 'react';
import { createRoot } from 'react-dom/client';

const SkillsPage = React.lazy(() => import('./features/skills/components/SkillsPage'));

function mount() {
  const el = document.getElementById('skills-page');
  if (!el) {
    console.error("Element 'skills-page' not found");
    return;
  }
  createRoot(el).render(
    <Suspense fallback={null}>
      <SkillsPage />
    </Suspense>,
  );
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', mount);
} else {
  mount();
}
