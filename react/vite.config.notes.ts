import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Collection Notes page build (separate from the main app build).
//
// The notes page uses Monaco (~3.8 MB) and lazy-loads it via `React.lazy`,
// which requires an ES-module output. We can't put that in the main build
// because main.js is loaded with a classic `<script>` tag — see the comment
// in `vite.config.ts`. This bundle is loaded with `<script type="module">`
// from aquillm/templates/aquillm/collection_notes.html.
//
// Output:
//   static/js/dist/collection_notes.js          (notes entry, ES module)
//   static/js/dist/chunks/NoteEditor-*.js        (Monaco lazy chunk)
//   static/js/dist/workers/editor.worker-*.js    (Monaco's web worker)
export default defineConfig({
  plugins: [react()],
  base: '/static/js/dist/',
  build: {
    outDir: '../aquillm/aquillm/static/js/dist/',
    assetsDir: '',
    manifest: 'manifest.notes.json',
    // Don't wipe sibling main-build outputs in the same dist directory.
    emptyOutDir: false,
    rollupOptions: {
      input: {
        collection_notes: './src/collection_notes_entry.tsx',
      },
      output: {
        entryFileNames: '[name].js',
        chunkFileNames: 'chunks/[name]-[hash].js',
        assetFileNames: '[name].[ext]',
      },
    },
  },
  worker: {
    format: 'es',
    rollupOptions: {
      output: {
        entryFileNames: 'workers/[name]-[hash].js',
        chunkFileNames: 'workers/chunks/[name]-[hash].js',
      },
    },
  },
})
