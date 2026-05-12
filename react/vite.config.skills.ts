import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Skills page build (separate from the main app build).
//
// The skills page uses Monaco (~3.8 MB) and lazy-loads it via `React.lazy`,
// which requires an ES-module output. We can't put that in the main build
// because main.js is loaded with a classic `<script>` tag — see the comment
// in `vite.config.ts`.
//
// Output:
//   static/js/dist/skills.js                    (entry, ES module)
//   static/js/dist/skills.css                   (entry CSS, if any)
//   static/js/dist/chunks/SkillEditor-*.js      (Monaco lazy chunk)
//   static/js/dist/workers/editor.worker-*.js   (Monaco's web worker)
export default defineConfig({
  plugins: [react()],
  base: '/static/js/dist/',
  build: {
    outDir: '../aquillm/aquillm/static/js/dist/',
    assetsDir: '',
    manifest: 'manifest.skills.json',
    // Don't wipe sibling main-build outputs in the same dist directory.
    emptyOutDir: false,
    rollupOptions: {
      input: {
        skills: './src/skills_entry.tsx',
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
