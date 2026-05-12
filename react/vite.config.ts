import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from "tailwindcss"

// Main app build: single entry, fully static, output as one self-contained
// classic-script-compatible bundle (no top-level import/export keywords).
// The skills page is built separately by vite.config.skills.ts so it can use
// `React.lazy` + Monaco without forcing this bundle into ES-module format.
export default defineConfig({
  plugins: [react()],
  base: '/static/js/dist/',
  build: {
    outDir: '../aquillm/aquillm/static/js/dist/',
    assetsDir: '',
    manifest: true,
    // Don't wipe outputs of the sibling skills build.
    emptyOutDir: false,
    rollupOptions: {
      input: {
        main: './src/main.tsx',
      },
      output: {
        entryFileNames: '[name].js',
        chunkFileNames: '[name].js',
        assetFileNames: '[name].[ext]',
      },
    },
  },
})
