import { defineConfig, type Plugin } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from "tailwindcss"
import { copyFileSync, mkdirSync, writeFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = dirname(fileURLToPath(import.meta.url))

// Copy the pdfjs worker into the build output dir so it can be served by
// Django's static handling at /static/js/dist/pdf.worker.min.mjs.
// Done via plugin (not `import workerSrc from '...?url'`) to keep the
// bundle free of `import.meta.url`, which would require <script type="module">.
function copyPdfWorker(): Plugin {
  return {
    name: 'copy-pdf-worker',
    apply: 'build',
    closeBundle() {
      const src = resolve(__dirname, 'node_modules/pdfjs-dist/build/pdf.worker.min.mjs')
      const dest = resolve(__dirname, '../aquillm/aquillm/static/js/dist/pdf.worker.min.mjs')
      mkdirSync(dirname(dest), { recursive: true })
      copyFileSync(src, dest)
    },
  }
}

// IIFE format inlines CSS into the JS bundle (it injects <style> tags at
// runtime via document.createElement("style")). But base.html still has a
// <link rel="stylesheet" href="js/dist/main.css">. Write a placeholder so
// that link doesn't 404. The actual styles live in main.js.
function placeholderMainCss(): Plugin {
  return {
    name: 'placeholder-main-css',
    apply: 'build',
    closeBundle() {
      const dest = resolve(__dirname, '../aquillm/aquillm/static/js/dist/main.css')
      mkdirSync(dirname(dest), { recursive: true })
      writeFileSync(
        dest,
        '/* CSS is bundled into main.js (IIFE format) and injected at runtime. */\n',
      )
    },
  }
}

export default defineConfig({
  plugins: [react(), copyPdfWorker(), placeholderMainCss()],
  base: '/static/js/dist/',
  build: {
    outDir: '../aquillm/aquillm/static/js/dist/',
    assetsDir: '',
    manifest: true,
    rollupOptions: {
      input: {
        main: './src/main.tsx',
      },
      output: {
        // The bundle is loaded via a classic <script> tag in
        // aquillm/templates/aquillm/base.html (not type="module"). IIFE
        // emits a self-executing function with no `export` / `import.meta`
        // syntax, so the file parses in that context. Rollup auto-rewrites
        // `import.meta.url` to `document.currentScript.src` for IIFE,
        // which keeps pdfjs-dist's internal usage working.
        format: 'iife',
        inlineDynamicImports: true,
        entryFileNames: '[name].js',
        chunkFileNames: '[name].js',
        assetFileNames: '[name].[ext]'
      }
    }
  },
})