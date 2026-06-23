import { pdfjs } from 'react-pdf';

// The worker file is copied to this static path by the `copy-pdf-worker`
// Vite plugin (see vite.config.ts). A literal string is used instead of
// Vite's `?url` import because that emits `import.meta.url`, which fails
// to parse when the bundle is loaded as a non-module <script>.
const WORKER_URL = '/static/js/dist/pdf.worker.min.mjs';

let configured = false;

export function configurePdfWorker(): void {
  if (configured) return;
  pdfjs.GlobalWorkerOptions.workerSrc = WORKER_URL;
  configured = true;
}
