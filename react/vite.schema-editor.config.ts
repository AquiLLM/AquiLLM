import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { resolve } from 'node:path';

export default defineConfig({
  plugins: [react()],
  base: '/',
  root: resolve(__dirname),
  server: {
    host: '127.0.0.1',
    port: 4173,
    strictPort: true,
  },
});
