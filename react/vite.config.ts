import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from "tailwindcss"

export default defineConfig({
  plugins: [react()],
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
        entryFileNames: '[name].js',
        chunkFileNames: '[name].js',
        assetFileNames: '[name].[ext]'
      }
    }
  },
})