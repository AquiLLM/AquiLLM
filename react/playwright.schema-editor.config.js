import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './tests',
  testMatch: 'collection-schema-editor.spec.ts',
  use: {
    baseURL: 'http://127.0.0.1:4173',
  },
  webServer: {
    command:
      'npx vite --config vite.schema-editor.config.ts --host 127.0.0.1 --port 4173 --strictPort',
    url: 'http://127.0.0.1:4173/tests/fixtures/collection-schema-editor.html',
    reuseExistingServer: false,
  },
});
