import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './tests',
  testMatch: 'chat-context.spec.ts',
  use: { baseURL: 'http://127.0.0.1:5178', channel: 'msedge' },
  webServer: {
    command: 'npm run dev -- --host 127.0.0.1 --port 5178 --strictPort',
    url: 'http://127.0.0.1:5178/static/js/dist/tests/fixtures/chat-context.html',
    reuseExistingServer: true,
  },
});
