import { defineConfig } from '@playwright/test';

const baseURL = process.env.RAG_PLAYWRIGHT_BASE_URL;
const storageState = process.env.RAG_PLAYWRIGHT_STORAGE_STATE;
if (baseURL !== 'http://app:8000' || !storageState ||
    !/^\/artifacts\/browser\/run-[a-f0-9-]+\/state\.json$/.test(storageState)) {
  throw new Error('Playwright requires the owned app URL and fresh task storage');
}

export default defineConfig({
  testDir: './tests',
  workers: 1,
  retries: 0,
  outputDir: storageState.replace('/state.json', '/results'),
  reporter: [['list']],
  use: { baseURL, storageState, browserName: 'chromium', trace: 'retain-on-failure', screenshot: 'only-on-failure' },
});
