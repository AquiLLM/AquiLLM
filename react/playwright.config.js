import { defineConfig } from '@playwright/test';

export default defineConfig({
  use: {
    baseURL: 'http://localhost:8080',
    storageState: 'auth.json',  // makes tests start logged in
  },
});