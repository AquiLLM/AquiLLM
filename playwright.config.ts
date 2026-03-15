import { defineConfig, devices } from '@playwright/test';

/**
 * playwright configuration for some AquiLLM end-to-end tests
 * our basic and/or key ideas:
 * - testDir: all E2E tests live under ./e2e
 * - baseURL: where the running app is reachable (defaults to localhost:8080)
 * - reporter: list output in console + HTML report artifact
 * - retries: CI retries to reduce random flake failures
 * - projects: run tests in Chromium (fast + standard baseline)
 */
export default defineConfig({
  testDir: './e2e',

  // this is the max time a single test can run
  timeout: 60_000,

  // and this is the max time a single expect can wait
  expect: { timeout: 10_000 },

  // to allow tests to run in parallel when there are multiple files/tests
  fullyParallel: true,

  // retries are sometimes good in CI bc timing can be kinda inconsistent
  retries: process.env.CI ? 2 : 0,

  // reporters:
  // - list: readable CLI output
  // - html: generates playwright-report/ 
  reporter: [['list'], ['html', { open: 'never' }]],

  // default options that're applied to every test
  use: {
    // base URL of the app
    // in CI or remote runs, we override with PLAYWRIGHT_BASE_URL
    baseURL: process.env.PLAYWRIGHT_BASE_URL || 'http://localhost:8080',

    // helpful debugging stuff
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
  },

  // browsers to run against
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],
});