import { defineConfig } from 'vitest/config'

// Standalone config for unit tests so the build-only Vite plugins (pdf worker
// copy, css placeholder, Django manifest emit) don't run during `vitest`.
// The matchers under test are pure functions, so the default node environment
// is sufficient — no jsdom needed.
export default defineConfig({
  test: {
    include: ['src/**/*.test.ts'],
    environment: 'node',
  },
})
