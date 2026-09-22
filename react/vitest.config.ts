import { defineConfig } from 'vitest/config'

// Standalone config for unit tests so the build-only Vite plugins (pdf worker
// copy, css placeholder, Django manifest emit) don't run during `vitest`.
// Most matchers under test are pure functions, so the default node environment
// remains sufficient. Component tests opt into jsdom per-file.
export default defineConfig({
  test: {
    include: ['src/**/*.test.ts', 'src/**/*.test.tsx'],
    environment: 'node',
  },
})
