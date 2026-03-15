import { test, expect } from '@playwright/test';

/**
 * mini smoke test:
 * proves the web app responds to a real browser request
 *
 * why this test exists:
 * - proves Playwright is set up correctly
 * - proves the running app is reachable at baseURL
 * - keeps initial E2E setup simple and non-flaky
 *
 * intentionally avoid login/data creation here until the tooling is stable
 */
test('homepage loads with GET', async ({ page }) => {
  await page.goto('/');

  // this is a lightweight assertion that we ended up on the app
  // if redirects happen (like to /accounts/login/), this should still pass as long as it's on localhost:8080
  await expect(page).toHaveURL(/localhost:8080/);
});