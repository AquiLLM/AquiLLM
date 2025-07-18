import { chromium } from '@playwright/test';
import fs from 'fs';

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();

  await page.goto('http://localhost:8080/accounts/login/');

  await page.fill('input[name="login"]', 'dev');
  await page.fill('input[name="password"]', 'rickbailey');
  await page.click('button[type="submit"]');

  // Wait for dashboard or some authenticated page
  await page.waitForURL('**/');

  // Save storage state (cookies + localStorage)
  await page.context().storageState({ path: 'auth.json' });

  await browser.close();
})();