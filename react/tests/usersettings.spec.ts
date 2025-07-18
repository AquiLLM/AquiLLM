import { test, expect } from '@playwright/test';

const themes = [
  'aquillm_default_dark',
  'aquillm_default_light',
  'aquillm_default_dark_accessible_chat',
  'aquillm_default_light_accessible_chat',
  'high_contrast',
];

const fonts = [
  'sans_serif',
  'verdana',
  'timesnewroman',
  'opendyslexic',
  'lexend',
  'comicsans',
];

test.describe('User Settings (Theme + Font) Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('http://localhost:8080/user-settings/');
    await expect(page.locator('#user-settings-root h2')).toHaveText('Select Your Theme');
  });

  for (const theme of themes) {
    test(`applies theme: ${theme}`, async ({ page }) => {
      await page.selectOption('select[name="color_scheme"]', theme);
      await page.getByTestId('save-theme-settings').click();
      await page.waitForTimeout(500); // wait for DOM update

      const classList = await page.evaluate(() => document.body.className);
      expect(classList).toContain(`theme-${theme}`);
    });
  }

  for (const font of fonts) {
    test(`applies font: ${font}`, async ({ page }) => {
      await page.selectOption('select[name="font_family"]', font);
      await page.getByTestId('save-theme-settings').click();
      await page.waitForTimeout(500); // wait for DOM update

      const classList = await page.evaluate(() => document.body.className);
      expect(classList).toContain(`font-${font}`);
    });
  }
});