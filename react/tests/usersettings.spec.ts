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

test('profile memories show provenance, correction, removal and pending external deletion', async ({ page }) => {
  let fact = { id: 7, fact: 'I prefer telescope A.', revision: 1, status: 'active', origin_label: 'Unknown', source_label: 'Unknown', reconciliation_status: 'not_needed' };
  let deleted = false;
  await page.route('**/api/profile-memory/**', async route => {
    const request = route.request();
    if (request.method() === 'PATCH') {
      expect(request.postDataJSON()).toEqual({ replacement: 'I prefer telescope B.', expected_revision: 1 });
      fact = { ...fact, id: 8, fact: 'I prefer telescope B.', origin_label: 'Manual', source_label: 'User correction' };
      return route.fulfill({ json: { fact } });
    }
    if (request.method() === 'DELETE') {
      expect(request.postDataJSON()).toEqual({ expected_revision: 1 });
      deleted = true;
      return route.fulfill({ json: { status: 'revoked', reconciliation_status: 'pending' } });
    }
    return route.fulfill({ json: { facts: deleted ? [] : [fact], reconciliations: deleted ? [{ id: 8, reconciliation_status: 'pending' }] : [] } });
  });
  await page.goto('/user-settings/');
  await expect(page.getByRole('heading', { name: 'Profile Memory', exact: true })).toBeVisible();
  await expect(page.getByText('Origin: Unknown · Source: Unknown', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Correct memory 7', exact: true }).click();
  await page.getByRole('textbox', { name: 'Replacement for memory 7', exact: true }).fill('I prefer telescope B.');
  await page.getByRole('button', { name: 'Save correction', exact: true }).click();
  await expect(page.getByText('I prefer telescope B.', { exact: true })).toBeVisible();
  await expect(page.getByText('I prefer telescope A.', { exact: true })).toHaveCount(0);
  await expect(page.getByText('Origin: Manual · Source: User correction', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Delete memory 8', exact: true }).click();
  await expect(page.getByText('I prefer telescope B.', { exact: true })).toHaveCount(0);
  await expect(page.getByText('No active profile memories.', { exact: true })).toBeVisible();
  await expect(page.getByText('External deletion pending.', { exact: true })).toBeVisible();
});

test('a stale memory revision surfaces a reload action', async ({ page }) => {
  await page.route('**/api/profile-memory/**', async route => {
    if (route.request().method() === 'DELETE')
      return route.fulfill({ status: 409, json: { error: 'Memory changed. Reload before editing.' } });
    return route.fulfill({ json: { facts: [{ id: 7, fact: 'I prefer telescope A.', revision: 1, origin_label: 'Unknown', source_label: 'Unknown', reconciliation_status: 'not_needed' }], reconciliations: [] } });
  });
  await page.goto('/user-settings/');
  await page.getByRole('button', { name: 'Delete memory 7', exact: true }).click();
  await expect(page.getByRole('alert')).toContainText('Memory changed. Reload before editing.');
  await expect(page.getByRole('button', { name: 'Reload memories', exact: true })).toBeVisible();
  await expect(page.getByText('I prefer telescope A.', { exact: true })).toBeVisible();
});

test.describe('User Settings (Theme + Font) Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/user-settings/');
    await expect(page.getByRole('heading', { name: 'Select Your Theme', exact: true })).toBeVisible();
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
