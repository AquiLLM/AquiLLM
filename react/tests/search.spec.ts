import { test, expect } from '@playwright/test';

test.describe('Vector Search Page (UI Only)', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('http://localhost:8080/aquillm/search/');
  });

  test('renders search inputs and collection checkboxes', async ({ page }) => {
    await expect(page.getByText('Search')).toBeVisible();
    await expect(page.getByLabel('Query:')).toBeVisible();
    await expect(page.getByLabel('Top K:')).toBeVisible();
    await expect(page.getByText('Collections:')).toBeVisible();
  });

  test('allows typing a query and setting Top K', async ({ page }) => {
    const queryBox = page.locator('textarea[placeholder="Send a message"]');
    await queryBox.fill('Quantum computing error rates');
    await expect(queryBox).toHaveValue('Quantum computing error rates');

    const topKInput = page.locator('input[type="number"]');
    await topKInput.fill('10');
    await expect(topKInput).toHaveValue('10');
  });

  test('allows selecting and unselecting collections', async ({ page }) => {
    const checkboxes = page.locator('fieldset input[type="checkbox"]');
    const count = await checkboxes.count();

    if (count > 0) {
      await checkboxes.nth(0).check();
      await expect(checkboxes.nth(0)).toBeChecked();

      await checkboxes.nth(0).uncheck();
      await expect(checkboxes.nth(0)).not.toBeChecked();
    }
  });
});
