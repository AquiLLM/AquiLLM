import { test, expect } from '@playwright/test';

test.describe('Base Page (/) UI Tests', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('http://localhost:8080/');
  });

  test('Sidebar opens and closes', async ({ page }) => {
    const sidebar = page.locator('#aq-sidebar');
    const toggleButton = page.locator('#menu-toggle');

    await expect(sidebar).toBeVisible();
    await toggleButton.click();
    await expect(sidebar).toHaveClass(/ml-\[-260px\]/);
    await toggleButton.click();
    await expect(sidebar).not.toHaveClass(/ml-\[-260px\]/);
  });

  test('Collections dropdown opens and closes', async ({ page }) => {
    const dropdownButton = page.locator('#conversation-dropdown-btn');
    const dropdown = page.locator('#conversation-dropdown');

    await dropdownButton.click();
    await expect(dropdown).toHaveCSS('height', '0px');

    await dropdownButton.click();
    await expect(dropdown).not.toHaveCSS('height', '0px');
  });

  test('User account menu appears and shows options', async ({ page }) => {
    const userButton = page.locator('#account-management-toggle-button');
    const accountMenu = page.locator('#account-menu-modal');

    await userButton.click();
    await expect(accountMenu).toBeVisible();
    await expect(accountMenu.getByText('Manage Account')).toBeVisible();
    await expect(accountMenu.getByRole('button', { name: 'Logout' })).toBeVisible();
  });

  test('Navbar contains AquiLLM logo and text', async ({ page }) => {
    const logoSvg = page.locator('#nav-logo');
    const logoText = page.locator('nav').getByText('AquiLLM');

    await expect(logoSvg).toBeVisible();
    await expect(logoText).toBeVisible();
  });

  test('Sidebar includes Menu and Utilities sections', async ({ page }) => {
    await expect(page.locator('#sidebar-header span')).toHaveText('Menu');
    await expect(page.locator('#aq-sidebar')).toContainText('Utilities');
  });
});
