import { test, expect } from '@playwright/test';

test.describe('Sidebar Navigation Tests', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to homepage
    await page.goto('http://localhost:8080/');
    await page.waitForLoadState('networkidle');
  });

  test('Sidebar contains expected navigation elements', async ({ page }) => {
    // Take a screenshot of the sidebar for debugging
    await page.screenshot({ path: 'test-results/sidebar-full.png', fullPage: true });
    
    // Check if the sidebar is visible
    const sidebar = page.locator('#aq-sidebar');
    if (await sidebar.isVisible()) {
      console.log('Sidebar is visible');
    } else {
      // Try to find a toggle button if sidebar isn't immediately visible
      const possibleToggleButtons = page.locator('button').all();
      console.log('Looking for sidebar toggle...');
      for (const button of await possibleToggleButtons) {
        const buttonId = await button.getAttribute('id');
        console.log('Found button with ID:', buttonId);
      }
    }
    
    // Log all visible text in the sidebar area for debugging
    const allText = await page.locator('body').textContent();
    console.log('Page text content:', allText);
  });
}); 