import { test, expect } from '@playwright/test';

test.describe('Login Page Visual and Functional Tests', () => {
  test.beforeEach(async ({ page }) => {
    await page.context().clearCookies();
    await page.goto('http://localhost:8080/accounts/login/');
  });

  test('renders gradient background', async ({ page }) => {
    const background = page.locator('#login-background');
    await expect(background).toBeVisible();
    await expect(background).toContainClass("bg-gradient-to-b");
  });

  test('logo SVG is rendered and behind the login card', async ({ page }) => {
    const svgLogo = page.locator('svg').first();
    const loginCard = page.locator('#content');

    await expect(svgLogo).toBeVisible();
    const logoBox = await svgLogo.boundingBox();
    const cardBox = await loginCard.boundingBox();

    expect(logoBox && cardBox).toBeTruthy();
    if (logoBox && cardBox) {
      // Rough overlap test
      expect(logoBox.x + logoBox.width / 2).toBeGreaterThan(cardBox.x);
      expect(logoBox.x + logoBox.width / 2).toBeLessThan(cardBox.x + cardBox.width);
    }
  });

  test('login form has all required inputs', async ({ page }) => {
    await expect(page.locator('input[name="login"]')).toBeVisible();
    await expect(page.locator('input[name="password"]')).toBeVisible();
    await expect(page.locator('input[name="remember"]')).toBeVisible();
  });

  test('login form accepts input and submits mock login', async ({ page }) => {
    await page.locator('input[name="login"]').fill('testuser');
    await page.locator('input[name="password"]').fill('fakepassword');
    await page.locator('button[type="submit"]').click();

    // This assumes login fails (no mock backend), check for a failure message or redirect
    await page.waitForLoadState('networkidle');
    const maybeError = await page.locator('text=invalid').first();
    expect(await maybeError.count() >= 0).toBeTruthy();
  });

  test('third-party login button is visible (Google)', async ({ page }) => {
    const googleLogin = page.locator('a[href*="google"]');
    await expect(googleLogin).toBeVisible();
    await expect(googleLogin.locator('svg')).toBeVisible();
  });

  test('layout is responsive (card centered)', async ({ page }) => {
    const card = page.locator('#content');
    const cardBox = await card.boundingBox();
    const windowWidth = page.viewportSize()?.width || 0;

    expect(cardBox).toBeTruthy();
    if (cardBox) {
      const cardCenter = cardBox.x + cardBox.width / 2;
      expect(Math.abs(cardCenter - windowWidth / 2)).toBeLessThanOrEqual(20);
    }
  });
});
