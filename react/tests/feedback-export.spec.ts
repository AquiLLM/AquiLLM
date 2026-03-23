import { test } from '@playwright/test';

/**
 * Full coverage needs a logged-in superuser session against a running stack.
 * Backend and CSV behavior are covered by Django tests in
 * apps/platform_admin/tests/test_feedback_csv_export_api.py.
 */
test.fixme('superuser sees Download Feedback CSV on email whitelist page', async ({ page }) => {
  await page.goto('http://localhost:8080/aquillm/email_whitelist/');
  await page.getByTestId('download-feedback-csv').click();
});
