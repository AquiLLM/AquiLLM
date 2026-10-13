import { test, expect } from '@playwright/test';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';

test('the owned app serves the current frontend image build', async ({ request }) => {
  const image = readFileSync('/app/aquillm/aquillm/static/js/dist/main.js');
  const response = await request.get('/static/js/dist/main.js');
  expect(response.ok()).toBe(true);
  const hash = (value: Buffer) => createHash('sha256').update(value).digest('hex');
  expect(hash(await response.body())).toBe(hash(image));
});
