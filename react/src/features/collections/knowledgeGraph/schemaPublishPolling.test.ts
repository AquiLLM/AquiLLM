import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { createPublishPollController, pollPublishStatus } from './schemaPublishPolling';

describe('pollPublishStatus', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('returns immediately on terminal success', async () => {
    const poll = vi.fn().mockResolvedValue('succeeded');
    await expect(pollPublishStatus({ poll, sleep: vi.fn() })).resolves.toBe('succeeded');
    expect(poll).toHaveBeenCalledTimes(1);
  });

  it('uses bounded backoff until terminal failure', async () => {
    const poll = vi
      .fn()
      .mockResolvedValueOnce('pending')
      .mockResolvedValueOnce('pending')
      .mockResolvedValueOnce('failed');
    const sleep = vi.fn().mockResolvedValue(undefined);
    const promise = pollPublishStatus({ poll, sleep, initialDelayMs: 100, maxDelayMs: 400, maxAttempts: 5 });
    await vi.runAllTimersAsync();
    await expect(promise).resolves.toBe('failed');
    expect(sleep).toHaveBeenCalledTimes(2);
    expect(sleep.mock.calls[0]?.[0]).toBe(100);
    expect(sleep.mock.calls[1]?.[0]).toBe(200);
  });

  it('stops when cancelled', async () => {
    const controller = new AbortController();
    const poll = vi.fn().mockResolvedValue('pending');
    const promise = pollPublishStatus({ poll, signal: controller.signal, sleep: vi.fn() });
    controller.abort();
    await expect(promise).resolves.toBe('cancelled');
  });

  it('exposes manual retry after exhaustion', async () => {
    const poll = vi.fn().mockResolvedValue('pending');
    const controller = createPublishPollController({ poll, maxAttempts: 2, sleep: vi.fn() });
    await expect(controller.promise).resolves.toBe('exhausted');
    poll.mockResolvedValueOnce('succeeded');
    await expect(controller.retry()).resolves.toBe('succeeded');
  });
});
