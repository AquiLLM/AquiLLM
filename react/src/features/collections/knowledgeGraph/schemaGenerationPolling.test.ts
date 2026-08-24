import { describe, expect, it, vi } from 'vitest';

import { createGenerationPollController, pollGenerationStatus } from './schemaGenerationPolling';

const status = (value: 'queued' | 'running' | 'succeeded' | 'failed') => ({
  run_id: 'run-1',
  status: value,
  error_code: value === 'failed' ? 'no_collection_text' : null,
  statistics: {},
});

describe('pollGenerationStatus', () => {
  it.each(['queued', 'running'] as const)('continues polling while generation is %s', async (pendingStatus) => {
    const poll = vi.fn().mockResolvedValueOnce(status(pendingStatus)).mockResolvedValueOnce(status('succeeded'));
    const sleep = vi.fn().mockResolvedValue(undefined);

    await expect(pollGenerationStatus({ poll, sleep })).resolves.toMatchObject({ status: 'succeeded' });
    expect(poll).toHaveBeenCalledTimes(2);
    expect(sleep).toHaveBeenCalledWith(100);
  });

  it.each(['succeeded', 'failed'] as const)('returns the terminal %s status without sleeping', async (terminalStatus) => {
    const poll = vi.fn().mockResolvedValue(status(terminalStatus));
    const sleep = vi.fn();

    await expect(pollGenerationStatus({ poll, sleep })).resolves.toMatchObject({ status: terminalStatus });
    expect(poll).toHaveBeenCalledTimes(1);
    expect(sleep).not.toHaveBeenCalled();
  });

  it('uses capped exponential backoff and stops after the maximum attempts', async () => {
    const poll = vi.fn().mockResolvedValue(status('running'));
    const sleep = vi.fn().mockResolvedValue(undefined);

    await expect(
      pollGenerationStatus({ poll, sleep, initialDelayMs: 50, maxDelayMs: 100, maxAttempts: 4 }),
    ).resolves.toEqual({ status: 'exhausted' });
    expect(poll).toHaveBeenCalledTimes(4);
    expect(sleep.mock.calls.map(([delay]) => delay)).toEqual([50, 100, 100]);
  });

  it('aborts without accepting a stale response', async () => {
    const controller = new AbortController();
    let resolvePoll: ((value: ReturnType<typeof status>) => void) | undefined;
    const poll = vi.fn(
      () =>
        new Promise<ReturnType<typeof status>>((resolve) => {
          resolvePoll = resolve;
        }),
    );
    const promise = pollGenerationStatus({ poll, signal: controller.signal, sleep: vi.fn() });

    controller.abort();
    resolvePoll?.(status('succeeded'));

    await expect(promise).resolves.toEqual({ status: 'cancelled' });
  });

  it('exposes cancellation through a controller', async () => {
    const poll = vi.fn().mockResolvedValue(status('queued'));
    const controller = createGenerationPollController({ poll, sleep: vi.fn() });

    controller.cancel();

    await expect(controller.promise).resolves.toEqual({ status: 'cancelled' });
  });
});
