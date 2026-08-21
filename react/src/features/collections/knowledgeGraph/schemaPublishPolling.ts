export type PublishPollStatus = 'pending' | 'succeeded' | 'failed';

export type PublishPollOutcome = PublishPollStatus | 'exhausted' | 'cancelled';

export interface PublishPollOptions {
  poll: () => Promise<PublishPollStatus>;
  signal?: AbortSignal;
  maxAttempts?: number;
  initialDelayMs?: number;
  maxDelayMs?: number;
  sleep?: (ms: number) => Promise<void>;
}

const defaultSleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

export async function pollPublishStatus(options: PublishPollOptions): Promise<PublishPollOutcome> {
  const {
    poll,
    signal,
    maxAttempts = 5,
    initialDelayMs = 100,
    maxDelayMs = 1000,
    sleep = defaultSleep,
  } = options;

  let delay = initialDelayMs;
  for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
    if (signal?.aborted) return 'cancelled';
    const status = await poll();
    if (status !== 'pending') return status;
    if (attempt === maxAttempts - 1) break;
    await sleep(delay);
    delay = Math.min(delay * 2, maxDelayMs);
  }
  return 'exhausted';
}

export interface PublishPollController {
  promise: Promise<PublishPollOutcome>;
  retry: () => Promise<PublishPollOutcome>;
  cancel: () => void;
}

export function createPublishPollController(options: PublishPollOptions): PublishPollController {
  const controller = new AbortController();
  const run = () =>
    pollPublishStatus({
      ...options,
      signal: controller.signal,
    });

  return {
    promise: run(),
    retry: run,
    cancel: () => controller.abort(),
  };
}
