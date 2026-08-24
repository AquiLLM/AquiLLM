import type { SchemaGenerationStatus } from './schemaTypes';

export type GenerationPollOutcome = SchemaGenerationStatus | { status: 'exhausted' | 'cancelled' };

export interface GenerationPollOptions {
  poll: (signal?: AbortSignal) => Promise<SchemaGenerationStatus>;
  signal?: AbortSignal;
  maxAttempts?: number;
  initialDelayMs?: number;
  maxDelayMs?: number;
  sleep?: (ms: number) => Promise<void>;
}

const defaultSleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

export async function pollGenerationStatus(options: GenerationPollOptions): Promise<GenerationPollOutcome> {
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
    if (signal?.aborted) return { status: 'cancelled' };
    const result = await poll(signal);
    if (signal?.aborted) return { status: 'cancelled' };
    if (result.status === 'succeeded' || result.status === 'failed') return result;
    if (attempt === maxAttempts - 1) break;
    await sleep(delay);
    if (signal?.aborted) return { status: 'cancelled' };
    delay = Math.min(delay * 2, maxDelayMs);
  }
  return { status: 'exhausted' };
}

export interface GenerationPollController {
  promise: Promise<GenerationPollOutcome>;
  retry: () => Promise<GenerationPollOutcome>;
  cancel: () => void;
}

export function createGenerationPollController(options: GenerationPollOptions): GenerationPollController {
  const controller = new AbortController();
  const run = () => pollGenerationStatus({ ...options, signal: controller.signal });

  return {
    promise: run(),
    retry: run,
    cancel: () => controller.abort(),
  };
}
