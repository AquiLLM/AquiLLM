/** One-time ownership transfer from the destination page's inline bootstrap. */
export type BufferedChatSocketEvent =
  | { type: 'open' | 'error'; event: Event }
  | { type: 'message'; event: MessageEvent }
  | { type: 'close'; event: CloseEvent };

export interface ChatSocketHandoff {
  socket: WebSocket;
  events: BufferedChatSocketEvent[];
  startedAt: number;
}

declare global {
  interface Window {
    aquiChatSocketBootstrap?: {
      take(convoId: string): ChatSocketHandoff | null;
    };
  }
}
