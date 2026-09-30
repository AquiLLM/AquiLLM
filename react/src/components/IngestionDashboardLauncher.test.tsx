// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import IngestionDashboardLauncher from './IngestionDashboardLauncher';

class Socket extends EventTarget {
  static sockets: Socket[] = [];
  constructor(public url: string) { super(); Socket.sockets.push(this); }
  close = vi.fn();
  emit(payload: unknown) { this.dispatchEvent(new MessageEvent('message', { data: JSON.stringify(payload) })); }
}
beforeEach(() => {
  Socket.sockets = [];
  vi.stubGlobal('WebSocket', Socket);
  Element.prototype.scrollIntoView = vi.fn();
  window.pageUrls = { document: '/document/%(doc_id)s/' };
});
afterEach(() => { cleanup(); vi.useRealTimers(); vi.unstubAllGlobals(); vi.restoreAllMocks(); });

it('keeps a dismissed ingestion panel closed on document replay, and opens for new work', () => {
  render(<IngestionDashboardLauncher wsUrl="ws://test/ingest/dashboard/" />);
  const start = { type: 'document.ingestion.start', documentId: 'doc-1', documentName: 'Paper' };
  const dashboards = () => Socket.sockets.filter(socket => socket.url.endsWith('/ingest/dashboard/'));
  act(() => dashboards()[0].emit(start));
  expect(dashboards()).toHaveLength(1);
  const wrapper = () => screen.getByRole('heading', { name: 'Ingestion Dashboard', hidden: true }).closest('.fixed')!.parentElement!;
  fireEvent.click(screen.getByRole('heading', { name: 'Ingestion Dashboard' }).parentElement!.querySelector('button')!);
  expect(wrapper().classList.contains('hidden')).toBe(true);
  act(() => dashboards()[0].emit(start));
  expect(wrapper().classList.contains('hidden')).toBe(true);
  expect(screen.getAllByRole('link', { name: 'Paper', hidden: true })).toHaveLength(1);
  act(() => dashboards()[0].emit({ ...start, documentId: 'doc-2', documentName: 'New paper' }));
  expect(wrapper().classList.contains('hidden')).toBe(false);
  expect(screen.getAllByRole('link', { name: 'Paper' })).toHaveLength(1);
  expect(screen.getByRole('link', { name: 'New paper' })).toBeTruthy();
  expect(dashboards()).toHaveLength(1);
});

it('reconnects without reopening a dismissed panel for an already seen document', () => {
  vi.useFakeTimers();
  render(<IngestionDashboardLauncher wsUrl="ws://test/ingest/dashboard/" />);
  const dashboards = () => Socket.sockets.filter(socket => socket.url.endsWith('/ingest/dashboard/'));
  const start = { type: 'document.ingestion.start', documentId: 'doc-1', documentName: 'Paper' };
  act(() => dashboards()[0].emit(start));
  fireEvent.click(screen.getByRole('heading', { name: 'Ingestion Dashboard' }).parentElement!.querySelector('button')!);
  act(() => dashboards()[0].dispatchEvent(new Event('close')));
  act(() => vi.advanceTimersByTime(2000));
  expect(dashboards()).toHaveLength(2);
  act(() => dashboards()[1].emit(start));
  const wrapper = screen.getByRole('heading', { name: 'Ingestion Dashboard', hidden: true }).closest('.fixed')!.parentElement!;
  expect(wrapper.classList.contains('hidden')).toBe(true);
  expect(screen.getAllByRole('link', { name: 'Paper', hidden: true })).toHaveLength(1);
});

it('bounds a flapping monitor even when each socket opens, and cancels timers on unmount', () => {
  vi.useFakeTimers();
  const view = render(<IngestionDashboardLauncher wsUrl="ws://test/ingest/dashboard/" />);
  for (let attempt = 0; attempt < 5; attempt++) {
    expect(Socket.sockets[attempt]).toBeTruthy();
    act(() => {
      Socket.sockets[attempt].dispatchEvent(new Event('open'));
      Socket.sockets[attempt].dispatchEvent(new Event('close'));
      vi.advanceTimersByTime(2000);
    });
  }
  expect(Socket.sockets).toHaveLength(5);
  act(() => vi.advanceTimersByTime(30000));
  expect(Socket.sockets).toHaveLength(5);
  expect(screen.getByText(/Could not reconnect/)).toBeTruthy();
  view.unmount();
  expect(vi.getTimerCount()).toBe(0);
});

it('does not reconnect or process retired events after unmount', () => {
  vi.useFakeTimers();
  const view = render(<IngestionDashboardLauncher wsUrl="ws://test/ingest/dashboard/" />);
  act(() => Socket.sockets[0].dispatchEvent(new Event('close')));
  view.unmount();
  act(() => {
    Socket.sockets[0].emit({ type: 'document.ingestion.start', documentId: 'stale', documentName: 'Stale' });
    vi.advanceTimersByTime(30000);
  });
  expect(Socket.sockets).toHaveLength(1);
  expect(vi.getTimerCount()).toBe(0);
});
