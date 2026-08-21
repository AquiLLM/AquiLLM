// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import CollectionModeNav from './CollectionModeNav';
import type { CollectionViewMode } from './collectionViewTypes';

function setLocation(pathname: string, search = '', hash = '') {
  window.history.replaceState(null, '', `${pathname}${search}${hash}`);
}

beforeEach(() => {
  setLocation('/collections/7', '', '');
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe('CollectionModeNav', () => {
  it('shows Files selected by default', () => {
    const onActiveModeChange = vi.fn();
    render(
      <CollectionModeNav activeMode="files" onActiveModeChange={onActiveModeChange} />
    );

    expect(screen.getByRole('tab', { name: 'Files' }).getAttribute('aria-selected')).toBe('true');
    expect(screen.getByRole('tab', { name: 'Knowledge Graph' }).getAttribute('aria-selected')).toBe(
      'false'
    );
  });

  it('deep-links to knowledge graph when view query param is present', () => {
    setLocation('/collections/7', '?view=knowledge-graph');
    const onActiveModeChange = vi.fn();
    render(
      <CollectionModeNav activeMode="knowledge-graph" onActiveModeChange={onActiveModeChange} />
    );

    expect(screen.getByRole('tab', { name: 'Knowledge Graph' }).getAttribute('aria-selected')).toBe(
      'true'
    );
  });

  it('falls back to Files for invalid view values via parent state', () => {
    setLocation('/collections/7', '?view=invalid-mode');
    render(<CollectionModeNav activeMode="files" onActiveModeChange={vi.fn()} />);
    expect(screen.getByRole('tab', { name: 'Files' }).getAttribute('aria-selected')).toBe('true');
  });

  it('updates URL with pushState when switching modes', () => {
    const pushState = vi.spyOn(window.history, 'pushState');
    const onActiveModeChange = vi.fn();
    setLocation('/collections/7', '?tab=details', '');

    render(
      <CollectionModeNav activeMode="files" onActiveModeChange={onActiveModeChange} />
    );

    fireEvent.click(screen.getByRole('tab', { name: 'Knowledge Graph' }));

    expect(pushState).toHaveBeenCalledWith(null, '', '/collections/7?tab=details&view=knowledge-graph');
    expect(onActiveModeChange).toHaveBeenCalledWith('knowledge-graph');
  });

  it('preserves unrelated query params when returning to Files', () => {
    const pushState = vi.spyOn(window.history, 'pushState');
    const onActiveModeChange = vi.fn();
    setLocation('/collections/7', '?tab=details&view=knowledge-graph');

    render(
      <CollectionModeNav activeMode="knowledge-graph" onActiveModeChange={onActiveModeChange} />
    );

    fireEvent.click(screen.getByRole('tab', { name: 'Files' }));

    expect(pushState).toHaveBeenCalledWith(null, '', '/collections/7?tab=details');
    expect(onActiveModeChange).toHaveBeenCalledWith('files');
  });

  it('responds to browser back and forward via popstate', () => {
    const onActiveModeChange = vi.fn();
    render(
      <CollectionModeNav activeMode="files" onActiveModeChange={onActiveModeChange} />
    );

    setLocation('/collections/7', '?view=knowledge-graph');
    window.dispatchEvent(new PopStateEvent('popstate'));

    expect(onActiveModeChange).toHaveBeenCalledWith('knowledge-graph');
  });

  it('blocks navigation when guardNavigation returns false', () => {
    const pushState = vi.spyOn(window.history, 'pushState');
    const onActiveModeChange = vi.fn();
    const guardNavigation = vi.fn(() => false);

    render(
      <CollectionModeNav
        activeMode="files"
        onActiveModeChange={onActiveModeChange}
        guardNavigation={guardNavigation}
      />
    );

    fireEvent.click(screen.getByRole('tab', { name: 'Knowledge Graph' }));

    expect(guardNavigation).toHaveBeenCalledWith({ type: 'mode', mode: 'knowledge-graph' });
    expect(pushState).not.toHaveBeenCalled();
    expect(onActiveModeChange).not.toHaveBeenCalled();
  });

  it('reverts browser navigation when guardNavigation rejects popstate', () => {
    const pushState = vi.spyOn(window.history, 'pushState');
    const onActiveModeChange = vi.fn();
    const guardNavigation = vi.fn(({ mode }: { mode: CollectionViewMode }) => mode !== 'knowledge-graph');

    render(
      <CollectionModeNav
        activeMode="files"
        onActiveModeChange={onActiveModeChange}
        guardNavigation={guardNavigation}
      />
    );

    setLocation('/collections/7', '?view=knowledge-graph');
    window.dispatchEvent(new PopStateEvent('popstate'));

    expect(onActiveModeChange).not.toHaveBeenCalled();
    expect(pushState).toHaveBeenCalledWith(null, '', '/collections/7');
  });
});
