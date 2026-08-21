// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { useEffect } from 'react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import CollectionSchemaNavigationGuard, {
  useSchemaNavigationGuard,
} from './CollectionSchemaNavigationGuard';

afterEach(() => cleanup());

function DirtyConsumer({ dirty }: { dirty: boolean }) {
  const { registerDirtyState } = useSchemaNavigationGuard();
  useEffect(() => {
    registerDirtyState({
      isDirty: dirty,
      discard: vi.fn(),
    });
    return () => registerDirtyState(null);
  }, [dirty, registerDirtyState]);
  return null;
}

describe('CollectionSchemaNavigationGuard', () => {
  it('allows navigation when the form buffer is clean', () => {
    const onProceedNavigation = vi.fn();
    let guardResult = true;

    function Probe() {
      const { guardNavigation } = useSchemaNavigationGuard();
      guardResult = guardNavigation({ type: 'mode', mode: 'files' });
      return null;
    }

    render(
      <CollectionSchemaNavigationGuard onProceedNavigation={onProceedNavigation}>
        <DirtyConsumer dirty={false} />
        <Probe />
      </CollectionSchemaNavigationGuard>,
    );

    expect(guardResult).toBe(true);
  });

  it('blocks navigation and shows discard dialog while dirty', () => {
    const onProceedNavigation = vi.fn();

    function Trigger() {
      const { guardNavigation, registerDirtyState } = useSchemaNavigationGuard();
      useEffect(() => {
        registerDirtyState({ isDirty: true, discard: vi.fn() });
        guardNavigation({ type: 'mode', mode: 'files' });
      }, [guardNavigation, registerDirtyState]);
      return null;
    }

    render(
      <CollectionSchemaNavigationGuard onProceedNavigation={onProceedNavigation}>
        <Trigger />
      </CollectionSchemaNavigationGuard>,
    );

    expect(screen.getByRole('dialog', { name: 'Unsaved schema changes' })).toBeTruthy();
  });

  it('registers beforeunload while dirty', () => {
    const preventDefault = vi.fn();
    render(
      <CollectionSchemaNavigationGuard>
        <DirtyConsumer dirty={true} />
      </CollectionSchemaNavigationGuard>,
    );

    const event = new Event('beforeunload') as BeforeUnloadEvent;
    Object.defineProperty(event, 'preventDefault', { value: preventDefault });
    window.dispatchEvent(event);
    expect(preventDefault).toHaveBeenCalled();
  });

  it('proceeds with blocked navigation after discarding unsaved form', () => {
    const discard = vi.fn();
    const onProceedNavigation = vi.fn();

    function DirtyRegister() {
      const { registerDirtyState } = useSchemaNavigationGuard();
      useEffect(() => {
        registerDirtyState({ isDirty: true, discard });
        return () => registerDirtyState(null);
      }, [registerDirtyState]);
      return null;
    }

    function Trigger() {
      const { guardNavigation } = useSchemaNavigationGuard();
      return (
        <button type="button" onClick={() => guardNavigation({ type: 'mode', mode: 'files' })}>
          Leave
        </button>
      );
    }

    render(
      <CollectionSchemaNavigationGuard onProceedNavigation={onProceedNavigation}>
        <DirtyRegister />
        <Trigger />
      </CollectionSchemaNavigationGuard>,
    );

    fireEvent.click(screen.getByRole('button', { name: 'Leave' }));
    fireEvent.click(screen.getByRole('button', { name: 'Discard unsaved form' }));
    expect(discard).toHaveBeenCalled();
    expect(onProceedNavigation).toHaveBeenCalledWith({ type: 'mode', mode: 'files' });
  });
});
