import React, { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';
import type { CollectionViewNavigationGuard, CollectionViewNavigationIntent } from './collectionViewTypes';
import SchemaModalShell from '../knowledgeGraph/SchemaModalShell';
import { buttonDangerClass, buttonSecondaryClass } from '../knowledgeGraph/schemaUiShared';

interface DirtyRegistration {
  isDirty: boolean;
  discard: () => void;
}

interface SchemaNavigationGuardContextValue {
  registerDirtyState: (registration: DirtyRegistration | null) => void;
  guardNavigation: CollectionViewNavigationGuard;
  requestSelectionChange: (next: () => void) => void;
}

const SchemaNavigationGuardContext = createContext<SchemaNavigationGuardContextValue | null>(null);

export function useSchemaNavigationGuard(): SchemaNavigationGuardContextValue {
  const value = useContext(SchemaNavigationGuardContext);
  if (!value) {
    throw new Error('useSchemaNavigationGuard must be used within CollectionSchemaNavigationGuard');
  }
  return value;
}

export interface CollectionSchemaNavigationGuardProps {
  children: React.ReactNode;
  onProceedNavigation?: (intent: CollectionViewNavigationIntent) => void;
}

const CollectionSchemaNavigationGuard: React.FC<CollectionSchemaNavigationGuardProps> = ({
  children,
  onProceedNavigation,
}) => {
  const registrationRef = useRef<DirtyRegistration | null>(null);
  const pendingActionRef = useRef<(() => void) | null>(null);
  const [dialogOpen, setDialogOpen] = useState(false);

  const registerDirtyState = useCallback((registration: DirtyRegistration | null) => {
    registrationRef.current = registration;
  }, []);

  const guardNavigation = useCallback<CollectionViewNavigationGuard>((intent) => {
    if (!registrationRef.current?.isDirty) return true;
    pendingActionRef.current = () => onProceedNavigation?.(intent);
    setDialogOpen(true);
    return false;
  }, [onProceedNavigation]);

  const requestSelectionChange = useCallback((next: () => void) => {
    if (!registrationRef.current?.isDirty) {
      next();
      return;
    }
    pendingActionRef.current = next;
    setDialogOpen(true);
  }, []);

  useEffect(() => {
    const handleBeforeUnload = (event: BeforeUnloadEvent) => {
      if (!registrationRef.current?.isDirty) return;
      event.preventDefault();
      event.returnValue = '';
    };
    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => window.removeEventListener('beforeunload', handleBeforeUnload);
  }, []);

  const value = useMemo(
    () => ({ registerDirtyState, guardNavigation, requestSelectionChange }),
    [guardNavigation, registerDirtyState, requestSelectionChange],
  );

  return (
    <SchemaNavigationGuardContext.Provider value={value}>
      {children}
      <SchemaModalShell
        isOpen={dialogOpen}
        title="Unsaved schema changes"
        onClose={() => {
          setDialogOpen(false);
          pendingActionRef.current = null;
        }}
        footer={
          <>
            <button
              type="button"
              className={buttonSecondaryClass}
              onClick={() => {
                setDialogOpen(false);
                pendingActionRef.current = null;
              }}
            >
              Keep editing
            </button>
            <button
              type="button"
              className={buttonDangerClass}
              onClick={() => {
                registrationRef.current?.discard();
                setDialogOpen(false);
                pendingActionRef.current?.();
                pendingActionRef.current = null;
              }}
            >
              Discard unsaved form
            </button>
          </>
        }
      >
        <p>You have unsaved schema edits. Discard them before leaving this definition or workspace mode?</p>
      </SchemaModalShell>
    </SchemaNavigationGuardContext.Provider>
  );
};

export default CollectionSchemaNavigationGuard;
