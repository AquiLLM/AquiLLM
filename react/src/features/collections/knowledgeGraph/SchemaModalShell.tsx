import React, { useEffect, useId, useRef } from 'react';

const FOCUSABLE =
  'a[href],button:not([disabled]),textarea:not([disabled]),input:not([disabled]),select:not([disabled]),[tabindex]:not([tabindex="-1"])';

export interface SchemaModalShellProps {
  isOpen: boolean;
  title: string;
  onClose: () => void;
  children: React.ReactNode;
  footer?: React.ReactNode;
  allowEscape?: boolean;
  initialFocusSelector?: string;
}

const SchemaModalShell: React.FC<SchemaModalShellProps> = ({
  isOpen,
  title,
  onClose,
  children,
  footer,
  allowEscape = true,
  initialFocusSelector,
}) => {
  const titleId = useId();
  const dialogRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (!isOpen) return;
    triggerRef.current = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const dialog = dialogRef.current;
    if (!dialog) return;

    const focusTarget =
      (initialFocusSelector ? dialog.querySelector<HTMLElement>(initialFocusSelector) : null) ??
      dialog.querySelector<HTMLElement>(FOCUSABLE);
    focusTarget?.focus();

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && allowEscape) {
        event.preventDefault();
        onClose();
        return;
      }
      if (event.key !== 'Tab' || !dialog) return;
      const nodes = [...dialog.querySelectorAll<HTMLElement>(FOCUSABLE)].filter(
        (node) => !node.hasAttribute('disabled') && node.tabIndex !== -1,
      );
      if (nodes.length === 0) return;
      const first = nodes[0];
      const last = nodes[nodes.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      triggerRef.current?.focus();
    };
  }, [allowEscape, initialFocusSelector, isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-[120] flex items-center justify-center bg-black/70 backdrop-blur-[8px] p-4">
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        className="w-full max-w-[640px] rounded-[24px] border border-border-high_contrast bg-scheme-shade_3 p-[20px] text-text-normal shadow-lg"
      >
        <h2 id={titleId} className="text-xl font-semibold mb-3">
          {title}
        </h2>
        <div>{children}</div>
        {footer ? <div className="mt-4 flex flex-wrap gap-2 justify-end">{footer}</div> : null}
      </div>
    </div>
  );
};

export default SchemaModalShell;
