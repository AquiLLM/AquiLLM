import React, { useEffect, useId, useRef, useState } from 'react';
import { getCookie } from '../../../utils/csrf';

interface RenameTarget {
  id: string;
  name: string;
  url: string;
  returnFocus: HTMLElement | null;
}

// The history menu is rendered by Django; one shared dialog handles its buttons.
export default function ChatRenameDialog() {
  const [target, setTarget] = useState<RenameTarget | null>(null);

  useEffect(() => {
    const open = (event: MouseEvent) => {
      const button = event.target instanceof Element
        ? event.target.closest<HTMLElement>('[data-rename-conversation]') : null;
      const id = button?.dataset.renameConversation;
      const url = button?.dataset.renameUrl;
      if (!id || !/^\d+$/.test(id) || !url) return;
      event.preventDefault();
      const name = document.querySelector(`[data-conversation-title="${id}"]`)?.textContent?.trim() ?? '';
      setTarget(current => current ?? {
        id, name, url,
        returnFocus: document.getElementById(`conversation-options-${id}`) ?? button,
      });
    };
    document.addEventListener('click', open);
    return () => document.removeEventListener('click', open);
  }, []);

  return target ? <RenameForm target={target} onClose={() => setTarget(null)} /> : null;
}

function RenameForm({ target, onClose }: { target: RenameTarget; onClose: () => void }) {
  const [name, setName] = useState(target.name);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const input = useRef<HTMLInputElement>(null);
  const dialog = useRef<HTMLDivElement>(null);
  const pending = useRef(false);
  const mounted = useRef(true);
  const request = useRef<AbortController | null>(null);
  const id = useId();

  useEffect(() => {
    mounted.current = true;
    const overflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    input.current?.focus();
    input.current?.select();
    return () => {
      mounted.current = false;
      request.current?.abort();
      document.body.style.overflow = overflow;
      target.returnFocus?.focus();
    };
  }, [target]);

  useEffect(() => {
    if (error && !saving) input.current?.focus();
  }, [error, saving]);

  const close = () => { if (!pending.current) onClose(); };
  const save = async (event: React.FormEvent) => {
    event.preventDefault();
    if (pending.current) return;
    const trimmed = name.trim();
    if (!trimmed || [...trimmed].length > 200) {
      setError('Enter a chat name between 1 and 200 characters.');
      input.current?.focus();
      return;
    }
    pending.current = true;
    // Keep keyboard events inside the dialog while all controls are disabled.
    dialog.current?.focus();
    setSaving(true);
    setError('');
    const controller = new AbortController();
    request.current = controller;
    const timeout = window.setTimeout(() => controller.abort(), 15000);
    try {
      const csrf = document.querySelector<HTMLInputElement>('[name=csrfmiddlewaretoken]')?.value || getCookie('csrftoken');
      const response = await fetch(target.url, {
        method: 'POST', credentials: 'same-origin', signal: controller.signal,
        headers: { 'Content-Type': 'application/json', 'X-CSRFToken': csrf },
        body: JSON.stringify({ name: trimmed }),
      });
      if (response.redirected || response.status === 401 || response.status === 403) {
        throw new Error('Your session could not be verified. Refresh the page and try again.');
      }
      const result = await response.json();
      if (!response.ok) throw new Error(typeof result.error === 'string' ? result.error : 'Could not rename this chat. Please try again.');
      if (String(result.id) !== target.id || typeof result.name !== 'string') throw new Error('Could not confirm the new name. Please try again.');
      if (!mounted.current) return;
      document.querySelectorAll(`[data-conversation-title="${target.id}"]`).forEach(element => { element.textContent = result.name; });
      document.querySelectorAll(`[data-conversation-options="${target.id}"]`).forEach(element => { element.setAttribute('aria-label', `Options for ${result.name}`); });
      onClose();
    } catch (failure) {
      if (mounted.current) setError(controller.signal.aborted
        ? 'Saving took too long. Please try again.'
        : failure instanceof Error && !(failure instanceof TypeError || failure instanceof SyntaxError)
          ? failure.message : 'Could not rename this chat. Please try again.');
    } finally {
      window.clearTimeout(timeout);
      pending.current = false;
      if (mounted.current) setSaving(false);
    }
  };

  return <div className="fixed inset-0 z-[120] flex items-center justify-center bg-black/50 p-4"
    onClick={event => { if (event.target === event.currentTarget) close(); }}>
    <div ref={dialog} role="dialog" aria-modal="true" aria-labelledby={`${id}-title`} tabIndex={-1}
      className="w-full max-w-[420px] rounded-xl border border-border-mid_contrast bg-scheme-shade_3 p-5 text-text-normal shadow-xl"
      onKeyDown={event => {
        if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); close(); }
        if (event.key !== 'Tab') return;
        const items = [...(dialog.current?.querySelectorAll<HTMLElement>('input:not(:disabled),button:not(:disabled)') ?? [])];
        const first = items[0], last = items[items.length - 1];
        if (!first) { event.preventDefault(); dialog.current?.focus(); }
        else if (event.shiftKey && (document.activeElement === first || document.activeElement === dialog.current)) { event.preventDefault(); last.focus(); }
        else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
      }}>
      <h2 id={`${id}-title`} className="mb-4 text-lg font-semibold">Rename chat</h2>
      <form onSubmit={save} noValidate aria-busy={saving}>
        <label htmlFor={`${id}-name`} className="mb-2 block text-sm font-medium">Chat name</label>
        <input ref={input} id={`${id}-name`} type="text" value={name} disabled={saving} autoComplete="off"
          aria-invalid={!!error} aria-describedby={error ? `${id}-error` : undefined}
          onChange={event => { setName(event.target.value); setError(''); }}
          className="h-11 w-full min-w-0 rounded-lg border border-border-high_contrast bg-scheme-shade_5 px-3 text-base outline-none focus:ring-2 focus:ring-accent disabled:opacity-60" />
        {error && <p id={`${id}-error`} role="alert" className="mt-2 text-sm">{error}</p>}
        <div className="mt-5 flex justify-end gap-2">
          <button type="button" disabled={saving} onClick={close}
            className="min-h-10 rounded-lg border border-border-mid_contrast px-4 text-sm hover:bg-scheme-shade_5 disabled:opacity-50">Cancel</button>
          <button type="submit" disabled={saving}
            className="min-h-10 rounded-lg border border-border-high_contrast bg-accent px-4 text-sm font-medium hover:bg-accent-light disabled:opacity-50">{saving ? 'Saving…' : 'Save'}</button>
        </div>
      </form>
    </div>
  </div>;
}
