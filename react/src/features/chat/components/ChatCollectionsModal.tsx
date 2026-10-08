import { useEffect, useId, useRef, useState } from 'react';
import { Folder, Search, Sparkles, X } from 'lucide-react';
import type { ChatContextCatalog, ContextSelection, SkillOverrides } from '../types';
import { descendantIds, skillEnabled } from '../utils/contextSelection';
import ChatContextCollections from './ChatContextCollections';
import ChatContextSkills from './ChatContextSkills';
import ChatContextSummary from './ChatContextSummary';

export interface ChatCollectionsModalProps {
  open: boolean;
  onClose: () => void;
  catalog: ChatContextCatalog;
  selectedCollections: Set<string>;
  skillOverrides: SkillOverrides;
  onApply: (selection: ContextSelection) => Promise<void>;
  loading?: boolean;
  error?: string;
  onRetry?: () => void;
  applyDisabled?: boolean;
}

// Mount a fresh draft per opening. Catalog retries do not overwrite unsaved edits.
export default function ChatCollectionsModal(props: ChatCollectionsModalProps) {
  return props.open ? <ContextDialog {...props} /> : null;
}

function ContextDialog({ onClose, catalog, selectedCollections, skillOverrides, onApply, loading = false, error = '', onRetry, applyDisabled = false }: ChatCollectionsModalProps) {
  const [selected, setSelected] = useState(() => new Set(selectedCollections));
  const [overrides, setOverrides] = useState(() => ({ ...skillOverrides }));
  const [tab, setTab] = useState<'collections' | 'skills'>('collections');
  const [query, setQuery] = useState('');
  const [associatedCollection, setAssociatedCollection] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState('');
  const dialog = useRef<HTMLDivElement>(null);
  const mounted = useRef(true);
  const id = useId();
  const skills = catalog.skills_enabled ? catalog.skills : [];
  const enabledSkills = skills.filter(skill => skillEnabled(skill, selected, overrides));
  const selectedAvailable = catalog.collections.filter(collection => selected.has(String(collection.id)));

  useEffect(() => {
    mounted.current = true;
    const opener = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const overflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    dialog.current?.querySelector<HTMLInputElement>('input[type="search"]')?.focus();
    return () => { mounted.current = false; document.body.style.overflow = overflow; opener?.focus(); };
  }, []);

  const switchTab = (next: 'collections' | 'skills') => { setTab(next); setQuery(''); setAssociatedCollection(null); };
  const toggleCollection = (collectionId: string, remove = false) => setSelected(previous => {
    const next = new Set(previous);
    const deselect = remove || next.has(collectionId);
    descendantIds(catalog.collections, collectionId).forEach(child => deselect ? next.delete(child) : next.add(child));
    return next;
  });
  const cancel = () => { if (!saving) onClose(); };
  const apply = async () => {
    if (saving || applyDisabled || loading || error) return;
    setSaving(true); setSaveError('');
    try {
      await onApply({ selectedCollections: new Set(selected), skillOverrides: { ...overrides } });
      if (mounted.current) onClose();
    } catch (failure) {
      if (mounted.current) setSaveError(failure instanceof Error ? failure.message : 'Could not save context. Please try again.');
    } finally { if (mounted.current) setSaving(false); }
  };
  return <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/50 p-2 sm:p-5" onClick={event => { if (event.target === event.currentTarget) cancel(); }}>
    <div ref={dialog} role="dialog" aria-modal="true" aria-labelledby={`${id}-title`} aria-describedby={`${id}-description`} tabIndex={-1}
      className="flex max-h-[94dvh] w-full max-w-[740px] flex-col overflow-hidden rounded-xl border border-border-mid_contrast bg-scheme-shade_3 text-text-normal shadow-xl"
      onKeyDown={event => {
        if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); cancel(); }
        if (event.key !== 'Tab') return;
        const items = [...(dialog.current?.querySelectorAll<HTMLElement>('button:not(:disabled), input:not(:disabled), [tabindex="0"]') ?? [])].filter(element => element.tabIndex >= 0);
        const first = items[0], last = items[items.length - 1];
        if (!first) { event.preventDefault(); dialog.current?.focus(); }
        else if (event.shiftKey && (document.activeElement === first || document.activeElement === dialog.current)) { event.preventDefault(); last.focus(); }
        else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
      }}>
      <header className="flex shrink-0 items-start justify-between gap-3 px-4 py-5 sm:px-6">
        <div><h2 id={`${id}-title`} className="text-xl font-semibold tracking-tight">Set up this chat</h2><p id={`${id}-description`} className="mt-1 text-[13px] text-text-low_contrast">Choose reference collections and response skills.</p></div>
        <button type="button" aria-label="Close picker" disabled={saving} onClick={cancel} className="flex h-9 w-9 shrink-0 items-center justify-center rounded hover:bg-scheme-shade_5 disabled:opacity-50"><X size={18} /></button>
      </header>
      <div className="min-h-0 overflow-y-auto border-t border-border-mid_contrast">
      <fieldset disabled={saving} className="min-w-0">
        <div className="grid min-w-0 grid-cols-1 sm:grid-cols-[minmax(0,1fr)_204px]">
          <div className="min-w-0 bg-scheme-shade_2 px-3 pb-5 sm:px-5">
            <div role="tablist" aria-label="Chat context" className="mb-4 flex gap-5 border-b border-border-mid_contrast">
              {(['collections', 'skills'] as const).map(value => <button key={value} type="button" role="tab" id={`${id}-${value}`} aria-controls={`${id}-panel`} aria-selected={tab === value} tabIndex={tab === value ? 0 : -1}
                className={`flex min-w-0 items-center gap-1.5 border-b-2 py-3 text-[13px] font-medium ${tab === value ? 'border-accent-dark text-text-normal' : 'border-transparent text-text-low_contrast'}`}
                onClick={() => switchTab(value)} onKeyDown={event => {
                  if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
                  event.preventDefault();
                  const next = event.key === 'Home' ? 'collections' : event.key === 'End' ? 'skills' : value === 'collections' ? 'skills' : 'collections';
                  switchTab(next); document.getElementById(`${id}-${next}`)?.focus();
                }}>
                {value === 'collections' ? <Folder size={16} /> : <Sparkles size={16} />}{value === 'collections' ? 'Collections' : 'Skills'}
                <span className="rounded bg-scheme-shade_4 px-1.5 text-[11px]">{value === 'collections' ? selectedAvailable.length : enabledSkills.length}</span>
              </button>)}
            </div>
            <div role="tabpanel" id={`${id}-panel`} aria-labelledby={`${id}-${tab}`}>
              <label className="flex items-center gap-2 rounded-lg border border-border-high_contrast bg-scheme-shade_3 px-2.5"><Search size={17} className="shrink-0 text-text-low_contrast" />
                <input type="search" aria-label={tab === 'collections' ? 'Search collections' : 'Search skills'} value={query} onChange={event => setQuery(event.target.value)}
                  placeholder={tab === 'collections' ? 'Find a collection…' : 'Find a skill or source…'} className="h-10 w-full min-w-0 border-0 bg-transparent text-base outline-none placeholder:text-text-low_contrast sm:text-sm" />
              </label>
              <p className="mb-4 mt-2 text-xs text-text-low_contrast">{tab === 'collections' ? 'Collections provide reference material. Included skills appear in Skills.' : 'Skills guide how the assistant responds in this chat.'}</p>
              {associatedCollection && tab === 'skills' && <button type="button" onClick={() => setAssociatedCollection(null)} className="mb-3 text-xs underline">Show all sources</button>}
              {loading ? <p role="status" className="py-5 text-sm">Loading collections and skills…</p> : error ? <div role="alert" className="py-4 text-sm"><p>{error}</p><button type="button" onClick={onRetry} className="mt-2 underline">Retry collections</button></div>
                : tab === 'collections' ? <ChatContextCollections catalog={catalog} selected={selected} query={query} onToggle={toggleCollection} onSkills={collectionId => { setTab('skills'); setQuery(''); setAssociatedCollection(collectionId); }} />
                : !catalog.skills_enabled ? <p className="py-5 text-sm text-text-low_contrast">Collection skills are disabled on this server. Your saved preferences are preserved.</p>
                : <ChatContextSkills skills={skills} selected={selected} overrides={overrides} query={query} associatedCollection={associatedCollection} onOverrides={setOverrides} />}
            </div>
          </div>
          <ChatContextSummary collections={selectedAvailable} skills={enabledSkills} onRemoveCollection={collectionId => toggleCollection(collectionId, true)} onDisableSkill={skillId => setOverrides(previous => ({ ...previous, [skillId]: false }))}
            onClear={() => { setSelected(new Set()); setOverrides(Object.fromEntries(skills.map(skill => [skill.id, false]))); }} />
        </div>
      </fieldset>
      </div>
      <footer className="shrink-0 border-t border-border-mid_contrast bg-scheme-shade_3 px-4 py-3.5 sm:px-5">
        {saveError && <p role="alert" className="mb-3 text-sm">{saveError}</p>}
        {applyDisabled && !saving && <p className="mb-2 text-xs text-text-low_contrast">Wait for the chat to be ready before applying changes.</p>}
        <div className="flex flex-wrap items-center justify-between gap-3"><span className="text-xs text-text-low_contrast" aria-live="polite">{selectedAvailable.length} collections · {enabledSkills.length} skills</span>
          <div className="flex gap-2"><button type="button" disabled={saving} onClick={cancel} className="min-h-10 rounded-lg border border-border-mid_contrast px-3 text-sm disabled:opacity-50">Cancel</button>
            <button type="button" disabled={saving || applyDisabled || loading || !!error} onClick={() => void apply()} className="min-h-10 rounded-lg border border-border-high_contrast bg-accent px-3 text-sm font-medium text-text-normal hover:bg-accent-light disabled:opacity-50">{saving ? 'Applying…' : 'Apply to chat'}</button></div>
        </div>
      </footer>
    </div>
  </div>;
}
