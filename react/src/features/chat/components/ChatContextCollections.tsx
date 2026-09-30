import { useState } from 'react';
import { ChevronDown, ChevronRight, Sparkles } from 'lucide-react';
import type { ChatContextCatalog } from '../types';

interface Props {
  catalog: ChatContextCatalog;
  selected: Set<string>;
  query: string;
  onToggle: (id: string) => void;
  onSkills: (id: string) => void;
}
export default function ChatContextCollections({ catalog, selected, query, onToggle, onSkills }: Props) {
  const [collapsed, setCollapsed] = useState(new Set<string>());
  const needle = query.trim().toLowerCase();
  const ids = new Set(catalog.collections.map(c => String(c.id)));
  const children = (id: string) => catalog.collections.filter(c => c.parent != null && String(c.parent) === id);
  const seen = new Set<string>();
  const rows: { collection: ChatContextCatalog['collections'][number]; depth: number }[] = [];
  const visit = (collection: ChatContextCatalog['collections'][number], depth: number) => {
    const id = String(collection.id);
    if (seen.has(id)) return;
    seen.add(id);
    rows.push({ collection, depth });
    if (!collapsed.has(id)) children(id).forEach(child => visit(child, depth + 1));
  };
  if (needle) catalog.collections.filter(c => `${c.name} ${c.path}`.toLowerCase().includes(needle))
    .forEach(collection => rows.push({ collection, depth: 0 }));
  else catalog.collections.filter(c => c.parent == null || !ids.has(String(c.parent))).forEach(c => visit(c, 0));

  if (!rows.length) return <p className="py-6 text-sm text-text-low_contrast">{needle ? 'No collections match your search.' : 'No collections are available.'}</p>;
  return <div className="space-y-1">{rows.map(({ collection, depth }) => {
    const id = String(collection.id);
    const count = catalog.skills_enabled ? catalog.skills.filter(s => s.default_collection_ids.includes(id)).length : 0;
    const hasChildren = children(id).length > 0;
    return <div key={id} style={{ marginLeft: `${Math.min(depth, 4) * 12}px` }}
      className={`flex min-w-0 flex-wrap items-center rounded-lg ${selected.has(id) ? 'bg-scheme-shade_4' : 'hover:bg-scheme-shade_3'} ${depth ? 'border-l border-border-mid_contrast' : ''}`}>
      {hasChildren && !needle && <button type="button" className="flex h-9 w-7 shrink-0 items-center justify-center rounded hover:bg-scheme-shade_5"
        aria-label={`${collapsed.has(id) ? 'Expand' : 'Collapse'} ${collection.name}`} aria-expanded={!collapsed.has(id)}
        onClick={() => setCollapsed(previous => { const next = new Set(previous); next.has(id) ? next.delete(id) : next.add(id); return next; })}>
        {collapsed.has(id) ? <ChevronRight size={15} /> : <ChevronDown size={15} />}
      </button>}
      <label className="flex min-w-0 flex-1 cursor-pointer items-start gap-2.5 px-2 py-3">
        <input type="checkbox" aria-label={collection.name} checked={selected.has(id)} onChange={() => onToggle(id)} className="mt-1 h-4 w-4 shrink-0 accent-accent" />
        <span className="min-w-0 break-words text-[13px]"><span className="block font-medium">{collection.name}</span>
          <span className="mt-1 block break-all text-xs text-text-low_contrast">{collection.path}{collection.is_skill_pack ? ' · Skill pack' : ''}</span>
        </span>
      </label>
      {count > 0 && <button type="button" onClick={() => onSkills(id)} aria-label={`Show ${collection.name} skills`}
        className="m-1 flex min-h-9 shrink-0 items-center gap-1 rounded bg-scheme-shade_2 px-2 text-xs hover:bg-scheme-shade_5"><Sparkles size={13} aria-hidden="true" />{count} {count === 1 ? 'skill' : 'skills'}</button>}
    </div>;
  })}</div>;
}
