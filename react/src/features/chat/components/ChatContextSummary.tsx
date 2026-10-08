import { Folder, Sparkles, X } from 'lucide-react';
import type { ContextCollection, ContextSkill } from '../types';

interface Props {
  collections: ContextCollection[];
  skills: ContextSkill[];
  onRemoveCollection: (id: string) => void;
  onDisableSkill: (id: string) => void;
  onClear: () => void;
}
export default function ChatContextSummary({ collections, skills, onRemoveCollection, onDisableSkill, onClear }: Props) {
  return <aside aria-label="Selected chat context" className="min-w-0 border-t border-border-mid_contrast p-4 sm:border-l sm:border-t-0">
    <div className="mb-5 flex items-center justify-between gap-2"><h3 className="text-[13px] font-semibold">In this chat</h3>
      <button type="button" onClick={onClear} className="min-h-8 text-xs underline underline-offset-4">Clear all</button></div>
    <div className="grid grid-cols-2 gap-4 sm:grid-cols-1 sm:gap-6">
      <section className="min-w-0"><h4 className="mb-3 flex items-center gap-1.5 text-xs text-text-low_contrast"><Folder size={15} />Collections · {collections.length}</h4>
        {!collections.length && <p className="text-xs text-text-low_contrast">None selected</p>}
        {collections.map(collection => <div key={collection.id} className="mb-3 flex items-start gap-1">
          <span className="min-w-0 flex-1 break-words text-xs">{collection.name}<small className="mt-1 block break-words text-[11px] text-text-low_contrast">{collection.path}</small></span>
          <button type="button" aria-label={`Remove ${collection.name}`} onClick={() => onRemoveCollection(String(collection.id))} className="-mt-1 flex h-8 w-8 shrink-0 items-center justify-center rounded hover:bg-scheme-shade_5"><X size={13} /></button>
        </div>)}
      </section>
      <section className="min-w-0"><h4 className="mb-3 flex items-center gap-1.5 text-xs text-text-low_contrast"><Sparkles size={15} />Skills · {skills.length}</h4>
        {!skills.length && <p className="text-xs text-text-low_contrast">None enabled</p>}
        {skills.map(skill => <div key={skill.id} className="mb-3 flex items-start gap-1">
          <span className="min-w-0 flex-1 break-words text-xs">{skill.name}<small className="mt-1 block break-words text-[11px] text-text-low_contrast">From {skill.collection_path}</small></span>
          <button type="button" aria-label={`Disable ${skill.name}`} onClick={() => onDisableSkill(skill.id)} className="-mt-1 flex h-8 w-8 shrink-0 items-center justify-center rounded hover:bg-scheme-shade_5"><X size={13} /></button>
        </div>)}
      </section>
    </div>
  </aside>;
}
