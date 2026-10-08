import { useState } from 'react';
import { ChevronDown, ChevronUp, Folder, Layers } from 'lucide-react';
import type { ContextSkill, SkillOverrides } from '../types';
import { skillEnabled } from '../utils/contextSelection';

interface Props {
  skills: ContextSkill[];
  selected: Set<string>;
  overrides: SkillOverrides;
  query: string;
  associatedCollection: string | null;
  onOverrides: (overrides: SkillOverrides) => void;
}
export default function ChatContextSkills({ skills, selected, overrides, query, associatedCollection, onOverrides }: Props) {
  const [expanded, setExpanded] = useState(new Set<string>());
  const groups = new Map<string, ContextSkill[]>();
  skills.forEach(skill => {
    const key = skill.pack_id ? `pack:${skill.pack_id}` : `collection:${skill.collection_id}`;
    groups.set(key, [...(groups.get(key) ?? []), skill]);
  });
  const needle = query.trim().toLowerCase();
  const matches = (skill: ContextSkill) => (!associatedCollection || skill.default_collection_ids.includes(associatedCollection)) &&
    `${skill.name} ${skill.description} ${skill.source_path} ${skill.collection_path} ${skill.pack_name ?? ''}`.toLowerCase().includes(needle);
  if (!skills.some(matches)) return <p className="py-6 text-sm text-text-low_contrast">{needle || associatedCollection ? 'No skills match your search.' : 'No collection skills are available.'}</p>;
  return <div className="space-y-5">{[...groups].map(([key, source]) => {
    const visible = source.filter(matches);
    if (!visible.length) return null;
    const first = source[0];
    const allEnabled = source.every(skill => skillEnabled(skill, selected, overrides));
    return <section key={key} aria-label={`Skills from ${first.collection_path}`}>
      <div className="mb-2 flex flex-wrap items-start justify-between gap-2 px-1">
        <div className="flex min-w-0 flex-1 gap-2">{first.pack_id ? <Layers size={17} className="mt-0.5 shrink-0" /> : <Folder size={17} className="mt-0.5 shrink-0" />}
          <div className="min-w-0"><h3 className="break-words text-[13px] font-semibold">{first.pack_id ? `Skill pack: ${first.pack_name}` : `From ${first.collection_name}`}</h3>
            <p className="mt-0.5 break-all text-[11px] text-text-low_contrast">{first.collection_path} · {source.length} {source.length === 1 ? 'skill' : 'skills'}</p>
          </div>
        </div>
        <button type="button" className="min-h-8 text-xs underline underline-offset-4" aria-label={`${allEnabled ? 'Disable' : 'Enable'} all from ${first.collection_path}`}
          onClick={() => onOverrides({ ...overrides, ...Object.fromEntries(source.map(skill => [skill.id, !allEnabled])) })}>{allEnabled ? 'Disable all' : 'Enable all'}</button>
      </div>
      {visible.map(skill => {
        const enabled = skillEnabled(skill, selected, overrides);
        const explicit = Object.prototype.hasOwnProperty.call(overrides, skill.id);
        return <div key={skill.id} className={`mb-2 overflow-hidden rounded-lg border border-border-mid_contrast ${enabled ? 'bg-scheme-shade_4' : 'bg-scheme-shade_2'}`}>
          <div className="flex items-start gap-1 p-3">
            <label className="flex min-w-0 flex-1 cursor-pointer items-start gap-2.5">
              <input type="checkbox" aria-label={skill.name} checked={enabled} onChange={event => onOverrides({ ...overrides, [skill.id]: event.target.checked })} className="mt-1 h-4 w-4 shrink-0 accent-accent" />
              <span className="min-w-0 break-words"><span className="block text-[13px] font-medium">{skill.name}</span>
                <span className="mt-1 block text-xs text-text-low_contrast">{skill.description}</span>
                <span className="mt-2 block text-[11px]">{explicit ? enabled ? 'Enabled for this chat' : 'Off for this chat' : enabled ? 'Included with collection selection' : 'Available from collection'}</span>
              </span>
            </label>
            <button type="button" aria-label={`Read ${skill.name} instructions`} aria-expanded={expanded.has(skill.id)}
              className="-mt-1 flex h-8 w-8 shrink-0 items-center justify-center rounded hover:bg-scheme-shade_5"
              onClick={() => setExpanded(previous => { const next = new Set(previous); next.has(skill.id) ? next.delete(skill.id) : next.add(skill.id); return next; })}>
              {expanded.has(skill.id) ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
            </button>
          </div>
          {expanded.has(skill.id) && <div className="border-t border-border-mid_contrast bg-scheme-shade_3 p-3 text-xs">
            <h4 className="mb-1 font-semibold text-text-low_contrast">Instructions</h4>
            <p className="mb-3 whitespace-pre-wrap break-words">{skill.instructions}</p>
            <h4 className="mb-1 font-semibold text-text-low_contrast">Source</h4>
            <p className="break-all font-mono text-[11px] text-text-low_contrast">{skill.source_path}</p>
            {explicit && <button type="button" className="mt-2 min-h-8 underline underline-offset-4" onClick={() => {
              const next = { ...overrides }; delete next[skill.id]; onOverrides(next);
            }}>Follow collection selection</button>}
          </div>}
        </div>;
      })}
    </section>;
  })}</div>;
}
