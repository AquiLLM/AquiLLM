import React, { Suspense, useCallback, useEffect, useMemo, useState } from 'react';
import {
  createSkill,
  deleteSkill,
  listSkills,
  updateSkill,
} from '../api/skillsApi';
import type { Skill } from '../types';

const SkillEditor = React.lazy(() => import('./SkillEditor'));

type Draft = {
  name: string;
  body: string;
  enabled: boolean;
};

function toDraft(s: Skill): Draft {
  return { name: s.name, body: s.body, enabled: s.enabled };
}

function isDirty(a: Draft, b: Draft): boolean {
  return a.name !== b.name || a.body !== b.body || a.enabled !== b.enabled;
}

export default function SkillsPage() {
  const [skills, setSkills] = useState<Skill[]>([]);
  const [skillsEnabled, setSkillsEnabled] = useState<boolean>(true);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [draft, setDraft] = useState<Draft | null>(null);
  const [original, setOriginal] = useState<Draft | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const refresh = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await listSkills();
      setSkills(data.skills);
      setSkillsEnabled(data.skills_enabled);
      if (selectedId == null && data.skills.length > 0) {
        setSelectedId(data.skills[0].id);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [selectedId]);

  useEffect(() => {
    void refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const selected = useMemo(
    () => skills.find(s => s.id === selectedId) ?? null,
    [skills, selectedId],
  );

  useEffect(() => {
    if (selected) {
      const d = toDraft(selected);
      setDraft(d);
      setOriginal(d);
    } else {
      setDraft(null);
      setOriginal(null);
    }
  }, [selected]);

  const dirty = !!(draft && original && isDirty(draft, original));

  const handleSelect = (id: number) => {
    if (dirty && !confirm('Discard unsaved changes?')) return;
    setSelectedId(id);
  };

  const handleNew = async () => {
    if (dirty && !confirm('Discard unsaved changes?')) return;
    const defaultName = (() => {
      const names = new Set(skills.map(s => s.name));
      let n = 1;
      while (names.has(`Untitled skill ${n}`)) n += 1;
      return `Untitled skill ${n}`;
    })();
    try {
      setBusy(true);
      setError(null);
      const created = await createSkill({ name: defaultName, body: '' });
      setSkills(prev => [...prev, created].sort((a, b) => a.name.localeCompare(b.name)));
      setSelectedId(created.id);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const handleSave = async () => {
    if (!selected || !draft) return;
    try {
      setBusy(true);
      setError(null);
      const updated = await updateSkill(selected.id, draft);
      setSkills(prev =>
        prev
          .map(s => (s.id === updated.id ? updated : s))
          .sort((a, b) => a.name.localeCompare(b.name)),
      );
      setOriginal(toDraft(updated));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const handleDiscard = () => {
    if (original) setDraft(original);
  };

  const handleDelete = async () => {
    if (!selected) return;
    if (!confirm(`Delete "${selected.name}"? This cannot be undone.`)) return;
    try {
      setBusy(true);
      setError(null);
      await deleteSkill(selected.id);
      const remaining = skills.filter(s => s.id !== selected.id);
      setSkills(remaining);
      setSelectedId(remaining.length > 0 ? remaining[0].id : null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex h-full w-full bg-scheme-shade_2 text-text-normal">
      <aside className="w-[260px] min-w-[260px] flex flex-col border-r border-border-mid_contrast bg-scheme-shade_3">
        <div className="flex items-center justify-between p-3 border-b border-border-low_contrast">
          <span className="font-bold">Skills</span>
          <button
            type="button"
            onClick={handleNew}
            disabled={busy}
            className="px-2 py-1 text-sm bg-accent hover:bg-accent-dark rounded disabled:opacity-50"
            title="New skill"
          >
            + New
          </button>
        </div>
        {!skillsEnabled && (
          <div className="m-3 p-2 text-sm border border-yellow-600 bg-yellow-900/30 rounded">
            Skills feature is disabled. Set <code>AQUILLM_SKILLS_ENABLED=1</code> for these skills
            to affect chats.
          </div>
        )}
        <div className="flex-1 overflow-y-auto">
          {loading ? (
            <div className="p-3 text-sm opacity-70">Loading…</div>
          ) : skills.length === 0 ? (
            <div className="p-3 text-sm opacity-70">No skills yet. Click "+ New" to create one.</div>
          ) : (
            <ul>
              {skills.map(s => {
                const isSelected = s.id === selectedId;
                return (
                  <li key={s.id}>
                    <button
                      type="button"
                      onClick={() => handleSelect(s.id)}
                      className={
                        'w-full text-left px-3 py-2 hover:bg-scheme-shade_5 transition-colors ' +
                        (isSelected ? 'bg-scheme-shade_5 font-bold ' : '')
                      }
                    >
                      <div className="flex items-center justify-between gap-2">
                        <span className="truncate">{s.name}</span>
                        {!s.enabled && (
                          <span className="text-xs opacity-60 flex-shrink-0">off</span>
                        )}
                      </div>
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
        </div>
      </aside>

      <main className="flex-1 flex flex-col min-w-0">
        {error && (
          <div className="m-3 p-2 text-sm border border-red bg-red/10 rounded">
            {error}{' '}
            <button type="button" className="underline" onClick={() => setError(null)}>
              dismiss
            </button>
          </div>
        )}

        {!selected || !draft ? (
          <div className="flex-1 flex items-center justify-center opacity-70">
            {skills.length === 0
              ? 'Create a skill to get started.'
              : 'Select a skill from the sidebar.'}
          </div>
        ) : (
          <>
            <div className="flex items-center gap-3 p-3 border-b border-border-low_contrast">
              <input
                type="text"
                value={draft.name}
                maxLength={120}
                onChange={e => setDraft({ ...draft, name: e.target.value })}
                placeholder="Skill name"
                className="flex-1 px-2 py-1 bg-scheme-shade_3 border border-border-mid_contrast rounded"
              />
              <label className="flex items-center gap-1 text-sm">
                <input
                  type="checkbox"
                  checked={draft.enabled}
                  onChange={e => setDraft({ ...draft, enabled: e.target.checked })}
                />
                Enabled
              </label>
              <button
                type="button"
                onClick={handleSave}
                disabled={!dirty || busy || draft.name.trim() === ''}
                className="px-3 py-1 bg-accent hover:bg-accent-dark rounded disabled:opacity-50"
              >
                Save
              </button>
              <button
                type="button"
                onClick={handleDiscard}
                disabled={!dirty || busy}
                className="px-3 py-1 bg-scheme-shade_5 hover:bg-scheme-shade_6 rounded disabled:opacity-50"
              >
                Discard
              </button>
              <button
                type="button"
                onClick={handleDelete}
                disabled={busy}
                className="px-3 py-1 bg-red text-text-normal rounded disabled:opacity-50"
              >
                Delete
              </button>
            </div>
            <div className="flex-1 min-h-0">
              <Suspense fallback={<div className="p-3 text-sm opacity-70">Loading editor…</div>}>
                <SkillEditor
                  value={draft.body}
                  onChange={body => setDraft(d => (d ? { ...d, body } : d))}
                />
              </Suspense>
            </div>
          </>
        )}
      </main>
    </div>
  );
}
