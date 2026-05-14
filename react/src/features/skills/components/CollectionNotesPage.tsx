// Full-page editor for per-collection notes.
//
// Lives in the `collection_notes` bundle alongside Monaco — NOT in main.js.
// See `vite.config.skills.ts` and `vite.config.ts` for why Monaco can't live
// in the main app bundle.
import React, { lazy, Suspense, useEffect, useState } from 'react';

import {
  getCollectionSkill,
  saveCollectionSkill,
  type CollectionSkill,
} from '../api/skillsApi';

const SkillEditor = lazy(() => import('./SkillEditor'));

const SOFT_WARNING_RATIO = 0.75;

export interface CollectionNotesPageProps {
  collectionId: number;
  collectionName: string;
  collectionUrl: string;
}

const CollectionNotesPage: React.FC<CollectionNotesPageProps> = ({
  collectionId,
  collectionName,
  collectionUrl,
}) => {
  const [data, setData] = useState<CollectionSkill | null>(null);
  const [body, setBody] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(true);
  const [saving, setSaving] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [dirty, setDirty] = useState<boolean>(false);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    setDirty(false);
    getCollectionSkill(collectionId)
      .then((result) => {
        if (cancelled) return;
        setData(result);
        setBody(result.body);
      })
      .catch((e: Error) => {
        if (!cancelled) setError(e.message);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [collectionId]);

  useEffect(() => {
    const beforeUnload = (e: BeforeUnloadEvent) => {
      if (dirty) {
        e.preventDefault();
        e.returnValue = '';
      }
    };
    window.addEventListener('beforeunload', beforeUnload);
    return () => window.removeEventListener('beforeunload', beforeUnload);
  }, [dirty]);

  const handleSave = async () => {
    if (!data) return;
    if (body.length > data.max_body_length) {
      setError(`Notes too long (max ${data.max_body_length} characters)`);
      return;
    }
    setSaving(true);
    setError(null);
    try {
      const updated = await saveCollectionSkill(collectionId, body);
      setData(updated);
      setBody(updated.body);
      setDirty(false);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Save failed');
    } finally {
      setSaving(false);
    }
  };

  const handleChange = (next: string) => {
    setBody(next);
    setDirty(true);
  };

  const handleBack = () => {
    if (dirty && !window.confirm('Discard unsaved changes?')) return;
    window.location.href = collectionUrl;
  };

  const maxLen = data?.max_body_length ?? 16000;
  const softLimit = Math.floor(maxLen * SOFT_WARNING_RATIO);
  const overSoft = body.length > softLimit;
  const overHard = body.length > maxLen;

  return (
    <div className="p-[24px] md:p-[32px] flex flex-col h-full">
      <div className="mb-4">
        <button
          onClick={handleBack}
          className="h-[36px] px-3 rounded-[18px] bg-scheme-shade_4 text-text-slightly_less_contrast border border-border-mid_contrast hover:bg-scheme-shade_5 hover:text-text-normal transition-colors cursor-pointer inline-flex items-center justify-center mb-[12px]"
        >
          {'← Back to collection'}
        </button>
        <h1 className="text-[2.05rem] font-semibold leading-none text-text-normal">
          Collection Notes — {collectionName}
        </h1>
        <p className="text-sm text-text-slightly_less_contrast mt-2 max-w-3xl">
          Markdown notes AquiLLM will keep in mind when answering questions about this
          collection. Use this for author lists, terminology, tool versions, and other
          domain knowledge you want the assistant to remember. The notes are appended to
          the chat system prompt whenever this collection is selected in a conversation.
        </p>
      </div>

      <div className="flex-1 min-h-0 border border-border-low_contrast rounded mb-3">
        {loading ? (
          <div className="h-full flex items-center justify-center text-text-slightly_less_contrast">
            Loading…
          </div>
        ) : (
          <Suspense
            fallback={
              <div className="h-full flex items-center justify-center text-text-slightly_less_contrast">
                Loading editor…
              </div>
            }
          >
            <SkillEditor value={body} onChange={handleChange} />
          </Suspense>
        )}
      </div>

      <div className="flex items-center justify-between border-t border-border-low_contrast pt-3">
        <div className="text-sm">
          <span
            className={
              overHard
                ? 'text-red'
                : overSoft
                ? 'text-yellow-400'
                : 'text-text-slightly_less_contrast'
            }
          >
            {body.length.toLocaleString()} / {maxLen.toLocaleString()} characters
          </span>
          {error && <span className="ml-4 text-red">{error}</span>}
          {data?.updated_at && !dirty && !error && (
            <span className="ml-4 text-text-slightly_less_contrast">
              Saved {new Date(data.updated_at).toLocaleString()}
              {data.updated_by && ` by ${data.updated_by}`}
            </span>
          )}
        </div>
        <div className="flex gap-2">
          <button
            onClick={handleBack}
            className="px-4 py-2 rounded-md border border-border-mid_contrast text-text-normal hover:bg-scheme-shade_5"
          >
            Back
          </button>
          <button
            onClick={() => void handleSave()}
            disabled={saving || loading || overHard || !dirty}
            className="px-4 py-2 rounded-md bg-accent text-white disabled:opacity-50 disabled:cursor-not-allowed hover:bg-accent-dark"
          >
            {saving ? 'Saving…' : 'Save'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default CollectionNotesPage;
