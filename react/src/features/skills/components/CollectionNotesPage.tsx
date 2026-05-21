// Full-page editor for per-collection notes.
//
// Lives in the `collection_notes` bundle alongside Monaco — NOT in main.js.
// See `vite.config.skills.ts` and `vite.config.ts` for why Monaco can't live
// in the main app bundle.
import React, { lazy, Suspense, useCallback, useEffect, useState } from 'react';

import {
  acceptSuggestion,
  dismissPendingFeedback,
  dismissSuggestion,
  generateSuggestion,
  getCollectionSkill,
  listPendingFeedback,
  listSuggestions,
  saveCollectionSkill,
  type CollectionSkill,
  type PendingFeedback,
  type SkillEditSuggestion,
} from '../api/skillsApi';

const SkillEditor = lazy(() => import('./SkillEditor'));
const NotesDiffEditor = lazy(() => import('./NotesDiffEditor'));

const SOFT_WARNING_RATIO = 0.75;

export interface CollectionNotesPageProps {
  collectionId: number;
  collectionName: string;
  collectionUrl: string;
}

type Mode = 'edit' | 'review';

const CollectionNotesPage: React.FC<CollectionNotesPageProps> = ({
  collectionId,
  collectionName,
  collectionUrl,
}) => {
  // ---- notes state ---------------------------------------------------------
  const [data, setData] = useState<CollectionSkill | null>(null);
  const [body, setBody] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(true);
  const [saving, setSaving] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [dirty, setDirty] = useState<boolean>(false);

  // ---- feedback / suggestions state ----------------------------------------
  const [pending, setPending] = useState<PendingFeedback[]>([]);
  const [suggestions, setSuggestions] = useState<SkillEditSuggestion[]>([]);
  const [generatingFor, setGeneratingFor] = useState<number | null>(null);
  const [dismissingFor, setDismissingFor] = useState<number | null>(null);
  const [showSuggestions, setShowSuggestions] = useState<boolean>(false);
  const [showFeedback, setShowFeedback] = useState<boolean>(false);

  // ---- review-mode state ---------------------------------------------------
  const [mode, setMode] = useState<Mode>('edit');
  const [activeReview, setActiveReview] = useState<SkillEditSuggestion | null>(null);
  const [proposedDraft, setProposedDraft] = useState<string>('');
  const [resolvingReview, setResolvingReview] = useState<boolean>(false);

  // ---- helpers -------------------------------------------------------------
  const refreshFeedback = useCallback(async () => {
    const [p, s] = await Promise.all([
      listPendingFeedback(collectionId),
      listSuggestions(collectionId),
    ]);
    setPending(p);
    setSuggestions(s);
  }, [collectionId]);

  // ---- initial load --------------------------------------------------------
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    setDirty(false);
    Promise.all([
      getCollectionSkill(collectionId),
      listPendingFeedback(collectionId).catch(() => [] as PendingFeedback[]),
      listSuggestions(collectionId).catch(() => [] as SkillEditSuggestion[]),
    ])
      .then(([result, p, s]) => {
        if (cancelled) return;
        setData(result);
        setBody(result.body);
        setPending(p);
        setSuggestions(s);
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

  // ---- unsaved-edits guard -------------------------------------------------
  useEffect(() => {
    const beforeUnload = (e: BeforeUnloadEvent) => {
      if (dirty || mode === 'review') {
        e.preventDefault();
        e.returnValue = '';
      }
    };
    window.addEventListener('beforeunload', beforeUnload);
    return () => window.removeEventListener('beforeunload', beforeUnload);
  }, [dirty, mode]);

  // ---- direct-edit handlers ------------------------------------------------
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

  // ---- pending-feedback handlers -------------------------------------------
  const handleDraftSuggestion = async (messageId: number) => {
    setGeneratingFor(messageId);
    setError(null);
    try {
      const sugg = await generateSuggestion(collectionId, messageId);
      await refreshFeedback();
      openReview(sugg);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Generation failed');
    } finally {
      setGeneratingFor(null);
    }
  };

  const handleDismissFeedback = async (messageId: number) => {
    if (!window.confirm('Dismiss this feedback without making a notes change?')) return;
    setDismissingFor(messageId);
    setError(null);
    try {
      await dismissPendingFeedback(collectionId, messageId);
      await refreshFeedback();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Dismiss failed');
    } finally {
      setDismissingFor(null);
    }
  };

  // ---- review-mode handlers ------------------------------------------------
  const openReview = (sugg: SkillEditSuggestion) => {
    setActiveReview(sugg);
    setProposedDraft(sugg.proposed_body);
    setMode('review');
  };

  const cancelReview = () => {
    if (
      proposedDraft !== activeReview?.proposed_body &&
      !window.confirm('Discard your edits to the proposal?')
    ) {
      return;
    }
    setActiveReview(null);
    setProposedDraft('');
    setMode('edit');
  };

  const handleAccept = async () => {
    if (!activeReview) return;
    setResolvingReview(true);
    setError(null);
    try {
      const override =
        proposedDraft !== activeReview.proposed_body ? proposedDraft : undefined;
      await acceptSuggestion(activeReview.id, override);
      // Reload notes (they were just updated server-side) + suggestions.
      const fresh = await getCollectionSkill(collectionId);
      setData(fresh);
      setBody(fresh.body);
      setDirty(false);
      await refreshFeedback();
      setActiveReview(null);
      setProposedDraft('');
      setMode('edit');
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Accept failed');
    } finally {
      setResolvingReview(false);
    }
  };

  const handleDismiss = async () => {
    if (!activeReview) return;
    setResolvingReview(true);
    setError(null);
    try {
      await dismissSuggestion(activeReview.id);
      await refreshFeedback();
      setActiveReview(null);
      setProposedDraft('');
      setMode('edit');
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Dismiss failed');
    } finally {
      setResolvingReview(false);
    }
  };

  // ---- derived -------------------------------------------------------------
  const maxLen = data?.max_body_length ?? 16000;
  const softLimit = Math.floor(maxLen * SOFT_WARNING_RATIO);
  const overSoft = body.length > softLimit;
  const overHard = body.length > maxLen;

  // ---- review mode render --------------------------------------------------
  if (mode === 'review' && activeReview) {
    const proposalOverHard = proposedDraft.length > maxLen;
    return (
      <div className="p-[24px] md:p-[32px] flex flex-col h-full">
        <div className="mb-4">
          <button
            onClick={cancelReview}
            className="h-[36px] px-3 rounded-[18px] bg-scheme-shade_4 text-text-slightly_less_contrast border border-border-mid_contrast hover:bg-scheme-shade_5 hover:text-text-normal transition-colors cursor-pointer inline-flex items-center justify-center mb-[12px]"
          >
            {'← Back to notes'}
          </button>
          <h1 className="text-[1.75rem] font-semibold leading-none text-text-normal">
            Review suggestion — {collectionName}
          </h1>
          <p className="text-sm text-text-slightly_less_contrast mt-2 max-w-3xl">
            Left: current notes (read-only). Right: proposed notes (editable). Tweak the right
            side if you want, then accept or dismiss. Accepting replaces the collection notes
            with what's on the right.
          </p>
        </div>

        <div className="flex-1 min-h-0 border border-border-low_contrast rounded mb-3">
          <Suspense
            fallback={
              <div className="h-full flex items-center justify-center text-text-slightly_less_contrast">
                Loading diff editor…
              </div>
            }
          >
            <NotesDiffEditor
              original={activeReview.notes_body_at_generation}
              modified={proposedDraft}
              onModifiedChange={setProposedDraft}
            />
          </Suspense>
        </div>

        <div className="flex items-center justify-between border-t border-border-low_contrast pt-3">
          <div className="text-sm">
            <span
              className={
                proposalOverHard ? 'text-red' : 'text-text-slightly_less_contrast'
              }
            >
              Proposal: {proposedDraft.length.toLocaleString()} / {maxLen.toLocaleString()} chars
            </span>
            {error && <span className="ml-4 text-red">{error}</span>}
          </div>
          <div className="flex gap-2">
            <button
              onClick={cancelReview}
              disabled={resolvingReview}
              className="px-4 py-2 rounded-md border border-border-mid_contrast text-text-normal hover:bg-scheme-shade_5 disabled:opacity-50"
            >
              Back
            </button>
            <button
              onClick={() => void handleDismiss()}
              disabled={resolvingReview}
              className="px-4 py-2 rounded-md border border-border-mid_contrast text-red hover:bg-scheme-shade_5 disabled:opacity-50"
              title="Throw out this draft. The feedback stays open for re-drafting."
            >
              Discard draft
            </button>
            <button
              onClick={() => void handleAccept()}
              disabled={resolvingReview || proposalOverHard}
              className="px-4 py-2 rounded-md bg-accent text-white disabled:opacity-50 disabled:cursor-not-allowed hover:bg-accent-dark"
            >
              {resolvingReview ? 'Saving…' : 'Accept'}
            </button>
          </div>
        </div>
      </div>
    );
  }

  // ---- edit mode render ----------------------------------------------------
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
          collection. The notes are appended to the chat system prompt whenever this
          collection is selected in a conversation.
        </p>
      </div>

      {/* Pending suggestions panel (collapsible) ----------------------------- */}
      {suggestions.length > 0 && (
        <div className="mb-3 rounded-md border border-accent border-opacity-40 bg-accent bg-opacity-5">
          <button
            type="button"
            onClick={() => setShowSuggestions((v) => !v)}
            className="w-full flex items-center justify-between gap-2 p-3 text-left text-text-normal hover:bg-accent hover:bg-opacity-10"
            aria-expanded={showSuggestions}
          >
            <span className="font-medium">
              {showSuggestions ? '▾' : '▸'} Pending suggestions ({suggestions.length})
            </span>
            <span className="text-xs text-text-slightly_less_contrast">
              {showSuggestions ? 'click to collapse' : 'click to view'}
            </span>
          </button>
          {showSuggestions && (
            <ul className="space-y-2 p-3 pt-0 max-h-[260px] overflow-y-auto">
              {suggestions.map((s) => (
                <li
                  key={s.id}
                  className="flex items-center justify-between gap-3 p-2 rounded bg-scheme-shade_4 border border-border-low_contrast"
                >
                  <div className="text-sm text-text-normal">
                    Drafted {s.created_at ? new Date(s.created_at).toLocaleString() : ''}
                    {s.generated_by && ` by ${s.generated_by}`}
                  </div>
                  <button
                    onClick={() => openReview(s)}
                    className="px-3 py-1 rounded-md bg-accent text-white text-sm hover:bg-accent-dark"
                  >
                    Review
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      {/* Pending feedback panel (collapsible) -------------------------------- */}
      {pending.length > 0 && (
        <div className="mb-3 rounded-md border border-border-low_contrast">
          <button
            type="button"
            onClick={() => setShowFeedback((v) => !v)}
            className="w-full flex items-center justify-between gap-2 p-3 text-left text-text-normal hover:bg-scheme-shade_4"
            aria-expanded={showFeedback}
          >
            <span className="font-medium">
              {showFeedback ? '▾' : '▸'} Pending feedback ({pending.length})
            </span>
            <span className="text-xs text-text-slightly_less_contrast">
              {showFeedback ? 'click to collapse' : 'click to view'}
            </span>
          </button>
          {showFeedback && (
            <div className="p-3 pt-0">
              <p className="text-xs text-text-slightly_less_contrast mb-2">
                Low-rated chat responses with user corrections.{' '}
                <strong>Draft suggestion</strong> asks the assistant to propose a notes
                update from this feedback. <strong>Dismiss feedback</strong> hides the row
                from the queue — the chat message and its rating stay in the database.
              </p>
              <ul className="space-y-2 max-h-[360px] overflow-y-auto">
                {pending.map((m) => (
                  <li
                    key={m.message_id}
                    className="p-2 rounded bg-scheme-shade_4 border border-border-low_contrast"
                  >
                    <div className="text-sm text-text-normal">
                      <span className="text-yellow-400 mr-2">{'★'.repeat(m.rating ?? 0)}</span>
                      <span className="text-text-slightly_less_contrast">
                        {m.conversation_name || `convo #${m.conversation_id}`}
                        {m.feedback_submitted_at &&
                          ` — ${new Date(m.feedback_submitted_at).toLocaleString()}`}
                      </span>
                    </div>
                    <div className="text-sm text-text-normal mt-1">
                      <span className="text-text-slightly_less_contrast">User said: </span>
                      {m.feedback_text}
                    </div>
                    <div className="text-xs text-text-slightly_less_contrast mt-1 italic">
                      Assistant said: {m.content_preview.slice(0, 200)}
                      {m.content_preview.length >= 200 ? '…' : ''}
                    </div>
                    <div className="mt-2 flex justify-end gap-2">
                      <button
                        onClick={() => void handleDismissFeedback(m.message_id)}
                        disabled={dismissingFor !== null || generatingFor !== null}
                        className="px-3 py-1 rounded-md border border-border-mid_contrast text-text-normal text-sm hover:bg-scheme-shade_5 disabled:opacity-50 disabled:cursor-not-allowed"
                        title="Hide this feedback from the queue. The chat message and its rating stay in the database."
                      >
                        {dismissingFor === m.message_id ? 'Dismissing…' : 'Dismiss feedback'}
                      </button>
                      <button
                        onClick={() => void handleDraftSuggestion(m.message_id)}
                        disabled={generatingFor !== null || dismissingFor !== null}
                        className="px-3 py-1 rounded-md bg-accent text-white text-sm hover:bg-accent-dark disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        {generatingFor === m.message_id ? 'Drafting…' : 'Draft suggestion'}
                      </button>
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      {/* Notes editor ------------------------------------------------------- */}
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
