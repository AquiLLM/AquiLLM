import React, { useEffect, useMemo, useRef, useState } from 'react';
import { X, ExternalLink } from 'lucide-react';
import formatUrl from '../../../utils/formatUrl';
import { getCsrfCookie } from '../../../main';
import type { CitationChunkDetail } from './citationTypes';
import { findNormalizedOffset, normalizeForSearch } from '../../../utils/textOffsetMatch';
import CitationPinButton from './CitationPinButton';

interface TextCitationModalProps {
  docId: string;
  chunkId: string;
  /** Assistant message UUID — enables LLM-narrowed highlight. */
  messageUuid?: string;
  /** Optional chunk prefetched by the provider, skipping the initial fetch. */
  preloadedChunk?: CitationChunkDetail | null;
  onClose: () => void;
  /** Pin state for the slide-out panel. */
  pinned?: boolean;
  onTogglePin?: () => void;
}

function formatTimestamp(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  const m = Math.floor(s / 60);
  const r = s % 60;
  return `${m}:${r.toString().padStart(2, '0')}`;
}

export const TextCitationModal: React.FC<TextCitationModalProps> = ({
  docId,
  chunkId,
  messageUuid,
  preloadedChunk,
  onClose,
  pinned = false,
  onTogglePin,
}) => {
  const [chunk, setChunk] = useState<CitationChunkDetail | null>(preloadedChunk ?? null);
  const [chunkError, setChunkError] = useState<string | null>(null);
  const [narrowQuote, setNarrowQuote] = useState<string | null>(null);
  const [narrowState, setNarrowState] = useState<'idle' | 'pending' | 'tightened' | 'failed'>('idle');

  const markRef = useRef<HTMLSpanElement | null>(null);
  const hasScrolledRef = useRef(false);

  // Escape closes the panel.
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onClose]);

  // Fetch chunk metadata if not preloaded.
  useEffect(() => {
    if (preloadedChunk) return;
    let cancelled = false;
    const apiPattern = window.apiUrls?.api_chunk_detail;
    if (!apiPattern) {
      setChunkError('Chunk detail API not configured.');
      return;
    }
    const url = formatUrl(apiPattern, { chunk_id: chunkId });
    fetch(url, { credentials: 'include' })
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((data: CitationChunkDetail) => {
        if (!cancelled) setChunk(data);
      })
      .catch((err) => {
        if (!cancelled) setChunkError(err.message || 'Failed to load chunk.');
      });
    return () => {
      cancelled = true;
    };
  }, [chunkId, preloadedChunk]);

  // LLM-narrow in parallel.
  useEffect(() => {
    if (!messageUuid) {
      setNarrowState('idle');
      return;
    }
    const apiUrl = window.apiUrls?.api_citation_narrow;
    if (!apiUrl) {
      setNarrowState('idle');
      return;
    }
    let cancelled = false;
    setNarrowState('pending');
    fetch(apiUrl, {
      method: 'POST',
      credentials: 'include',
      headers: {
        'Content-Type': 'application/json',
        'X-CSRFToken': getCsrfCookie(),
      },
      body: JSON.stringify({ message_uuid: messageUuid, chunk_id: Number(chunkId) }),
    })
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => {
        if (cancelled) return;
        const quote = data && typeof data.quote === 'string' ? data.quote.trim() : '';
        if (quote) {
          setNarrowQuote(quote);
          hasScrolledRef.current = false;
          setNarrowState('tightened');
        } else {
          setNarrowState('failed');
        }
      })
      .catch(() => {
        if (!cancelled) setNarrowState('failed');
      });
    return () => {
      cancelled = true;
    };
  }, [messageUuid, chunkId]);

  const fallbackDocUrl = useMemo(() => {
    if (!window.pageUrls?.document) return null;
    return `${formatUrl(window.pageUrls.document, { doc_id: docId })}?chunk=${chunkId}`;
  }, [docId, chunkId]);

  // Compute the highlight range within the displayed full_text. The API
  // may have windowed full_text; subtract text_offset so positions line
  // up with the local string.
  const highlight = useMemo(() => {
    if (!chunk) return null;
    const text = chunk.document.full_text;
    const localStart = chunk.start_position - chunk.document.text_offset;
    const localEnd = chunk.end_position - chunk.document.text_offset;
    let chunkStart = Math.max(0, Math.min(localStart, text.length));
    let chunkEnd = Math.max(chunkStart, Math.min(localEnd, text.length));

    // The offsets are trusted as long as the slice actually matches the chunk
    // content. If they've drifted from the displayed text (windowing edge,
    // re-extraction, etc.) the offset highlight would be silently wrong, so
    // fall back to locating chunk.content directly — same approach the PDF
    // modal uses.
    const sliceNorm = normalizeForSearch(text.slice(chunkStart, chunkEnd)).trim();
    const contentNorm = normalizeForSearch(chunk.content).trim();
    if (contentNorm && sliceNorm !== contentNorm) {
      const located = findNormalizedOffset(text, chunk.content);
      if (located) {
        chunkStart = located.start;
        chunkEnd = located.end;
      }
    }

    let tightStart: number | null = null;
    let tightEnd: number | null = null;
    if (narrowQuote) {
      const window_ = text.slice(chunkStart, chunkEnd);
      const hit = findNormalizedOffset(window_, narrowQuote);
      if (hit) {
        tightStart = chunkStart + hit.start;
        tightEnd = chunkStart + hit.end;
      }
    }
    return { chunkStart, chunkEnd, tightStart, tightEnd };
  }, [chunk, narrowQuote]);

  // Scroll the highlight into view once it's in the DOM.
  useEffect(() => {
    if (!highlight) return;
    if (hasScrolledRef.current) return;
    let attempts = 0;
    const maxAttempts = 30;
    let timer: number | null = null;
    const tick = () => {
      attempts += 1;
      if (markRef.current) {
        markRef.current.scrollIntoView({ block: 'center', behavior: 'smooth' });
        hasScrolledRef.current = true;
        return;
      }
      if (attempts < maxAttempts) {
        timer = window.setTimeout(tick, 80);
      }
    };
    timer = window.setTimeout(tick, 80);
    return () => {
      if (timer !== null) window.clearTimeout(timer);
    };
  }, [highlight, narrowQuote]);

  const headerStatus = useMemo(() => {
    if (!chunk) return '';
    if (narrowState === 'pending') return ' · narrowing…';
    if (narrowState === 'tightened') {
      return highlight?.tightStart != null ? ' · tightened' : ' · narrow miss';
    }
    return '';
  }, [chunk, narrowState, highlight]);

  const renderBody = () => {
    if (chunkError) {
      return (
        <div className="p-6 text-text-normal">
          <p className="font-semibold mb-2">Couldn't load citation.</p>
          <p className="text-sm text-text-low_contrast">{chunkError}</p>
          {fallbackDocUrl && (
            <a
              href={fallbackDocUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1 mt-3 text-accent hover:underline text-sm"
            >
              Open document page <ExternalLink className="w-3.5 h-3.5" />
            </a>
          )}
        </div>
      );
    }

    if (!chunk) {
      return <div className="p-6 text-text-low_contrast text-sm">Loading citation…</div>;
    }
    if (!highlight) return null;

    const text = chunk.document.full_text;
    const { chunkStart, chunkEnd, tightStart, tightEnd } = highlight;

    // Build the rendered segments. When narrowed, we nest a tighter
    // <mark> inside the wider chunk highlight.
    const before = text.slice(0, chunkStart);
    const chunkText = text.slice(chunkStart, chunkEnd);
    const after = text.slice(chunkEnd);

    let chunkRendered: React.ReactNode;
    if (tightStart != null && tightEnd != null) {
      const relStart = tightStart - chunkStart;
      const relEnd = tightEnd - chunkStart;
      chunkRendered = (
        <mark className="bg-yellow-200/40 text-inherit">
          {chunkText.slice(0, relStart)}
          <span
            ref={markRef}
            className="bg-yellow-300/80 text-inherit rounded-sm"
            data-citation-hit=""
          >
            {chunkText.slice(relStart, relEnd)}
          </span>
          {chunkText.slice(relEnd)}
        </mark>
      );
    } else {
      chunkRendered = (
        <mark
          ref={markRef as unknown as React.RefObject<HTMLElement>}
          className="bg-yellow-300/70 text-inherit rounded-sm"
          data-citation-hit=""
        >
          {chunkText}
        </mark>
      );
    }

    return (
      <div className="flex-1 min-h-0 overflow-auto bg-scheme-shade_5 p-4">
        <pre className="whitespace-pre-wrap font-mono text-sm leading-relaxed text-text-normal">
          {before}
          {chunkRendered}
          {after}
        </pre>
      </div>
    );
  };

  const titleSuffix = useMemo(() => {
    if (!chunk) return '';
    if (chunk.start_time != null) return ` · ${formatTimestamp(chunk.start_time)}`;
    return '';
  }, [chunk]);

  return (
    <div className="bg-scheme-shade_3 h-full flex flex-col overflow-hidden">
      <div className="flex items-center justify-between px-4 py-3 border-b border-border-mid_contrast">
        <div className="min-w-0">
          <div className="text-text-normal font-semibold truncate">
            {chunk?.document.title || 'Citation'}
          </div>
          {chunk && (
            <div className="text-xs text-text-low_contrast truncate">
              Chunk {chunk.chunk_number}
              {titleSuffix}
              {headerStatus}
            </div>
          )}
          {chunk?.document.source_url && (
            <a
              href={chunk.document.source_url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs text-accent hover:underline truncate inline-block max-w-full"
              title={chunk.document.source_url}
            >
              {chunk.document.source_url}
            </a>
          )}
        </div>
        <div className="flex items-center gap-1 flex-shrink-0">
          {fallbackDocUrl && (
            <a
              href={fallbackDocUrl}
              target="_blank"
              rel="noopener noreferrer"
              aria-label="Open document page in new tab"
              className="p-1 rounded hover:bg-scheme-shade_4 text-text-normal"
            >
              <ExternalLink className="w-4 h-4" />
            </a>
          )}
          {onTogglePin && <CitationPinButton pinned={pinned} onToggle={onTogglePin} />}
          <button
            type="button"
            onClick={onClose}
            aria-label="Close citation panel"
            className="p-1 rounded hover:bg-scheme-shade_4 text-text-normal"
          >
            <X className="w-5 h-5" />
          </button>
        </div>
      </div>
      {renderBody()}
    </div>
  );
};

export default TextCitationModal;
