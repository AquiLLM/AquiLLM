
import React from 'react';
import { SummaryMetrics } from './types';

interface SummaryCardsProps {
  summary: SummaryMetrics | null;
  loading: boolean;
  error: string | null;
}

// the rating bar colours go from red at 1 to green at 5
const RATING_COLORS: Record<number, string> = {
  1: 'bg-red',
  2: 'bg-red',
  3: 'bg-accent',
  4: 'bg-green',
  5: 'bg-green',
};

function formatDate(iso: string | null): string {
  if (!iso) return '—';
  try {
    return new Date(iso).toLocaleDateString(undefined, {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    });
  } catch {
    return iso;
  }
}

interface MetricCardProps {
  label: string;
  value: React.ReactNode;
}

const MetricCard: React.FC<MetricCardProps> = ({ label, value }) => (
  <div className="bg-scheme-shade_3 border border-border-mid_contrast rounded-[12px] p-4 flex flex-col gap-1 min-w-0">
    <span className="text-xs text-text-low_contrast uppercase tracking-wide">{label}</span>
    <span className="text-2xl font-semibold text-text-normal truncate">{value}</span>
  </div>
);

const SummaryCards: React.FC<SummaryCardsProps> = ({ summary, loading, error }) => {
  if (error) {
    return (
      <div className="rounded-[12px] bg-scheme-shade_3 border border-border-mid_contrast p-4 text-red">
        failed to load summary: {error}
      </div>
    );
  }

  if (loading || !summary) {
    return (
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
        {Array.from({ length: 6 }).map((_, i) => (
          <div
            key={i}
            className="bg-scheme-shade_3 border border-border-mid_contrast rounded-[12px] p-4 h-[80px] animate-pulse"
          />
        ))}
      </div>
    );
  }

  const avgDisplay = summary.avg_rating !== null
    ? summary.avg_rating.toFixed(2)
    : '—';

  const dateRange =
    summary.date_min || summary.date_max
      ? `${formatDate(summary.date_min)} – ${formatDate(summary.date_max)}`
      : '—';

  // compute max value across ratings so we can scale bars
  const ratingEntries: Array<[number, number]> = [1, 2, 3, 4, 5].map(r => [
    r,
    summary.rating_distribution[String(r)] ?? 0,
  ]);
  const maxRatingCount = Math.max(...ratingEntries.map(([, c]) => c), 1);

  return (
    <div className="flex flex-col gap-4">
      {/* metric cards row */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
        <MetricCard label="Total feedback"    value={summary.total_count.toLocaleString()} />
        <MetricCard label="Rated"             value={summary.rated_count.toLocaleString()} />
        <MetricCard label="Avg rating"        value={avgDisplay} />
        <MetricCard label="With text"         value={summary.has_text_count.toLocaleString()} />
        <MetricCard
          label="Date range"
          value={
            <span className="text-base font-medium leading-tight">
              {dateRange}
            </span>
          }
        />
        <MetricCard
          label="Unrated"
          value={(summary.total_count - summary.rated_count).toLocaleString()}
        />
      </div>

      {/* rating distribution */}
      <div className="bg-scheme-shade_3 border border-border-mid_contrast rounded-[12px] p-4">
        <span className="text-xs text-text-low_contrast uppercase tracking-wide block mb-3">
          Rating distribution
        </span>
        <div className="flex items-end gap-2 h-[56px]">
          {ratingEntries.map(([rating, count]) => {
            const pct = maxRatingCount > 0 ? (count / maxRatingCount) * 100 : 0;
            return (
              <div key={rating} className="flex flex-col items-center flex-1 gap-1">
                <span className="text-xs text-text-less_contrast">{count}</span>
                <div className="w-full flex items-end" style={{ height: '36px' }}>
                  <div
                    className={`w-full rounded-t-[4px] transition-all ${RATING_COLORS[rating] ?? 'bg-accent'}`}
                    style={{ height: `${Math.max(pct, count > 0 ? 8 : 0)}%` }}
                  />
                </div>
                <span className="text-xs text-text-low_contrast">{rating}★</span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default SummaryCards;