import React, { useState, useCallback } from 'react';
import { FilterState, EMPTY_FILTERS, DashboardProps } from './types';
import { useFilteredData } from './useFilteredData';
import FilterBar from './FilterBar';
import LivePRQLDisplay from './LivePRQLDisplay';
import SummaryCards from './SummaryCards';
import FeedbackTable from './FeedbackTable';
import ExportButton from './ExportButton';
import PRQLPanel from './PRQLPanel';

const FeedbackDashboard: React.FC<DashboardProps> = ({
  apiRows,
  apiSummary,
  apiFilters,
  apiExport,
}) => {
  const [filters, setFilters] = useState<FilterState>(EMPTY_FILTERS);
  const [showPrqlGuide, setShowPrqlGuide] = useState(false);

  const handleFilterChange = useCallback((updated: Partial<FilterState>) => {
    setFilters(prev => ({ ...prev, ...updated }));
  }, []);

  const handleReset = useCallback(() => {
    setFilters(EMPTY_FILTERS);
  }, []);

  const handlePageChange = useCallback((page: number) => {
    setFilters(prev => ({ ...prev, page }));
  }, []);

  const {
    summary,
    rows,
    filterOptions,
    totalCount,
    totalPages,
    currentPage,
    prql,
    loadingRows,
    loadingSummary,
    loadingOptions,
    errorRows,
    errorSummary,
    errorOptions,
    exportQueryString,
  } = useFilteredData(filters, apiRows, apiSummary, apiFilters);

  return (
    <div className="flex flex-col h-full max-h-full overflow-y-auto">
      <div className="flex items-center justify-between px-6 pt-6 pb-4 border-b border-border-low_contrast flex-shrink-0">
        <div>
          <h1 className="text-xl font-semibold text-text-normal">Feedback Dashboard</h1>
          <p className="text-sm text-text-low_contrast mt-1">
            Superuser-only analytics view of all user feedback
          </p>
        </div>

        <ExportButton
          apiExport={apiExport}
          exportQueryString={exportQueryString}
          totalCount={totalCount}
        />
      </div>

      <div className="flex flex-col gap-4 px-6 py-5 flex-grow min-h-0">
        {errorOptions && (
          <div className="rounded-[8px] bg-scheme-shade_3 border border-border-mid_contrast p-3 text-red text-sm">
            could not load filter options: {errorOptions}
          </div>
        )}

        <FilterBar
          filters={filters}
          filterOptions={filterOptions}
          optionsLoading={loadingOptions}
          onChange={handleFilterChange}
          onReset={handleReset}
        />

        <div style={{ flexShrink: 0, width: '100%' }}>
          <LivePRQLDisplay prql={prql} loading={loadingRows} />
        </div>

        <div className="border border-border-mid_contrast rounded-[12px] overflow-hidden">
          <button
            onClick={() => setShowPrqlGuide(value => !value)}
            className="w-full flex items-center justify-between px-4 py-3 bg-scheme-shade_4 hover:bg-scheme-shade_5 transition-colors text-left"
          >
            <div className="flex items-center gap-2">
              <span className="font-mono text-xs font-bold text-accent bg-accent/10 px-2 py-0.5 rounded">
                PRQL Guide
              </span>
              <span className="text-xs text-text-low_contrast">
                {showPrqlGuide
                  ? 'click to collapse'
                  : 'click to open display-only PRQL reference'}
              </span>
            </div>
            <span className="text-text-low_contrast text-sm">
              {showPrqlGuide ? '▾' : '▸'}
            </span>
          </button>

          {showPrqlGuide && (
            <div className="p-4 bg-scheme-shade_3 border-t border-border-mid_contrast">
              <PRQLPanel currentPrql={prql} loading={loadingRows} />
            </div>
          )}
        </div>

        <SummaryCards
          summary={summary}
          loading={loadingSummary}
          error={errorSummary}
        />

        <FeedbackTable
          rows={rows}
          loading={loadingRows}
          error={errorRows}
          totalCount={totalCount}
          totalPages={totalPages}
          currentPage={currentPage}
          pageSize={filters.page_size}
          onPageChange={handlePageChange}
        />
      </div>
    </div>
  );
};

export default FeedbackDashboard;
