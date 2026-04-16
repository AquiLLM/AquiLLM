import React, { useState, useCallback } from 'react';
import { FilterState, EMPTY_FILTERS, DashboardProps } from './types';
import { useFilteredData } from './useFilteredData';
import FilterBar from './FilterBar';
import SummaryCards from './SummaryCards';
import FeedbackTable from './FeedbackTable';
import ExportButton from './ExportButton';
import PRQLPanel from './PRQLPanel';

// extend DashboardProps to include the prql endpoint
interface ExtendedDashboardProps extends DashboardProps {
  apiPrql?: string;
}

type TabId = 'dashboard' | 'prql';

const FeedbackDashboard: React.FC<ExtendedDashboardProps> = ({
  apiRows,
  apiSummary,
  apiFilters,
  apiExport,
  apiPrql,
}) => {
  const [filters, setFilters] = useState<FilterState>(EMPTY_FILTERS);
  const [activeTab, setActiveTab] = useState<TabId>('dashboard');

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
    loadingRows,
    loadingSummary,
    loadingOptions,
    errorRows,
    errorSummary,
    errorOptions,
    exportQueryString,
  } = useFilteredData(filters, apiRows, apiSummary, apiFilters);

  const tabs: Array<{ id: TabId; label: string }> = [
    { id: 'dashboard', label: 'Dashboard' },
    { id: 'prql',      label: 'PRQL Console' },
  ];

  return (
    <div className="flex flex-col h-full max-h-full overflow-y-auto">
      {/* page header */}
      <div className="flex items-center justify-between px-6 pt-6 pb-0 flex-shrink-0">
        <div>
          <h1 className="text-xl font-semibold text-text-normal">Feedback Dashboard</h1>
          <p className="text-sm text-text-low_contrast mt-1">
            Superuser-only analytics view of all user feedback
          </p>
        </div>
        {activeTab === 'dashboard' && (
          <ExportButton
            apiExport={apiExport}
            exportQueryString={exportQueryString}
            totalCount={totalCount}
          />
        )}
      </div>

      {/* tab bar */}
      <div className="flex gap-1 px-6 pt-4 pb-0 border-b border-border-low_contrast flex-shrink-0">
        {tabs.map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`px-4 py-2 text-sm font-medium rounded-t-[8px] transition-colors border-b-2 -mb-px ${
              activeTab === tab.id
                ? 'border-accent text-accent bg-scheme-shade_3'
                : 'border-transparent text-text-low_contrast hover:text-text-normal hover:bg-scheme-shade_4'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      <div className="flex flex-col gap-5 px-6 py-5 flex-grow min-h-0">
        {activeTab === 'dashboard' && (
          <>
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
          </>
        )}

        {activeTab === 'prql' && apiPrql && (
          <PRQLPanel apiPrql={apiPrql} />
        )}

        {activeTab === 'prql' && !apiPrql && (
          <div className="text-sm text-text-low_contrast p-4 bg-scheme-shade_3 border border-border-low_contrast rounded-[10px]">
            PRQL endpoint not configured. Add apiPrql to the component props.
          </div>
        )}
      </div>
    </div>
  );
};

export default FeedbackDashboard;