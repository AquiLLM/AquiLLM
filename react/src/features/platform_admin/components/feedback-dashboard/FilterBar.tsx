// FilterBar.tsx
// filter controls for the KQL feedback dashboard
// each control maps to a KQL where clause in buildKQLFromFilters
// dropdowns are populated from the api_feedback_filter_options endpoint
// text inputs are debounced inside useFilterBar
import React from 'react';
import type { FilterOptions } from './useFilterOptions';
import type { FilterState } from './useFilterBar';

interface FilterBarProps {
  filters: FilterState;
  options: FilterOptions;
  optionsLoading: boolean;
  hasActiveFilters: boolean;
  onFilterChange: (key: keyof FilterState, value: string) => void;
  onReset: () => void;
  /** When true, suppresses the built-in "Filters" header row (use inside DashboardSection) */
  hideHeader?: boolean;
}

// shared input class
const inputCls =
  'w-full px-2 py-1.5 rounded bg-scheme-shade_2 element-border text-sm text-text-normal ' +
  'focus:outline-none focus:ring-2 focus:ring-blue-500/50 placeholder-text-muted';

const labelCls = 'block text-xs font-semibold text-text-normal mb-1';

interface SelectProps {
  label: string;
  value: string;
  onChange: (v: string) => void;
  options: Array<{ value: string; label: string }>;
  placeholder: string;
}

const FilterSelect: React.FC<SelectProps> = ({ label, value, onChange, options, placeholder }) => (
  <div>
    <label className={labelCls}>{label}</label>
    <select
      value={value}
      onChange={e => onChange(e.target.value)}
      className={inputCls}
    >
      <option value="">{placeholder}</option>
      {options.map(o => (
        <option key={o.value} value={o.value}>{o.label}</option>
      ))}
    </select>
  </div>
);

interface TextInputProps {
  label: string;
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
  type?: string;
}

const FilterTextInput: React.FC<TextInputProps> = ({
  label, value, onChange, placeholder, type = 'text',
}) => (
  <div>
    <label className={labelCls}>{label}</label>
    <input
      type={type}
      value={value}
      onChange={e => onChange(e.target.value)}
      placeholder={placeholder ?? ''}
      className={inputCls}
    />
  </div>
);

const FilterBar: React.FC<FilterBarProps> = ({
  filters,
  options,
  optionsLoading,
  hasActiveFilters,
  onFilterChange,
  onReset,
  hideHeader = false,
}) => {
  const userOptions = options.users.map(u => ({
    value: String(u.id),
    label: u.username,
  }));
  const modelOptions = options.models.map(m => ({ value: m, label: m }));
  const toolOptions  = options.tool_names.map(t => ({ value: t, label: t }));
  const roleOptions  = options.roles.map(r => ({ value: r, label: r }));

  const ratingOptions = [1, 2, 3, 4, 5].map(r => ({
    value: String(r),
    label: `${r} star${r !== 1 ? 's' : ''}`,
  }));

  const hasFeedbackOptions = [
    { value: 'true',  label: 'Has text' },
    { value: 'false', label: 'No text'  },
  ];

  // hide min/max when exact_rating is set to avoid visual confusion
  const showRatingRange = !filters.exact_rating;

  return (
    <div className={hideHeader ? '' : 'mb-5 rounded-lg bg-scheme-shade_3 element-border'}>
      {/* header — only shown when not embedded inside a DashboardSection */}
      {!hideHeader && (
        <div className="flex items-center justify-between px-4 py-2.5 border-b border-scheme-contrast/20">
          <span className="text-sm font-semibold text-text-normal">Filters</span>
          <div className="flex items-center gap-3">
            {optionsLoading && (
              <span className="text-xs text-text-muted">loading options…</span>
            )}
            {hasActiveFilters && (
              <button
                onClick={onReset}
                className="text-xs text-text-muted hover:text-text-normal transition-colors px-2 py-1 rounded hover:bg-scheme-shade_5"
              >
                Clear all
              </button>
            )}
          </div>
        </div>
      )}

      {/* When embedded, show loading / clear all inline above the grid */}
      {hideHeader && (optionsLoading || hasActiveFilters) && (
        <div className="flex items-center gap-3 px-4 pt-3 text-xs">
          {optionsLoading && (
            <span className="text-text-muted">loading options…</span>
          )}
          {hasActiveFilters && (
            <button
              onClick={onReset}
              className="text-text-muted hover:text-text-normal transition-colors px-2 py-1 rounded hover:bg-scheme-shade_5"
            >
              Clear all
            </button>
          )}
        </div>
      )}

      {/* controls grid */}
      <div className="px-4 py-3 grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6 gap-3">
        {/* date range */}
        <FilterTextInput
          label="From date"
          value={filters.date_from}
          onChange={v => onFilterChange('date_from', v)}
          type="date"
        />
        <FilterTextInput
          label="To date"
          value={filters.date_to}
          onChange={v => onFilterChange('date_to', v)}
          type="date"
        />

        {/* user */}
        <FilterSelect
          label="User"
          value={filters.user_id}
          onChange={v => onFilterChange('user_id', v)}
          options={userOptions}
          placeholder="All users"
        />

        {/* rating */}
        <FilterSelect
          label="Exact rating"
          value={filters.exact_rating}
          onChange={v => onFilterChange('exact_rating', v)}
          options={ratingOptions}
          placeholder="Any rating"
        />
        {showRatingRange && (
          <FilterSelect
            label="Min rating"
            value={filters.min_rating}
            onChange={v => onFilterChange('min_rating', v)}
            options={ratingOptions}
            placeholder="No minimum"
          />
        )}
        {showRatingRange && (
          <FilterSelect
            label="Max rating"
            value={filters.max_rating}
            onChange={v => onFilterChange('max_rating', v)}
            options={ratingOptions}
            placeholder="No maximum"
          />
        )}

        {/* feedback text search */}
        <FilterTextInput
          label="Feedback text"
          value={filters.feedback_text_search}
          onChange={v => onFilterChange('feedback_text_search', v)}
          placeholder="Search feedback…"
        />

        {/* role */}
        <FilterSelect
          label="Role"
          value={filters.role}
          onChange={v => onFilterChange('role', v)}
          options={roleOptions}
          placeholder="All roles"
        />

        {/* model */}
        <FilterSelect
          label="Model"
          value={filters.model}
          onChange={v => onFilterChange('model', v)}
          options={modelOptions}
          placeholder="All models"
        />

        {/* tool */}
        <FilterSelect
          label="Tool"
          value={filters.tool_call_name}
          onChange={v => onFilterChange('tool_call_name', v)}
          options={toolOptions}
          placeholder="All tools"
        />

        {/* has feedback text */}
        <FilterSelect
          label="Has feedback text"
          value={filters.has_feedback_text}
          onChange={v => onFilterChange('has_feedback_text', v)}
          options={hasFeedbackOptions}
          placeholder="Either"
        />

      </div>
    </div>
  );
};

export default FilterBar;