
import React from 'react';
import { FilterState, FilterOptions, EMPTY_FILTERS } from './types';

interface FilterBarProps {
  filters: FilterState;
  filterOptions: FilterOptions | null;
  optionsLoading: boolean;
  onChange: (updated: Partial<FilterState>) => void;
  onReset: () => void;
}

// shared input class tokens so every control looks consistent
const inputClass =
  'w-full px-3 py-[6px] rounded-[8px] bg-scheme-shade_4 border border-border-mid_contrast ' +
  'text-text-normal text-sm focus:outline-none focus:border-border-high_contrast transition-colors';

const labelClass = 'text-xs text-text-low_contrast uppercase tracking-wide block mb-1';

interface SelectProps {
  label: string;
  value: string;
  onChange: (v: string) => void;
  options: Array<{ value: string; label: string }>;
  placeholder?: string;
}

const FilterSelect: React.FC<SelectProps> = ({ label, value, onChange, options, placeholder }) => (
  <div className="flex flex-col min-w-0">
    <label className={labelClass}>{label}</label>
    <select
      value={value}
      onChange={e => onChange(e.target.value)}
      className={inputClass}
    >
      <option value="">{placeholder ?? `All ${label.toLowerCase()}`}</option>
      {options.map(o => (
        <option key={o.value} value={o.value}>
          {o.label}
        </option>
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

const FilterTextInput: React.FC<TextInputProps> = ({ label, value, onChange, placeholder, type = 'text' }) => (
  <div className="flex flex-col min-w-0">
    <label className={labelClass}>{label}</label>
    <input
      type={type}
      value={value}
      onChange={e => onChange(e.target.value)}
      placeholder={placeholder ?? ''}
      className={inputClass}
    />
  </div>
);

const FilterBar: React.FC<FilterBarProps> = ({
  filters,
  filterOptions,
  optionsLoading,
  onChange,
  onReset,
}) => {
  // check whether any filter is active so we can show/hide the reset button
  const hasActiveFilters = (
    filters.start_date !== EMPTY_FILTERS.start_date ||
    filters.end_date !== EMPTY_FILTERS.end_date ||
    filters.user_id !== EMPTY_FILTERS.user_id ||
    filters.exact_rating !== EMPTY_FILTERS.exact_rating ||
    filters.min_rating !== EMPTY_FILTERS.min_rating ||
    filters.max_rating !== EMPTY_FILTERS.max_rating ||
    filters.feedback_text_search !== EMPTY_FILTERS.feedback_text_search ||
    filters.conversation_name_search !== EMPTY_FILTERS.conversation_name_search ||
    filters.role !== EMPTY_FILTERS.role ||
    filters.model !== EMPTY_FILTERS.model ||
    filters.tool_call_name !== EMPTY_FILTERS.tool_call_name ||
    filters.has_feedback_text !== EMPTY_FILTERS.has_feedback_text
  );

  const userOptions = (filterOptions?.users ?? []).map(u => ({
    value: String(u.id),
    label: u.username,
  }));

  const roleOptions = (filterOptions?.roles ?? []).map(r => ({
    value: r,
    label: r,
  }));

  const modelOptions = (filterOptions?.models ?? []).map(m => ({
    value: m,
    label: m,
  }));

  const toolOptions = (filterOptions?.tool_names ?? []).map(t => ({
    value: t,
    label: t,
  }));

  const ratingOptions = [1, 2, 3, 4, 5].map(r => ({
    value: String(r),
    label: `${r} star${r !== 1 ? 's' : ''}`,
  }));

  const hasFeedbackTextOptions = [
    { value: 'true',  label: 'Has text' },
    { value: 'false', label: 'No text' },
  ];

  // when exact_rating is set we hide min/max to avoid confusion
  const showRangeRating = !filters.exact_rating;

  return (
    <div className="bg-scheme-shade_3 border border-border-mid_contrast rounded-[12px] p-4">
      <div className="flex items-center justify-between mb-3">
        <span className="text-sm font-medium text-text-normal">Filters</span>
        {hasActiveFilters && (
          <button
            onClick={onReset}
            className="text-xs text-text-less_contrast hover:text-text-normal transition-colors px-2 py-1 rounded-[6px] hover:bg-scheme-shade_5"
          >
            Clear all
          </button>
        )}
      </div>

      {optionsLoading && (
        <div className="text-xs text-text-low_contrast mb-2">loading filter options…</div>
      )}

      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6 gap-3">

        {/* date range */}
        <FilterTextInput
          label="From date"
          value={filters.start_date}
          onChange={v => onChange({ start_date: v, page: 1 })}
          type="date"
        />
        <FilterTextInput
          label="To date"
          value={filters.end_date}
          onChange={v => onChange({ end_date: v, page: 1 })}
          type="date"
        />

        {/* user */}
        <FilterSelect
          label="User"
          value={filters.user_id}
          onChange={v => onChange({ user_id: v, page: 1 })}
          options={userOptions}
          placeholder="All users"
        />

        {/* rating — exact takes priority; when set, range is hidden */}
        <FilterSelect
          label="Exact rating"
          value={filters.exact_rating}
          onChange={v => onChange({ exact_rating: v, min_rating: '', max_rating: '', page: 1 })}
          options={ratingOptions}
          placeholder="Any rating"
        />

        {showRangeRating && (
          <FilterSelect
            label="Min rating"
            value={filters.min_rating}
            onChange={v => onChange({ min_rating: v, page: 1 })}
            options={ratingOptions}
            placeholder="No minimum"
          />
        )}

        {showRangeRating && (
          <FilterSelect
            label="Max rating"
            value={filters.max_rating}
            onChange={v => onChange({ max_rating: v, page: 1 })}
            options={ratingOptions}
            placeholder="No maximum"
          />
        )}

        {/* text searches */}
        <FilterTextInput
          label="Feedback text search"
          value={filters.feedback_text_search}
          onChange={v => onChange({ feedback_text_search: v, page: 1 })}
          placeholder="Search feedback…"
        />
        <FilterTextInput
          label="Conversation name"
          value={filters.conversation_name_search}
          onChange={v => onChange({ conversation_name_search: v, page: 1 })}
          placeholder="Search conversation…"
        />

        {/* dropdowns */}
        <FilterSelect
          label="Role"
          value={filters.role}
          onChange={v => onChange({ role: v, page: 1 })}
          options={roleOptions}
          placeholder="All roles"
        />
        <FilterSelect
          label="Model"
          value={filters.model}
          onChange={v => onChange({ model: v, page: 1 })}
          options={modelOptions}
          placeholder="All models"
        />
        <FilterSelect
          label="Tool"
          value={filters.tool_call_name}
          onChange={v => onChange({ tool_call_name: v, page: 1 })}
          options={toolOptions}
          placeholder="All tools"
        />
        <FilterSelect
          label="Has feedback text"
          value={filters.has_feedback_text}
          onChange={v => onChange({ has_feedback_text: v, page: 1 })}
          options={hasFeedbackTextOptions}
          placeholder="Either"
        />

        {/* page size */}
        <FilterSelect
          label="Rows per page"
          value={String(filters.page_size)}
          onChange={v => onChange({ page_size: Number(v), page: 1 })}
          options={[
            { value: '25',  label: '25 rows' },
            { value: '50',  label: '50 rows' },
            { value: '100', label: '100 rows' },
            { value: '200', label: '200 rows' },
          ]}
        />
      </div>
    </div>
  );
};

export default FilterBar;
