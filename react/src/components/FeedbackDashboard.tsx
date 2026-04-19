import React, { useEffect, useMemo, useState } from "react";

type UserOption = {
  id: number;
  username: string;
};

type FilterOptions = {
  users: UserOption[];
  roles: string[];
  models: string[];
  tool_names: string[];
  ratings: number[];
};

type SummaryResponse = {
  total_count: number;
  rated_count: number;
  avg_rating: number | null;
  rating_distribution: Record<string, number>;
  has_text_count: number;
  date_min: string | null;
  date_max: string | null;
};

type Row = {
  id: number;
  message_uuid: string;
  conversation_id: number;
  conversation_name: string | null;
  user_id: number;
  username: string;
  rating: number | null;
  feedback_text: string | null;
  feedback_submitted_at: string | null;
  created_at: string | null;
  effective_date: string | null;
  role: string;
  content_snippet: string | null;
  model: string | null;
  tool_call_name: string | null;
  usage: number | null;
  has_feedback_text: boolean;
};

type RowsResponse = {
  rows: Row[];
  page: number;
  page_size: number;
  total_count: number;
  total_pages: number;
};

type DashboardProps = {
  apiRows: string;
  apiSummary: string;
  apiFilters: string;
  apiExport: string;
};

type FilterState = {
  start_date: string;
  end_date: string;
  user_id: string;
  exact_rating: string;
  feedback_text_search: string;
  conversation_name_search: string;
  role: string;
  model: string;
  tool_call_name: string;
  has_feedback_text: string;
  page: number;
  page_size: number;
};

const defaultFilters: FilterState = {
  start_date: "",
  end_date: "",
  user_id: "",
  exact_rating: "",
  feedback_text_search: "",
  conversation_name_search: "",
  role: "",
  model: "",
  tool_call_name: "",
  has_feedback_text: "",
  page: 1,
  page_size: 25,
};

function formatDate(value: string | null): string {
  if (!value) return "—";
  try {
    return new Date(value).toLocaleString();
  } catch {
    return value;
  }
}

function buildQuery(filters: FilterState): URLSearchParams {
  const params = new URLSearchParams();

  if (filters.start_date) params.set("start_date", filters.start_date);
  if (filters.end_date) params.set("end_date", filters.end_date);
  if (filters.user_id) params.set("user_id", filters.user_id);
  if (filters.exact_rating) params.set("exact_rating", filters.exact_rating);
  if (filters.feedback_text_search) {
    params.set("feedback_text_search", filters.feedback_text_search);
  }
  if (filters.conversation_name_search) {
    params.set("conversation_name_search", filters.conversation_name_search);
  }
  if (filters.role) params.set("role", filters.role);
  if (filters.model) params.set("model", filters.model);
  if (filters.tool_call_name) params.set("tool_call_name", filters.tool_call_name);
  if (filters.has_feedback_text) {
    params.set("has_feedback_text", filters.has_feedback_text);
  }

  params.set("page", String(filters.page));
  params.set("page_size", String(filters.page_size));

  return params;
}

export default function FeedbackDashboard({
  apiRows,
  apiSummary,
  apiFilters,
  apiExport,
}: DashboardProps) {
  const [draftFilters, setDraftFilters] = useState<FilterState>(defaultFilters);
  const [appliedFilters, setAppliedFilters] = useState<FilterState>(defaultFilters);

  const [filterOptions, setFilterOptions] = useState<FilterOptions>({
    users: [],
    roles: [],
    models: [],
    tool_names: [],
    ratings: [],
  });

  const [summary, setSummary] = useState<SummaryResponse | null>(null);
  const [rowsData, setRowsData] = useState<RowsResponse | null>(null);

  const [loadingFilters, setLoadingFilters] = useState(true);
  const [loadingSummary, setLoadingSummary] = useState(true);
  const [loadingRows, setLoadingRows] = useState(true);
  const [error, setError] = useState<string>("");

  const queryString = useMemo(() => buildQuery(appliedFilters).toString(), [appliedFilters]);

  useEffect(() => {
    let cancelled = false;

    async function loadFilterOptions() {
      setLoadingFilters(true);
      try {
        const response = await fetch(apiFilters, {
          method: "GET",
          credentials: "same-origin",
        });

        if (!response.ok) {
          throw new Error(`Filter request failed with status ${response.status}`);
        }

        const data: FilterOptions = await response.json();
        if (!cancelled) {
          setFilterOptions(data);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to load filter options");
        }
      } finally {
        if (!cancelled) {
          setLoadingFilters(false);
        }
      }
    }

    loadFilterOptions();
    return () => {
      cancelled = true;
    };
  }, [apiFilters]);

  useEffect(() => {
    let cancelled = false;

    async function loadSummaryAndRows() {
      setLoadingSummary(true);
      setLoadingRows(true);
      setError("");

      try {
        const summaryUrl = queryString ? `${apiSummary}?${queryString}` : apiSummary;
        const rowsUrl = queryString ? `${apiRows}?${queryString}` : apiRows;

        const [summaryResp, rowsResp] = await Promise.all([
          fetch(summaryUrl, { method: "GET", credentials: "same-origin" }),
          fetch(rowsUrl, { method: "GET", credentials: "same-origin" }),
        ]);

        if (!summaryResp.ok) {
          throw new Error(`Summary request failed with status ${summaryResp.status}`);
        }
        if (!rowsResp.ok) {
          throw new Error(`Rows request failed with status ${rowsResp.status}`);
        }

        const summaryData: SummaryResponse = await summaryResp.json();
        const rowsDataResp: RowsResponse = await rowsResp.json();

        if (!cancelled) {
          setSummary(summaryData);
          setRowsData(rowsDataResp);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to load dashboard data");
        }
      } finally {
        if (!cancelled) {
          setLoadingSummary(false);
          setLoadingRows(false);
        }
      }
    }

    loadSummaryAndRows();

    return () => {
      cancelled = true;
    };
  }, [apiRows, apiSummary, queryString]);

  function updateDraft<K extends keyof FilterState>(key: K, value: FilterState[K]) {
    setDraftFilters((prev) => ({ ...prev, [key]: value }));
  }

  function applyFilters() {
    setAppliedFilters({
      ...draftFilters,
      page: 1,
    });
  }

  function resetFilters() {
    setDraftFilters(defaultFilters);
    setAppliedFilters(defaultFilters);
  }

  function goToPage(page: number) {
    setAppliedFilters((prev) => ({
      ...prev,
      page,
    }));
  }

  function exportRows() {
    const params = buildQuery(appliedFilters);
    const url = params.toString() ? `${apiExport}?${params.toString()}` : apiExport;
    window.location.href = url;
  }

  const ratingDistributionEntries = summary
    ? Object.entries(summary.rating_distribution).sort(
        ([a], [b]) => Number(a) - Number(b)
      )
    : [];

  return (
    <div className="p-6 md:p-8 text-text-normal">
      <div className="mb-6">
        <h1 className="text-3xl font-semibold mb-2">Feedback Dashboard</h1>
        <p className="text-sm text-text-less_contrast">
          Superuser-only analytics for message feedback across users and conversations.
        </p>
      </div>

      {error ? (
        <div className="mb-6 rounded-xl border border-red p-4 bg-scheme-shade_3">
          <p className="font-medium text-red">Dashboard error</p>
          <p className="text-sm mt-1">{error}</p>
        </div>
      ) : null}

      <div className="rounded-2xl border border-border-mid_contrast bg-scheme-shade_3 p-5 mb-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold">Filters</h2>
          {loadingFilters ? (
            <span className="text-sm text-text-less_contrast">Loading filters...</span>
          ) : null}
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
          <div>
            <label className="block text-sm mb-1">Start date</label>
            <input
              type="date"
              value={draftFilters.start_date}
              onChange={(e) => updateDraft("start_date", e.target.value)}
              className="w-full rounded-lg bg-scheme-shade_4 border border-border-high_contrast p-2"
            />
          </div>

          <div>
            <label className="block text-sm mb-1">End date</label>
            <input
              type="date"
              value={draftFilters.end_date}
              onChange={(e) => updateDraft("end_date", e.target.value)}
              className="w-full rounded-lg bg-scheme-shade_4 border border-border-high_contrast p-2"
            />
          </div>

          <div>
            <label className="block text-sm mb-1">User</label>
            <select
              value={draftFilters.user_id}
              onChange={(e) => updateDraft("user_id", e.target.value)}
              className="w-full rounded-lg bg-scheme-shade_4 border border-border-high_contrast p-2"
            >
              <option value="">All users</option>
              {filterOptions.users.map((user) => (
                <option key={user.id} value={user.id}>
                  {user.username}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm mb-1">Exact rating</label>
            <select
              value={draftFilters.exact_rating}
              onChange={(e) => updateDraft("exact_rating", e.target.value)}
              className="w-full rounded-lg bg-scheme-shade_4 border border-border-high_contrast p-2"
            >
              <option value="">All ratings</option>
              {filterOptions.ratings.map((rating) => (
                <option key={rating} value={rating}>
                  {rating}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm mb-1">Feedback text</label>
            <input
              type="text"
              value={draftFilters.feedback_text_search}
              onChange={(e) => updateDraft("feedback_text_search", e.target.value)}
              placeholder="Search feedback text"
              className="w-full rounded-lg bg-scheme-shade_4 border border-border-high_contrast p-2"
            />
          </div>

          <div>
            <label className="block text-sm mb-1">Conversation name</label>
            <input
              type="text"
              value={draftFilters.conversation_name_search}
              onChange={(e) => updateDraft("conversation_name_search", e.target.value)}
              placeholder="Search conversation"
              className="w-full rounded-lg bg-scheme-shade_4 border border-border-high_contrast p-2"
            />
          </div>

          <div>
            <label className="block text-sm mb-1">Role</label>
            <select
              value={draftFilters.role}
              onChange={(e) => updateDraft("role", e.target.value)}
              className="w-full rounded-lg bg-scheme-shade_4 border border-border-high_contrast p-2"
            >
              <option value="">All roles</option>
              {filterOptions.roles.map((role) => (
                <option key={role} value={role}>
                  {role}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm mb-1">Model</label>
            <select
              value={draftFilters.model}
              onChange={(e) => updateDraft("model", e.target.value)}
              className="w-full rounded-lg bg-scheme-shade_4 border border-border-high_contrast p-2"
            >
              <option value="">All models</option>
              {filterOptions.models.map((model) => (
                <option key={model} value={model}>
                  {model}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm mb-1">Tool call name</label>
            <select
              value={draftFilters.tool_call_name}
              onChange={(e) => updateDraft("tool_call_name", e.target.value)}
              className="w-full rounded-lg bg-scheme-shade_4 border border-border-high_contrast p-2"
            >
              <option value="">All tools</option>
              {filterOptions.tool_names.map((toolName) => (
                <option key={toolName} value={toolName}>
                  {toolName}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm mb-1">Has feedback text</label>
            <select
              value={draftFilters.has_feedback_text}
              onChange={(e) => updateDraft("has_feedback_text", e.target.value)}
              className="w-full rounded-lg bg-scheme-shade_4 border border-border-high_contrast p-2"
            >
              <option value="">All rows</option>
              <option value="true">Text only</option>
              <option value="false">No text</option>
            </select>
          </div>

          <div>
            <label className="block text-sm mb-1">Page size</label>
            <select
              value={draftFilters.page_size}
              onChange={(e) => updateDraft("page_size", Number(e.target.value))}
              className="w-full rounded-lg bg-scheme-shade_4 border border-border-high_contrast p-2"
            >
              {[10, 25, 50, 100].map((size) => (
                <option key={size} value={size}>
                  {size}
                </option>
              ))}
            </select>
          </div>
        </div>

        <div className="mt-5 flex flex-wrap gap-3">
          <button
            onClick={applyFilters}
            className="px-4 py-2 rounded-lg bg-scheme-shade_5 hover:bg-scheme-shade_6 border border-border-mid_contrast"
          >
            Apply filters
          </button>
          <button
            onClick={resetFilters}
            className="px-4 py-2 rounded-lg bg-scheme-shade_4 hover:bg-scheme-shade_5 border border-border-mid_contrast"
          >
            Reset
          </button>
          <button
            onClick={exportRows}
            className="px-4 py-2 rounded-lg bg-scheme-shade_4 hover:bg-scheme-shade_5 border border-border-mid_contrast"
          >
            Export CSV
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-5 gap-4 mb-6">
        <div className="rounded-2xl border border-border-mid_contrast bg-scheme-shade_3 p-4">
          <div className="text-sm text-text-less_contrast mb-1">Total feedback rows</div>
          <div className="text-2xl font-semibold">
            {loadingSummary || !summary ? "…" : summary.total_count}
          </div>
        </div>

        <div className="rounded-2xl border border-border-mid_contrast bg-scheme-shade_3 p-4">
          <div className="text-sm text-text-less_contrast mb-1">Rated rows</div>
          <div className="text-2xl font-semibold">
            {loadingSummary || !summary ? "…" : summary.rated_count}
          </div>
        </div>

        <div className="rounded-2xl border border-border-mid_contrast bg-scheme-shade_3 p-4">
          <div className="text-sm text-text-less_contrast mb-1">Average rating</div>
          <div className="text-2xl font-semibold">
            {loadingSummary || !summary
              ? "…"
              : summary.avg_rating === null
              ? "—"
              : summary.avg_rating}
          </div>
        </div>

        <div className="rounded-2xl border border-border-mid_contrast bg-scheme-shade_3 p-4">
          <div className="text-sm text-text-less_contrast mb-1">Rows with text</div>
          <div className="text-2xl font-semibold">
            {loadingSummary || !summary ? "…" : summary.has_text_count}
          </div>
        </div>

        <div className="rounded-2xl border border-border-mid_contrast bg-scheme-shade_3 p-4">
          <div className="text-sm text-text-less_contrast mb-1">Date range</div>
          <div className="text-sm leading-6">
            {loadingSummary || !summary ? (
              "Loading…"
            ) : (
              <>
                <div>{summary.date_min ? formatDate(summary.date_min) : "—"}</div>
                <div>{summary.date_max ? formatDate(summary.date_max) : "—"}</div>
              </>
            )}
          </div>
        </div>
      </div>

      <div className="rounded-2xl border border-border-mid_contrast bg-scheme-shade_3 p-5 mb-6">
        <h2 className="text-xl font-semibold mb-4">Rating distribution</h2>
        <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
          {ratingDistributionEntries.map(([rating, count]) => (
            <div
              key={rating}
              className="rounded-xl border border-border-mid_contrast bg-scheme-shade_4 p-4"
            >
              <div className="text-sm text-text-less_contrast mb-1">Rating {rating}</div>
              <div className="text-2xl font-semibold">
                {loadingSummary ? "…" : count}
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="rounded-2xl border border-border-mid_contrast bg-scheme-shade_3 p-5">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
          <h2 className="text-xl font-semibold">Feedback rows</h2>
          <div className="text-sm text-text-less_contrast">
            {loadingRows || !rowsData
              ? "Loading rows..."
              : `${rowsData.total_count} total rows`}
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="min-w-full text-sm border-separate border-spacing-y-2">
            <thead>
              <tr className="text-left">
                <th className="p-2">User</th>
                <th className="p-2">Conversation</th>
                <th className="p-2">Rating</th>
                <th className="p-2">Feedback</th>
                <th className="p-2">Snippet</th>
                <th className="p-2">Role</th>
                <th className="p-2">Model</th>
                <th className="p-2">Tool</th>
                <th className="p-2">Effective date</th>
              </tr>
            </thead>
            <tbody>
              {!loadingRows && rowsData && rowsData.rows.length === 0 ? (
                <tr>
                  <td colSpan={9} className="p-4 text-text-less_contrast">
                    No feedback rows matched the current filters.
                  </td>
                </tr>
              ) : null}

              {rowsData?.rows.map((row) => (
                <tr key={row.id} className="bg-scheme-shade_4">
                  <td className="p-2 rounded-l-xl align-top">{row.username}</td>
                  <td className="p-2 align-top">{row.conversation_name || "—"}</td>
                  <td className="p-2 align-top">{row.rating ?? "—"}</td>
                  <td className="p-2 align-top whitespace-pre-wrap">
                    {row.feedback_text || "—"}
                  </td>
                  <td className="p-2 align-top whitespace-pre-wrap">
                    {row.content_snippet || "—"}
                  </td>
                  <td className="p-2 align-top">{row.role}</td>
                  <td className="p-2 align-top">{row.model || "—"}</td>
                  <td className="p-2 align-top">{row.tool_call_name || "—"}</td>
                  <td className="p-2 rounded-r-xl align-top">
                    {formatDate(row.effective_date)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {rowsData ? (
          <div className="mt-5 flex items-center justify-between gap-3">
            <div className="text-sm text-text-less_contrast">
              Page {rowsData.page} of {rowsData.total_pages}
            </div>

            <div className="flex gap-2">
              <button
                onClick={() => goToPage(Math.max(1, rowsData.page - 1))}
                disabled={rowsData.page <= 1}
                className="px-4 py-2 rounded-lg border border-border-mid_contrast disabled:opacity-50"
              >
                Previous
              </button>
              <button
                onClick={() =>
                  goToPage(Math.min(rowsData.total_pages, rowsData.page + 1))
                }
                disabled={rowsData.page >= rowsData.total_pages}
                className="px-4 py-2 rounded-lg border border-border-mid_contrast disabled:opacity-50"
              >
                Next
              </button>
            </div>
          </div>
        ) : null}
      </div>
    </div>
  );
}
