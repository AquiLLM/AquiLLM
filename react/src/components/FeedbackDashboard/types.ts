
export interface FeedbackRow {
  id: number;
  message_uuid: string;
  conversation_id: number;
  conversation_name: string | null;
  user_id: number;
  username: string;
  rating: number | null;
  feedback_text: string | null;
  feedback_submitted_at: string | null;
  created_at: string;
  effective_date: string;
  role: string;
  content_snippet: string;
  model: string | null;
  tool_call_name: string | null;
  usage: number;
  has_feedback_text: boolean;
}

export interface SummaryMetrics {
  total_count: number;
  rated_count: number;
  avg_rating: number | null;
  rating_distribution: Record<string, number>;
  has_text_count: number;
  date_min: string | null;
  date_max: string | null;
}

export interface FilterOptions {
  users: Array<{ id: number; username: string }>;
  roles: string[];
  models: string[];
  tool_names: string[];
  ratings: number[];
}

export interface RowsResponse {
  rows: FeedbackRow[];
  page: number;
  page_size: number;
  total_count: number;
  total_pages: number;
}

// FilterState uses string for all scalar fields because html inputs are strings.
// the backend accepts empty string as "no filter" so we never need to convert
// to null before sending — we just omit keys whose value is empty string.
export interface FilterState {
  start_date: string;
  end_date: string;
  user_id: string;
  min_rating: string;
  max_rating: string;
  exact_rating: string;
  feedback_text_search: string;
  conversation_name_search: string;
  role: string;
  model: string;
  tool_call_name: string;
  has_feedback_text: string; // "true" | "false" | ""
  page: number;
  page_size: number;
}

export const EMPTY_FILTERS: FilterState = {
  start_date: '',
  end_date: '',
  user_id: '',
  min_rating: '',
  max_rating: '',
  exact_rating: '',
  feedback_text_search: '',
  conversation_name_search: '',
  role: '',
  model: '',
  tool_call_name: '',
  has_feedback_text: '',
  page: 1,
  page_size: 50,
};

export interface DashboardProps {
  apiRows: string;
  apiSummary: string;
  apiFilters: string;
  apiExport: string;
}