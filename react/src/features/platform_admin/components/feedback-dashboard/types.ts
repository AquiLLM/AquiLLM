export type QueryResultRow = {
  cells: (string | number | boolean | null)[];
  conversation_id?: string;
  message_uuid?: string;
};

export type ChartData = {
  labels: string[];
  datasets: { label: string; data: (number | null)[] }[];
};

export type QueryResponse = {
  query_text: string;
  rows: QueryResultRow[];
  columns: string[];
  is_row_level: boolean;
  chart_data: ChartData | null;
  row_count: number;
  error?: string;
  notice?: string | null;
};

export type ThreadMessage = {
  message_uuid: string;
  role: 'user' | 'assistant' | 'tool' | string;
  content: string;
  model: string | null;
  sequence_number: number;
  created_at: string | null;
  rating: number | null;
  feedback_text: string | null;
  tool_call_name: string | null;
  tool_call_input: unknown;
  tool_name: string | null;
  result_dict: { exception?: unknown; result?: unknown } | null;
  for_whom: string | null;
};
