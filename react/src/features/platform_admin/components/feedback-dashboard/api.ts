import type { QueryResponse, ThreadMessage } from './types';

export async function runQuery(encodedQuery: string): Promise<QueryResponse> {
  const url = window.apiUrls.api_feedback_dashboard_query;
  const res = await fetch(`${url}?q=${encodeURIComponent(encodedQuery)}`);
  const data = await res.json();
  if (!res.ok) {
    return {
      query_text: data.query_text || '',
      rows: [],
      columns: [],
      is_row_level: true,
      chart_data: null,
      row_count: 0,
      error: data.error || 'Request failed',
    };
  }
  return data as QueryResponse;
}

export async function fetchConversation(convId: string): Promise<ThreadMessage[]> {
  const url = window.apiUrls.api_feedback_dashboard_conversation;
  const res = await fetch(`${url}?id=${encodeURIComponent(convId)}`);
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.error || 'Failed to load conversation');
  }
  return (data.messages || []) as ThreadMessage[];
}

// UTF-8-safe base64 encode (matches Python's base64.b64encode on UTF-8 bytes).
export function b64encode(str: string): string {
  return btoa(
    encodeURIComponent(str).replace(/%([0-9A-F]{2})/g, (_, h) =>
      String.fromCharCode(parseInt(h, 16)),
    ),
  );
}

export function b64decode(str: string): string {
  try {
    return decodeURIComponent(
      Array.from(atob(str))
        .map((c) => '%' + c.charCodeAt(0).toString(16).padStart(2, '0'))
        .join(''),
    );
  } catch {
    return '';
  }
}
