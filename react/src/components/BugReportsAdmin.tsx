import React, { useState, useEffect } from 'react';
import { Trash2, ChevronDown, ChevronRight, ExternalLink } from 'lucide-react';
import formatUrl from '../utils/formatUrl';
import { getCsrfCookie } from '../main';

interface BugReportSummary {
  id: number;
  title: string;
  user: string | null;
  source: string;
  trace_id: string;
  created_at: string;
}

interface BugReportDetail {
  id: number;
  title: string;
  description: string;
  url: string;
  user: string | null;
  source: string;
  user_agent: string;
  trace_id: string;
  grafana_trace_url: string;
  grafana_user_traces_url: string;
  created_at: string;
}

const BugReportsAdmin: React.FC = () => {
  const [reports, setReports] = useState<BugReportSummary[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(25);
  const [sourceFilter, setSourceFilter] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [expandedId, setExpandedId] = useState<number | null>(null);
  const [detail, setDetail] = useState<BugReportDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);

  const fetchReports = async () => {
    setLoading(true);
    try {
      let url = window.apiUrls.api_bug_reports_list + `?page=${page}`;
      if (sourceFilter) url += `&source=${sourceFilter}`;
      const resp = await fetch(url, { credentials: 'include' });
      if (resp.ok) {
        const data = await resp.json();
        setReports(data.results);
        setTotal(data.total);
        setPageSize(data.page_size);
      }
    } catch {
      // ignore
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchReports();
  }, [page, sourceFilter]);

  const handleExpand = async (id: number) => {
    if (expandedId === id) {
      setExpandedId(null);
      setDetail(null);
      return;
    }
    setExpandedId(id);
    setDetail(null);
    setDetailLoading(true);
    try {
      const url = formatUrl(window.apiUrls.api_bug_report_detail, { report_id: id });
      const resp = await fetch(url, { credentials: 'include' });
      if (resp.ok) {
        setDetail(await resp.json());
      }
    } catch {
      // ignore
    } finally {
      setDetailLoading(false);
    }
  };

  const handleDelete = async (id: number) => {
    try {
      const url = formatUrl(window.apiUrls.api_bug_report_delete, { report_id: id });
      const resp = await fetch(url, {
        method: 'DELETE',
        credentials: 'include',
        headers: { 'X-CSRFToken': getCsrfCookie() },
      });
      if (resp.ok) {
        setReports(prev => prev.filter(r => r.id !== id));
        if (expandedId === id) {
          setExpandedId(null);
          setDetail(null);
        }
      }
    } catch {
      // ignore
    }
  };

  const totalPages = Math.ceil(total / pageSize);

  const formatTime = (iso: string) => {
    const d = new Date(iso);
    return d.toLocaleString();
  };

  return (
    <div className="w-full p-4 md:px-8">
      <h2 className="mb-6 text-center text-xl font-bold text-text-normal">Bug Reports</h2>

      <div className="mx-auto max-w-4xl">
        {/* Filter */}
        <div className="mb-4 flex items-center gap-3">
          <label className="text-text-low_contrast text-sm">Source:</label>
          <select
            value={sourceFilter}
            onChange={(e) => { setSourceFilter(e.target.value); setPage(1); }}
            className="p-2 bg-scheme-shade_4 border border-border-mid_contrast rounded text-text-normal text-sm"
          >
            <option value="">All</option>
            <option value="user">User</option>
            <option value="exception">Exception</option>
          </select>
          <span className="text-text-low_contrast text-sm ml-auto">{total} report{total !== 1 ? 's' : ''}</span>
        </div>

        {loading && <div className="text-text-low_contrast">Loading...</div>}

        {/* Table */}
        <div className="border border-border-mid_contrast rounded-lg overflow-hidden">
          <table className="w-full text-sm text-text-normal">
            <thead>
              <tr className="bg-scheme-shade_4 border-b border-border-mid_contrast">
                <th className="p-3 text-left w-8"></th>
                <th className="p-3 text-left">Title</th>
                <th className="p-3 text-left w-28">User</th>
                <th className="p-3 text-left w-24">Source</th>
                <th className="p-3 text-left w-44">Created</th>
                <th className="p-3 text-left w-12"></th>
              </tr>
            </thead>
            <tbody>
              {reports.map(r => (
                <React.Fragment key={r.id}>
                  <tr
                    className="border-b border-border-mid_contrast hover:bg-scheme-shade_4 cursor-pointer transition-colors"
                    onClick={() => handleExpand(r.id)}
                  >
                    <td className="p-3">
                      {expandedId === r.id ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
                    </td>
                    <td className="p-3 font-medium">{r.title}</td>
                    <td className="p-3 text-text-low_contrast">{r.user || '—'}</td>
                    <td className="p-3">
                      <span className={`px-2 py-0.5 rounded text-xs font-semibold ${
                        r.source === 'exception'
                          ? 'bg-scheme-shade_5 text-red'
                          : 'bg-scheme-shade_5 text-text-normal'
                      }`}>
                        {r.source}
                      </span>
                    </td>
                    <td className="p-3 text-text-low_contrast">{formatTime(r.created_at)}</td>
                    <td className="p-3">
                      <button
                        onClick={(e) => { e.stopPropagation(); handleDelete(r.id); }}
                        className="bg-transparent border-none cursor-pointer p-1 text-text-low_contrast hover:text-red-400 transition-colors"
                        aria-label="Delete report"
                      >
                        <Trash2 size={16} />
                      </button>
                    </td>
                  </tr>
                  {expandedId === r.id && (
                    <tr>
                      <td colSpan={6} className="bg-scheme-shade_4 p-4">
                        {detailLoading ? (
                          <div className="text-text-low_contrast">Loading details...</div>
                        ) : detail ? (
                          <div className="flex flex-col gap-4">
                            {detail.description && (
                              <div>
                                <h4 className="text-text-low_contrast text-xs uppercase mb-1">Description</h4>
                                <p className="text-text-normal whitespace-pre-wrap">{detail.description}</p>
                              </div>
                            )}

                            <div>
                              <h4 className="text-text-low_contrast text-xs uppercase mb-1">URL</h4>
                              <p className="text-text-normal text-sm break-all">{detail.url || '—'}</p>
                            </div>

                            {detail.user_agent && (
                              <div>
                                <h4 className="text-text-low_contrast text-xs uppercase mb-1">User Agent</h4>
                                <p className="text-text-normal text-xs break-all">{detail.user_agent}</p>
                              </div>
                            )}

                            {/* Grafana Trace Links */}
                            {(detail.trace_id || detail.grafana_user_traces_url) && (
                              <div>
                                <h4 className="text-text-low_contrast text-xs uppercase mb-2">Traces</h4>
                                <div className="flex flex-wrap gap-2">
                                  {detail.trace_id && (
                                    <a
                                      href={detail.grafana_trace_url}
                                      target="_blank"
                                      rel="noopener noreferrer"
                                      className="inline-flex items-center gap-2 px-3 py-2 bg-scheme-shade_3 border border-border-mid_contrast rounded text-sm text-accent hover:bg-scheme-shade_5 transition-colors"
                                    >
                                      <ExternalLink size={14} />
                                      View this trace
                                      <span className="text-text-low_contrast text-xs font-mono ml-1">{detail.trace_id.slice(0, 16)}...</span>
                                    </a>
                                  )}
                                  {detail.grafana_user_traces_url && (
                                    <a
                                      href={detail.grafana_user_traces_url}
                                      target="_blank"
                                      rel="noopener noreferrer"
                                      className="inline-flex items-center gap-2 px-3 py-2 bg-scheme-shade_3 border border-border-mid_contrast rounded text-sm text-accent hover:bg-scheme-shade_5 transition-colors"
                                    >
                                      <ExternalLink size={14} />
                                      User's recent traces
                                    </a>
                                  )}
                                </div>
                              </div>
                            )}

                          </div>
                        ) : null}
                      </td>
                    </tr>
                  )}
                </React.Fragment>
              ))}
              {!loading && reports.length === 0 && (
                <tr>
                  <td colSpan={6} className="p-8 text-center text-text-low_contrast">
                    No bug reports found.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="flex justify-center gap-2 mt-4">
            <button
              disabled={page <= 1}
              onClick={() => setPage(p => p - 1)}
              className="px-3 py-1 rounded bg-scheme-shade_4 border border-border-mid_contrast text-text-normal text-sm cursor-pointer disabled:opacity-50"
            >
              Previous
            </button>
            <span className="px-3 py-1 text-text-low_contrast text-sm">
              Page {page} of {totalPages}
            </span>
            <button
              disabled={page >= totalPages}
              onClick={() => setPage(p => p + 1)}
              className="px-3 py-1 rounded bg-scheme-shade_4 border border-border-mid_contrast text-text-normal text-sm cursor-pointer disabled:opacity-50"
            >
              Next
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default BugReportsAdmin;
