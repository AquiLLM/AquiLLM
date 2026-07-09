import React from 'react';

interface LivePRQLDisplayProps {
  prql: string;
  loading: boolean;
}

const LivePRQLDisplay: React.FC<LivePRQLDisplayProps> = ({ prql, loading }) => {
  return (
    <div style={{
      border: '1px solid #888',
      borderRadius: '10px',
      overflow: 'hidden',
      flexShrink: 0,
      width: '100%',
      backgroundColor: 'var(--color-scheme-shade-3, #f0f0f0)',
    }}>
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '8px 16px',
        borderBottom: '1px solid #ccc',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{
            fontFamily: 'monospace',
            fontSize: '11px',
            fontWeight: 700,
            padding: '2px 8px',
            borderRadius: '4px',
            backgroundColor: '#4a90d9',
            color: 'white',
          }}>
            PRQL
          </span>
          <span style={{ fontSize: '11px', color: '#888' }}>
            {loading
              ? 'updating...'
              : prql
              ? 'live query — updates with every filter change'
              : 'loading query...'}
          </span>
        </div>
        <span style={{ fontSize: '11px', color: '#888', fontFamily: 'monospace' }}>
          compiled by prql-python → SQL → PostgreSQL
        </span>
      </div>

      <div style={{
        minHeight: '100px',
        backgroundColor: 'var(--color-scheme-shade-4, #e8e8e8)',
      }}>
        {loading ? (
          <div style={{
            padding: '12px 16px',
            fontFamily: 'monospace',
            fontSize: '11px',
            color: '#888',
            minHeight: '100px',
            display: 'flex',
            alignItems: 'center',
          }}>
            updating...
          </div>
        ) : prql ? (
          <pre style={{
            margin: 0,
            padding: '12px 16px',
            fontSize: '12px',
            fontFamily: 'monospace',
            lineHeight: '1.6',
            overflowX: 'auto',
            overflowY: 'auto',
            whiteSpace: 'pre',
            maxHeight: '300px',
            color: 'var(--color-accent-DEFAULT, #4a90d9)',
            minHeight: '60px',
          }}>
            {prql}
          </pre>
        ) : (
          <div style={{
            padding: '12px 16px',
            fontFamily: 'monospace',
            fontSize: '11px',
            color: '#888',
            minHeight: '100px',
            display: 'flex',
            alignItems: 'center',
          }}>
            waiting for first query...
          </div>
        )}
      </div>

      <div style={{
        padding: '6px 16px',
        borderTop: '1px solid #ccc',
        backgroundColor: 'var(--color-scheme-shade-3, #f0f0f0)',
      }}>
        <p style={{
          fontSize: '11px',
          color: '#888',
          margin: 0,
          lineHeight: '1.4',
        }}>
          Lines starting with <code style={{ fontFamily: 'monospace' }}>#</code> are
          display annotations — text search and pagination applied as SQL after compilation.
        </p>
      </div>
    </div>
  );
};

export default LivePRQLDisplay;