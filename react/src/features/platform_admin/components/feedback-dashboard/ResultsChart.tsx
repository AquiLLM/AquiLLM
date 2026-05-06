import React, { useEffect, useRef } from 'react';
import type { ChartData } from './types';

declare global {
  interface Window {
    Chart?: new (canvas: HTMLCanvasElement, config: unknown) => { destroy: () => void };
  }
}

const PALETTE = [
  'rgba(99,179,237,0.75)',
  'rgba(154,230,180,0.75)',
  'rgba(252,196,100,0.75)',
  'rgba(252,129,129,0.75)',
  'rgba(183,148,246,0.75)',
  'rgba(251,182,206,0.75)',
];

type Props = { data: ChartData };

const ResultsChart: React.FC<Props> = ({ data }) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const chartRef = useRef<{ destroy: () => void } | null>(null);

  useEffect(() => {
    if (!canvasRef.current || !window.Chart) return;

    const isDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    const tickColor = isDark ? '#bbb' : '#555';
    const gridColor = isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.08)';

    const datasets = data.datasets.map((ds, i) => ({
      label: ds.label,
      data: ds.data,
      backgroundColor: PALETTE[i % PALETTE.length],
      borderRadius: 4,
    }));

    chartRef.current = new window.Chart(canvasRef.current, {
      type: 'bar',
      data: { labels: data.labels, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: true,
        plugins: {
          legend: { labels: { color: tickColor, font: { size: 12 } } },
        },
        scales: {
          x: { ticks: { color: tickColor }, grid: { color: gridColor } },
          y: { ticks: { color: tickColor }, grid: { color: gridColor }, beginAtZero: true },
        },
      },
    });

    return () => {
      chartRef.current?.destroy();
      chartRef.current = null;
    };
  }, [data]);

  return (
    <div className="mb-6 p-4 rounded-lg bg-scheme-shade_3 element-border">
      <canvas ref={canvasRef} style={{ maxHeight: 340 }} />
    </div>
  );
};

export default ResultsChart;
