interface CollectionGraphScopeProps {
  connected: number;
  unconnected: number;
  bounded: boolean;
  includeUnconnected: boolean;
  onChange: (includeUnconnected: boolean) => void;
}

export function CollectionGraphScope({
  connected, unconnected, bounded, includeUnconnected, onChange,
}: CollectionGraphScopeProps) {
  return (
    <div className="flex flex-wrap items-center gap-2 rounded-[18px] border border-border-low_contrast bg-scheme-shade_4 p-[10px]">
      <div className="flex gap-2" aria-label="Instance graph scope">
        <button
          type="button"
          aria-pressed={!includeUnconnected}
          onClick={() => onChange(false)}
          className={`h-[36px] rounded-[18px] border px-4 ${!includeUnconnected ? 'bg-accent text-white border-accent' : 'border-border-mid_contrast'}`}
        >
          Connected
        </button>
        <button
          type="button"
          aria-label="Include unconnected"
          aria-pressed={includeUnconnected}
          onClick={() => onChange(true)}
          className={`h-[36px] rounded-[18px] border px-4 ${includeUnconnected ? 'bg-accent text-white border-accent' : 'border-border-mid_contrast'}`}
        >
          All entities
        </button>
      </div>
      <span className="text-sm text-text-lower_contrast">
        {connected} connected · {unconnected}{' '}unconnected
        {bounded ? ' in this bounded result' : ''}
      </span>
    </div>
  );
}
