import type { VisualizationEdge, VisualizationNode } from './collectionGraphTypes';

interface CollectionGraphDetailsProps {
  selected: VisualizationNode | VisualizationEdge | null;
}

export function CollectionGraphDetails({ selected }: CollectionGraphDetailsProps) {
  return (
    <aside className="rounded-[18px] border border-border-low_contrast bg-scheme-shade_4 p-4">
      {!selected ? (
        <p>Select a node or relation to inspect its details and evidence.</p>
      ) : (
        <>
          <h2 className="text-lg font-semibold">{selected.label}</h2>
          <p className="text-sm text-text-lower_contrast">{selected.kind}</p>
          {'type' in selected && (
            <p className="mt-2">Type: {selected.type}</p>
          )}
          {selected.description && (
            <p className="mt-2">{selected.description}</p>
          )}
          {selected.confidence !== undefined && (
            <p className="mt-2">
              Confidence: {(selected.confidence * 100).toFixed(1)}%
            </p>
          )}
          {'supportCount' in selected && selected.supportCount !== undefined && (
            <p className="mt-1">Supporting passages: {selected.supportCount}</p>
          )}
          {'evidence' in selected && selected.evidence && selected.evidence.length > 0 && (
            <div className="mt-4 space-y-3">
              <h3 className="font-semibold">Evidence</h3>
              {selected.evidence.map((evidence) => (
                <details
                  key={`${evidence.document_id}:${evidence.chunk_id}:${evidence.start}`}
                  className="rounded border border-border-low_contrast p-2"
                  open
                >
                  <summary className="cursor-pointer text-sm">
                    Document {evidence.document_id} · chunk {evidence.chunk_id}
                  </summary>
                  <p className="mt-2 whitespace-pre-wrap text-sm">{evidence.excerpt}</p>
                </details>
              ))}
            </div>
          )}
        </>
      )}
    </aside>
  );
}
