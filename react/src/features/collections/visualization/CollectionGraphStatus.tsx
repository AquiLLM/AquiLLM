import type { CollectionGraphEnvelope } from './collectionGraphTypes';
import { failureMessage } from './collectionGraphPresentation';

interface CollectionGraphStatusProps {
  instance: CollectionGraphEnvelope | null;
  message: string | null;
  rebuilding: boolean;
  onRebuild: () => void;
}

export function CollectionGraphStatus({ instance, message, rebuilding, onRebuild }: CollectionGraphStatusProps) {
  return (
    <>
      {instance?.progress && instance.progress.total > 0 && (
        <div role="status" className="rounded-[18px] border border-border-mid_contrast bg-scheme-shade_4 p-4">
          <p>{instance.progress.active} of {instance.progress.total} document graphs complete.</p>
          <p className="mt-1 text-sm text-text-lower_contrast">
            {instance.progress.ingesting} ingesting · {instance.progress.pending} waiting · {instance.progress.building} processing · {instance.progress.failed} failed
          </p>
          {instance.progress.failures.map((failure) => (
            <p key={failure.code} className="mt-1 text-sm">
              {failure.count} {failure.count === 1 ? 'document' : 'documents'}: {failureMessage(failure.code)}
            </p>
          ))}
        </div>
      )}
      {message && (
        <div className="rounded-[18px] border border-border-mid_contrast bg-scheme-shade_4 p-4">
          <p>{message}</p>
          {instance?.status.error_code && (
            <p className="mt-1 text-sm text-text-lower_contrast">
              Build error: <code>{instance.status.error_code}</code>
            </p>
          )}
          {instance?.permissions.can_rebuild && (
            <button
              type="button"
              onClick={onRebuild}
              disabled={rebuilding || instance.status.state === 'building'}
              className="mt-3 h-[36px] rounded-[18px] border border-border-mid_contrast px-4 disabled:opacity-50"
            >
              {rebuilding ? 'Starting rebuild…' : 'Rebuild graph'}
            </button>
          )}
        </div>
      )}
    </>
  );
}
