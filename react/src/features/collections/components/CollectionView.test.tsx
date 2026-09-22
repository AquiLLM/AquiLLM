// @vitest-environment jsdom

import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import CollectionView from './CollectionView';

// Drive the upload completion boundary without uploading a file to a server.
let completeUpload: () => void;
vi.mock('../../../components/IngestRow', () => ({
  default: ({ onUploadSuccess }: { onUploadSuccess: () => void }) => {
    completeUpload = onUploadSuccess;
    return <button onClick={onUploadSuccess}>Complete upload</button>;
  },
}));

const inheritedPermission = {
  direct: false,
  source_collection_id: 2,
  source_collection_name: 'Parent Research',
  permission_level: 'VIEW',
};

function payload(id = 7, name = 'Research', permissionSource: unknown = inheritedPermission) {
  return {
    collection: {
      id, name, parent: null, path: '/research',
      created_at: '2026-01-01T00:00:00Z', updated_at: '2026-01-02T00:00:00Z',
    },
    children: [],
    documents: [{ id: 11, title: 'Study notes', type: 'document', created_at: '2026-01-01' }],
    permission_source: permissionSource,
  };
}

const response = (body: unknown, ok = true) => ({ ok, json: async () => body }) as Response;
let collectionResponse: () => Promise<Response>;

beforeEach(() => {
  window.apiUrls = {
    api_collection: '/api/collection/%(col_id)s/',
    api_collections: '/api/collections/',
  };
  window.pageUrls = { collection: '/collection/%(col_id)s/' };
  collectionResponse = async () => response(payload());
  vi.stubGlobal('fetch', vi.fn((input: RequestInfo | URL) => {
    if (String(input).startsWith('/api/collection/')) return collectionResponse();
    if (String(input) === '/api/collections/') return Promise.resolve(response({ collections: [] }));
    return Promise.reject(new Error(`Unexpected fetch: ${String(input)}`));
  }));
  vi.spyOn(console, 'error').mockImplementation(() => undefined);
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe('CollectionView data recovery', () => {
  it('recovers after a failed refresh instead of retaining its error', async () => {
    render(<CollectionView collectionId="7" />);
    await screen.findByRole('heading', { name: 'Research' });
    collectionResponse = async () => response({ error: 'Forbidden' }, false);
    fireEvent.click(screen.getByRole('button', { name: 'Complete upload' }));
    await screen.findByText('Error: Forbidden');
    expect(screen.queryByText('Study notes')).toBeNull();

    collectionResponse = async () => response(payload());
    await act(async () => completeUpload());

    await screen.findByRole('heading', { name: 'Research' });
    expect(screen.queryByText('Error: Forbidden')).toBeNull();
  });

  it('clears inherited permission when a refreshed response no longer supplies it', async () => {
    render(<CollectionView collectionId="7" />);
    await screen.findByText('Parent Research');
    // A response with no permission_source differs from a retained inherited grant.
    collectionResponse = async () => response({ ...payload(7, 'Updated research'), permission_source: undefined });

    fireEvent.click(screen.getByRole('button', { name: 'Complete upload' }));

    await screen.findByRole('heading', { name: 'Updated research' });
    expect(screen.queryByText('Parent Research')).toBeNull();
  });

  it('loads a different collection after an error without carrying the old permission', async () => {
    const { rerender } = render(<CollectionView collectionId="7" />);
    await screen.findByText('Parent Research');
    collectionResponse = async () => response({ error: 'Access revoked' }, false);
    fireEvent.click(screen.getByRole('button', { name: 'Complete upload' }));
    await screen.findByText('Error: Access revoked');

    collectionResponse = async () => response({ ...payload(12, 'Other research'), permission_source: undefined, documents: [] });
    rerender(<CollectionView collectionId="12" />);

    await screen.findByRole('heading', { name: 'Other research' });
    expect(screen.queryByText('Parent Research')).toBeNull();
    expect(screen.queryByText('Study notes')).toBeNull();
  });
});
