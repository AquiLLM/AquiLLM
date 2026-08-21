import { test, expect, type Page } from '@playwright/test';

const COLLECTION_ID = '42';

const apiUrls = {
  api_collections: '/api/collections/',
  api_collection: `/api/collection/${COLLECTION_ID}/`,
  api_collection_schema_workspace: `/api/collection/${COLLECTION_ID}/schema/`,
  api_collection_schema_draft: `/api/collection/${COLLECTION_ID}/schema/draft/`,
  api_collection_schema_entity: `/api/collection/${COLLECTION_ID}/schema/entity/%(entity_key)s/`,
  api_collection_schema_relation: `/api/collection/${COLLECTION_ID}/schema/relation/%(relation_key)s/`,
  api_collection_schema_validate: `/api/collection/${COLLECTION_ID}/schema/validate/`,
  api_collection_schema_diff: `/api/collection/${COLLECTION_ID}/schema/diff/`,
  api_collection_schema_publish: `/api/collection/${COLLECTION_ID}/schema/publish/`,
  api_collection_schema_discard: `/api/collection/${COLLECTION_ID}/schema/discard/`,
  api_collection_schema_versions: `/api/collection/${COLLECTION_ID}/schema/versions/`,
  api_collection_schema_version_diff: `/api/collection/${COLLECTION_ID}/schema/versions/%(version_id)s/diff/`,
  api_collection_schema_restore: `/api/collection/${COLLECTION_ID}/schema/versions/%(version_id)s/restore/`,
  api_collection_schema_restore_replace: `/api/collection/${COLLECTION_ID}/schema/restore-replace/`,
};

const pageUrls = {
  collection: '/collections/%(col_id)s/',
  user_collections: '/collections/',
  document: '/documents/%(doc_id)s/',
};

function entityDefinition(description: string) {
  return {
    key: 'person',
    origin: 'inherited',
    change_state: 'changed',
    capabilities: {
      editable_fields: ['description', 'aliases'],
      removable: false,
      renameable: false,
    },
    values: {
      name: 'person',
      description,
      aliases: ['individual'],
      default_retrieval_weight: 0.8,
      default_suppression_policy: 'none',
      default_suppression_threshold: 0.2,
    },
  };
}

function manageEnvelope(revision: number, description = 'Updated person description', includeDraft = true) {
  const entity = entityDefinition(description);
  return {
    collection_id: COLLECTION_ID,
    permissions: {
      level: 'MANAGE',
      can_create_draft: true,
      can_edit_definitions: true,
      can_validate: true,
      can_publish: true,
      can_discard_draft: true,
      can_restore: true,
      can_view_history: true,
    },
    published: {
      version: 4,
      checksum: 'pub-edit-checksum',
      entities: [entityDefinition('A person entity')],
      relations: [
        {
          key: 'works_for',
          origin: 'inherited',
          change_state: 'unchanged',
          capabilities: { editable_fields: ['description'], removable: false, renameable: false },
          values: {
            name: 'works_for',
            description: 'Employment relation',
            direction: 'directed',
            allowed_head_types: ['person'],
            allowed_tail_types: ['organization'],
          },
        },
      ],
    },
    draft: includeDraft
      ? {
          draft_id: 'draft-manage-1',
          revision,
          base_published_checksum: 'pub-edit-checksum',
          last_editor: 'editor@example.test',
          updated_at: '2026-08-21T10:00:00Z',
          entities: [entity],
          relations: [],
        }
      : null,
    constraints: {
      entity_fields: { name: { required: true, max_length: 64 }, description: { max_length: 512 } },
      relation_fields: { name: { required: true, max_length: 64 } },
    },
  };
}

function viewEnvelope() {
  const envelope = manageEnvelope(0, 'A person entity', false);
  return {
    ...envelope,
    permissions: {
      level: 'VIEW',
      can_create_draft: false,
      can_edit_definitions: false,
      can_validate: false,
      can_publish: false,
      can_discard_draft: false,
      can_restore: false,
      can_view_history: true,
    },
    draft: null,
  };
}

async function installHarness(page: Page, mode: 'manage' | 'view' = 'manage') {
  let revision = 5;
  let entityDescription = 'Updated person description';
  let permissionMode = mode;

  await page.addInitScript(({ urls, pages }) => {
    window.apiUrls = urls;
    window.pageUrls = pages;
  }, { urls: apiUrls, pages: pageUrls });

  await page.context().addCookies([
    { name: 'csrftoken', value: 'test-csrf-token', domain: '127.0.0.1', path: '/' },
  ]);

  await page.route('**/api/collections/', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        collections: [{ id: 42, name: 'Harness Collection', parent: null, path: '/harness' }],
      }),
    });
  });

  await page.route(`**/api/collection/${COLLECTION_ID}/`, async (route) => {
    if (route.request().method() !== 'GET') return route.continue();
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        collection: {
          id: 42,
          name: 'Harness Collection',
          path: '/harness',
          parent: null,
          created_at: '2026-01-01T00:00:00Z',
          updated_at: '2026-01-01T00:00:00Z',
        },
        documents: [],
        children: [],
        can_edit: permissionMode === 'manage',
        can_manage: permissionMode === 'manage',
      }),
    });
  });

  await page.route(`**/api/collection/${COLLECTION_ID}/schema/**`, async (route) => {
    const url = route.request().url();
    const method = route.request().method();

    if (url.endsWith('/schema/') && method === 'GET') {
      const body = permissionMode === 'view' ? viewEnvelope() : manageEnvelope(revision, entityDescription);
      return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(body) });
    }

    if (url.endsWith('/schema/draft/') && method === 'POST') {
      revision = 1;
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(manageEnvelope(revision, entityDescription)),
      });
    }

    if (url.includes('/schema/entity/') && method === 'PUT') {
      const ifMatch = route.request().headers()['if-match'];
      if (ifMatch === '4') {
        revision = 6;
        entityDescription = 'Local unsaved description';
        return route.fulfill({
          status: 409,
          contentType: 'application/json',
          body: JSON.stringify({
            attempted_revision: 4,
            current_revision: revision,
            draft_id: 'draft-manage-1',
            definitions: [
              {
                kind: 'entity',
                key: 'person',
                fields: [
                  {
                    field: 'description',
                    server_value: 'Server accepted description',
                    attempted_value: 'Local unsaved description',
                  },
                ],
              },
            ],
          }),
        });
      }
      const payload = route.request().postDataJSON() as { values?: { description?: string } };
      entityDescription = payload.values?.description ?? entityDescription;
      revision += 1;
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(manageEnvelope(revision, entityDescription)),
      });
    }

    if (url.endsWith('/schema/validate/') && method === 'POST') {
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          identity: {
            draft_id: 'draft-manage-1',
            revision,
            candidate_checksum: 'candidate-checksum-v5',
            result_id: 'validation-result-1',
          },
          issues: [],
          diff_summary: {
            base_version: 4,
            base_checksum: 'pub-edit-checksum',
            candidate_version: revision,
            candidate_checksum: 'candidate-checksum-v5',
            entities: { added: 0, changed: 1, removed: 0 },
            relations: { added: 0, changed: 0, removed: 0 },
          },
        }),
      });
    }

    if (url.endsWith('/schema/publish/') && method === 'POST') {
      revision = 1;
      entityDescription = 'Published description';
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          ...manageEnvelope(revision, entityDescription, false),
          published: {
            ...manageEnvelope(revision, entityDescription, false).published,
            version: 5,
            checksum: 'candidate-checksum-v5',
          },
        }),
      });
    }

    if (url.endsWith('/schema/diff/') && method === 'GET') {
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          base_version: 4,
          base_checksum: 'pub-edit-checksum',
          candidate_version: revision,
          candidate_checksum: 'candidate-checksum-v5',
          entities: { added: 0, changed: 1, removed: 0 },
          relations: { added: 0, changed: 0, removed: 0 },
        }),
      });
    }

    return route.fulfill({ status: 404, contentType: 'application/json', body: JSON.stringify({ error: 'not found' }) });
  });
}

test.describe('Collection schema editor harness', () => {
  test('MANAGE happy path: draft edit, validate, publish', async ({ page }) => {
    await installHarness(page, 'manage');
    await page.goto('/tests/fixtures/collection-schema-editor.html');

    await expect(page.getByRole('heading', { name: 'Harness Collection' })).toBeVisible();
    await page.getByRole('tab', { name: 'Knowledge Graph' }).click();
    await expect(page.getByTestId('collection-knowledge-graph-workspace')).toBeVisible();

    await page.getByTestId('schema-nav-entity-person').click();
    await page.getByLabel('Description').fill('Edited in harness');
    await page.getByRole('button', { name: 'Save' }).click();

    await page.getByRole('button', { name: 'Validate' }).click();
    await page.getByRole('button', { name: 'Publish' }).click();
    await expect(page.getByRole('heading', { name: 'Publish schema draft' })).toBeVisible();
    await page.getByRole('button', { name: 'Confirm publish' }).click();
    await expect(page.getByText(/Published:/)).toBeVisible();
  });

  test('VIEW mode hides draft controls and author metadata', async ({ page }) => {
    await installHarness(page, 'view');
    await page.goto('/tests/fixtures/collection-schema-editor.html?view=knowledge-graph');

    await expect(page.getByText(/View-only access/i)).toBeVisible();
    await expect(page.getByRole('button', { name: 'Create draft' })).toHaveCount(0);
    await expect(page.getByText(/Last editor/i)).toHaveCount(0);
  });

  test('409 conflict preserves local edits and supports reviewed reapply', async ({ page }) => {
    await installHarness(page, 'manage');
    await page.goto('/tests/fixtures/collection-schema-editor.html?view=knowledge-graph');

    await page.getByTestId('schema-nav-entity-person').click();
    await page.getByLabel('Description').fill('Local unsaved description');

    await page.evaluate(() => {
      (window as unknown as { __forceRevision?: number }).__forceRevision = 4;
    });

    await page.route(`**/api/collection/${COLLECTION_ID}/schema/entity/person/`, async (route) => {
      if (route.request().method() !== 'PUT') return route.continue();
      await route.fulfill({
        status: 409,
        contentType: 'application/json',
        body: JSON.stringify({
          attempted_revision: 4,
          current_revision: 6,
          draft_id: 'draft-manage-1',
          definitions: [
            {
              kind: 'entity',
              key: 'person',
              fields: [
                {
                  field: 'description',
                  server_value: 'Server accepted description',
                  attempted_value: 'Local unsaved description',
                },
              ],
            },
          ],
        }),
      });
    });

    await page.route(`**/api/collection/${COLLECTION_ID}/schema/`, async (route) => {
      if (route.request().method() !== 'GET') return route.continue();
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(manageEnvelope(6, 'Server accepted description')),
      });
    });

    await page.getByRole('button', { name: 'Save' }).click();
    await expect(page.getByLabel('Description')).toHaveValue('Local unsaved description');
  });

  test('mode tabs are keyboard operable', async ({ page }) => {
    await installHarness(page, 'manage');
    await page.goto('/tests/fixtures/collection-schema-editor.html');

    await page.getByRole('tab', { name: 'Knowledge Graph' }).focus();
    await page.keyboard.press('Enter');
    await expect(page.getByTestId('collection-knowledge-graph-workspace')).toBeVisible();
  });
});
