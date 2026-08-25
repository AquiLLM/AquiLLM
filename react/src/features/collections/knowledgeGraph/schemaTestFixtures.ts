import type {
  CollectionSchemaEnvelope,
  EntityTypeDefinition,
  RelationTypeDefinition,
  SchemaConflictInfo,
  SchemaHistoryPage,
  SchemaValidationConstraints,
  ValidationResult,
} from './schemaTypes';

const constraints: SchemaValidationConstraints = {
  entity_fields: {
    name: { required: true, max_length: 64 },
    description: { max_length: 512 },
    default_retrieval_weight: { min: 0, max: 1 },
    default_suppression_threshold: { min: 0, max: 1 },
  },
  relation_fields: {
    name: { required: true, max_length: 64 },
    direction: { allowed_values: ['directed', 'undirected'] },
  },
};

const publishedEntity: EntityTypeDefinition = {
  key: 'person',
  origin: 'inherited',
  change_state: 'unchanged',
  capabilities: {
    editable_fields: ['description', 'aliases'],
    removable: false,
    renameable: false,
  },
  values: {
    name: 'person',
    description: 'A person entity',
    aliases: ['individual'],
    default_retrieval_weight: 0.8,
    default_suppression_policy: 'none',
    default_suppression_threshold: 0.2,
  },
};

const publishedRelation: RelationTypeDefinition = {
  key: 'works_for',
  origin: 'inherited',
  change_state: 'unchanged',
  capabilities: {
    editable_fields: ['description'],
    removable: false,
    renameable: false,
  },
  values: {
    name: 'works_for',
    description: 'Employment relation',
    direction: 'directed',
    allowed_head_types: ['person'],
    allowed_tail_types: ['organization'],
  },
};

const draftEntity: EntityTypeDefinition = {
  ...publishedEntity,
  change_state: 'changed',
  values: {
    ...publishedEntity.values,
    description: 'Updated person description',
  },
};

export const viewPublishedEnvelope: CollectionSchemaEnvelope = {
  collection_id: 'col-view',
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
  published: {
    version: 3,
    checksum: 'pub-view-checksum',
    entities: [publishedEntity],
    relations: [publishedRelation],
  },
  draft: null,
  constraints,
};

export const editDraftEnvelope: CollectionSchemaEnvelope = {
  collection_id: 'col-edit',
  permissions: {
    level: 'EDIT',
    can_create_draft: true,
    can_edit_definitions: true,
    can_validate: true,
    can_publish: false,
    can_discard_draft: false,
    can_restore: false,
    can_view_history: true,
  },
  published: {
    version: 4,
    checksum: 'pub-edit-checksum',
    entities: [publishedEntity],
    relations: [publishedRelation],
  },
  draft: {
    draft_id: 'draft-edit-1',
    revision: 2,
    base_published_checksum: 'pub-edit-checksum',
    last_editor: 'editor@example.test',
    updated_at: '2026-08-21T10:00:00Z',
    entities: [draftEntity],
    relations: [publishedRelation],
  },
  constraints,
};

export const emptyEditableEnvelope: CollectionSchemaEnvelope = {
  ...editDraftEnvelope,
  collection_id: 'col-empty',
  published: {
    version: 0,
    checksum: '',
    entities: [],
    relations: [],
  },
  draft: null,
};

export const manageDraftEnvelope: CollectionSchemaEnvelope = {
  ...editDraftEnvelope,
  collection_id: 'col-manage',
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
  draft: {
    ...editDraftEnvelope.draft!,
    draft_id: 'draft-manage-1',
    revision: 5,
  },
};

export const validationResultFixture: ValidationResult = {
  identity: {
    draft_id: 'draft-manage-1',
    revision: 5,
    candidate_checksum: 'candidate-checksum-v5',
    result_id: 'validation-result-1',
  },
  issues: [
    {
      code: 'alias_duplicate',
      location: 'entity.person.aliases',
      message: 'Alias already exists',
      severity: 'warning',
    },
  ],
  diff_summary: {
    base_version: 4,
    base_checksum: 'pub-edit-checksum',
    candidate_version: 5,
    candidate_checksum: 'candidate-checksum-v5',
    entities: { added: 0, changed: 1, removed: 0 },
    relations: { added: 0, changed: 0, removed: 0 },
  },
};

export const historyPageFixture: SchemaHistoryPage = {
  versions: [
    {
      version: 4,
      checksum: 'pub-edit-checksum',
      published_at: '2026-08-20T12:00:00Z',
      summary: 'Published person description baseline',
    },
    {
      version: 3,
      checksum: 'pub-view-checksum',
      published_at: '2026-08-19T12:00:00Z',
      summary: 'Initial inherited schema',
    },
  ],
  next_cursor: 'cursor-v2',
  has_more: true,
};

export const conflictInfoFixture: SchemaConflictInfo = {
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
};

export function envelopeAfterConflict(envelope: CollectionSchemaEnvelope): CollectionSchemaEnvelope {
  return {
    ...envelope,
    draft: envelope.draft
      ? {
          ...envelope.draft,
          revision: conflictInfoFixture.current_revision,
          entities: envelope.draft.entities.map((entity) =>
            entity.key === 'person'
              ? {
                  ...entity,
                  values: {
                    ...entity.values,
                    description: 'Server accepted description',
                  },
                }
              : entity,
          ),
        }
      : null,
  };
}
