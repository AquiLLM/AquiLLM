export type SchemaPermissionLevel = 'VIEW' | 'EDIT' | 'MANAGE';

export type SchemaDefinitionOrigin = 'inherited' | 'collection' | 'generated';

export type SchemaDefinitionKind = 'entity' | 'relation';

export type SchemaDefinitionChangeState = 'unchanged' | 'added' | 'changed' | 'removed';

export interface SchemaDefinitionCapabilities {
  editable_fields: string[];
  removable: boolean;
  renameable: boolean;
}

export interface EntityTypeValues {
  name: string;
  description: string;
  aliases: string[];
  default_retrieval_weight: number;
  default_suppression_policy: string;
  default_suppression_threshold: number;
}

export interface RelationTypeValues {
  name: string;
  description: string;
  direction: 'directed' | 'undirected';
  allowed_head_types: string[];
  allowed_tail_types: string[];
}

export interface EntityTypeDefinition {
  key: string;
  origin: SchemaDefinitionOrigin;
  change_state: SchemaDefinitionChangeState;
  capabilities: SchemaDefinitionCapabilities;
  values: EntityTypeValues;
}

export interface RelationTypeDefinition {
  key: string;
  origin: SchemaDefinitionOrigin;
  change_state: SchemaDefinitionChangeState;
  capabilities: SchemaDefinitionCapabilities;
  values: RelationTypeValues;
}

export interface DraftIdentity {
  draft_id: string;
  revision: number;
}

export interface PublishedSchemaSnapshot {
  version: number;
  checksum: string;
}

export interface DraftSnapshot extends DraftIdentity {
  base_published_checksum: string;
  last_editor: string;
  updated_at: string;
  entities: EntityTypeDefinition[];
  relations: RelationTypeDefinition[];
}

export interface SchemaDiffCounts {
  added: number;
  changed: number;
  removed: number;
}

export interface SchemaDiffSummary {
  base_version: number;
  base_checksum: string;
  candidate_version: number;
  candidate_checksum: string;
  entities: SchemaDiffCounts;
  relations: SchemaDiffCounts;
}

export interface ValidationIdentity {
  draft_id: string;
  revision: number;
  candidate_checksum: string;
  result_id: string;
}

export interface ValidationIssue {
  code: string;
  location: string;
  message: string;
  severity: 'error' | 'warning';
}

export interface ValidationResult {
  identity: ValidationIdentity;
  issues: ValidationIssue[];
  diff_summary: SchemaDiffSummary;
}

export type PublishStatus = 'idle' | 'pending' | 'polling' | 'succeeded' | 'failed';

export type SchemaGenerationRunStatus = 'queued' | 'running' | 'succeeded' | 'failed';

export interface SchemaGenerationStart {
  run_id: string;
  status: SchemaGenerationRunStatus;
  status_url: string;
}

export interface SchemaGenerationStatus {
  run_id: string;
  status: SchemaGenerationRunStatus;
  error_code: string | null;
  statistics: Record<string, unknown>;
  workspace?: CollectionSchemaEnvelope;
}

export interface SchemaGenerationState {
  status: 'idle' | 'starting' | SchemaGenerationRunStatus;
  runId?: string;
  errorCode?: string | null;
  statistics?: Record<string, unknown>;
}

export interface PublishOperation {
  draft_id: string;
  revision: number;
  candidate_checksum: string;
  validation_result_id: string;
  status_url?: string;
}

export interface SchemaHistoryVersion {
  version: number;
  checksum: string;
  published_at: string;
  summary: string;
}

export interface SchemaHistoryPage {
  versions: SchemaHistoryVersion[];
  next_cursor: string | null;
  has_more: boolean;
}

export type CollectionSchemaClientErrorKind =
  | 'session_expired'
  | 'forbidden'
  | 'not_found'
  | 'revision_conflict'
  | 'validation_failed'
  | 'rate_limited'
  | 'server_error'
  | 'schema_unavailable'
  | 'invalid_response'
  | 'network_error';

export interface SchemaFieldConstraint {
  required?: boolean;
  min?: number;
  max?: number;
  max_length?: number;
  pattern?: string;
  allowed_values?: string[];
}

export interface SchemaValidationConstraints {
  entity_fields: Record<string, SchemaFieldConstraint>;
  relation_fields: Record<string, SchemaFieldConstraint>;
}

export interface SchemaPermissionsSnapshot {
  level: SchemaPermissionLevel;
  can_create_draft: boolean;
  can_edit_definitions: boolean;
  can_validate: boolean;
  can_publish: boolean;
  can_discard_draft: boolean;
  can_restore: boolean;
  can_view_history: boolean;
}

export interface CollectionSchemaEnvelope {
  collection_id: string;
  permissions: SchemaPermissionsSnapshot;
  published: PublishedSchemaSnapshot & {
    entities: EntityTypeDefinition[];
    relations: RelationTypeDefinition[];
  };
  draft: DraftSnapshot | null;
  constraints: SchemaValidationConstraints;
}

export interface SchemaConflictField {
  field: string;
  server_value: unknown;
  attempted_value: unknown;
}

export interface SchemaConflictDefinition {
  kind: SchemaDefinitionKind;
  key: string;
  fields: SchemaConflictField[];
}

export interface SchemaConflictInfo {
  attempted_revision: number;
  current_revision: number;
  draft_id: string;
  definitions: SchemaConflictDefinition[];
}

export type SelectedSchemaDefinition =
  | { kind: 'entity'; definition: EntityTypeDefinition }
  | { kind: 'relation'; definition: RelationTypeDefinition };
