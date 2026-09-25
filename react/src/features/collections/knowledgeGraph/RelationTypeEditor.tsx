import React, { useId, useMemo, useState } from 'react';
import type {
  EntityTypeDefinition,
  RelationTypeDefinition,
  SchemaValidationConstraints,
} from './schemaTypes';
import { validateRelationForm } from './schemaFieldValidation';
import SchemaEntityTypePicker from './SchemaEntityTypePicker';
import SchemaFieldErrorSummary from './SchemaFieldErrorSummary';
import SchemaModalShell from './SchemaModalShell';
import {
  buttonDangerClass,
  buttonPrimaryClass,
  buttonSecondaryClass,
  changeStateLabel,
  inputClass,
  originLabel,
  panelClass,
  textareaClass,
} from './schemaUiShared';

export interface RelationTypeEditorProps {
  collectionName: string;
  draftRevision: number | null;
  definition: RelationTypeDefinition;
  entityTypes: EntityTypeDefinition[];
  constraints: SchemaValidationConstraints;
  values: Record<string, unknown>;
  serverErrors?: Record<string, string>;
  conflictFields?: string[];
  dirty: boolean;
  pending: boolean;
  readOnly: boolean;
  onFieldChange: (field: string, value: unknown) => void;
  onSave: () => void;
  onRevert: () => void;
  onRemove?: () => void;
  onCancel?: () => void;
}

function isEditable(definition: RelationTypeDefinition, field: string, readOnly: boolean): boolean {
  if (readOnly) return false;
  if (field === 'name') return definition.capabilities.renameable;
  return definition.capabilities.editable_fields.includes(field);
}

const RelationTypeEditor: React.FC<RelationTypeEditorProps> = ({
  collectionName,
  draftRevision,
  definition,
  entityTypes,
  constraints,
  values,
  serverErrors = {},
  conflictFields = [],
  dirty,
  pending,
  readOnly,
  onFieldChange,
  onSave,
  onRevert,
  onRemove,
  onCancel,
}) => {
  const headingId = useId();
  const [removeOpen, setRemoveOpen] = useState(false);
  const localErrors = useMemo(
    () => validateRelationForm(values, constraints.relation_fields),
    [constraints.relation_fields, values],
  );
  const errors = { ...localErrors, ...serverErrors };
  const headTypes = Array.isArray(values.allowed_head_types)
    ? (values.allowed_head_types as string[])
    : [];
  const tailTypes = Array.isArray(values.allowed_tail_types)
    ? (values.allowed_tail_types as string[])
    : [];

  const focusField = (field: string) => {
    document.getElementById(`relation-field-${field}`)?.focus();
  };

  return (
    <section className={`${panelClass} p-[16px]`} aria-labelledby={headingId}>
      <header className="mb-3">
        <h2 id={headingId} className="text-lg font-semibold">
          Relation type: {definition.values.name}
        </h2>
        <p className="text-sm text-text-lower_contrast">
          {originLabel(definition.origin)} · {changeStateLabel(definition.change_state)}
        </p>
      </header>

      <SchemaFieldErrorSummary errors={errors} onFocusField={focusField} headingId={`${headingId}-errors`} />

      <div className="space-y-3">
        <div>
          <label htmlFor="relation-field-name" className="text-sm font-medium">
            Name
          </label>
          <input
            id="relation-field-name"
            className={inputClass}
            value={String(values.name ?? '')}
            disabled={!isEditable(definition, 'name', readOnly) || pending}
            onChange={(event) => onFieldChange('name', event.target.value)}
          />
        </div>
        <div>
          <label htmlFor="relation-field-description" className="text-sm font-medium">
            Description
          </label>
          <textarea
            id="relation-field-description"
            className={textareaClass}
            value={String(values.description ?? '')}
            disabled={!isEditable(definition, 'description', readOnly) || pending}
            onChange={(event) => onFieldChange('description', event.target.value)}
          />
        </div>
        <div>
          <label htmlFor="relation-field-direction" className="text-sm font-medium">
            Direction
          </label>
          <select
            id="relation-field-direction"
            className={inputClass}
            value={String(values.direction ?? 'directed')}
            disabled={!isEditable(definition, 'direction', readOnly) || pending}
            onChange={(event) => onFieldChange('direction', event.target.value)}
          >
            <option value="directed">Directed</option>
            <option value="undirected">Undirected</option>
          </select>
        </div>
        <SchemaEntityTypePicker
          id="relation-field-allowed_head_types"
          label="Allowed head entity types"
          entityTypes={entityTypes}
          selectedKeys={headTypes}
          disabled={!isEditable(definition, 'allowed_head_types', readOnly) || pending}
          required
          error={errors.allowed_head_types ?? null}
          onChange={(keys) => onFieldChange('allowed_head_types', keys)}
        />
        <SchemaEntityTypePicker
          id="relation-field-allowed_tail_types"
          label="Allowed tail entity types"
          entityTypes={entityTypes}
          selectedKeys={tailTypes}
          disabled={!isEditable(definition, 'allowed_tail_types', readOnly) || pending}
          required
          error={errors.allowed_tail_types ?? null}
          onChange={(keys) => onFieldChange('allowed_tail_types', keys)}
        />
      </div>

      {conflictFields.length > 0 ? (
        <p className="text-sm text-amber-200 mt-3" role="status">
          Conflicting fields: {conflictFields.join(', ')}
        </p>
      ) : null}

      {!readOnly ? (
        <div className="mt-4 flex flex-wrap gap-2">
          <button type="button" className={buttonPrimaryClass} disabled={!dirty || pending || Object.keys(errors).length > 0} onClick={onSave}>
            {pending ? 'Saving…' : 'Save'}
          </button>
          <button type="button" className={buttonSecondaryClass} disabled={!dirty || pending} onClick={onRevert}>
            Revert
          </button>
          {definition.capabilities.removable && onRemove ? (
            <button type="button" className={buttonDangerClass} disabled={pending} onClick={() => setRemoveOpen(true)}>
              Remove
            </button>
          ) : null}
          {onCancel ? (
            <button type="button" className={buttonSecondaryClass} disabled={pending} onClick={onCancel}>
              Cancel
            </button>
          ) : null}
        </div>
      ) : (
        <p className="mt-4 text-sm text-text-lower_contrast">Read-only: server capabilities prevent editing.</p>
      )}

      <SchemaModalShell
        isOpen={removeOpen}
        title="Remove relation type"
        onClose={() => setRemoveOpen(false)}
        footer={
          <>
            <button type="button" className={buttonSecondaryClass} onClick={() => setRemoveOpen(false)}>
              Cancel
            </button>
            <button
              type="button"
              className={buttonDangerClass}
              onClick={() => {
                setRemoveOpen(false);
                onRemove?.();
              }}
            >
              Confirm removal
            </button>
          </>
        }
      >
        <p>
          Remove <strong>{definition.values.name}</strong> from the shared draft for{' '}
          <strong>{collectionName}</strong> at revision <strong>{draftRevision ?? '—'}</strong>?
        </p>
      </SchemaModalShell>
    </section>
  );
};

export default RelationTypeEditor;
