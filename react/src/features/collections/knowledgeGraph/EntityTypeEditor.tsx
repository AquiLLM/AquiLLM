import React, { useId, useMemo, useState } from 'react';
import type { EntityTypeDefinition, SchemaValidationConstraints } from './schemaTypes';
import { validateEntityForm } from './schemaFieldValidation';
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

export interface EntityTypeEditorProps {
  collectionName: string;
  draftRevision: number | null;
  definition: EntityTypeDefinition;
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

function isEditable(definition: EntityTypeDefinition, field: string, readOnly: boolean): boolean {
  if (readOnly) return false;
  if (field === 'name') return definition.capabilities.renameable;
  return definition.capabilities.editable_fields.includes(field);
}

const EntityTypeEditor: React.FC<EntityTypeEditorProps> = ({
  collectionName,
  draftRevision,
  definition,
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
    () => validateEntityForm(values, constraints.entity_fields),
    [constraints.entity_fields, values],
  );
  const errors = { ...localErrors, ...serverErrors };
  const aliases = Array.isArray(values.aliases) ? (values.aliases as string[]) : [];

  const focusField = (field: string) => {
    document.getElementById(`entity-field-${field}`)?.focus();
  };

  const updateAlias = (index: number, next: string) => {
    const copy = [...aliases];
    copy[index] = next;
    onFieldChange('aliases', copy);
  };

  const addAlias = () => onFieldChange('aliases', [...aliases, '']);
  const removeAlias = (index: number) => onFieldChange('aliases', aliases.filter((_, i) => i !== index));

  return (
    <section className={`${panelClass} p-[16px]`} aria-labelledby={headingId}>
      <header className="mb-3">
        <h2 id={headingId} className="text-lg font-semibold">
          Entity type: {definition.values.name}
        </h2>
        <p className="text-sm text-text-lower_contrast">
          {originLabel(definition.origin)} · {changeStateLabel(definition.change_state)}
        </p>
      </header>

      <SchemaFieldErrorSummary errors={errors} onFocusField={focusField} headingId={`${headingId}-errors`} />

      <div className="space-y-3">
        <div>
          <label htmlFor="entity-field-name" className="text-sm font-medium">
            Name
          </label>
          <input
            id="entity-field-name"
            className={inputClass}
            value={String(values.name ?? '')}
            disabled={!isEditable(definition, 'name', readOnly) || pending}
            aria-invalid={Boolean(errors.name)}
            onChange={(event) => onFieldChange('name', event.target.value)}
          />
        </div>
        <div>
          <label htmlFor="entity-field-description" className="text-sm font-medium">
            Description
          </label>
          <textarea
            id="entity-field-description"
            className={textareaClass}
            value={String(values.description ?? '')}
            disabled={!isEditable(definition, 'description', readOnly) || pending}
            aria-invalid={Boolean(errors.description)}
            onChange={(event) => onFieldChange('description', event.target.value)}
          />
        </div>
        <div>
          <span className="text-sm font-medium">Aliases</span>
          <ul className="space-y-2 mt-1 list-none p-0 m-0">
            {aliases.map((alias, index) => (
              <li key={`alias-${index}`} className="flex gap-2">
                <input
                  id={index === 0 ? 'entity-field-aliases' : undefined}
                  className={inputClass}
                  value={alias}
                  disabled={!isEditable(definition, 'aliases', readOnly) || pending}
                  onChange={(event) => updateAlias(index, event.target.value)}
                />
                {isEditable(definition, 'aliases', readOnly) ? (
                  <button type="button" className={buttonSecondaryClass} onClick={() => removeAlias(index)}>
                    Remove
                  </button>
                ) : null}
              </li>
            ))}
          </ul>
          {isEditable(definition, 'aliases', readOnly) ? (
            <button type="button" className={`${buttonSecondaryClass} mt-2`} onClick={addAlias}>
              Add alias
            </button>
          ) : null}
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div>
            <label htmlFor="entity-field-default_retrieval_weight" className="text-sm font-medium">
              Retrieval weight
            </label>
            <input
              id="entity-field-default_retrieval_weight"
              type="number"
              step="0.01"
              min={0}
              max={1}
              className={inputClass}
              value={Number(values.default_retrieval_weight ?? 0)}
              disabled={!isEditable(definition, 'default_retrieval_weight', readOnly) || pending}
              onChange={(event) => onFieldChange('default_retrieval_weight', Number(event.target.value))}
            />
          </div>
          <div>
            <label htmlFor="entity-field-default_suppression_threshold" className="text-sm font-medium">
              Suppression threshold
            </label>
            <input
              id="entity-field-default_suppression_threshold"
              type="number"
              step="0.01"
              min={0}
              max={1}
              className={inputClass}
              value={Number(values.default_suppression_threshold ?? 0)}
              disabled={!isEditable(definition, 'default_suppression_threshold', readOnly) || pending}
              onChange={(event) => onFieldChange('default_suppression_threshold', Number(event.target.value))}
            />
          </div>
        </div>
        <div>
          <label htmlFor="entity-field-default_suppression_policy" className="text-sm font-medium">
            Suppression policy
          </label>
          <input
            id="entity-field-default_suppression_policy"
            className={inputClass}
            value={String(values.default_suppression_policy ?? '')}
            disabled={!isEditable(definition, 'default_suppression_policy', readOnly) || pending}
            onChange={(event) => onFieldChange('default_suppression_policy', event.target.value)}
          />
        </div>
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
        title="Remove entity type"
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

export default EntityTypeEditor;
