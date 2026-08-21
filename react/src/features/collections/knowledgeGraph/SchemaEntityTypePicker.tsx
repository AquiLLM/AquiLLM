import React, { useMemo, useState } from 'react';
import type { EntityTypeDefinition } from './schemaTypes';
import { inputClass } from './schemaUiShared';

export interface SchemaEntityTypePickerProps {
  id: string;
  label: string;
  entityTypes: EntityTypeDefinition[];
  selectedKeys: string[];
  onChange: (keys: string[]) => void;
  disabled?: boolean;
  required?: boolean;
  error?: string | null;
}

const SchemaEntityTypePicker: React.FC<SchemaEntityTypePickerProps> = ({
  id,
  label,
  entityTypes,
  selectedKeys,
  onChange,
  disabled = false,
  required = false,
  error,
}) => {
  const [filter, setFilter] = useState('');
  const filtered = useMemo(() => {
    const query = filter.trim().toLowerCase();
    if (!query) return entityTypes;
    return entityTypes.filter(
      (entity) =>
        entity.key.toLowerCase().includes(query) ||
        entity.values.name.toLowerCase().includes(query),
    );
  }, [entityTypes, filter]);

  const toggle = (key: string) => {
    if (disabled) return;
    onChange(
      selectedKeys.includes(key)
        ? selectedKeys.filter((item) => item !== key)
        : [...selectedKeys, key],
    );
  };

  return (
    <fieldset className="space-y-2" aria-describedby={error ? `${id}-error` : undefined}>
      <legend className="text-sm font-medium text-text-normal mb-1">
        {label}
        {required ? ' (required)' : ''}
      </legend>
      <input
        type="search"
        aria-label={`Search ${label}`}
        className={inputClass}
        value={filter}
        onChange={(event) => setFilter(event.target.value)}
        disabled={disabled}
      />
      <ul
        role="listbox"
        aria-multiselectable="true"
        aria-label={label}
        className="max-h-[180px] overflow-y-auto rounded-[16px] border border-border-mid_contrast p-2 space-y-1 m-0 list-none"
      >
        {filtered.map((entity) => {
          const selected = selectedKeys.includes(entity.key);
          return (
            <li key={entity.key} role="presentation">
              <button
                type="button"
                role="option"
                aria-selected={selected}
                disabled={disabled}
                className={`w-full text-left px-3 py-2 rounded-[12px] text-sm cursor-pointer border ${
                  selected
                    ? 'bg-accent/20 border-accent text-text-normal'
                    : 'bg-scheme-shade_5 border-transparent hover:border-border-mid_contrast'
                }`}
                onClick={() => toggle(entity.key)}
              >
                {entity.values.name}
                <span className="text-text-lower_contrast ml-2">({entity.key})</span>
              </button>
            </li>
          );
        })}
      </ul>
      {error ? (
        <p id={`${id}-error`} className="text-sm text-red-200">
          {error}
        </p>
      ) : null}
    </fieldset>
  );
};

export default SchemaEntityTypePicker;
