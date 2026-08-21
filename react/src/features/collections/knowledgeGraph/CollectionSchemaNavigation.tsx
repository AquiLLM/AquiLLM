import React, { useMemo, useState } from 'react';
import type {
  EntityTypeDefinition,
  RelationTypeDefinition,
  SchemaDefinitionChangeState,
  SchemaDefinitionKind,
  SchemaDefinitionOrigin,
} from './schemaTypes';
import { buttonPrimaryClass, changeStateBadgeClass, changeStateLabel, inputClass, originLabel, panelClass } from './schemaUiShared';

export type SchemaNavKindFilter = 'all' | 'entity' | 'relation';
export type SchemaNavOriginFilter = 'all' | SchemaDefinitionOrigin;
export type SchemaNavChangeFilter = 'all' | SchemaDefinitionChangeState;

export interface CollectionSchemaNavigationProps {
  entities: EntityTypeDefinition[];
  relations: RelationTypeDefinition[];
  selectedKind: SchemaDefinitionKind | null;
  selectedKey: string | null;
  onSelect: (kind: SchemaDefinitionKind, key: string) => void;
  canAddEntity: boolean;
  canAddRelation: boolean;
  onAddEntity?: () => void;
  onAddRelation?: () => void;
}

function relationSummary(relation: RelationTypeDefinition): string {
  const head = relation.values.allowed_head_types.join(', ') || '—';
  const tail = relation.values.allowed_tail_types.join(', ') || '—';
  return `${relation.values.direction}: ${head} → ${tail}`;
}

const CollectionSchemaNavigation: React.FC<CollectionSchemaNavigationProps> = ({
  entities,
  relations,
  selectedKind,
  selectedKey,
  onSelect,
  canAddEntity,
  canAddRelation,
  onAddEntity,
  onAddRelation,
}) => {
  const [query, setQuery] = useState('');
  const [kindFilter, setKindFilter] = useState<SchemaNavKindFilter>('all');
  const [originFilter, setOriginFilter] = useState<SchemaNavOriginFilter>('all');
  const [changeFilter, setChangeFilter] = useState<SchemaNavChangeFilter>('all');

  const filteredEntities = useMemo(() => {
    const q = query.trim().toLowerCase();
    return entities.filter((entity) => {
      if (kindFilter === 'relation') return false;
      if (originFilter !== 'all' && entity.origin !== originFilter) return false;
      if (changeFilter !== 'all' && entity.change_state !== changeFilter) return false;
      if (!q) return true;
      return (
        entity.key.toLowerCase().includes(q) ||
        entity.values.name.toLowerCase().includes(q) ||
        entity.values.description.toLowerCase().includes(q)
      );
    });
  }, [changeFilter, entities, kindFilter, originFilter, query]);

  const filteredRelations = useMemo(() => {
    const q = query.trim().toLowerCase();
    return relations.filter((relation) => {
      if (kindFilter === 'entity') return false;
      if (originFilter !== 'all' && relation.origin !== originFilter) return false;
      if (changeFilter !== 'all' && relation.change_state !== changeFilter) return false;
      if (!q) return true;
      return (
        relation.key.toLowerCase().includes(q) ||
        relation.values.name.toLowerCase().includes(q) ||
        relationSummary(relation).toLowerCase().includes(q)
      );
    });
  }, [changeFilter, kindFilter, originFilter, query, relations]);

  const renderRow = (
    kind: SchemaDefinitionKind,
    key: string,
    title: string,
    subtitle: string,
    origin: SchemaDefinitionOrigin,
    changeState: SchemaDefinitionChangeState,
  ) => {
    const selected = selectedKind === kind && selectedKey === key;
    return (
      <li key={`${kind}-${key}`} role="presentation">
        <button
          type="button"
          role="option"
          aria-selected={selected}
          data-testid={`schema-nav-${kind}-${key}`}
          className={`w-full text-left px-3 py-2 rounded-[14px] border cursor-pointer transition-colors ${
            selected
              ? 'bg-accent/20 border-accent'
              : 'bg-scheme-shade_5 border-transparent hover:border-border-mid_contrast'
          }`}
          onClick={() => onSelect(kind, key)}
        >
          <div className="font-medium text-sm">{title}</div>
          <div className="text-xs text-text-lower_contrast truncate">{subtitle}</div>
          <div className="text-xs mt-1 flex gap-2">
            <span>{originLabel(origin)}</span>
            <span className={changeStateBadgeClass(changeState)}>{changeStateLabel(changeState)}</span>
          </div>
        </button>
      </li>
    );
  };

  return (
    <section className={`${panelClass} p-[12px] flex flex-col min-h-[320px]`} aria-label="Schema definitions">
      <div className="space-y-2 mb-3">
        <label htmlFor="schema-nav-search" className="text-sm font-medium">
          Search definitions
        </label>
        <input
          id="schema-nav-search"
          type="search"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          className={inputClass}
          placeholder="Search by name or description"
        />
        <div className="flex flex-wrap gap-2 text-xs">
          <select
            aria-label="Filter by kind"
            className={`${inputClass} !w-auto`}
            value={kindFilter}
            onChange={(event) => setKindFilter(event.target.value as SchemaNavKindFilter)}
          >
            <option value="all">All kinds</option>
            <option value="entity">Entity types</option>
            <option value="relation">Relation types</option>
          </select>
          <select
            aria-label="Filter by origin"
            className={`${inputClass} !w-auto`}
            value={originFilter}
            onChange={(event) => setOriginFilter(event.target.value as SchemaNavOriginFilter)}
          >
            <option value="all">All origins</option>
            <option value="inherited">Inherited</option>
            <option value="collection">Collection</option>
          </select>
          <select
            aria-label="Filter by draft status"
            className={`${inputClass} !w-auto`}
            value={changeFilter}
            onChange={(event) => setChangeFilter(event.target.value as SchemaNavChangeFilter)}
          >
            <option value="all">All statuses</option>
            <option value="unchanged">Unchanged</option>
            <option value="added">Added</option>
            <option value="changed">Changed</option>
            <option value="removed">Removed</option>
          </select>
        </div>
      </div>

      {(canAddEntity || canAddRelation) && (
        <div className="flex flex-wrap gap-2 mb-3">
          {canAddEntity ? (
            <button type="button" className={buttonPrimaryClass} onClick={onAddEntity}>
              Add entity type
            </button>
          ) : null}
          {canAddRelation ? (
            <button type="button" className={buttonPrimaryClass} onClick={onAddRelation}>
              Add relation type
            </button>
          ) : null}
        </div>
      )}

      <ul role="listbox" aria-label="Schema definition results" className="flex-1 overflow-y-auto space-y-2 m-0 p-0 list-none">
        {filteredEntities.map((entity) =>
          renderRow(
            'entity',
            entity.key,
            entity.values.name,
            entity.values.description || 'No description',
            entity.origin,
            entity.change_state,
          ),
        )}
        {filteredRelations.map((relation) =>
          renderRow(
            'relation',
            relation.key,
            relation.values.name,
            relationSummary(relation),
            relation.origin,
            relation.change_state,
          ),
        )}
        {filteredEntities.length === 0 && filteredRelations.length === 0 ? (
          <li className="text-sm text-text-lower_contrast px-2 py-4">No definitions match the current filters.</li>
        ) : null}
      </ul>
    </section>
  );
};

export default CollectionSchemaNavigation;
