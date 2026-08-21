import type { SchemaDefinitionChangeState, SchemaDefinitionOrigin } from './schemaTypes';

export const panelClass =
  'bg-scheme-shade_4 border border-border-low_contrast rounded-[20px] text-text-normal';
export const inputClass =
  'w-full h-[36px] px-3 rounded-[18px] bg-scheme-shade_5 border border-border-mid_contrast text-text-normal text-sm';
export const textareaClass =
  'w-full min-h-[88px] px-3 py-2 rounded-[18px] bg-scheme-shade_5 border border-border-mid_contrast text-text-normal text-sm';
export const buttonPrimaryClass =
  'h-[36px] px-3 rounded-[18px] bg-accent text-white border border-accent hover:opacity-90 transition-opacity cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed';
export const buttonSecondaryClass =
  'h-[36px] px-3 rounded-[18px] bg-scheme-shade_5 text-text-normal border border-border-mid_contrast hover:bg-scheme-shade_6 transition-colors cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed';
export const buttonDangerClass =
  'h-[36px] px-3 rounded-[18px] bg-scheme-shade_5 text-red-300 border border-red-400/40 hover:bg-scheme-shade_6 transition-colors cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed';

export function originLabel(origin: SchemaDefinitionOrigin): string {
  return origin === 'inherited' ? 'Inherited' : 'Collection';
}

export function changeStateLabel(state: SchemaDefinitionChangeState): string {
  switch (state) {
    case 'added':
      return 'Added in draft';
    case 'changed':
      return 'Changed in draft';
    case 'removed':
      return 'Removed in draft';
    default:
      return 'Unchanged';
  }
}

export function changeStateBadgeClass(state: SchemaDefinitionChangeState): string {
  switch (state) {
    case 'added':
      return 'text-accent';
    case 'changed':
      return 'text-amber-300';
    case 'removed':
      return 'text-red-300';
    default:
      return 'text-text-lower_contrast';
  }
}
