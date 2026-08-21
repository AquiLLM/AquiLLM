import type { SchemaFieldConstraint } from './schemaTypes';

export function validateFieldValue(
  field: string,
  value: unknown,
  constraint?: SchemaFieldConstraint,
): string | null {
  if (!constraint) return null;
  if (constraint.required && (value === null || value === undefined || value === '')) {
    return `${field} is required`;
  }
  if (typeof value === 'string') {
    if (constraint.max_length !== undefined && value.length > constraint.max_length) {
      return `${field} must be at most ${constraint.max_length} characters`;
    }
    if (constraint.pattern && !new RegExp(constraint.pattern).test(value)) {
      return `${field} has an invalid format`;
    }
  }
  if (typeof value === 'number') {
    if (constraint.min !== undefined && value < constraint.min) {
      return `${field} must be at least ${constraint.min}`;
    }
    if (constraint.max !== undefined && value > constraint.max) {
      return `${field} must be at most ${constraint.max}`;
    }
  }
  if (typeof value === 'string' && constraint.allowed_values && !constraint.allowed_values.includes(value)) {
    return `${field} must be one of: ${constraint.allowed_values.join(', ')}`;
  }
  return null;
}

export function validateEntityForm(
  values: Record<string, unknown>,
  constraints: Record<string, SchemaFieldConstraint>,
): Record<string, string> {
  const errors: Record<string, string> = {};
  for (const [field, constraint] of Object.entries(constraints)) {
    const message = validateFieldValue(field, values[field], constraint);
    if (message) errors[field] = message;
  }
  return errors;
}

export function validateRelationForm(
  values: Record<string, unknown>,
  constraints: Record<string, SchemaFieldConstraint>,
): Record<string, string> {
  const errors = validateEntityForm(values, constraints);
  const head = values.allowed_head_types;
  const tail = values.allowed_tail_types;
  if (Array.isArray(head) && head.length === 0) {
    errors.allowed_head_types = 'Select at least one head entity type';
  }
  if (Array.isArray(tail) && tail.length === 0) {
    errors.allowed_tail_types = 'Select at least one tail entity type';
  }
  return errors;
}
