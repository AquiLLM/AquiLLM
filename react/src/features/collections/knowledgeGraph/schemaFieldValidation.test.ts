import { expect, it } from 'vitest';
import { validateFieldValue } from './schemaFieldValidation';

it('rejects provider-reserved type names from server constraints', () => {
  expect(validateFieldValue('name', 'entities', { disallowed_values: ['entities'] })).toBe('name is reserved');
  expect(validateFieldValue('name', 'person', { disallowed_values: ['entities'] })).toBeNull();
});
