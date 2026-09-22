import { describe, it, expect } from 'vitest';
import { findNormalizedOffset, normalizeForSearch } from './textOffsetMatch';

describe('normalizeForSearch', () => {
  it('collapses whitespace, lowercases, and unifies smart quotes', () => {
    expect(normalizeForSearch('The  \n quick “Fox’s”')).toBe('the quick "fox\'s"');
  });
});

describe('findNormalizedOffset', () => {
  it('returns offsets into the original haystack for an exact match', () => {
    const hay = 'alpha beta gamma delta';
    const hit = findNormalizedOffset(hay, 'beta gamma');
    expect(hit).not.toBeNull();
    expect(hay.slice(hit!.start, hit!.end)).toBe('beta gamma');
  });

  it('matches across collapsed whitespace but maps back to original spans', () => {
    // Haystack has a newline + double space the needle does not.
    const hay = 'Results show that\n  the model improves recall.';
    const hit = findNormalizedOffset(hay, 'that the model improves');
    expect(hit).not.toBeNull();
    // The matched original text preserves the original whitespace.
    expect(hay.slice(hit!.start, hit!.end)).toBe('that\n  the model improves');
  });

  it('matches despite smart-quote and case drift, preserving original chars', () => {
    const hay = 'She said “It Works” yesterday.';
    const hit = findNormalizedOffset(hay, 'it works');
    expect(hit).not.toBeNull();
    // Original casing and the smart quote glyph are preserved in the span.
    expect(hay.slice(hit!.start, hit!.end)).toBe('It Works');
  });

  it('returns null when the needle is absent', () => {
    expect(findNormalizedOffset('alpha beta gamma', 'nonexistent phrase')).toBeNull();
  });

  it('returns null for an empty/whitespace needle', () => {
    expect(findNormalizedOffset('alpha beta', '   ')).toBeNull();
  });
});
