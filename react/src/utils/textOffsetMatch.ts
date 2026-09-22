/**
 * Normalized substring matching for the text citation modal.
 *
 * The LLM-narrowed quote (and the chunk content) may differ from the rendered
 * document text only by collapsed whitespace, smart quotes, or case. These
 * helpers find such a needle inside a haystack while returning offsets into the
 * ORIGINAL (un-normalized) haystack, so the caller can highlight the real text.
 */

/** Normalize whitespace and case for substring search so a quote that
 *  differs only in collapsed spaces / smart quotes still matches. */
export function normalizeForSearch(s: string): string {
  return s
    .normalize('NFKC')
    .replace(/[‘’‚‛]/g, "'")
    .replace(/[“”„‟]/g, '"')
    .replace(/\s+/g, ' ')
    .toLowerCase();
}

/** Find `needle` inside `haystack` using normalized matching but return
 *  an offset into the ORIGINAL haystack. Returns null on miss. */
export function findNormalizedOffset(
  haystack: string,
  needle: string,
): { start: number; end: number } | null {
  const normNeedle = normalizeForSearch(needle).trim();
  if (!normNeedle) return null;
  // Build a parallel array of original-index per normalized character.
  const indexMap: number[] = [];
  let norm = '';
  let prevWasSpace = false;
  for (let i = 0; i < haystack.length; i++) {
    const ch = haystack[i].normalize('NFKC');
    for (const c of ch) {
      let mapped = c;
      if (c === '‘' || c === '’' || c === '‚' || c === '‛') mapped = "'";
      else if (c === '“' || c === '”' || c === '„' || c === '‟') mapped = '"';
      if (/\s/.test(mapped)) {
        if (prevWasSpace) continue;
        prevWasSpace = true;
        norm += ' ';
        indexMap.push(i);
      } else {
        prevWasSpace = false;
        norm += mapped.toLowerCase();
        indexMap.push(i);
      }
    }
  }
  const hit = norm.indexOf(normNeedle);
  if (hit < 0) return null;
  const start = indexMap[hit];
  const endIdx = hit + normNeedle.length - 1;
  if (endIdx >= indexMap.length) return null;
  const end = indexMap[endIdx] + 1; // exclusive
  return { start, end };
}
