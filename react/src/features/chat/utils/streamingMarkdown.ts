import { Parser } from 'htmlparser2';

export interface SafeSplit {
  safe: string;
  pending: string;
}

type Mode = 'NORMAL' | 'FENCE' | 'DISPLAY_MATH' | 'INLINE_MATH';

const FENCE_OPEN = /^( {0,3})(`{3,}|~{3,})/;

function isEscaped(s: string, idx: number): boolean {
  let count = 0;
  let i = idx - 1;
  while (i >= 0 && s[i] === '\\') {
    count++;
    i--;
  }
  return count % 2 === 1;
}

function findUnescaped(s: string, needle: string, from: number): number {
  let i = from;
  while (i <= s.length - needle.length) {
    const j = s.indexOf(needle, i);
    if (j < 0) return -1;
    if (!isEscaped(s, j)) return j;
    i = j + 1;
  }
  return -1;
}

function isInlineMathOpen(s: string, i: number): boolean {
  const next = s[i + 1];
  if (next === undefined) return false;
  if (next === '$') return false;
  if (/\s/.test(next)) return false;
  if (/\d/.test(next)) return false;
  const prev = s[i - 1];
  if (prev !== undefined && /[A-Za-z0-9]/.test(prev)) return false;
  return true;
}

function isInlineMathClose(s: string, i: number): boolean {
  if (s[i + 1] === '$') return false;
  const prev = s[i - 1];
  if (prev === undefined || /\s/.test(prev)) return false;
  const next = s[i + 1];
  if (next !== undefined && /\d/.test(next)) return false;
  return true;
}

function maskRanges(text: string, ranges: Array<[number, number]>): string {
  if (ranges.length === 0) return text;
  const arr = text.split('');
  for (const [start, end] of ranges) {
    for (let k = start; k < end && k < arr.length; k++) {
      if (arr[k] !== '\n') arr[k] = ' ';
    }
  }
  return arr.join('');
}

const VOID_ELEMENTS = new Set([
  'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input',
  'link', 'meta', 'param', 'source', 'track', 'wbr',
]);

function firstUnclosedHtmlOffset(text: string): number {
  const stack: Array<{ name: string; start: number }> = [];
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let p: any;
  p = new Parser(
    {
      onopentag(name: string) {
        if (VOID_ELEMENTS.has(name)) return;
        stack.push({ name, start: p.startIndex });
      },
      onclosetag(name: string, isImplied: boolean) {
        if (isImplied) return;
        for (let k = stack.length - 1; k >= 0; k--) {
          if (stack[k].name === name) {
            stack.splice(k, 1);
            break;
          }
        }
      },
    },
    { lowerCaseTags: true, recognizeSelfClosing: true },
  );
  p.write(text);
  p.end();

  let earliest = stack.length > 0 ? stack[0].start : -1;

  // Detect mid-typing tag at end ("<", "</", "<!-" etc with no matching ">").
  const lastLt = text.lastIndexOf('<');
  if (lastLt >= 0 && text.indexOf('>', lastLt) < 0) {
    const after = text.slice(lastLt + 1, lastLt + 4);
    if (after === '' || /^[a-zA-Z!/]/.test(after)) {
      if (earliest < 0 || lastLt < earliest) earliest = lastLt;
    }
  }

  return earliest;
}

export function splitOnUnclosedMarkdown(content: string): SafeSplit {
  if (!content) return { safe: '', pending: '' };

  let mode: Mode = 'NORMAL';
  let safeEnd = 0;
  let pendingStart = -1;
  let fenceMarker = '';
  const codeRegions: Array<[number, number]> = [];

  const len = content.length;
  let i = 0;
  let lineStart = 0;
  let fenceBodyStart = -1;

  while (i < len) {
    const ch = content[i];

    if (mode === 'FENCE') {
      if (ch === '\n') {
        const nextLineStart = i + 1;
        const nextLineEnd = content.indexOf('\n', nextLineStart);
        const nextLine = content.slice(
          nextLineStart,
          nextLineEnd < 0 ? len : nextLineEnd,
        );
        const closer = new RegExp(
          '^ {0,3}' + fenceMarker[0] + '{' + fenceMarker.length + ',}\\s*$',
        );
        if (closer.test(nextLine)) {
          mode = 'NORMAL';
          const closerEnd = nextLineEnd < 0 ? len : nextLineEnd + 1;
          if (fenceBodyStart >= 0) {
            codeRegions.push([fenceBodyStart, closerEnd]);
          }
          fenceBodyStart = -1;
          safeEnd = closerEnd;
          i = closerEnd;
          lineStart = i;
          continue;
        }
        i = nextLineStart;
        lineStart = nextLineStart;
        continue;
      }
      i += 1;
      continue;
    }

    if (mode === 'DISPLAY_MATH') {
      const closeIdx = findUnescaped(content, '$$', i);
      if (closeIdx < 0) {
        return {
          safe: content.slice(0, pendingStart),
          pending: content.slice(pendingStart),
        };
      }
      mode = 'NORMAL';
      i = closeIdx + 2;
      safeEnd = i;
      pendingStart = -1;
      const lastNl = content.lastIndexOf('\n', i - 1);
      lineStart = lastNl < 0 ? 0 : lastNl + 1;
      continue;
    }

    if (mode === 'INLINE_MATH') {
      if (ch === '\\') {
        i += 2;
        continue;
      }
      if (ch === '$' && !isEscaped(content, i) && isInlineMathClose(content, i)) {
        mode = 'NORMAL';
        i += 1;
        safeEnd = i;
        pendingStart = -1;
        continue;
      }
      i += 1;
      continue;
    }

    if (ch === '\n') {
      i += 1;
      lineStart = i;
      safeEnd = i;
      continue;
    }

    if (i === lineStart || (i - lineStart <= 3 && /^ +$/.test(content.slice(lineStart, i)))) {
      const fm = content.slice(lineStart).match(FENCE_OPEN);
      if (fm && i === lineStart + fm[1].length) {
        mode = 'FENCE';
        fenceMarker = fm[2];
        pendingStart = lineStart;
        fenceBodyStart = lineStart;
        const nl = content.indexOf('\n', i);
        if (nl < 0) {
          return {
            safe: content + '\n' + fenceMarker + '\n',
            pending: '',
          };
        }
        i = nl;
        continue;
      }
    }

    if (ch === '\\') {
      i += 2;
      safeEnd = Math.min(i, len);
      continue;
    }

    if (ch === '`') {
      let runLen = 1;
      while (content[i + runLen] === '`') runLen += 1;
      const closeIdx = content.indexOf('`'.repeat(runLen), i + runLen);
      if (closeIdx < 0) {
        i += runLen;
        safeEnd = i;
        continue;
      }
      const spanEnd = closeIdx + runLen;
      codeRegions.push([i, spanEnd]);
      i = spanEnd;
      safeEnd = i;
      continue;
    }

    if (ch === '$' && !isEscaped(content, i)) {
      if (content[i + 1] === '$') {
        const closeIdx = findUnescaped(content, '$$', i + 2);
        if (closeIdx < 0) {
          mode = 'DISPLAY_MATH';
          pendingStart = i;
          i += 2;
          continue;
        }
        i = closeIdx + 2;
        safeEnd = i;
        continue;
      }
      if (isInlineMathOpen(content, i)) {
        let j = i + 1;
        let found = -1;
        while (j < len) {
          if (content[j] === '\\') {
            j += 2;
            continue;
          }
          if (content[j] === '$' && !isEscaped(content, j) && isInlineMathClose(content, j)) {
            found = j;
            break;
          }
          j += 1;
        }
        if (found < 0) {
          mode = 'INLINE_MATH';
          pendingStart = i;
          i += 1;
          continue;
        }
        i = found + 1;
        safeEnd = i;
        continue;
      }
      i += 1;
      continue;
    }

    i += 1;
    safeEnd = i;
  }

  let safe: string;
  if (mode === 'NORMAL') {
    safe = content;
  } else if (mode === 'FENCE') {
    const leadingNl = content.endsWith('\n') ? '' : '\n';
    safe = content + leadingNl + fenceMarker + '\n';
    // Mask the unclosed-fence body so HTML detector ignores its contents.
    if (fenceBodyStart >= 0) {
      codeRegions.push([fenceBodyStart, safe.length]);
    }
  } else {
    return {
      safe: content.slice(0, pendingStart < 0 ? safeEnd : pendingStart),
      pending: content.slice(pendingStart < 0 ? safeEnd : pendingStart),
    };
  }

  // HTML balance check on safe (with code regions masked).
  const masked = maskRanges(safe, codeRegions);
  const htmlCut = firstUnclosedHtmlOffset(masked);
  if (htmlCut >= 0 && htmlCut < safe.length) {
    return {
      safe: safe.slice(0, htmlCut),
      pending: content.slice(htmlCut),
    };
  }

  return { safe, pending: '' };
}
