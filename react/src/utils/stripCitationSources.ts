import { DOC_CHUNK_CITATION_RE } from './linkifyRagCitations';

const SOURCES_HEADING = /^ {0,3}(?:#{1,6}[ \t]+)?(?:\*\*|__)?Sources:?(?:\*\*|__)?:?[ \t]*(?:#+[ \t]*)?$/i;

/** Hide only a trailing citation inventory already represented by MessageSources.
 * Keep the original message for citation lookup, storage, and model context.
 */
export function stripCitationSources(content: string): string {
  const lines = content.split('\n');
  let fence: string | undefined;
  let heading = -1;
  for (let index = 0; index < lines.length; index++) {
    const line = lines[index].replace(/\r$/, '');
    const marker = /^ {0,3}(`{3,}|~{3,})(.*)$/.exec(line);
    if (marker) {
      if (!fence) fence = marker[1];
      else if (marker[1][0] === fence[0] && marker[1].length >= fence.length && !marker[2].trim()) {
        fence = undefined;
      }
      continue;
    }
    if (!fence && SOURCES_HEADING.test(line)) heading = index;
  }
  if (heading < 0) return content;

  let start = heading + 1;
  // Also accept a Markdown setext heading: "Sources" followed by dashes.
  if (/^\s*(?:-{3,}|={3,})\s*$/.test(lines[start] ?? '')) start++;
  let hasCitation = false;
  for (const line of lines.slice(start)) {
    if (!line.trim()) continue;
    // Four columns of indentation can be Markdown code, including tab stops.
    if (/^(?: {4}| {0,3}\t)/.test(line)) return content;
    const entry = line.trim().replace(/^(?:[-*+]|\d+[.)])\s+/, '');
    let isCitation = false;
    const remainder = entry.replace(DOC_CHUNK_CITATION_RE, () => {
      isCitation = true;
      return '';
    });
    // Prose, links, code, and partial streaming tokens are never discarded.
    if (!isCitation || !/^[\s,;.]*$/.test(remainder)) return content;
    hasCitation = true;
  }
  if (hasCitation) return lines.slice(0, heading).join('\n').trimEnd();
  return content;
}
