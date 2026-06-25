/**
 * Normalize hrefs for same-origin browser loads (e.g. <img src>).
 * Root-relative paths always use the current page origin so they are not
 * affected by a future <base href> or non-standard embed contexts.
 */
export function resolveSiteAbsoluteUrl(href: string | undefined): string | undefined {
  if (href == null || typeof href !== 'string') {
    return href;
  }
  const t = href.trim();
  if (!t || t.startsWith('data:') || /^[a-z][a-z0-9+.-]*:/i.test(t)) {
    return href;
  }
  if (t.startsWith('//')) {
    return `${window.location.protocol}${t}`;
  }
  if (t.startsWith('/')) {
    return `${window.location.origin}${t}`;
  }
  return href;
}
