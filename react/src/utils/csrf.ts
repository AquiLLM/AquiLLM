/** Read a cookie value by name. Standalone (no dependency on main.tsx) so it
 *  can be used from separate entry bundles like the Collection Notes page. */
export function getCookie(name: string): string {
  const prefix = `${name}=`;
  const parts = document.cookie ? document.cookie.split('; ') : [];
  for (const part of parts) {
    if (part.startsWith(prefix)) {
      return decodeURIComponent(part.slice(prefix.length));
    }
  }
  return '';
}
