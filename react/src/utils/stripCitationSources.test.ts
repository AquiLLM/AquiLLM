import { describe, expect, it } from 'vitest';
import { stripCitationSources } from './stripCitationSources';

const ref = '[doc:11111111-1111-4111-8111-111111111111 chunk:17]';
const answer = `A supported answer ${ref}.`;

describe('stripCitationSources', () => {
  it.each(['Sources:', '**Sources:**', '**Sources**:', '## Sources', '### Sources:', 'Sources\n-------'])
    ('hides a trailing citation-only list headed %s', (heading) => {
      expect(stripCitationSources(`${answer}\n\n${heading}\n- ${ref}\n`)).toBe(answer);
    });

  it('handles numbered and bare citations with Windows line endings', () => {
    expect(stripCitationSources(`${answer}\r\n\r\nSources:\r\n1. ${ref}\r\n${ref}\r\n`)).toBe(answer);
  });

  it.each([
    `${answer}\n\nSources:\n- The paper describes measurement uncertainty ${ref}.`,
    `${answer}\n\nSources of uncertainty:\n- ${ref}`,
    `${answer}\n\nSources:\n- ${ref}\n\n## Caveats\nMeasurements are incomplete.`,
    `${answer}\n\nSources:\n- [External reference](https://example.com)`,
    `${answer}\n\nSources:`,
    `${answer}\n\nSources:\n- [doc:111`,
    `${answer}\n\n\`\`\`markdown\nSources:\n- ${ref}\n\`\`\``,
    `${answer}\n\n\`\`\`markdown\nSources:\n- ${ref}`,
    `${answer}\n\n~~~markdown\nSources:\n- ${ref}`,
    `${answer}\n\n    Sources:\n    - ${ref}`,
    `${answer}\n\nSources:\n\n    ${ref}`,
    `Sources\n=======\n\n\t${ref}`,
    `${answer}\n\nSources:\n\n  \t${ref}`,
  ])('preserves prose, external references, incomplete sections and code: %s', (content) => {
    expect(stripCitationSources(content)).toBe(content);
  });
});
