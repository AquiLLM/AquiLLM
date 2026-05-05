import { describe, it, expect } from 'vitest';
import { splitOnUnclosedMarkdown } from './streamingMarkdown';

describe('splitOnUnclosedMarkdown', () => {
  describe('plain text', () => {
    it('returns empty for empty input', () => {
      expect(splitOnUnclosedMarkdown('')).toEqual({ safe: '', pending: '' });
    });

    it('passes plain text through unchanged', () => {
      expect(splitOnUnclosedMarkdown('Hello world')).toEqual({
        safe: 'Hello world',
        pending: '',
      });
    });

    it('passes multi-paragraph plain text through', () => {
      const text = 'First paragraph.\n\nSecond paragraph.';
      expect(splitOnUnclosedMarkdown(text)).toEqual({ safe: text, pending: '' });
    });
  });

  describe('display math ($$...$$)', () => {
    it('hides unclosed display math at end', () => {
      const r = splitOnUnclosedMarkdown('Hello $$x = \\frac{');
      expect(r.safe).toBe('Hello ');
      expect(r.pending).toBe('$$x = \\frac{');
    });

    it('hides bare unclosed $$', () => {
      const r = splitOnUnclosedMarkdown('text $$');
      expect(r.safe).toBe('text ');
      expect(r.pending).toBe('$$');
    });

    it('passes through closed display math', () => {
      const text = 'Use $$x^2 + y^2 = z^2$$ here';
      expect(splitOnUnclosedMarkdown(text)).toEqual({ safe: text, pending: '' });
    });

    it('passes through multi-line closed display math', () => {
      const text = '$$\nE = mc^2\n$$\nDone.';
      expect(splitOnUnclosedMarkdown(text)).toEqual({ safe: text, pending: '' });
    });

    it('hides multi-line unclosed display math', () => {
      const r = splitOnUnclosedMarkdown('$$\nE = mc');
      expect(r.safe).toBe('');
      expect(r.pending).toBe('$$\nE = mc');
    });

    it('handles two consecutive closed display math blocks', () => {
      const text = '$$a$$ between $$b$$';
      expect(splitOnUnclosedMarkdown(text)).toEqual({ safe: text, pending: '' });
    });

    it('hides second unclosed display math', () => {
      const r = splitOnUnclosedMarkdown('$$a$$ then $$b');
      expect(r.safe).toBe('$$a$$ then ');
      expect(r.pending).toBe('$$b');
    });

    it('respects escaped \\$ inside display math', () => {
      const r = splitOnUnclosedMarkdown('$$ a \\$$ b');
      expect(r.safe).toBe('');
      expect(r.pending).toBe('$$ a \\$$ b');
    });
  });

  describe('inline math ($...$)', () => {
    it('passes through closed inline math', () => {
      const text = 'value $x = 1$ here';
      expect(splitOnUnclosedMarkdown(text)).toEqual({ safe: text, pending: '' });
    });

    it('hides unclosed inline math', () => {
      const r = splitOnUnclosedMarkdown('value $x = 1');
      expect(r.safe).toBe('value ');
      expect(r.pending).toBe('$x = 1');
    });

    it('does not treat $5 currency as math', () => {
      const text = 'price is $5 today';
      expect(splitOnUnclosedMarkdown(text)).toEqual({ safe: text, pending: '' });
    });

    it('does not treat cost$5 as math (alphanumeric prefix)', () => {
      const text = 'cost$5 stays plain';
      expect(splitOnUnclosedMarkdown(text)).toEqual({ safe: text, pending: '' });
    });

    it('does not open inline math on $ followed by space', () => {
      const text = 'amount $ x is text';
      expect(splitOnUnclosedMarkdown(text)).toEqual({ safe: text, pending: '' });
    });
  });

  describe('fenced code (```)', () => {
    it('appends synthetic close to unclosed fence at end', () => {
      const r = splitOnUnclosedMarkdown('text\n```py\ncode here');
      expect(r.pending).toBe('');
      expect(r.safe).toBe('text\n```py\ncode here\n```\n');
    });

    it('appends synthetic close when only opener present', () => {
      const r = splitOnUnclosedMarkdown('```py');
      expect(r.pending).toBe('');
      expect(r.safe).toBe('```py\n```\n');
    });

    it('passes through closed fenced code unchanged', () => {
      const text = 'before\n```py\nx = 1\n```\nafter';
      expect(splitOnUnclosedMarkdown(text)).toEqual({ safe: text, pending: '' });
    });

    it('treats $$ inside fenced code as literal (no math triggered)', () => {
      const text = 'pre\n```\nlet a = $$x$$\n```\npost';
      expect(splitOnUnclosedMarkdown(text)).toEqual({ safe: text, pending: '' });
    });

    it('handles tilde fences', () => {
      const text = '~~~\ncode\n~~~';
      expect(splitOnUnclosedMarkdown(text)).toEqual({ safe: text, pending: '' });
    });

    it('appends synthetic close for unclosed tilde fence', () => {
      const r = splitOnUnclosedMarkdown('~~~\ncode here');
      expect(r.pending).toBe('');
      expect(r.safe).toBe('~~~\ncode here\n~~~\n');
    });
  });

  describe('inline code (`)', () => {
    it('passes closed inline code through', () => {
      const text = 'use `foo` here';
      expect(splitOnUnclosedMarkdown(text)).toEqual({ safe: text, pending: '' });
    });

    it('treats $$ inside inline code as literal', () => {
      const text = 'inline `$$x$$` is literal';
      expect(splitOnUnclosedMarkdown(text)).toEqual({ safe: text, pending: '' });
    });
  });

  describe('HTML balance', () => {
    it('passes balanced HTML through', () => {
      const text = '<details><summary>Title</summary>body</details>';
      expect(splitOnUnclosedMarkdown(text)).toEqual({ safe: text, pending: '' });
    });

    it('hides content from first unclosed start tag', () => {
      const r = splitOnUnclosedMarkdown('intro <details><summary>X</summary>body');
      expect(r.safe).toBe('intro ');
      expect(r.pending).toBe('<details><summary>X</summary>body');
    });

    it('passes through void tags without close', () => {
      const text = 'line1<br>line2';
      expect(splitOnUnclosedMarkdown(text)).toEqual({ safe: text, pending: '' });
    });

    it('hides mid-typing tag at end', () => {
      const r = splitOnUnclosedMarkdown('text before <deta');
      expect(r.safe).toBe('text before ');
      expect(r.pending).toBe('<deta');
    });

    it('does not flag HTML inside fenced code', () => {
      const text = 'before\n```html\n<details>\n```\nafter';
      expect(splitOnUnclosedMarkdown(text)).toEqual({ safe: text, pending: '' });
    });

    it('does not flag HTML inside inline code', () => {
      const text = 'tag is `<details>` literal';
      expect(splitOnUnclosedMarkdown(text)).toEqual({ safe: text, pending: '' });
    });

    it('handles self-closing tag', () => {
      const text = '<img src="x" />after';
      expect(splitOnUnclosedMarkdown(text)).toEqual({ safe: text, pending: '' });
    });
  });

  describe('mixed cases', () => {
    it('hides unclosed math but renders prior fence', () => {
      const r = splitOnUnclosedMarkdown('```\nx\n```\nthen $$y');
      expect(r.safe).toBe('```\nx\n```\nthen ');
      expect(r.pending).toBe('$$y');
    });

    it('hides unclosed HTML after closed math', () => {
      const r = splitOnUnclosedMarkdown('$$x$$ <details>more');
      expect(r.safe).toBe('$$x$$ ');
      expect(r.pending).toBe('<details>more');
    });

    it('handles fence containing dollar signs and html', () => {
      const text = '```\n$$ <details>\n```';
      expect(splitOnUnclosedMarkdown(text)).toEqual({ safe: text, pending: '' });
    });

    it('renders unclosed fence body containing HTML as code (synthetic close)', () => {
      const r = splitOnUnclosedMarkdown('intro\n```html\n<table>\n  <tr>');
      expect(r.pending).toBe('');
      expect(r.safe).toBe('intro\n```html\n<table>\n  <tr>\n```\n');
    });

    it('renders unclosed fence body containing partial HTML', () => {
      const r = splitOnUnclosedMarkdown('```\n<details><summary>X');
      expect(r.pending).toBe('');
      expect(r.safe).toBe('```\n<details><summary>X\n```\n');
    });
  });

  describe('progressive streaming snapshots', () => {
    const final = '## Title\n\nText with $$x = 1$$ and `code`.\n\n```py\nprint(1)\n```\n\nDone.';

    it('hides display math during partial typing', () => {
      const partial = '## Title\n\nText with $$x = ';
      const r = splitOnUnclosedMarkdown(partial);
      expect(r.safe).toBe('## Title\n\nText with ');
      expect(r.pending).toBe('$$x = ');
    });

    it('reveals display math once $$ closes', () => {
      const partial = '## Title\n\nText with $$x = 1$$';
      expect(splitOnUnclosedMarkdown(partial)).toEqual({ safe: partial, pending: '' });
    });

    it('renders fence body live with synthetic close', () => {
      const partial = '## Title\n\nText with $$x = 1$$ and `code`.\n\n```py\nprint(';
      const r = splitOnUnclosedMarkdown(partial);
      expect(r.pending).toBe('');
      expect(r.safe).toBe(partial + '\n```\n');
    });

    it('passes final complete content through', () => {
      expect(splitOnUnclosedMarkdown(final)).toEqual({ safe: final, pending: '' });
    });
  });
});
