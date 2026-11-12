import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { afterEach, describe, expect, it } from 'vitest'
import { collectNotices } from './thirdPartyNotices'

const roots: string[] = []
function fixture(name: string, license?: string) {
  const root = mkdtempSync(join(tmpdir(), 'aquillm-notices-'))
  roots.push(root)
  const dir = join(root, 'node_modules', name)
  mkdirSync(dir, { recursive: true })
  writeFileSync(join(dir, 'package.json'), JSON.stringify({ name, version: '1.2.3', license: 'MIT' }))
  writeFileSync(join(dir, 'index.js'), 'export const value = 1')
  if (license) writeFileSync(join(dir, 'LICENSE'), license)
  return { root, dir }
}
afterEach(() => roots.splice(0).forEach(root => rmSync(root, { recursive: true, force: true })))

describe('browser third-party notices', () => {
  it('preserves the separate KaTeX font attribution and reserved font names', () => {
    const root = fileURLToPath(new URL('..', import.meta.url))
    const result = collectNotices([], [join(root, 'node_modules/katex')], join(root, 'license-supplements'))
    expect(result).toContain('Design Science')
    expect(result).toContain('Reserved Font Name KaTeX_AMS')
    expect(result).toContain('SIL OPEN FONT LICENSE Version 1.1')
  })
  it('retains full license and NOTICE text once per package', () => {
    const { dir } = fixture('@example/widget', 'Copyright Widget authors\nPermission is granted.')
    writeFileSync(join(dir, 'NOTICE'), 'Includes work by Original authors.')
    const result = collectNotices([join(dir, 'index.js'), join(dir, 'index.js')])
    expect(result).toContain('Copyright Widget authors\nPermission is granted.')
    expect(result).toContain('Includes work by Original authors.')
    expect(result.match(/@example\/widget@1.2.3/g)).toHaveLength(1)
  })

  it('fails the build when shipped code has no license text', () => {
    const { dir } = fixture('missing-license')
    expect(() => collectNotices([join(dir, 'index.js')])).toThrow(/missing-license/)
  })

  it('includes separately copied assets as well as bundled modules', () => {
    const { dir } = fixture('worker', 'Worker license terms')
    expect(collectNotices([], [dir])).toContain('Worker license terms')
    expect(collectNotices([join(dir, 'index.js')], [dir]).match(/worker@1.2.3/g)).toHaveLength(1)
  })

  it('only uses an upstream license supplement for its exact reviewed version', () => {
    const { root, dir } = fixture('missing-license')
    const supplements = join(root, 'supplements')
    mkdirSync(supplements)
    writeFileSync(join(supplements, 'missing-license@1.2.3.txt'), 'Upstream MIT terms')
    expect(collectNotices([join(dir, 'index.js')], [], supplements)).toContain('Upstream MIT terms')
    writeFileSync(join(dir, 'package.json'), JSON.stringify({ name: 'missing-license', version: '2.0.0' }))
    expect(() => collectNotices([join(dir, 'index.js')], [], supplements)).toThrow(/missing-license/)
  })
})
