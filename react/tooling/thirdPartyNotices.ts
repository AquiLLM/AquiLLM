import { existsSync, readFileSync, readdirSync } from 'node:fs'
import { join, relative } from 'node:path'
import type { Plugin } from 'vite'

const noticeFile = /^(licen[cs]e|copying|notice|ofl)([._-]|$)/i

function noticeFiles(directory: string): string[] {
  return readdirSync(directory, { withFileTypes: true }).flatMap(entry => {
    const path = join(directory, entry.name)
    if (entry.isDirectory() && entry.name !== 'node_modules' && !entry.name.startsWith('.')) {
      return noticeFiles(path)
    }
    return entry.isFile() && noticeFile.test(entry.name) ? [path] : []
  }).sort()
}

export function collectNotices(moduleIds: string[], extraPackages: string[] = [], supplements?: string): string {
  const packages = new Set(extraPackages.map(path => path.replaceAll('\\', '/')))
  for (const id of moduleIds) {
    const match = id.replaceAll('\\', '/').replace(/^\0/, '').match(/^(.*\/node_modules\/(?:@[^/]+\/)?[^/]+)/)
    if (match) packages.add(match[1])
  }
  const missing: string[] = []
  const sections = [...packages].map(directory => {
    const manifest = JSON.parse(readFileSync(join(directory, 'package.json'), 'utf8'))
    const files = noticeFiles(directory)
    const supplement = supplements && join(supplements, `${manifest.name.replaceAll('/', '__')}@${manifest.version}.txt`)
    if (supplement && existsSync(supplement)) files.push(supplement)
    if (!files.length) missing.push(`${manifest.name}@${manifest.version}`)
    const texts = [...new Set(files.map(path => readFileSync(path, 'utf8').trim()))]
    return `${manifest.name}@${manifest.version}\nLicense: ${manifest.license ?? 'See terms below'}\n`
      + texts.join('\n\n')
  }).sort()
  if (missing.length) throw new Error(`Missing license text for ${missing.join(', ')}; retain upstream notices before shipping.`)
  return 'Third-party licenses for the AquiLLM browser distribution.\n'
    + 'Each component retains its own license.\n\n' + sections.join('\n\n' + '='.repeat(72) + '\n\n') + '\n'
}

export function thirdPartyNotices(reactRoot: string): Plugin {
  return {
    name: 'aquillm-third-party-notices',
    apply: 'build',
    generateBundle(_options, bundle) {
      const ids = Object.values(bundle).flatMap(item => item.type === 'chunk' ? Object.keys(item.modules) : [])
      // The worker is copied outside Rollup; KaTeX ships font assets too.
      const extras = ['pdfjs-dist', 'katex'].map(name => join(reactRoot, 'node_modules', name))
      for (const directory of extras) {
        if (!existsSync(directory)) this.error(`Missing distribution dependency: ${relative(reactRoot, directory)}`)
      }
      this.emitFile({
        type: 'asset', fileName: 'THIRD_PARTY_LICENSES.txt',
        source: collectNotices(ids, extras, join(reactRoot, 'license-supplements')),
      })
    },
  }
}
