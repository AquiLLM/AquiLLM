// Fixed entry point for the isolated wrapper. Positional Vitest arguments are
// substring filters, so use a fresh config with an exact include list instead.
'use strict';

const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const { spawnSync } = require('node:child_process');

function main() {
  const root = fs.realpathSync(process.cwd());
  if (root !== '/app/react' || process.argv.length !== 3) {
    throw new Error('isolated frontend entry point required');
  }
  const specs = JSON.parse(process.argv[2]);
  if (!Array.isArray(specs) || !specs.length || new Set(specs).size !== specs.length) {
    throw new Error('nonempty, unique test paths required');
  }
  for (const spec of specs) {
    if (typeof spec !== 'string' || !/^src\/[A-Za-z0-9_./-]+\.test\.tsx?$/.test(spec)
        || spec.split('/').some(part => !part || part === '.' || part === '..')) {
      throw new Error('literal test path required');
    }
    const target = path.join(root, spec);
    if (fs.realpathSync(target) !== target || !fs.statSync(target).isFile()) {
      throw new Error('test path must be an existing frontend file');
    }
  }
  const config = path.join(root, `.rag-vitest-${crypto.randomUUID()}.config.mts`);
  fs.writeFileSync(config,
    'import base from "./vitest.config.ts";\n' +
    `export default {...base,test:{...base.test,include:${JSON.stringify(specs)}}};\n`,
    { flag: 'wx' });
  try {
    const result = spawnSync(process.execPath,
      ['node_modules/vitest/vitest.mjs', 'run', '--config', config],
      { cwd: root, stdio: 'inherit' });
    if (result.error) throw result.error;
    return result.status ?? 1;
  } finally {
    fs.unlinkSync(config);
  }
}

try {
  process.exitCode = main();
} catch (error) {
  console.error('Isolated frontend tests failed:', error.message);
  process.exitCode = 2;
}
