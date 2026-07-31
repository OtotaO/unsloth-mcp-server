#!/usr/bin/env node
//
// Publish-safety gate.
//
// build/ is gitignored, so a clean checkout has no compiled output. Without a
// `prepack` script, `npm publish` silently shipped a tarball containing only
// README.md + package.json — no executable at all — while package.json's `bin`
// still pointed at build/index.js. `npx unsloth-mcp-server` would have failed
// with ENOENT on first use, and npm does not allow republishing 1.0.0.
//
// This script packs the package exactly as `npm publish` would (so `prepack`
// runs) and asserts that every path `bin` and `main` point at is actually in
// the tarball, and that the entry point is a runnable CLI.
//
// Run: npm run check:pack

import { execFileSync } from 'node:child_process'
import { readFileSync, existsSync, statSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const pkg = JSON.parse(readFileSync(path.join(root, 'package.json'), 'utf8'))

const required = new Set()
if (pkg.main) required.add(pkg.main.replace(/^\.\//, ''))
for (const target of Object.values(pkg.bin ?? {})) required.add(target.replace(/^\.\//, ''))

if (required.size === 0) {
  console.error('⛔ check-pack: package.json declares neither `main` nor `bin` — nothing to verify.')
  process.exit(2)
}

// --json makes npm emit the file list; prepack runs, which is the whole point.
const raw = execFileSync('npm', ['pack', '--dry-run', '--json'], { cwd: root, encoding: 'utf8' })
const packed = JSON.parse(raw.slice(raw.indexOf('[')))[0]
const shipped = new Set(packed.files.map(f => f.path))

const missing = [...required].filter(f => !shipped.has(f))
if (missing.length) {
  console.error('⛔ check-pack: the published tarball would be missing its entry point(s):\n')
  for (const f of missing) console.error(`   ${f}  (declared in package.json main/bin)`)
  console.error(`\n   Tarball would contain ${packed.files.length} file(s):`)
  for (const f of packed.files) console.error(`     ${f.path}`)
  console.error('\n   Fix: ensure `prepack` builds the artifact before packing.')
  process.exit(1)
}

// The bin entry must be executable and self-describing, or `npx` still breaks.
for (const target of Object.values(pkg.bin ?? {})) {
  const abs = path.join(root, target)
  if (!existsSync(abs)) continue // built into the tarball but pruned locally — pack already proved it
  const firstLine = readFileSync(abs, 'utf8').split('\n', 1)[0]
  if (!firstLine.startsWith('#!')) {
    console.error(`⛔ check-pack: ${target} has no shebang — it cannot run as a CLI.`)
    process.exit(1)
  }
  if (!(statSync(abs).mode & 0o111)) {
    console.error(`⛔ check-pack: ${target} is not executable (chmod +x).`)
    process.exit(1)
  }
}

console.log(
  `✅ check-pack: tarball has ${packed.files.length} file(s) and includes ${[...required].join(', ')}.`
)
