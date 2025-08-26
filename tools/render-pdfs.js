#!/usr/bin/env node
const fs = require('fs');
const path = require('path');
const glob = require('glob');
const puppeteer = require('puppeteer');

// Simple arg parsing
const argv = require('minimist')(process.argv.slice(2), {
  string: ['buildDir', 'outDir', 'baseUrl'],
  default: { buildDir: '_build/html', outDir: 'exports/pdfs', baseUrl: 'http://localhost:8000' }
});

const buildDir = path.resolve(argv.buildDir);
const outDir = path.resolve(argv.outDir);
const baseUrl = argv.baseUrl.replace(/\/$/, '');

if (!fs.existsSync(buildDir)) {
  console.error('Build directory does not exist:', buildDir);
  process.exit(1);
}

async function renderAll() {
  // Find source markdown files that request PDF export in frontmatter
  const mdPattern = '**/*.md';
  const mdFiles = glob.sync(mdPattern, { nodir: true, ignore: ['**/node_modules/**', '**/_build/**', 'tools/**', '.git/**'] });
  const pagesToRender = [];

  for (const md of mdFiles) {
    const src = fs.readFileSync(md, 'utf8');
    const fmMatch = src.match(/^---\n([\s\S]*?)\n---/m);
    if (!fmMatch) continue;
    const fm = fmMatch[1];
    // crude check for exports: - format: pdf inside frontmatter
    if (/exports\s*:\s*[\s\S]*?format\s*:\s*pdf/i.test(fm)) {
      // candidate HTML paths relative to buildDir
      const rel = md.replace(/\\\\/g, '/');
      const htmlCandidate1 = path.join(buildDir, rel.replace(/\.md$/i, '.html'));
      const htmlCandidate2 = path.join(buildDir, rel.replace(/\.md$/i, '/index.html'));
      if (fs.existsSync(htmlCandidate1)) pagesToRender.push({ md, html: htmlCandidate1 });
      else if (fs.existsSync(htmlCandidate2)) pagesToRender.push({ md, html: htmlCandidate2 });
      else {
        console.warn('No built HTML found for', md, '\n  tried:', htmlCandidate1, htmlCandidate2);
      }
    }
  }

  if (pagesToRender.length === 0) {
    console.error('No pages found requesting PDF export via frontmatter.');
    process.exit(1);
  }

  const browser = await puppeteer.launch({ args: ['--no-sandbox','--disable-setuid-sandbox'] });
  const page = await browser.newPage();
  await page.setViewport({ width: 1200, height: 800 });

  const map = {};
  for (const entry of pagesToRender) {
    const file = entry.html;
    const rel = path.relative(buildDir, file);
    const url = baseUrl + '/' + rel.replace(/\\\\/g, '/');

    // Determine output path: keep directory structure, write .pdf
    const parsed = path.parse(rel);
    let outPath;
    if (parsed.base.toLowerCase() === 'index.html') {
      const dir = parsed.dir === '' ? '' : parsed.dir;
      const pdfName = (dir === '' ? 'index' : dir.replace(/\\//g, '-')) + '.pdf';
      outPath = path.join(outDir, pdfName);
    } else {
      const dir = parsed.dir;
      const pdfName = parsed.name + '.pdf';
      outPath = path.join(outDir, dir, pdfName);
    }

    fs.mkdirSync(path.dirname(outPath), { recursive: true });

    console.log(`Rendering ${url} -> ${outPath}`);
    try {
      await page.goto(url, { waitUntil: 'networkidle0', timeout: 120000 });
      await page.pdf({ path: outPath, format: 'A4', printBackground: true, margin: { top: '20mm', bottom: '20mm', left: '15mm', right: '15mm' } });
    } catch (err) {
      console.error(`Failed to render ${url}:`, err.message);
    }
    // Add map entries: prefer site route like '/foo/bar' -> pdf url relative to site
    const sitePath = '/' + rel.replace(/\\\\/g, '/').replace(/index\.html$/, '').replace(/\.html$/, '');
    // PDF url relative to site root
    const pdfUrl = path.relative(buildDir, outPath).replace(/\\\\/g, '/');
    map[sitePath] = '/' + pdfUrl;
  }

  // write index.json into outDir parent so site can serve it under /exports/pdfs/index.json
  const indexJsonPath = path.join(outDir, 'index.json');
  fs.mkdirSync(path.dirname(indexJsonPath), { recursive: true });
  fs.writeFileSync(indexJsonPath, JSON.stringify(map, null, 2), 'utf8');

  await browser.close();
  console.log('Done. PDFs written to', outDir);
}

renderAll().catch(err => { console.error(err); process.exit(1); });
