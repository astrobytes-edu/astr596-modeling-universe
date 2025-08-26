
# Render per-page PDFs from the built MyST site

## Requirements

- Node 18+ (or system Node compatible with Puppeteer)
- `npm install` in this `tools/` directory to install dependencies

## Usage

1. Build the site HTML: `myst build --html` (output is `_build/html` by default)

2. Serve the build directory (option A) or use a local static server (option B)

   - Option A — quick Python server (from repo root):

     ```bash
     python -m http.server --directory _build/html 8000
     ```

   - Option B — use any static server; set `--baseUrl` accordingly

3. Run the renderer (from `tools/`):

   ```bash
   npm install
   node render-pdfs.js --buildDir ../_build/html --outDir ../exports/pdfs --baseUrl http://localhost:8000
   ```

## Output

- PDFs are written to `exports/pdfs/` preserving a readable directory-like naming scheme.

## Notes

- Puppeteer downloads a Chromium binary; this can be large. If you prefer, adjust `puppeteer` settings to use system Chrome by setting `PUPPETEER_EXECUTABLE_PATH`.
- This approach avoids requiring a LaTeX toolchain and works across platforms.
