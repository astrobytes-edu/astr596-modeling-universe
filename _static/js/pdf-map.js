// Fetch a JSON map of html paths -> pdf URLs and inject download link when present
(async function () {
  try {
    const mapUrl = '/exports/pdfs/index.json';
    const res = await fetch(mapUrl);
    if (!res.ok) return;
    const map = await res.json();
    // normalize pathname: prefer pathname without trailing slash
    const pathname = window.location.pathname.replace(/index\.html$/, '').replace(/\/$/, '');
    // Candidate keys: pathname, pathname + '/', pathname + '.html'
    const candidates = [pathname, pathname + '/', pathname + '.html'];
    for (const key of candidates) {
      if (map[key]) {
        const a = document.createElement('a');
        a.href = map[key];
        a.innerText = 'Download PDF';
        a.className = 'pdf-download btn';
        // inject into page header if available
        const header = document.querySelector('.page-header') || document.querySelector('header') || document.body;
        header.insertBefore(a, header.firstChild);
        break;
      }
    }
  } catch (err) {
    console.debug('pdf-map failed', err);
  }
})();
