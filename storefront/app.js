async function load() {
  const target = document.getElementById('target').value;
  const task = document.getElementById('task').value;
  const q = document.getElementById('q').value;
  const params = new URLSearchParams();
  if (target) params.set('target', target);
  if (task) params.set('task', task);
  if (q) params.set('q', q);
  const res = await fetch('/models?' + params.toString());
  const items = await res.json();
  const facets = await (await fetch('/facets')).json();
  document.getElementById('facets').innerHTML =
    `<p>${facets.total} artifacts · targets ${JSON.stringify(facets.target)} · tools ${JSON.stringify(facets.tool)}</p>`;
  const el = document.getElementById('results');
  if (!items.length) { el.innerHTML = '<p>No artifacts match.</p>'; return; }
  el.innerHTML = items.map(a => {
    const b = a.benchmark || {};
    const lat = (b.latency_ms || {}).p95 ?? '?';
    return `<div class="card"><strong>${a.model_id}</strong><br>
      ${a.target} · ${a.task} · ${a.tool} · ${a.bit_width}bit · ${a.format}<br>
      ${a.size_bytes} bytes · state ${a.state} · p95 ${lat} ms · quality delta ${b.quality_delta ?? '?'}<br>
      <a href="/models/${a.id}/pull">Pull ${a.id}</a> ·
      <a href="/models/${a.id}">Details</a> ·
      <a href="/models/${a.id}/deploy-snippet">Deploy snippet</a></div>`;
  }).join('');
}
load();
