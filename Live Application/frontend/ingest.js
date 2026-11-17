// Data Ingestion page: one card per dataset (employee leave policy, bank documents).
//   Ingest -> POST   /api/ingest/{domain}   (document-hash / chunk-hash checks; bank: + guardrails)
//   Delete -> DELETE /api/vectors/{domain}  (every vector of that dataset; the other is untouched)

const NAMES = { leave: "employee leave policy", bank: "bank data" };
const cards = () => document.querySelectorAll(".dataset-card");
const card = (domain) => document.querySelector(`.dataset-card[data-domain="${domain}"]`);
let busy = false;

function renderIngest(domain, report, title) {
  const s = report.summary;
  const bank = !!s.guardrails;
  const tile = (value, label, color = "") => `
    <div class="stat ${color}"><div class="v">${value}</div><div class="k">${label}</div></div>`;

  const rows = report.documents.map((d) => `
    <tr>
      <td>${esc(d.file)}</td>
      ${bank ? `<td>${esc(d.customer || "–")}</td><td><span class="badge cls">${esc(d.classification || "–")}</span></td>
               <td class="num" title="${esc(Object.entries(d.pii_types || {}).map(([k, v]) => `${k}: ${v}`).join(", "))}">${d.pii_masked ?? "–"}</td>`
             : `<td>${d.year || "–"}</td>`}
      <td><code title="SHA-256 of the file">${esc(d.doc_hash)}…</code></td>
      <td><span class="badge ${esc(d.status)}">${esc(d.status)}</span></td>
      <td class="num">${d.pages}</td>
      <td class="num">${d.chunks}</td>
      <td class="num">${d.added ? `<span class="plus">+${d.added}</span>` : 0}</td>
      <td class="num">${d.existing}</td>
      <td class="num">${d.removed ? `<span class="minus">−${d.removed}</span>` : 0}</td>
      <td class="note">${esc(d.note)}</td>
    </tr>`).join("");

  const pii = Object.entries(s.pii_types || {}).map(([k, v]) => `${k} ${v}`).join(" · ");
  card(domain).querySelector(".ingest-result").innerHTML = `
    <p class="ingest-meta"><strong>${esc(title)}</strong> · ${esc(s.finished_at)} · ${s.seconds}s ·
      Pinecone index <code>${esc(s.index || "?")}</code> · namespace <code>${esc(s.namespace)}</code></p>
    <div class="stats">
      ${tile(s.documents, "Documents processed", "blue")}
      ${tile(s.new, "New documents", "violet")}
      ${tile(s.changed, "Changed documents", "amber")}
      ${tile(s.unchanged, "Unchanged (hash match)", "green")}
      ${tile(s.chunks, "Total chunks")}
      ${tile(s.chunks_added, "Chunks embedded & added", "violet")}
      ${tile(s.chunks_existing, "Chunks already in DB", "green")}
      ${tile(s.chunks_removed, "Chunks removed", s.chunks_removed ? "red" : "grey")}
      ${tile(s.vectors_in_db, "Vectors in Pinecone", "blue")}
      ${s.chunks_loaded_from_db != null ? tile(s.chunks_loaded_from_db, "Chunks loaded from DB for retrieval", "green") : ""}
      ${bank ? tile(s.pii_masked, "🛡 PII values masked", "pink") : ""}
      ${bank ? tile(s.quarantined, "🛡 Chunks quarantined (injection)", s.quarantined ? "red" : "grey") : ""}
      ${bank ? tile(s.customers, "Customers", "blue") : ""}
    </div>
    ${bank && pii ? `<p class="ingest-meta">🛡 Masked before embedding: ${esc(pii)}</p>` : ""}
    <details open>
      <summary>Per-document details (${report.documents.length})</summary>
      <div class="table-wrap">
        <table class="doc-table">
          <thead><tr>
            <th>File</th>${bank ? "<th>Customer</th><th>Classification</th><th class=\"num\">PII masked</th>" : "<th>Year</th>"}
            <th>Doc hash</th><th>Status</th><th class="num">Pages</th>
            <th class="num">Chunks</th><th class="num">Added</th><th class="num">Existing</th>
            <th class="num">Removed</th><th>Note</th>
          </tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
    </details>`;
}

function renderDeleted(domain, r) {
  card(domain).querySelector(".ingest-result").innerHTML = `
    <div class="deleted-box">
      <strong>🗑 ${esc(r.name)} deleted</strong> · ${esc(r.finished_at)} · ${r.seconds}s<br>
      ${r.deleted} vectors removed from Pinecone index <code>${esc(r.index)}</code>, namespace
      <code>${esc(r.namespace)}</code> · ${r.vectors_in_db} left. The other dataset was not touched.
      Click <em>Ingest</em> to load the documents again.
    </div>`;
}

function setBusy(on) {
  busy = on;
  document.querySelectorAll(".dataset-card button").forEach((b) => { b.disabled = on; });
}

async function call(domain, action) {
  const btn = card(domain).querySelector(`[data-action="${action}"]`);
  const result = card(domain).querySelector(".ingest-result");
  if (action === "delete" && !confirm(`Delete ALL ${NAMES[domain]} vectors from Pinecone?\n\n` +
      "Only this dataset's namespace is emptied; you can ingest the documents again afterwards.")) return;

  const label = btn.textContent;
  setBusy(true);
  btn.textContent = action === "ingest" ? "Ingesting…" : "Deleting…";
  result.innerHTML = `
    <div class="skeleton"><span></span><span></span></div>
    <p class="running">${action === "ingest"
      ? (domain === "bank" ? "Reading documents, masking PII, scanning for injection, hashing, checking Pinecone…"
                           : "Reading documents, hashing, checking Pinecone…")
      : "Deleting vectors from Pinecone…"}</p>`;
  try {
    const res = await fetch(action === "ingest" ? `/api/ingest/${domain}` : `/api/vectors/${domain}`,
                            { method: action === "ingest" ? "POST" : "DELETE" });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
    if (action === "ingest") renderIngest(domain, data, "Ingestion finished");
    else renderDeleted(domain, data);
  } catch (err) {
    result.innerHTML = `<div class="error-box">${esc(err.message || err)}</div>`;
  } finally {
    btn.textContent = label;
    setBusy(false);
    refreshCounts();
  }
}

function showCounts(s) {
  for (const [domain, d] of Object.entries(s.domains || {})) {
    const c = card(domain);
    if (!c) continue;
    c.querySelector(".ds-count").textContent = d.needs_ingestion && !d.vectors
      ? "empty – not ingested yet"
      : `${d.vectors} vectors · ${d.documents} documents in the vector DB`;
    c.querySelector(".ns").textContent = d.namespace;
  }
}

async function refreshCounts() {
  try { showCounts(await (await fetch("/api/status")).json()); } catch {}
}

cards().forEach((c) => c.querySelectorAll("button[data-action]").forEach((b) =>
  b.addEventListener("click", () => !busy && call(c.dataset.domain, b.dataset.action))));

pollStatus((s) => {
  setBusy(false);
  showCounts(s);
  for (const [domain, d] of Object.entries(s.domains || {}))
    if (d.last_ingest) renderIngest(domain, d.last_ingest, "Last ingestion");
});
