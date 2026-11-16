// Data Ingestion page: calls POST /api/ingest and shows the report
// (documents processed, document-hash / chunk-hash results, chunk counts).

function renderIngest(report, title) {
  const s = report.summary;
  const tile = (value, label, color = "") => `
    <div class="stat ${color}"><div class="v">${value}</div><div class="k">${label}</div></div>`;

  const rows = report.documents.map((d) => `
    <tr>
      <td>${esc(d.file)}</td>
      <td>${d.year || "–"}</td>
      <td><code title="SHA-256 of the file">${esc(d.doc_hash)}…</code></td>
      <td><span class="badge ${esc(d.status)}">${esc(d.status)}</span></td>
      <td class="num">${d.pages}</td>
      <td class="num">${d.chunks}</td>
      <td class="num">${d.added ? `<span class="plus">+${d.added}</span>` : 0}</td>
      <td class="num">${d.existing}</td>
      <td class="num">${d.removed ? `<span class="minus">−${d.removed}</span>` : 0}</td>
      <td class="note">${esc(d.note)}</td>
    </tr>`).join("");

  $("ingestResult").innerHTML = `
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
    </div>
    <details open>
      <summary>Per-document details (${report.documents.length})</summary>
      <div class="table-wrap">
        <table class="doc-table">
          <thead><tr>
            <th>File</th><th>Year</th><th>Doc hash</th><th>Status</th><th class="num">Pages</th>
            <th class="num">Chunks</th><th class="num">Added</th><th class="num">Existing</th>
            <th class="num">Removed</th><th>Note</th>
          </tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
    </details>`;
}

async function runIngest() {
  const btn = $("ingestBtn");
  btn.disabled = true;
  btn.textContent = "Ingesting…";
  $("ingestResult").innerHTML = `
    <div class="skeleton"><span></span><span></span></div>
    <p class="running" style="color:var(--ing-2)">Reading documents, hashing, checking Pinecone…</p>`;
  try {
    const res = await fetch("/api/ingest", { method: "POST" });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
    renderIngest(data, "Ingestion finished");
  } catch (err) {
    $("ingestResult").innerHTML = `<div class="error-box">${esc(err.message || err)}</div>`;
  } finally {
    btn.disabled = false;
    btn.textContent = "Ingest documents";
  }
}

$("ingestBtn").addEventListener("click", runIngest);

pollStatus((s) => {
  $("ingestBtn").disabled = false;
  if (s.last_ingest) renderIngest(s.last_ingest, "Last ingestion");
});
