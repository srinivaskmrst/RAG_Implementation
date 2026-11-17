// Shared by both pages: helpers + the status pill in the top bar.

const $ = (id) => document.getElementById(id);

function esc(text) {
  return String(text ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

// Polls /api/status until the backend is ready, then calls onReady(status) once.
async function pollStatus(onReady) {
  try {
    const s = await (await fetch("/api/status")).json();
    const box = $("status");
    if (s.ready) {
      const sets = Object.values(s.domains || {});
      const counts = sets.map((d) => `${d.name}: ${d.chunks}`).join(" · ");
      const empty = sets.filter((d) => d.needs_ingestion).map((d) => d.name);
      box.className = empty.length ? "status loading" : "status ready";
      $("statusText").textContent = empty.length
        ? `${counts} chunks · ${empty.join(", ")} need ingestion – open Data Ingestion`
        : `Ready · ${counts} chunks in Pinecone '${s.index}' · ${s.model}`;
      onReady(s);
      return;
    }
    if (s.error) {
      box.className = "status error";
      $("statusText").textContent = `Backend error: ${s.error}`;
      return;
    }
    box.className = "status loading";
    $("statusText").textContent = `Loading: ${s.status}…`;
  } catch {
    $("status").className = "status error";
    $("statusText").textContent = "Backend not reachable";
  }
  setTimeout(() => pollStatus(onReady), 1500);
}
