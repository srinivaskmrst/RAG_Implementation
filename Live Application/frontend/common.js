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
      box.className = "status ready";
      $("statusText").textContent = `Ready · ${s.chunks} chunks · ${s.model}`;
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
