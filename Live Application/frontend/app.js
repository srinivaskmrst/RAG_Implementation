// RAG Playground front end: calls /api/rag/traditional and /api/rag/hybrid in parallel
// and renders each result in its own section as soon as it arrives.

const MODES = ["traditional", "hybrid"];
let ready = false;

// ---------------------------------------------------------------- small helpers
// Minimal Markdown -> HTML (headings, bold/italic, code, bullet & numbered lists, citations)
function markdown(md) {
  const inline = (s) => esc(s)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*\s][^*]*)\*/g, "$1<em>$2</em>")
    .replace(/\[(Source \d+(?:,\s*Source \d+)*|HR-LV-\d{4}[^\]]*)\]/g, '<span class="cite">[$1]</span>');

  const out = [];
  let list = null;
  const closeList = () => { if (list) { out.push(`</${list}>`); list = null; } };

  for (const raw of String(md || "").split(/\r?\n/)) {
    const line = raw.trimEnd();
    let m;
    if (!line.trim()) { closeList(); continue; }
    if ((m = line.match(/^\s*#{1,6}\s+(.*)/))) { closeList(); out.push(`<h4>${inline(m[1])}</h4>`); continue; }
    if ((m = line.match(/^\s*[-*•]\s+(.*)/))) {
      if (list !== "ul") { closeList(); out.push("<ul>"); list = "ul"; }
      out.push(`<li>${inline(m[1])}</li>`); continue;
    }
    if ((m = line.match(/^\s*\d+[.)]\s+(.*)/))) {
      if (list !== "ol") { closeList(); out.push("<ol>"); list = "ol"; }
      out.push(`<li>${inline(m[1])}</li>`); continue;
    }
    closeList();
    out.push(`<p>${inline(line)}</p>`);
  }
  closeList();
  return out.join("");
}

function setMeta(section, html) { section.querySelector(".meta").innerHTML = html; }

// top-K inputs: [element id, request field, min, max]
const TOPK_FIELDS = [
  ["keywordTopK", "keyword_top_k", 1, 50],
  ["vectorTopK", "vector_top_k", 1, 50],
  ["finalTopK", "final_top_k", 1, 20],
];

function readTopK() {
  const values = {};
  for (const [id, key, min, max] of TOPK_FIELDS) {
    const input = $(id);
    const n = Number(input.value);
    if (!Number.isInteger(n) || n < min || n > max) {
      input.focus();
      throw new Error(`${input.closest("label").querySelector("strong").textContent} must be a whole number from ${min} to ${max}.`);
    }
    values[key] = n;
  }
  try { localStorage.setItem("topk", JSON.stringify(values)); } catch {}   // remember for next visit
  return values;
}

function restoreTopK() {
  try {
    const saved = JSON.parse(localStorage.getItem("topk") || "{}");
    for (const [id, key] of TOPK_FIELDS) if (saved[key]) $(id).value = saved[key];
  } catch {}
}

// ---------------------------------------------------------------- model dropdowns
// Two dropdowns (Ollama / Google Gemini). Exactly ONE model is used per question: the one in the
// active card (radio button). Picking a model in a dropdown makes that card active.
const PROVIDERS = ["ollama", "gemini"];
let activeProvider = null;

function loadChoice() {
  try { return JSON.parse(localStorage.getItem("llm") || "{}"); } catch { return {}; }
}
function saveChoice() {
  try {
    localStorage.setItem("llm", JSON.stringify({
      provider: activeProvider, ollama: $("ollamaSelect").value, gemini: $("geminiSelect").value }));
  } catch {}
}

function setActive(p) {
  if (!p || $(`${p}Select`).disabled) return;          // provider not available
  activeProvider = p;
  $(p === "ollama" ? "useOllama" : "useGemini").checked = true;
  $("cardOllama").classList.toggle("active", p === "ollama");
  $("cardGemini").classList.toggle("active", p === "gemini");
  saveChoice();
}

function fillOllama(info) {
  const select = $("ollamaSelect");
  if (!info.available) {
    select.innerHTML = "<option>Ollama not available</option>";
    select.disabled = true;
    $("ollamaHint").textContent = info.error;
    return [];
  }
  const installed = info.models.filter((m) => m.installed);
  const missing = info.models.filter((m) => !m.installed);
  const label = (m) => `${m.name}${m.size_gb ? ` · ${m.size_gb} GB` : ""}${m.note ? ` — ${m.note}` : ""}`;
  select.innerHTML =
    `<optgroup label="Downloaded (${installed.length})">${installed.map((m) =>
      `<option value="${esc(m.name)}">${esc(label(m))}${m.name === info.default ? " (default)" : ""}</option>`).join("")}</optgroup>` +
    (missing.length ? `<optgroup label="Not downloaded – run: ollama pull <name>">${missing.map((m) =>
      `<option value="${esc(m.name)}" disabled>${esc(label(m))}</option>`).join("")}</optgroup>` : "");
  select.disabled = installed.length === 0;
  $("ollamaHint").innerHTML = missing.length
    ? `${installed.length} downloaded. For a greyed-out model run <code>ollama pull ${esc(missing[0].name)}</code>, then press ↻.`
    : `${installed.length} models downloaded.`;
  return installed.map((m) => m.name);
}

function fillGemini(info) {
  const select = $("geminiSelect");
  if (!info.available) {
    select.innerHTML = "<option>Gemini not available</option>";
    select.disabled = true;
    $("geminiHint").textContent = info.error;
    return [];
  }
  select.innerHTML = `<optgroup label="Gemini models (${info.models.length})">${info.models.map((m) =>
    `<option value="${esc(m.name)}">${esc(m.name)}${m.display_name && m.display_name !== m.name
      ? ` — ${esc(m.display_name)}` : ""}${m.name === info.default ? " (default)" : ""}</option>`).join("")}</optgroup>`;
  select.disabled = info.models.length === 0;
  $("geminiHint").textContent = `${info.models.length} models available for your API key. ` +
    "Free tier: limited requests per day per model; some (e.g. Pro) may need a paid plan.";
  return info.models.map((m) => m.name);
}

async function loadModels() {
  $("ollamaSelect").disabled = $("geminiSelect").disabled = true;
  try {
    const res = await fetch("/api/models");
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);

    const names = { ollama: fillOllama(data.ollama), gemini: fillGemini(data.gemini) };
    const saved = loadChoice();
    for (const p of PROVIDERS) {                      // restore each dropdown's last value
      const wanted = [saved[p], data[p].default].find((n) => names[p].includes(n));
      if (wanted) $(`${p}Select`).value = wanted;
    }
    const order = [saved.provider, data.default.provider, ...PROVIDERS];
    setActive(order.find((p) => p && names[p] && names[p].length));
  } catch (err) {
    $("ollamaHint").textContent = $("geminiHint").textContent = err.message || String(err);
  }
}

function options() {
  return {
    provider: activeProvider,
    model: activeProvider ? $(`${activeProvider}Select`).value : null,
    rerank: $("optRerank").checked,
    corrective: $("optCorrective").checked,
    agentic: $("optAgentic").checked,
    ...readTopK(),
  };
}

function topkBadge(k) {
  if (!k) return "";
  const text = k.keyword != null ? `K ${k.keyword}+${k.vector} → ${k.final}` : `K ${k.vector} → ${k.final}`;
  return `<span class="badge plain" title="keyword top K + vector top K → final top K">${text}</span>`;
}

function techniqueBadges(opts) {
  const names = [["rerank", "Rerank"], ["corrective", "Corrective"], ["agentic", "Agentic"]]
    .filter(([k]) => opts[k]).map(([k, n]) => `<span class="badge ${k}">${n}</span>`);
  return names.join("") || '<span class="badge plain">Base</span>';
}

// ---------------------------------------------------------------- rendering
function renderLoading(section, opts) {
  setMeta(section, (opts.model ? `<span class="badge model ${esc(opts.provider)}">${esc(opts.model)}</span>` : "") + techniqueBadges(opts));
  section.querySelector(".body").innerHTML = `
    <div class="skeleton"><span></span><span></span><span></span></div>
    <p class="running">Running pipeline${opts.agentic ? " (agent may search several times)" : ""}…</p>
    ${opts.provider === "ollama" ? `<p class="running muted">Local model on CPU: this can take ${
      opts.agentic || opts.corrective ? "a few minutes" : "up to a minute"}.</p>` : ""}`;
}

function renderError(section, message) {
  section.querySelector(".body").innerHTML = `<div class="error-box">${esc(message)}</div>`;
}

// Reranking: ranking before (search order) vs after (cross-encoder order), one table per search
function renderReranking(reranks) {
  if (!reranks || !reranks.length) return "";
  const move = (m) => m > 0 ? `<span class="mv up" title="moved up ${m}">▲ ${m}</span>`
                    : m < 0 ? `<span class="mv down" title="moved down ${-m}">▼ ${-m}</span>`
                    : `<span class="mv same" title="same position">=</span>`;

  const table = (rr) => {
    const before = [...rr.rows].sort((a, b) => a.before - b.before);
    const beforeList = before.map((x) => `
      <li class="${x.kept ? "kept" : ""}"><span class="pos">${x.before}</span>
        <span class="lbl" title="${esc(x.snippet)}">${esc(x.label.replace("Leave Policy ", ""))}</span>
        <span class="sc">${x.retrieval_score}</span></li>`).join("");
    const afterRows = rr.rows.map((x) => `
      <tr class="${x.kept ? "kept" : "dropped"}">
        <td class="num"><span class="pos">${x.after}</span></td>
        <td class="num">${x.before}</td>
        <td>${move(x.move)}</td>
        <td><span title="${esc(x.snippet)}">${esc(x.label.replace("Leave Policy ", ""))}</span><small>${esc(x.snippet.slice(0, 60))}…</small></td>
        <td class="num">${x.retrieval_score}</td>
        <td class="num"><strong>${x.rerank_score}</strong></td>
        <td>${x.kept ? '<span class="badge ok" title="sent to the LLM">kept</span>' : '<span class="badge plain-dark">dropped</span>'}</td>
      </tr>`).join("");

    return `
      <p class="rr-query">Query: “${esc(rr.query)}”${rr.year ? ` · year ${rr.year}` : ""} ·
        ${rr.candidates} candidates → kept top ${rr.kept}</p>
      <div class="rr-grid">
        <div class="rr-before">
          <h5>Before reranking <small>(search order, ${esc(rr.retrieval_score_name)} score ·
            <span class="pos-legend">orange</span> = kept after reranking)</small></h5>
          <ol class="rr-list">${beforeList}</ol>
        </div>
        <div class="rr-after">
          <h5>After reranking <small>(cross-encoder score, asked year first)</small></h5>
          <div class="table-wrap">
            <table class="doc-table rr-table">
              <thead><tr><th class="num">New #</th><th class="num">Old #</th><th>Move</th><th>Chunk</th>
                <th class="num">${esc(rr.retrieval_score_name)}</th><th class="num">rerank</th><th></th></tr></thead>
              <tbody>${afterRows}</tbody>
            </table>
          </div>
        </div>
      </div>`;
  };

  return `
    <details class="rerank-box" open>
      <summary>Reranking: before → after${reranks.length > 1 ? ` (${reranks.length} searches)` : ""}</summary>
      ${reranks.map((rr, i) => reranks.length > 1
        ? `<details class="rr-search" ${i === 0 ? "open" : ""}><summary>Search ${i + 1}</summary>${table(rr)}</details>`
        : table(rr)).join("")}
    </details>`;
}

function renderResult(section, r) {
  const verdictClass = { CORRECT: "ok", AMBIGUOUS: "warn", INCORRECT: "bad" }[r.verdict] || "";
  setMeta(section, `
    ${r.techniques.length ? r.techniques.map((t) => `<span class="badge ${esc(t.toLowerCase())}">${esc(t)}</span>`).join("")
                          : '<span class="badge plain">Base</span>'}
    ${r.verdict ? `<span class="badge ${verdictClass}">${esc(r.verdict)}</span>` : ""}
    ${r.model ? `<span class="badge model ${esc(r.provider || "")}" title="${esc(r.model)}">${esc(r.model.replace(/ \(.*\)$/, ""))}</span>` : ""}
    ${topkBadge(r.top_k)}
    <span class="badge plain">${r.seconds}s</span>
    <span class="badge plain">${r.llm_calls} LLM call${r.llm_calls === 1 ? "" : "s"}</span>`);

  const steps = r.steps.map((s) => `
    <li class="${esc(s.kind)}">
      <strong>${esc(s.title)}</strong><span class="t">${s.t}s</span>
      ${s.detail ? `<span class="d">${esc(s.detail)}</span>` : ""}
    </li>`).join("");

  const gradeClass = { relevant: "ok", partial: "warn", irrelevant: "bad" };
  const sources = r.sources.map((s, i) => `
    <div class="source ${esc(s.grade || "")}">
      <div class="head">
        <strong>[${i + 1}] Leave Policy ${s.year} · p.${s.page}</strong>
        ${s.score != null ? `<span class="badge score">${esc(s.score_name)} ${s.score}</span>` : ""}
        ${s.grade ? `<span class="badge ${gradeClass[s.grade] || ""}">${esc(s.grade)}</span>` : ""}
      </div>
      <p>${esc(s.text)}</p>
    </div>`).join("");

  section.querySelector(".body").innerHTML = `
    <div class="answer">${markdown(r.answer)}</div>
    ${renderReranking(r.reranking)}
    <details open>
      <summary>Pipeline steps (${r.steps.length})</summary>
      <ol class="steps">${steps}</ol>
    </details>
    <details>
      <summary>Sources (${r.sources.length})</summary>
      <div class="sources">${sources || '<p class="muted">No sources used.</p>'}</div>
    </details>`;
}

// ---------------------------------------------------------------- run
async function runMode(mode, question, opts) {
  const section = $(mode);
  renderLoading(section, opts);
  try {
    const res = await fetch(`/api/rag/${mode}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, ...opts }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail ? JSON.stringify(data.detail).replace(/^"|"$/g, "") : `HTTP ${res.status}`);
    renderResult(section, data);
  } catch (err) {
    renderError(section, err.message || String(err));
  }
}

async function run(event) {
  event?.preventDefault();
  const question = $("question").value.trim();
  if (!ready || question.length < 2) return;

  let opts;
  try {
    opts = options();
  } catch (err) {
    selectedModes().forEach((m) => renderError($(m), err.message));
    return;
  }
  $("askBtn").disabled = true;
  $("askBtn").textContent = "Running…";
  await Promise.all(selectedModes().map((m) => runMode(m, question, opts)));
  $("askBtn").disabled = false;
  $("askBtn").textContent = RUN_LABELS[runChoice()];
}

// ---------------------------------------------------------------- which pipelines to run
const RUN_LABELS = { traditional: "Run Traditional RAG", hybrid: "Run Hybrid RAG", both: "Run both pipelines" };

function runChoice() {
  return document.querySelector('input[name="runMode"]:checked')?.value || "both";
}

function selectedModes() {
  const choice = runChoice();
  return choice === "both" ? MODES : [choice];
}

// show only the selected result panel(s) and update the button text
function updateRunMode() {
  const choice = runChoice();
  for (const m of MODES) $(m).hidden = !selectedModes().includes(m);
  document.querySelector(".results").classList.toggle("single", choice !== "both");
  if (!$("askBtn").textContent.startsWith("Running")) $("askBtn").textContent = RUN_LABELS[choice];
  try { localStorage.setItem("runMode", choice); } catch {}
}

function restoreRunMode() {
  let saved = null;
  try { saved = localStorage.getItem("runMode"); } catch {}
  const input = document.querySelector(`input[name="runMode"][value="${saved}"]`);
  if (input) input.checked = true;
  updateRunMode();
}

// ---------------------------------------------------------------- wiring
$("askForm").addEventListener("submit", run);
$("question").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) run(e);
});
document.querySelectorAll("#samples .chip").forEach((chip) =>
  chip.addEventListener("click", () => { $("question").value = chip.textContent; $("question").focus(); }));

for (const p of PROVIDERS) {
  $(`${p}Select`).addEventListener("change", () => setActive(p));       // picking a model selects its card
  $(`${p}Select`).addEventListener("focus", () => setActive(p));
  $(p === "ollama" ? "useOllama" : "useGemini").addEventListener("change", () => setActive(p));
}
$("modelRefresh").addEventListener("click", loadModels);
document.querySelectorAll('input[name="runMode"]').forEach((r) => r.addEventListener("change", updateRunMode));

restoreTopK();
restoreRunMode();
loadModels();
pollStatus((s) => {
  ready = true;
  $("askBtn").disabled = false;
});
