// RAG Playground front end: calls /api/rag/traditional and /api/rag/hybrid in parallel
// and renders each result in its own section as soon as it arrives.
// Two datasets: "leave" (employee leave policy) and "bank" (bank documents, with guardrails),
// each with its own "Process RAG" button.

const MODES = ["traditional", "hybrid"];
const DATASETS = { leave: "🧑‍💼 Leave policy", bank: "🏦 Bank documents" };
let ready = false;
let lastDomain = "leave";

// ---------------------------------------------------------------- small helpers
// Minimal Markdown -> HTML (headings, bold/italic, code, bullet & numbered lists, citations)
function markdown(md) {
  const inline = (s) => esc(s)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*\s][^*]*)\*/g, "$1<em>$2</em>")
    .replace(/\[(Source \d+(?:,\s*Source \d+)*|HR-LV-\d{4}[^\]]*)\]/g, '<span class="cite">[$1]</span>')
    .replace(/\[(Wikipedia:[^\]]+)\]/g, '<span class="cite web">[$1]</span>');

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

// ---------------------------------------------------------------- Google Gemini model dropdown
// The app uses Google Gemini only. The model picked here is sent with every question
// (and remembered for the next visit).
function savedModel() {
  try { return localStorage.getItem("geminiModel"); } catch { return null; }
}

function fillGemini(info) {
  const select = $("geminiSelect");
  if (!info.available) {
    select.innerHTML = "<option>Gemini not available</option>";
    select.disabled = true;
    $("geminiHint").textContent = info.error || "Add GEMINI_API_KEY to .env";
    return;
  }
  select.innerHTML = `<optgroup label="Gemini models (${info.models.length})">${info.models.map((m) =>
    `<option value="${esc(m.name)}">${esc(m.name)}${m.display_name && m.display_name !== m.name
      ? ` — ${esc(m.display_name)}` : ""}${m.name === info.default ? " (default)" : ""}</option>`).join("")}</optgroup>`;
  select.disabled = info.models.length === 0;
  const names = info.models.map((m) => m.name);
  const wanted = [savedModel(), info.default].find((n) => names.includes(n));
  if (wanted) select.value = wanted;
  $("geminiHint").textContent = `${info.models.length} models available for your API key. ` +
    "Free tier: limited requests per day per model; some (e.g. Pro) may need a paid plan.";
}

async function loadModels() {
  $("geminiSelect").disabled = true;
  try {
    const res = await fetch("/api/models");
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
    fillGemini(data);
  } catch (err) {
    $("geminiHint").textContent = err.message || String(err);
  }
}

// ---------------------------------------------------------------- bank role (guardrails)
function fillBankAccess(status) {
  const roles = status.bank_roles || [];
  const customers = (status.domains && status.domains.bank && status.domains.bank.customers) || [];
  let saved = {};
  try { saved = JSON.parse(localStorage.getItem("bankAccess") || "{}"); } catch {}
  $("bankRole").innerHTML = roles.map((r) =>
    `<option value="${esc(r.value)}" title="${esc(r.access)}">${esc(r.label)} – ${esc(r.access)}</option>`).join("");
  $("bankCustomer").innerHTML = customers.length
    ? customers.map((c) => `<option value="${esc(c)}">${esc(c)}</option>`).join("")
    : '<option value="">(ingest the bank documents first)</option>';
  if (roles.some((r) => r.value === saved.role)) $("bankRole").value = saved.role;
  if (customers.includes(saved.customer)) $("bankCustomer").value = saved.customer;
  updateCustomerField();
}

function updateCustomerField() {
  const isCustomer = $("bankRole").value === "customer";
  $("bankCustomer").disabled = !isCustomer;
  $("customerField").classList.toggle("off", !isCustomer);
  try {
    localStorage.setItem("bankAccess", JSON.stringify({ role: $("bankRole").value, customer: $("bankCustomer").value }));
  } catch {}
}

function options(domain) {
  return {
    domain,
    role: $("bankRole").value || undefined,
    customer: $("bankRole").value === "customer" ? $("bankCustomer").value : null,
    model: $("geminiSelect").disabled ? null : $("geminiSelect").value,   // Google Gemini model
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
  if (opts.domain === "bank") names.push('<span class="badge guardrails">Guardrails</span>');
  return names.join("") || '<span class="badge plain">Base</span>';
}

function datasetBadge(domain) {
  return `<span class="badge dataset ${esc(domain)}">${esc(DATASETS[domain] || domain)}</span>`;
}

// ---------------------------------------------------------------- rendering
function renderLoading(section, opts) {
  const bank = opts.domain === "bank";
  setMeta(section, datasetBadge(opts.domain) +
    (opts.model ? `<span class="badge model gemini">${esc(opts.model)}</span>` : "") +
    techniqueBadges(opts));
  section.querySelector(".body").innerHTML = `
    <div class="skeleton"><span></span><span></span><span></span></div>
    <p class="running">Running pipeline on ${esc(DATASETS[opts.domain])}${opts.agentic
      ? (bank ? " – Gemini agent is using the internal bank tools (guardrails on)"
              : " – Gemini agent is searching the company policy and live Wikipedia")
      : bank ? " – guardrails on" : ""}…</p>`;
}

// Guardrails report (bank documents): every input / output check, access filter, audit entry
function renderGuardrails(g) {
  if (!g) return "";
  const icon = { pass: "✓", masked: "◐", blocked: "✕" };
  const checks = g.checks.map((c) => `
    <li class="g-${esc(c.status)}">
      <span class="g-icon">${icon[c.status] || "•"}</span>
      <span class="g-stage">${esc(c.stage)}</span>
      <strong>${esc(c.name)}</strong>
      <span class="g-status">${esc(c.status)}</span>
      ${c.detail ? `<span class="d">${esc(c.detail)}</span>` : ""}
    </li>`).join("");
  const s = g.stats || {};
  return `
    <details class="guard-box ${g.blocked ? "blocked" : ""}" open>
      <summary>🛡 Guardrails · ${esc(g.role)} · ${g.blocked ? "request BLOCKED" : "passed"}</summary>
      <p class="guard-meta">Access: <strong>${esc(g.access)}</strong>${g.audit_id
        ? ` · audit entry <code>${esc(g.audit_id)}</code> in <code>${esc(g.audit_log)}</code>` : ""}</p>
      <ul class="guard-checks">${checks}</ul>
      ${g.tool_calls && g.tool_calls.length ? `
        <p class="guard-meta">Agent tool calls (all inside the role filter, written to the audit log):</p>
        <ul class="guard-checks">${g.tool_calls.map((t) => `
          <li class="g-${t.result === "denied" ? "blocked" : "pass"}">
            <span class="g-icon">${t.result === "denied" ? "✕" : "⚙"}</span>
            <strong>${esc(t.tool)}</strong><span class="d">${esc(t.args)} → ${esc(t.result)}</span>
          </li>`).join("")}</ul>` : ""}
      ${g.blocked ? "" : `<p class="guard-meta">Retrieval: ${s.chunks_checked} chunks screened ·
        ${s.access_dropped} dropped by access · ${s.injection_dropped} dropped (injection) ·
        ${s.chunks_remasked} re-masked · ${s.output_masked} PII value(s) masked in the answer</p>`}
    </details>`;
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
    ${datasetBadge(r.domain)}
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
  const sources = r.sources.map((s, i) => s.kind === "web" ? `
    <div class="source web">
      <div class="head">
        <strong>[${i + 1}] 🌐 Wikipedia: ${esc(s.title)}</strong>
        <span class="badge web">live web API</span>
        ${/^https:\/\/[a-z]+\.wikipedia\.org\//.test(s.url || "")
          ? `<a href="${esc(s.url)}" target="_blank" rel="noopener noreferrer">open article ↗</a>` : ""}
      </div>
      <p>${esc(s.text)}…</p>
    </div>` : `
    <div class="source ${esc(s.grade || "")}">
      <div class="head">
        <strong>[${i + 1}] ${esc(s.title || `Leave Policy ${s.year} · p.${s.page}`)}</strong>
        ${s.classification ? `<span class="badge cls">${esc(s.classification)}</span>` : ""}
        ${s.score != null ? `<span class="badge score">${esc(s.score_name)} ${s.score}</span>` : ""}
        ${s.grade ? `<span class="badge ${gradeClass[s.grade] || ""}">${esc(s.grade)}</span>` : ""}
      </div>
      <p>${esc(s.text)}</p>
    </div>`).join("");

  section.querySelector(".body").innerHTML = `
    <div class="answer ${r.guardrails && r.guardrails.blocked ? "blocked" : ""}">${markdown(r.answer)}</div>
    ${renderGuardrails(r.guardrails)}
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

const RUN_BUTTONS = { leave: "runLeave", bank: "runBank" };
let running = false;

// domain = "leave" (Process RAG – Employee leave policy) or "bank" (Process RAG – Bank documents)
async function run(domain) {
  const question = $("question").value.trim();
  if (!ready || running || question.length < 2) {
    if (question.length < 2) $("question").focus();
    return;
  }
  let opts;
  try {
    opts = options(domain);
  } catch (err) {
    selectedModes().forEach((m) => renderError($(m), err.message));
    return;
  }
  lastDomain = domain;
  running = true;
  const btn = $(RUN_BUTTONS[domain]);
  const label = btn.innerHTML;
  Object.values(RUN_BUTTONS).forEach((id) => { $(id).disabled = true; });
  btn.textContent = "Running…";
  await Promise.all(selectedModes().map((m) => runMode(m, question, opts)));
  btn.innerHTML = label;
  Object.values(RUN_BUTTONS).forEach((id) => { $(id).disabled = false; });
  running = false;
}

// ---------------------------------------------------------------- which pipelines to run
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
$("askForm").addEventListener("submit", (e) => { e.preventDefault(); run("leave"); });
$("runBank").addEventListener("click", () => run("bank"));
$("question").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); run(lastDomain); }
});
// two sample-question dropdowns: picking one fills the question box, sets the options it needs
// (Agentic for the ✦ questions, the bank role / customer for bank questions) and highlights the
// matching "Process RAG" button. Only one dropdown shows a selection at a time.
const QUESTION_PICKERS = ["leaveQuestions", "bankQuestions"];
QUESTION_PICKERS.forEach((id) =>
  $(id).addEventListener("change", () => {
    const select = $(id);
    const opt = select.selectedOptions[0];
    if (!opt || !opt.value) return;
    QUESTION_PICKERS.filter((other) => other !== id).forEach((other) => { $(other).value = ""; });
    $("question").value = opt.value.replace(/^\(as [^)]*\)\s*/, "");      // "(as customer …)" is a hint only
    if (opt.dataset.agentic) $("optAgentic").checked = true;                       // needs the agent
    if (opt.dataset.role && [...$("bankRole").options].some((o) => o.value === opt.dataset.role)) {
      $("bankRole").value = opt.dataset.role;                                       // demo: switch role
      if (opt.dataset.customer) $("bankCustomer").value = opt.dataset.customer;
      updateCustomerField();
    }
    lastDomain = select.dataset.domain;
    Object.entries(RUN_BUTTONS).forEach(([d, b]) => $(b).classList.toggle("suggested", d === lastDomain));
    $("question").focus();
  }));
// typing your own question clears the dropdown selection
$("question").addEventListener("input", () => QUESTION_PICKERS.forEach((id) => { $(id).value = ""; }));
$("bankRole").addEventListener("change", updateCustomerField);
$("bankCustomer").addEventListener("change", updateCustomerField);

$("geminiSelect").addEventListener("change", () => {                    // remember the model
  try { localStorage.setItem("geminiModel", $("geminiSelect").value); } catch {}
});
$("modelRefresh").addEventListener("click", loadModels);
document.querySelectorAll('input[name="runMode"]').forEach((r) => r.addEventListener("change", updateRunMode));

restoreTopK();
restoreRunMode();
loadModels();
pollStatus((s) => {
  ready = true;
  fillBankAccess(s);
  Object.values(RUN_BUTTONS).forEach((id) => { $(id).disabled = false; });
});
