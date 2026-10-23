"use strict";
// 10-K Copilot web page. All model output is escaped before display; links are only
// shown when they point to sec.gov; an access key, if the server requires one, is kept
// for this browser tab only (sessionStorage).

const SOURCE_NAMES = { vector: "report text", cypher: "the knowledge graph", sql: "financial data" };
const EXAMPLES = [
  "Which companies name AMD as a competitor, and what was each one's net income in its latest fiscal year?",
  "How does NVIDIA describe the impact of export controls?",
  "Which company had the highest operating margin in its latest fiscal year?",
  "In which countries does AMD have subsidiaries?"];
const CITATION = /\[([A-Z]{2,5}_\d{4}_[0-9A-Z]+_\d{4}|SQL|GRAPH)\]/g;
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"']/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
const listWords = a => a.length < 2 ? a.join("") : a.slice(0, -1).join(", ") + " and " + a[a.length - 1];
const safeUrl = u => (typeof u === "string" && u.startsWith("https://www.sec.gov/")) ? u : null;

function headers() {
  const h = {"Content-Type": "application/json"};
  const key = sessionStorage.getItem("copilotKey");
  if (key) h["X-API-Key"] = key;
  return h;
}

async function call(url, options = {}) {
  let r = await fetch(url, {...options, headers: headers()});
  if (r.status === 401) {                           // the server requires an access key
    const key = window.prompt("This copilot needs an access key:");
    if (!key) throw new Error("An access key is required");
    sessionStorage.setItem("copilotKey", key.trim());
    r = await fetch(url, {...options, headers: headers()});
  }
  let body = {};
  try { body = await r.json(); } catch { /* keep empty */ }
  if (r.status === 429) throw new Error(body.detail || "Too many questions right now. Wait a minute and ask again");
  if (!r.ok) throw new Error(typeof body.detail === "string" ? body.detail : "The question could not be processed");
  return body;
}

document.addEventListener("DOMContentLoaded", () => {
  EXAMPLES.forEach(t => {
    const b = document.createElement("button");
    b.type = "button"; b.textContent = t;
    b.addEventListener("click", () => { $("q").value = t; ask(); });
    $("suggestions").appendChild(b);
  });
  const q = $("q");
  q.addEventListener("input", () => {
    q.style.height = "auto"; q.style.height = q.scrollHeight + "px";
    $("send").disabled = q.value.trim().length < 3;
  });
  q.addEventListener("keydown", e => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); if (!$("send").disabled) ask(); }
  });
  $("form").addEventListener("submit", e => { e.preventDefault(); ask(); });
  $("close").addEventListener("click", closePanel);
  document.addEventListener("keydown", e => { if (e.key === "Escape") closePanel(); });
});

async function ask() {
  const q = $("q"), question = q.value.trim();
  if (question.length < 3) return;
  $("welcome").hidden = true;
  q.value = ""; q.style.height = "auto"; $("send").disabled = true; q.disabled = true;

  const turn = document.createElement("div");
  turn.className = "turn";
  turn.innerHTML = `<div class="question">${esc(question)}</div>
    <div class="reply pending"><svg class="mark"><use href="#mark"/></svg>
    <div><div class="skeleton"></div><div class="skeleton"></div><div class="skeleton"></div></div></div>`;
  $("chat").appendChild(turn);
  turn.scrollIntoView({behavior: "smooth", block: "start"});
  const reply = turn.querySelector(".reply"), slot = reply.lastElementChild;

  try {
    const d = await call("/ask", {method: "POST", body: JSON.stringify({question})});
    slot.innerHTML = render(d);
    slot.querySelectorAll("[data-source]").forEach(el =>
      el.addEventListener("click", () => openSource(el.dataset.source, d)));
  } catch (err) {
    slot.innerHTML = `<div class="error">No answer: ${esc(err.message)}.</div>`;
  }
  reply.classList.remove("pending");
  q.disabled = false; q.focus();
}

function render(d) {
  const order = (d.citations || []).map(c => c.id);
  const number = id => order.indexOf(id) + 1;
  let text = esc(d.answer).replace(CITATION, (m, id) =>
    number(id) ? `<button class="chip" data-source="${esc(id)}" aria-label="Source ${number(id)}">${number(id)}</button>` : m);
  text = text.replace(/\(unverified\)/g, '<span class="unverified">(unverified source)</span>');
  const sources = (d.citations || []).map((c, i) => {
    const url = safeUrl(c.url);
    const label = url ? `<a href="${esc(url)}" target="_blank" rel="noopener noreferrer">${esc(c.label)}</a>` : esc(c.label);
    return `<li><span>${i + 1}</span><span>${label}</span></li>`;
  }).join("");
  const used = listWords((d.route || []).map(t => SOURCE_NAMES[t] || "another source"));
  return `<div class="answer">${text}</div>
    ${sources ? `<ol class="sources">${sources}</ol>` : ""}
    <div class="how">Answered in ${Number(d.seconds).toFixed(1)} s from ${esc(used)}.</div>`;
}

async function openSource(id, d) {
  const c = (d.citations || []).find(x => x.id === id);
  $("panel-title").textContent = c ? c.label : id;
  const body = $("panel-body");
  if (id === "SQL" || id === "GRAPH") {
    body.textContent = id === "SQL"
      ? "Figures come from the companies' XBRL financial data filed with the SEC, queried with SQL."
      : "Relationships come from the knowledge graph, built from the 10-K text and each report's list of subsidiaries.";
  } else {
    body.textContent = "Loading the passage…";
    try {
      const p = await call("/sources/" + encodeURIComponent(id));
      body.textContent = p.text;
      const url = safeUrl(p.url);
      if (url) {
        const a = document.createElement("a");
        a.href = url; a.target = "_blank"; a.rel = "noopener noreferrer"; a.textContent = "Open the full report on sec.gov";
        body.appendChild(document.createElement("br")); body.appendChild(a);
      }
    } catch (err) { body.textContent = "The passage could not be loaded: " + err.message + "."; }
  }
  $("panel").classList.add("open"); $("panel").setAttribute("aria-hidden", "false"); $("close").focus();
}

function closePanel() {
  $("panel").classList.remove("open"); $("panel").setAttribute("aria-hidden", "true");
  $("q").focus();
}
