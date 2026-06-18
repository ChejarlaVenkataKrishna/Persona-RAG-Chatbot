// ---------- tabs ----------
const tabBtns = document.querySelectorAll(".tab-btn");
const panels = document.querySelectorAll(".tab-panel");
tabBtns.forEach(btn => {
  btn.addEventListener("click", () => {
    tabBtns.forEach(b => b.classList.remove("active"));
    panels.forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    const panel = document.getElementById("tab-" + btn.dataset.tab);
    panel.classList.add("active");
    if (btn.dataset.tab === "topics" && !panel.dataset.loaded) {
      loadTopics();
      panel.dataset.loaded = "1";
    }
  });
});

// ---------- persona rendering ----------
const persona = JSON.parse(document.getElementById("persona-data").textContent);

function renderCommStyle() {
  const el = document.getElementById("commStyle");
  const s = persona.communication_style || {};
  const rows = [
    ["Avg message length", `${s.avg_message_length_words ?? "—"} words (${s.message_length_style ?? "—"})`],
    ["Tone", s.tone ?? "—"],
    ["Exclamation marks", `${s.exclamation_marks_per_100_msgs ?? 0} / 100 msgs`],
    ["Question marks", `${s.question_marks_per_100_msgs ?? 0} / 100 msgs`],
    ["Emoji usage", `${s.emoji_usage_per_100_msgs ?? 0} / 100 msgs`],
    ["Casual/slang terms", `${s.slang_casual_terms_per_100_msgs ?? 0} / 100 msgs`],
  ];
  el.innerHTML = rows.map(([k, v]) => `<div class="kv-row"><span class="k">${k}</span><span class="v">${v}</span></div>`).join("");
}

function renderTraits() {
  const el = document.getElementById("traitsList");
  const traits = persona.personality_traits || [];
  if (!traits.length) { el.innerHTML = '<p class="muted">No strong trait signals detected.</p>'; return; }
  el.innerHTML = traits.map(t =>
    `<div class="chip">${t.trait}<span class="count">${t.signal_count}</span></div>`
  ).join("");
}

function renderFactGroups(containerId, dataObj) {
  const el = document.getElementById(containerId);
  const keys = Object.keys(dataObj || {});
  if (!keys.length) { el.innerHTML = '<p class="muted">No signals detected in the analyzed conversations.</p>'; return; }
  el.innerHTML = keys.map(key => {
    const items = dataObj[key].slice(0, 5);
    const itemsHtml = items.map(it => `
      <div class="fact-item">
        ${escapeHtml(it.value)} ${it.count > 1 ? `<span class="muted">×${it.count}</span>` : ""}
        <span class="quote">"${escapeHtml(it.example_quote)}"</span>
      </div>
    `).join("");
    return `<div class="fact-group"><h4>${key.replace(/_/g, " ")}</h4>${itemsHtml}</div>`;
  }).join("");
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str ?? "";
  return div.innerHTML;
}

renderCommStyle();
renderTraits();
renderFactGroups("habitsList", persona.habits);
renderFactGroups("factsList", persona.personal_facts);

// ---------- topics ----------
async function loadTopics() {
  const el = document.getElementById("topicsList");
  el.innerHTML = '<p class="muted">Loading…</p>';
  const res = await fetch("/api/topics");
  const topics = await res.json();
  el.innerHTML = topics.map(t => `
    <div class="topic-row">
      <div class="topic-id">#${t.topic_id}</div>
      <div class="topic-range">msgs ${t.start}–${t.end}<br>(${t.num_messages})</div>
      <div class="topic-summary">${escapeHtml(t.summary)}</div>
    </div>
  `).join("");
}

// ---------- chat ----------
const chatForm = document.getElementById("chatForm");
const chatInput = document.getElementById("chatInput");
const chatLog = document.getElementById("chatLog");
const sourcesPanel = document.getElementById("sourcesPanel");

function appendMessage(role, html) {
  const wrap = document.createElement("div");
  wrap.className = `msg ${role}`;
  wrap.innerHTML = `<div class="bubble">${html}</div>`;
  chatLog.appendChild(wrap);
  chatLog.scrollTop = chatLog.scrollHeight;
  return wrap;
}

function renderSources(sources, mode) {
  if (!sources) { sourcesPanel.innerHTML = '<h3>Retrieved sources</h3><p class="muted">No sources for this answer.</p>'; return; }
  let html = "<h3>Retrieved sources</h3>";
  (sources.topics || []).forEach(t => {
    html += `<div class="source-item"><span class="src-label">topic checkpoint · score ${t.score.toFixed(2)}</span>${escapeHtml(t.summary)}</div>`;
  });
  (sources.chunks || []).forEach(c => {
    html += `<div class="source-item"><span class="src-label">message chunk · score ${c.score.toFixed(2)}</span>${escapeHtml(c.text.slice(0, 180))}…</div>`;
  });
  (sources.checkpoints || []).forEach(c => {
    html += `<div class="source-item"><span class="src-label">100-msg checkpoint · score ${c.score.toFixed(2)}</span>${escapeHtml(c.summary)}</div>`;
  });
  if (html === "<h3>Retrieved sources</h3>") html += '<p class="muted">Nothing matched strongly for this query.</p>';
  sourcesPanel.innerHTML = html;
}

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const query = chatInput.value.trim();
  if (!query) return;
  appendMessage("user", escapeHtml(query));
  chatInput.value = "";
  const thinking = appendMessage("bot", "…thinking…");

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });
    const data = await res.json();
    thinking.querySelector(".bubble").innerHTML = escapeHtml(data.answer).replace(/\n/g, "<br>");
    const tagRow = document.createElement("div");
    tagRow.className = "tag-row";
    tagRow.innerHTML = `<span class="tag">${data.mode}</span><span class="tag">${data.method}</span>`;
    thinking.querySelector(".bubble").appendChild(tagRow);
    renderSources(data.sources, data.mode);
  } catch (err) {
    thinking.querySelector(".bubble").textContent = "Something went wrong reaching the chatbot API.";
  }
});
