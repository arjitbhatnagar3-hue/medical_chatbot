// Chat logic for MediBot. Served same-origin by FastAPI, so API_BASE is relative.
const API_BASE = "";
const STORAGE_KEY = "medibot_history";

const messagesEl = document.getElementById("messages");
const form = document.getElementById("chat-form");
const input = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");
const banner = document.getElementById("banner");
const statusDot = document.getElementById("status-dot");
const statusText = document.getElementById("status-text");

const SUGGESTIONS = [
  "What are the common symptoms of dengue fever?",
  "How is type 2 diabetes usually diagnosed?",
  "What are the side effects of aspirin?",
  "How should minor burns be treated at home?"
];

function nowTime() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function showWelcome() {
  const wrap = document.createElement("div");
  wrap.className = "welcome";
  wrap.id = "welcome";
  wrap.innerHTML = `
    <div class="big-logo">
      <svg viewBox="0 0 24 24" width="36" height="36" fill="none" stroke="#0a1428" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12h4l2 5 4-12 2 7h6"/></svg>
    </div>
    <h2>Ask your medical book</h2>
    <p>Get clear answers with page references. Try one of these to start:</p>
    <div class="chips">
      ${SUGGESTIONS.map(s => `<button class="suggest">${s}</button>`).join("")}
    </div>`;
  messagesEl.appendChild(wrap);
  wrap.querySelectorAll(".suggest").forEach(btn => {
    btn.addEventListener("click", () => { input.value = btn.textContent; form.requestSubmit(); });
  });
}

function removeWelcome() {
  const w = document.getElementById("welcome");
  if (w) w.remove();
}

function avatar(role) {
  if (role === "user") return `<div class="avatar user">🧑</div>`;
  return `<div class="avatar bot">
    <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="#0a1428" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12h4l2 5 4-12 2 7h6"/></svg></div>`;
}

function addMessage(role, text, sources, save = true) {
  removeWelcome();
  const wrap = document.createElement("div");
  wrap.className = "msg " + role;
  wrap.innerHTML = avatar(role) + `<div class="col"></div>`;
  const col = wrap.querySelector(".col");

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  col.appendChild(bubble);

  const time = document.createElement("div");
  time.className = "time";
  time.textContent = nowTime();
  col.appendChild(time);

  if (role === "bot" && sources && sources.length) {
    const cites = document.createElement("div");
    cites.className = "cites";
    sources.forEach(s => {
      const chip = document.createElement("span");
      chip.className = "chip";
      const page = (s.page !== undefined && s.page !== null && s.page !== "unknown") ? " · p." + s.page : "";
      chip.textContent = "📄 " + (s.source || "unknown") + page;
      cites.appendChild(chip);
    });
    col.appendChild(cites);
  }

  messagesEl.appendChild(wrap);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  if (save) saveHistory();
}

function showTyping() {
  const wrap = document.createElement("div");
  wrap.className = "msg bot";
  wrap.id = "typing";
  wrap.innerHTML = avatar("bot") + `<div class="col"><div class="bubble typing"><span></span><span></span><span></span></div></div>`;
  messagesEl.appendChild(wrap);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}
function removeTyping() { const t = document.getElementById("typing"); if (t) t.remove(); }

function saveHistory() {
  const items = [...messagesEl.querySelectorAll(".msg")].map(m => {
    const bubble = m.querySelector(".bubble");
    const isUser = m.classList.contains("user");
    const sources = [...m.querySelectorAll(".chip")].map(c => ({ source: c.textContent }));
    return { role: isUser ? "user" : "bot", text: bubble ? bubble.textContent : "", sources };
  }).filter(i => i.text && i.text.indexOf("typing") === -1);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(items));
}

function restoreHistory() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) { showWelcome(); return; }
    const items = JSON.parse(raw);
    if (!items.length) { showWelcome(); return; }
    items.forEach(i => addMessage(i.role, i.text, i.sources || [], false));
  } catch (_) { showWelcome(); }
}

function showBanner(msg) { banner.textContent = msg; banner.classList.add("show"); }
function hideBanner() { banner.classList.remove("show"); }

async function checkHealth() {
  try {
    const res = await fetch(API_BASE + "/health");
    const data = await res.json();
    if (data.status === "ok") {
      statusDot.classList.add("online");
      statusText.textContent = "online";
      hideBanner();
    } else {
      statusDot.classList.remove("online");
      statusText.textContent = "error";
      showBanner("Backend issue: " + data.detail);
    }
  } catch (_) {
    statusDot.classList.remove("online");
    statusText.textContent = "offline";
    showBanner("Cannot reach the backend at " + (API_BASE || location.origin) + "/health");
  }
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = input.value.trim();
  if (!q) return;

  addMessage("user", q);
  input.value = "";
  autoGrow();
  input.disabled = true;
  sendBtn.disabled = true;
  showTyping();

  try {
    const res = await fetch(API_BASE + "/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q })
    });
    removeTyping();
    if (!res.ok) {
      let detail = res.statusText;
      try { detail = (await res.json()).detail || detail; } catch (_) {}
      addMessage("bot", "⚠️ " + detail);
    } else {
      const data = await res.json();
      addMessage("bot", data.answer, data.sources || []);
    }
  } catch (err) {
    removeTyping();
    addMessage("bot", "⚠️ Network error — is the backend running?\n(" + err.message + ")");
  } finally {
    input.disabled = false;
    sendBtn.disabled = false;
    input.focus();
  }
});

function autoGrow() {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 140) + "px";
}
input.addEventListener("input", autoGrow);
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); }
});

document.getElementById("new-chat").addEventListener("click", () => {
  localStorage.removeItem(STORAGE_KEY);
  messagesEl.innerHTML = "";
  showWelcome();
  input.focus();
});

// Init
restoreHistory();
checkHealth();
setInterval(checkHealth, 30000);
