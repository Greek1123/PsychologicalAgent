const state = {
  latestResponse: null,
};

const $ = (id) => document.getElementById(id);

function setStatus(message, type = "info") {
  const target = $("statusLine");
  target.textContent = message;
  target.style.background = type === "error" ? "#ffe9e9" : "#edf4ff";
  target.style.color = type === "error" ? "#c73232" : "#1d4ed8";
}

function parseContext() {
  const raw = $("studentContext").value.trim();
  if (!raw) return {};
  return JSON.parse(raw);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function addMessage(role, text) {
  const log = $("chatLog");
  const empty = log.querySelector(".empty-state");
  if (empty) empty.remove();

  const item = document.createElement("div");
  item.className = `message ${role}`;
  item.innerHTML = `
    <div class="message-role">${role === "user" ? "学生" : "Agent"}</div>
    <div class="bubble">${escapeHtml(text)}</div>
  `;
  log.appendChild(item);
  log.scrollTop = log.scrollHeight;
}

function renderList(id, items, emptyText = "暂无数据") {
  const target = $(id);
  target.innerHTML = "";
  const values = Array.isArray(items) ? items : [];
  if (!values.length) {
    const li = document.createElement("li");
    li.textContent = emptyText;
    target.appendChild(li);
    return;
  }
  values.forEach((value) => {
    const li = document.createElement("li");
    li.textContent = value;
    target.appendChild(li);
  });
}

function riskTag(level) {
  const clean = level || "unknown";
  return `<span class="tag ${escapeHtml(clean)}">${escapeHtml(clean)}</span>`;
}

function renderResources(resources) {
  const target = $("campusResources");
  target.innerHTML = "";
  if (!resources?.length) {
    target.innerHTML = '<p class="muted">暂无校园资源推荐。</p>';
    return;
  }
  resources.forEach((resource) => {
    const item = document.createElement("div");
    item.className = "resource-item";
    item.innerHTML = `
      <strong>${escapeHtml(resource.title)}</strong>
      <div class="muted">${escapeHtml(resource.category)} · ${escapeHtml(resource.relevance_reason)}</div>
      <div>${escapeHtml(resource.summary)}</div>
    `;
    target.appendChild(item);
  });
}

function renderTrace(trace) {
  const target = $("entropyTrace");
  target.innerHTML = "";
  if (!trace?.length) {
    target.innerHTML = '<p class="muted">暂无会话轨迹。</p>';
    return;
  }
  trace.forEach((point, index) => {
    const item = document.createElement("div");
    item.className = "trace-item";
    item.innerHTML = `
      <strong>第 ${index + 1} 次 · 熵值 ${escapeHtml(point.score)}</strong>
      <div class="muted">${escapeHtml(point.balance_state)} · ${escapeHtml(point.created_at || "")}</div>
      <div>${escapeHtml((point.dominant_drivers || []).join(" / "))}</div>
    `;
    target.appendChild(item);
  });
}

function renderCareQueue(queue) {
  const target = $("careQueue");
  target.innerHTML = "";
  const items = queue?.items || [];
  if (!items.length) {
    target.innerHTML = '<p class="muted">当前没有队列条目。</p>';
    return;
  }
  items.forEach((item) => {
    const human = item.evidence?.human_intervention;
    const block = document.createElement("div");
    block.className = "queue-item";
    block.innerHTML = `
      <strong>${escapeHtml(item.session_id)} ${riskTag(item.priority)}</strong>
      <div>路线：${escapeHtml(item.route)}</div>
      <div>动作：${escapeHtml(item.recommended_action)}</div>
      <div class="muted">人工：${escapeHtml(human?.status || "-")} · 熵值：${escapeHtml(item.latest_entropy_score ?? "-")}</div>
    `;
    target.appendChild(block);
  });
}

function renderResponse(data) {
  state.latestResponse = data;
  const risk = data.risk || {};
  const entropy = data.entropy || {};
  const trend = entropy.trend || {};
  const strategy = data.intervention_strategy || {};
  const dynamic = data.dynamic_adjustment || {};
  const referral = data.referral_decision || {};
  const flags = data.system_flags || {};

  $("riskLevel").innerHTML = riskTag(risk.level);
  $("entropyScore").textContent = entropy.score ?? "-";
  $("balanceState").textContent = entropy.balance_state || "-";
  $("entropyDelta").textContent = trend.delta === null || trend.delta === undefined
    ? "-"
    : `${trend.delta > 0 ? "+" : ""}${trend.delta} ${trend.direction || ""}`;

  $("reductionRationale").textContent = data.entropy_reduction?.rationale || "暂无策略说明。";
  renderList("reductionActions", data.entropy_reduction?.core_actions || []);
  $("strategyId").textContent = strategy.strategy_id || "-";
  $("dynamicAction").textContent = dynamic.action || "-";
  $("referralState").textContent = `${referral.should_refer ? "yes" : "no"} / ${referral.urgency || "none"}`;
  $("systemFlags").textContent = flags.manual_referral_recommended ? `yes: ${(flags.reasons || []).join(", ")}` : "no";
  renderResources(data.campus_resources || []);
  $("rawJson").textContent = JSON.stringify(data, null, 2);
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || `请求失败：${response.status}`);
  }
  return data;
}

async function submitText() {
  try {
    const text = $("messageInput").value.trim();
    if (!text) throw new Error("请输入文本。");
    setStatus("生成中...");
    addMessage("user", text);
    const data = await requestJson("/api/v1/support/text", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: $("sessionId").value.trim(),
        text,
        student_context: parseContext(),
      }),
    });
    addMessage("assistant", data.reply_text || "暂无回复。");
    renderResponse(data);
    await loadSession(false);
    await loadRoleView(false);
    setStatus("已更新");
  } catch (error) {
    setStatus(error.message, "error");
  }
}

async function submitAudio() {
  try {
    const file = $("audioFile").files[0];
    if (!file) throw new Error("请选择音频文件。");
    setStatus("处理音频中...");
    const form = new FormData();
    form.append("file", file);
    form.append("session_id", $("sessionId").value.trim());
    form.append("student_context", JSON.stringify(parseContext()));
    const data = await requestJson("/api/v1/support/audio", { method: "POST", body: form });
    addMessage("user", data.transcript || `[音频] ${file.name}`);
    addMessage("assistant", data.reply_text || "暂无回复。");
    renderResponse(data);
    await loadSession(false);
    setStatus("语音已更新");
  } catch (error) {
    setStatus(error.message, "error");
  }
}

async function loadSession(showStatus = true) {
  try {
    const sessionId = $("sessionId").value.trim();
    if (!sessionId) throw new Error("请输入 session_id。");
    const data = await requestJson(`/api/v1/sessions/${encodeURIComponent(sessionId)}`);
    renderTrace(data.entropy_trace || []);
    if (showStatus) setStatus("历史已刷新");
  } catch (error) {
    setStatus(error.message, "error");
  }
}

async function loadRoleView(showStatus = true) {
  try {
    const sessionId = $("sessionId").value.trim();
    const role = $("roleSelect").value;
    if (!sessionId) throw new Error("请输入 session_id。");
    const data = await requestJson(`/api/v1/sessions/${encodeURIComponent(sessionId)}/view?role=${encodeURIComponent(role)}`);
    $("roleViewJson").textContent = JSON.stringify(data, null, 2);
    if (showStatus) setStatus(`已加载 ${role} 视图`);
  } catch (error) {
    setStatus(error.message, "error");
  }
}

async function clearSession() {
  try {
    const sessionId = $("sessionId").value.trim();
    if (!sessionId) throw new Error("请输入 session_id。");
    await requestJson(`/api/v1/sessions/${encodeURIComponent(sessionId)}`, { method: "DELETE" });
    $("chatLog").innerHTML = '<div class="empty-state">会话已清空。</div>';
    renderTrace([]);
    $("roleViewJson").textContent = "暂无数据";
    setStatus("会话已清空");
  } catch (error) {
    setStatus(error.message, "error");
  }
}

async function markHumanIntervention() {
  try {
    const sessionId = $("sessionId").value.trim();
    if (!sessionId) throw new Error("请输入 session_id。");
    const responseId = state.latestResponse?.response_id || "";
    const data = await requestJson(`/api/v1/sessions/${encodeURIComponent(sessionId)}/human-interventions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        response_id: responseId,
        status: "acknowledged",
        handler_id: "demo-counselor",
        note: "前端粗略工作台人工确认。",
        next_action: "same_day_review",
        tags: ["frontend_demo"],
      }),
    });
    $("roleViewJson").textContent = JSON.stringify(data, null, 2);
    await loadCareQueue(false);
    setStatus("已添加人工确认");
  } catch (error) {
    setStatus(error.message, "error");
  }
}

async function loadCareQueue(showStatus = true) {
  try {
    const data = await requestJson("/api/v1/analytics/care-queue?include_low_priority=true&include_resolved=true");
    renderCareQueue(data);
    if (showStatus) setStatus("队列已刷新");
  } catch (error) {
    setStatus(error.message, "error");
  }
}

async function refreshReadiness() {
  try {
    const data = await requestJson("/api/v1/ops/readiness");
    $("readinessJson").textContent = JSON.stringify(data, null, 2);
    setStatus(`自检：${data.status}`);
  } catch (error) {
    setStatus(error.message, "error");
  }
}

async function refreshContract() {
  try {
    const data = await requestJson("/api/v1/frontend/contract");
    $("roleViewJson").textContent = JSON.stringify(data, null, 2);
    setStatus("契约已加载");
  } catch (error) {
    setStatus(error.message, "error");
  }
}

function bindEvents() {
  $("submitText").addEventListener("click", submitText);
  $("submitAudio").addEventListener("click", submitAudio);
  $("loadSession").addEventListener("click", () => loadSession(true));
  $("loadRoleView").addEventListener("click", () => loadRoleView(true));
  $("clearSession").addEventListener("click", clearSession);
  $("markHuman").addEventListener("click", markHumanIntervention);
  $("loadCareQueue").addEventListener("click", () => loadCareQueue(true));
  $("refreshReadiness").addEventListener("click", refreshReadiness);
  $("refreshContract").addEventListener("click", refreshContract);
  $("roleSelect").addEventListener("change", () => loadRoleView(true));
  document.querySelectorAll("[data-prompt]").forEach((button) => {
    button.addEventListener("click", () => {
      $("messageInput").value = button.dataset.prompt;
    });
  });
}

bindEvents();
refreshReadiness();
loadCareQueue(false);
