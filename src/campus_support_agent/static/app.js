const entropyToggle = document.getElementById("entropyToggle");
const inputMode = document.getElementById("inputMode");
const textMode = document.getElementById("textMode");
const audioMode = document.getElementById("audioMode");
const statusLine = document.getElementById("statusLine");

function setStatus(message, isError = false) {
  statusLine.textContent = message;
  statusLine.style.color = isError ? "#b74f2c" : "#756759";
}

function parseContext() {
  const raw = document.getElementById("studentContext").value.trim();
  if (!raw) {
    return {};
  }
  return JSON.parse(raw);
}

function renderList(elementId, items, fallback = "暂无数据") {
  const target = document.getElementById(elementId);
  target.innerHTML = "";
  if (!items || items.length === 0) {
    const li = document.createElement("li");
    li.textContent = fallback;
    target.appendChild(li);
    return;
  }

  items.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    target.appendChild(li);
  });
}

function renderEntropyDimensions(dimensions) {
  const target = document.getElementById("entropyDimensions");
  target.innerHTML = "";
  if (!dimensions) {
    target.innerHTML = '<p class="muted">暂无熵维度。</p>';
    return;
  }

  const labels = {
    emotion_intensity: "情绪强度",
    emotional_volatility: "情绪波动",
    cognitive_load: "认知负荷",
    physiological_imbalance: "生理失衡",
    social_support_tension: "社会张力",
    risk_pressure: "风险压力",
  };

  Object.entries(dimensions).forEach(([key, value]) => {
    const card = document.createElement("div");
    card.className = "dimension-card";
    card.innerHTML = `<span>${labels[key] || key}</span><strong>${value}</strong>`;
    target.appendChild(card);
  });
}

function renderCampusResources(resources) {
  const target = document.getElementById("campusResources");
  target.innerHTML = "";
  if (!resources || resources.length === 0) {
    target.innerHTML = '<p class="muted">暂无资源命中。</p>';
    return;
  }

  resources.forEach((resource) => {
    const card = document.createElement("div");
    card.className = "resource-item";
    card.innerHTML = `
      <strong>${resource.title}</strong>
      <div class="resource-meta">${resource.category} · ${resource.relevance_reason}</div>
      <p>${resource.summary}</p>
    `;
    target.appendChild(card);
  });
}

function renderTrace(trace) {
  const target = document.getElementById("entropyTrace");
  target.innerHTML = "";
  if (!trace || trace.length === 0) {
    target.innerHTML = '<p class="muted">暂无会话轨迹。</p>';
    return;
  }

  trace.forEach((point, index) => {
    const item = document.createElement("div");
    item.className = "trace-item";
    item.innerHTML = `
      <strong>第 ${index + 1} 个熵点 · Score ${point.score}</strong>
      <div class="trace-meta">Level ${point.level} · ${point.balance_state}</div>
      <div>${(point.dominant_drivers || []).join(" / ")}</div>
    `;
    target.appendChild(item);
  });
}

function renderHistory(history) {
  const target = document.getElementById("conversationHistory");
  target.innerHTML = "";
  if (!history || history.length === 0) {
    target.innerHTML = '<p class="muted">暂无对话历史。</p>';
    return;
  }

  history.forEach((item) => {
    const block = document.createElement("div");
    block.className = "history-item";
    block.innerHTML = `
      <span class="history-role">${item.role}</span>
      <span>${item.content}</span>
    `;
    target.appendChild(block);
  });
}

function renderResponse(data) {
  document.getElementById("riskLevel").textContent = data.risk?.level || "-";
  document.getElementById("entropyScore").textContent = data.entropy?.score ?? "-";
  document.getElementById("balanceState").textContent = data.entropy?.balance_state || "-";

  const trend = data.entropy?.trend;
  const deltaText = trend && trend.delta !== null && trend.delta !== undefined
    ? `${trend.delta > 0 ? "+" : ""}${trend.delta} (${trend.direction})`
    : "-";
  document.getElementById("entropyDelta").textContent = deltaText;

  document.getElementById("supportSummary").textContent = data.plan?.summary || "暂无摘要。";
  document.getElementById("reductionRationale").textContent = data.entropy_reduction?.rationale || "暂无减熵策略。";
  renderList("reductionActions", data.entropy_reduction?.core_actions || []);
  renderList("immediateSupport", data.plan?.immediate_support || []);
  renderList("campusActions", data.plan?.campus_actions || []);
  renderCampusResources(data.campus_resources || []);
  renderEntropyDimensions(data.entropy?.dimensions || null);
  document.getElementById("rawJson").textContent = JSON.stringify(data, null, 2);
}

async function submitText() {
  try {
    setStatus("正在发送文本请求...");
    const payload = {
      session_id: document.getElementById("sessionId").value.trim(),
      text: document.getElementById("messageInput").value.trim(),
      student_context: parseContext(),
    };

    const response = await fetch("/api/v1/support/text", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "文本请求失败。");
    }
    renderResponse(data);
    setStatus("文本请求完成。");
  } catch (error) {
    setStatus(error.message || "文本请求失败。", true);
  }
}

async function submitAudio() {
  try {
    const file = document.getElementById("audioFile").files[0];
    if (!file) {
      throw new Error("请先选择音频文件。");
    }

    setStatus("正在上传音频...");
    const formData = new FormData();
    formData.append("file", file);
    formData.append("session_id", document.getElementById("sessionId").value.trim());
    formData.append("student_context", JSON.stringify(parseContext()));

    const response = await fetch("/api/v1/support/audio", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "语音请求失败。");
    }
    renderResponse(data);
    setStatus("语音请求完成。");
  } catch (error) {
    setStatus(error.message || "语音请求失败。", true);
  }
}

async function loadSession() {
  try {
    const sessionId = document.getElementById("sessionId").value.trim();
    if (!sessionId) {
      throw new Error("请先填写 session_id。");
    }

    setStatus("正在读取会话历史...");
    const response = await fetch(`/api/v1/sessions/${encodeURIComponent(sessionId)}`);
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "读取会话历史失败。");
    }
    renderHistory(data.conversation_history || []);
    renderTrace(data.entropy_trace || []);
    setStatus("会话历史已加载。");
  } catch (error) {
    setStatus(error.message || "读取会话历史失败。", true);
  }
}

async function clearSession() {
  try {
    const sessionId = document.getElementById("sessionId").value.trim();
    if (!sessionId) {
      throw new Error("请先填写 session_id。");
    }

    setStatus("正在清空会话...");
    const response = await fetch(`/api/v1/sessions/${encodeURIComponent(sessionId)}`, {
      method: "DELETE",
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "清空会话失败。");
    }
    renderHistory([]);
    renderTrace([]);
    setStatus(`会话 ${sessionId} 已清空。`);
  } catch (error) {
    setStatus(error.message || "清空会话失败。", true);
  }
}

function syncMode() {
  const mode = inputMode.value;
  textMode.classList.toggle("hidden", mode !== "text");
  audioMode.classList.toggle("hidden", mode !== "audio");
}

function syncEntropyVisibility() {
  document.body.classList.toggle("entropy-hidden", !entropyToggle.checked);
}

function clearOutput() {
  document.getElementById("riskLevel").textContent = "-";
  document.getElementById("entropyScore").textContent = "-";
  document.getElementById("balanceState").textContent = "-";
  document.getElementById("entropyDelta").textContent = "-";
  document.getElementById("supportSummary").textContent = "等待结果。";
  document.getElementById("reductionRationale").textContent = "等待结果。";
  renderList("reductionActions", [], "暂无数据");
  renderList("immediateSupport", [], "暂无数据");
  renderList("campusActions", [], "暂无数据");
  renderCampusResources([]);
  renderEntropyDimensions(null);
  renderHistory([]);
  renderTrace([]);
  document.getElementById("rawJson").textContent = "暂无数据";
  setStatus("结果区已清空。");
}

// 这里保持前端逻辑尽量轻，方便后续替换成正式学生端或管理端。
document.getElementById("submitText").addEventListener("click", submitText);
document.getElementById("submitAudio").addEventListener("click", submitAudio);
document.getElementById("loadSession").addEventListener("click", loadSession);
document.getElementById("clearSession").addEventListener("click", clearSession);
document.getElementById("clearOutput").addEventListener("click", clearOutput);
inputMode.addEventListener("change", syncMode);
entropyToggle.addEventListener("change", syncEntropyVisibility);

syncMode();
syncEntropyVisibility();
