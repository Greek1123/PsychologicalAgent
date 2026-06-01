import { useMemo, useState } from "react";
import { CampusSupportApi, CampusSupportApiError } from "./campusSupportApi";

const DEFAULT_SESSION_ID = "demo-student-001";
const DEFAULT_PROMPTS = [
  "明天早上考试，我现在完全睡不着，越想越慌。",
  "室友每天晚上外放视频，我提醒过一次，她好像不太高兴，我现在也不敢说了。",
  "我好难受，我想去天台冷静一下",
];

function getApiBaseUrl() {
  return import.meta.env.VITE_CAMPUS_AGENT_API_BASE_URL || "http://127.0.0.1:8000";
}

function formatApiError(error) {
  if (error instanceof CampusSupportApiError) {
    return typeof error.detail === "string" ? error.detail : `后端请求失败：${error.status}`;
  }
  return error instanceof Error ? error.message : "请求失败，请稍后重试。";
}

export default function StudentChatExample() {
  const api = useMemo(() => new CampusSupportApi(getApiBaseUrl()), []);
  const [sessionId, setSessionId] = useState(DEFAULT_SESSION_ID);
  const [text, setText] = useState("明天早上考试，我现在完全睡不着，越想越慌。");
  const [messages, setMessages] = useState([]);
  const [latestDisplay, setLatestDisplay] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function sendText() {
    const cleanText = text.trim();
    if (!cleanText || loading) return;

    setLoading(true);
    setError("");
    setMessages((current) => [...current, { role: "user", content: cleanText }]);

    try {
      const response = await api.sendText({
        session_id: sessionId.trim() || DEFAULT_SESSION_ID,
        text: cleanText,
        student_context: {
          grade: "大二",
          campus: "main",
        },
      });
      const display = api.toStudentDisplayModel(response);
      setLatestDisplay(display);
      setMessages((current) => [...current, { role: "assistant", content: display.replyText }]);
      setText("");
    } catch (requestError) {
      setError(formatApiError(requestError));
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(event) {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
      sendText();
    }
  }

  return (
    <main className="student-chat-page">
      <section className="chat-panel">
        <header className="chat-header">
          <h1>校园心理支持 Agent</h1>
          <label>
            <span>Session</span>
            <input value={sessionId} onChange={(event) => setSessionId(event.target.value)} />
          </label>
        </header>

        <div className="prompt-row">
          {DEFAULT_PROMPTS.map((prompt) => (
            <button key={prompt} type="button" onClick={() => setText(prompt)}>
              {prompt.length > 14 ? `${prompt.slice(0, 14)}...` : prompt}
            </button>
          ))}
        </div>

        <div className="message-list" aria-live="polite">
          {messages.length === 0 ? (
            <p className="empty-state">发送一条消息后，这里会显示对话。</p>
          ) : (
            messages.map((message, index) => (
              <article key={`${message.role}-${index}`} className={`message ${message.role}`}>
                <strong>{message.role === "user" ? "学生" : "Agent"}</strong>
                <p>{message.content}</p>
              </article>
            ))
          )}
        </div>

        {latestDisplay?.emergencyNotice ? (
          <aside className="safety-card urgent">
            <strong>安全提示</strong>
            <p>{latestDisplay.emergencyNotice}</p>
          </aside>
        ) : null}

        {latestDisplay?.humanReferral ? (
          <aside className="safety-card">
            <strong>人工支持</strong>
            <p>{latestDisplay.humanReferral}</p>
          </aside>
        ) : null}

        {latestDisplay?.coreActions.length ? (
          <section className="support-actions">
            <h2>可以先做的事</h2>
            <ul>
              {latestDisplay.coreActions.map((action) => (
                <li key={action}>{action}</li>
              ))}
            </ul>
          </section>
        ) : null}

        {error ? <p className="error-line">{error}</p> : null}

        <label className="composer">
          <span>输入</span>
          <textarea value={text} onChange={(event) => setText(event.target.value)} onKeyDown={handleKeyDown} rows={4} />
        </label>
        <button className="send-button" type="button" disabled={loading || !text.trim()} onClick={sendText}>
          {loading ? "生成中..." : "发送"}
        </button>
      </section>

      <aside className="status-panel">
        <h2>当前状态</h2>
        <dl>
          <div>
            <dt>风险</dt>
            <dd>{latestDisplay?.riskLevel || "-"}</dd>
          </div>
          <div>
            <dt>心理熵</dt>
            <dd>{latestDisplay?.entropyScore ?? "-"}</dd>
          </div>
          <div>
            <dt>平衡状态</dt>
            <dd>{latestDisplay?.balanceState || "-"}</dd>
          </div>
        </dl>
      </aside>
    </main>
  );
}
