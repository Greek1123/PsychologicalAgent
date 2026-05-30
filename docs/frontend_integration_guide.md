# 前端联调交接说明

本文档面向前端组员。当前后端已经提供文本、语音、会话、角色视图、人工队列和部署自检接口；前端正式页面可以先按这里的最小契约接入，不需要理解后端内部的心理熵计算细节。

## 本地启动

Windows CMD：

```bat
cd /d D:\psychologicalAgent
set LLM_PROVIDER=mock
set STT_PROVIDER=mock
uvicorn campus_support_agent.main:app --app-dir src --host 127.0.0.1 --port 8000
```

PowerShell：

```powershell
cd D:\psychologicalAgent
$env:LLM_PROVIDER="mock"
$env:STT_PROVIDER="mock"
uvicorn campus_support_agent.main:app --app-dir src --host 127.0.0.1 --port 8000
```

如果 `8000` 被占用，换成 `8001`：

```powershell
uvicorn campus_support_agent.main:app --app-dir src --host 127.0.0.1 --port 8001
```

如果前端是 Vite/React 独立启动，例如 `http://127.0.0.1:5173`，后端默认已经允许常见本地前端端口。需要自定义时设置：

```powershell
$env:FRONTEND_ALLOWED_ORIGINS="http://127.0.0.1:5173,http://localhost:5173"
```

粗略测试页面：

```text
http://127.0.0.1:8000/app
```

接口文档：

```text
http://127.0.0.1:8000/docs
```

## 前端最小接入

学生对话页第一版只需要接一个接口：

```text
POST /api/v1/support/text
```

请求：

```json
{
  "session_id": "demo-student-001",
  "text": "明天早上考试，我现在完全睡不着，越想越慌。",
  "student_context": {
    "grade": "大二",
    "major": "计算机科学",
    "campus": "main"
  },
  "conversation_history": []
}
```

前端主聊天气泡只展示：

```text
reply_text
```

学生侧可以少量展示：

```text
safety.emergency_notice
safety.human_referral
entropy.balance_state
entropy_reduction.core_actions
campus_resources
```

不要在学生端默认展示：

```text
system_flags
hidden_clinical_goal
backend_reason
backend_actions
```

这些字段适合研究面板、咨询师面板或本地调试。

## 文本接口 fetch 示例

```js
async function sendSupportText(text, sessionId = "demo-student-001") {
  const response = await fetch("http://127.0.0.1:8000/api/v1/support/text", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId,
      text,
      student_context: {
        grade: "大二",
        campus: "main",
      },
      conversation_history: [],
    }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || `request failed: ${response.status}`);
  }

  const data = await response.json();
  return {
    replyText: data.reply_text,
    riskLevel: data.risk?.level,
    entropyScore: data.entropy?.score,
    balanceState: data.entropy?.balance_state,
    coreActions: data.entropy_reduction?.core_actions || [],
    emergencyNotice: data.safety?.emergency_notice,
    humanReferral: data.safety?.human_referral,
    raw: data,
  };
}
```

## 语音接口

语音接口用于上传音频文件，后端会先转写，再走同一套 Agent 支持链路。

```text
POST /api/v1/support/audio
Content-Type: multipart/form-data
```

表单字段：

```text
file: 音频文件，必填
session_id: 会话 ID，可选但建议传
student_context: JSON 字符串，可选
conversation_history: JSON 数组字符串，可选
```

示例：

```js
async function sendAudio(file, sessionId = "demo-student-001") {
  const form = new FormData();
  form.append("file", file);
  form.append("session_id", sessionId);
  form.append("student_context", JSON.stringify({ grade: "大二", campus: "main" }));

  const response = await fetch("http://127.0.0.1:8000/api/v1/support/audio", {
    method: "POST",
    body: form,
  });
  return await response.json();
}
```

当前本地演示默认 `STT_PROVIDER=mock`，所以语音转写主要用于流程联调；真实语音质量要等接入正式 STT 服务。

## 角色视图与隐私边界

前端如果要做学生端、咨询师端、研究端或管理端，优先使用后端角色视图，而不是自己从完整响应里删字段。

```text
GET /api/v1/sessions/{session_id}/view?role=student
GET /api/v1/sessions/{session_id}/view?role=counselor
GET /api/v1/sessions/{session_id}/view?role=research
GET /api/v1/sessions/{session_id}/view?role=admin
```

建议：

- `student`：学生端页面。
- `counselor`：咨询师/辅导员工作台。
- `research`：研究统计面板，隐藏自由文本。
- `admin`：本地开发和审计，不给普通学生端使用。

## 人工队列

咨询师或辅导员端可以读取待关注队列：

```text
GET /api/v1/analytics/care-queue?include_low_priority=true&include_resolved=true
```

当人工已经查看或处理时，调用：

```text
POST /api/v1/sessions/{session_id}/human-interventions
```

请求：

```json
{
  "response_id": "support_xxx",
  "status": "acknowledged",
  "handler_id": "counselor-001",
  "note": "已查看高优先级队列，准备线下跟进。",
  "next_action": "same_day_review",
  "tags": ["manual_followup"]
}
```

允许的 `status`：

```text
acknowledged
in_progress
escalated
resolved
closed
```

## 自动联调检查

后端启动后，可以运行：

```powershell
python scripts\smoke_frontend_handoff.py --base-url http://127.0.0.1:8000
```

脚本会检查：

- `/health`
- `/api/v1/frontend/contract`
- `/api/v1/ops/readiness`
- `POST /api/v1/support/text`
- `GET /api/v1/sessions/{session_id}/view?role=student`
- `GET /api/v1/analytics/care-queue`
- contract 中的 CORS origin 配置

报告会写入：

```text
reports/frontend_handoff_smoke/
```

## 前端展示优先级

第一版学生端：

1. 输入框、发送按钮、会话气泡。
2. 主回复展示 `reply_text`。
3. 高危时额外展示 `safety.emergency_notice` 和 `safety.human_referral`。
4. 可折叠面板展示 `entropy_reduction.core_actions` 和 `campus_resources`。

第二版工作台：

1. 风险徽标：`risk.level`。
2. 心理熵趋势：`entropy.score`、`entropy.balance_state`、`entropy.trend`。
3. 干预状态：`intervention_strategy`、`dynamic_adjustment`、`referral_decision`。
4. 人工队列和人工处理记录。

## 当前模型说明

本地接口默认可以用 mock 模式跑完整流程：

```text
LLM_PROVIDER=mock
STT_PROVIDER=mock
```

需要给组员说明的是：前端不直接加载模型文件，只调用后端 API。模型路径和 provider 配置由后端环境变量控制；前端只需要知道当前后端地址和接口字段。
