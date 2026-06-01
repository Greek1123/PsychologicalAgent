# Frontend Handoff

这个目录给前端组员复制使用，不属于后端运行时依赖。

## 文件

- `campusSupportApi.ts`：TypeScript API client，封装后端文本、语音、角色视图、care queue 和人工处理记录。
- `StudentChatExample.jsx`：React 学生端聊天页示例，展示如何调用 client、维护消息列表、显示高危安全提示和熵减行动。

## 推荐复制方式

在 Vite/React 项目中：

```text
src/api/campusSupportApi.ts
src/pages/StudentChatExample.jsx
```

然后在 `.env.local` 中配置：

```text
VITE_CAMPUS_AGENT_API_BASE_URL=http://127.0.0.1:8000
```

后端需要允许前端端口：

```powershell
$env:FRONTEND_ALLOWED_ORIGINS="http://127.0.0.1:5173,http://localhost:5173"
```

## 学生端展示边界

学生端主界面优先展示：

- `reply_text`
- `safety.emergency_notice`
- `safety.human_referral`
- `entropy_reduction.core_actions`
- `campus_resources`

不要默认展示：

- `system_flags`
- `hidden_clinical_goal`
- `backend_reason`
- `backend_actions`

这些字段留给研究面板、咨询师面板或本地调试。
