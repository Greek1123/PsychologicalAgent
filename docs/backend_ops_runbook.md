# 后端运维与交付 Runbook

本文档用于组员接手、答辩复现和本地部署前检查。它只记录当前后端已经具备的可执行命令和受保护接口；本地生成的日志、报告、备份和导出包默认不提交到 GitHub。

## 1. 启动前检查

先确认依赖、模型、知识库、数据库目录、日志目录、CORS、管理员 API key 和数据保留配置：

```powershell
python scripts\check_deployment_readiness.py
```

服务启动后也可以读取同一类信息：

```powershell
curl http://127.0.0.1:8000/api/v1/ops/readiness
```

如果配置了 `ADMIN_API_KEY`，受保护接口需要带：

```powershell
curl -H "X-Admin-API-Key: <ADMIN_API_KEY>" http://127.0.0.1:8000/api/v1/ops/data-governance
```

## 2. 本地启动

Mock 模式适合前后端联调和演示：

```powershell
set LLM_PROVIDER=mock
set STT_PROVIDER=mock
uvicorn campus_support_agent.main:app --app-dir src --host 127.0.0.1 --port 8000
```

本地 checkpoint 模式优先使用启动脚本，它会先执行 readiness 检查：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_local_checkpoint_api.ps1
```

打开粗略测试工作台：

```text
http://127.0.0.1:8000/app
```

## 3. 核心接口自检

前端契约：

```powershell
curl http://127.0.0.1:8000/api/v1/frontend/contract
```

文本支持入口：

```powershell
curl -X POST http://127.0.0.1:8000/api/v1/support/text -H "Content-Type: application/json" -d "{\"session_id\":\"smoke-001\",\"text\":\"我最近压力很大，睡不好。\"}"
```

处理层健康：

```powershell
curl -H "X-Admin-API-Key: <ADMIN_API_KEY>" http://127.0.0.1:8000/api/v1/analytics/processing-health
```

数据治理总览：

```powershell
curl -H "X-Admin-API-Key: <ADMIN_API_KEY>" http://127.0.0.1:8000/api/v1/ops/data-governance
```

## 4. 数据库维护流程

只读完整性检查：

```powershell
python scripts\check_deployment_readiness.py
curl -H "X-Admin-API-Key: <ADMIN_API_KEY>" http://127.0.0.1:8000/api/v1/ops/database-integrity
```

创建备份：

```powershell
python scripts\backup_sqlite_database.py --db data\campus_agent.db --label before-maintenance
```

预览过期数据清理，默认不删除：

```powershell
python scripts\cleanup_expired_data.py --db data\campus_agent.db
```

推荐的安全维护入口，默认 dry-run：

```powershell
python scripts\maintain_sqlite_database.py --db data\campus_agent.db
```

确认后执行“先备份再清理”：

```powershell
python scripts\maintain_sqlite_database.py --db data\campus_agent.db --apply --label before-cleanup
```

如果数据库完整性是 `watch`，维护脚本默认拒绝写入；确实要保留现场并继续时，显式加：

```powershell
python scripts\maintain_sqlite_database.py --db data\campus_agent.db --apply --allow-watch --label watch-cleanup
```

## 5. 人工干预工作台接口

查看 care queue：

```powershell
curl -H "X-Admin-API-Key: <ADMIN_API_KEY>" "http://127.0.0.1:8000/api/v1/analytics/care-queue?include_low_priority=true&role=counselor"
```

筛选“我的待办”：

```powershell
curl -H "X-Admin-API-Key: <ADMIN_API_KEY>" "http://127.0.0.1:8000/api/v1/analytics/care-queue?workflow_state=assigned&owner=counselor-001&role=counselor"
```

单个会话认领：

```powershell
curl -X POST -H "X-Admin-API-Key: <ADMIN_API_KEY>" -H "Content-Type: application/json" http://127.0.0.1:8000/api/v1/sessions/<SESSION_ID>/human-interventions/action -d "{\"action\":\"claim\",\"handler_id\":\"counselor-001\"}"
```

批量认领：

```powershell
curl -X POST -H "X-Admin-API-Key: <ADMIN_API_KEY>" -H "Content-Type: application/json" http://127.0.0.1:8000/api/v1/analytics/care-queue/actions/batch -d "{\"session_ids\":[\"s1\",\"s2\"],\"action\":\"claim\",\"handler_id\":\"counselor-001\"}"
```

查询审计事件：

```powershell
curl -H "X-Admin-API-Key: <ADMIN_API_KEY>" "http://127.0.0.1:8000/api/v1/ops/audit-events?limit=50"
```

## 6. 导出与交付材料

导出脱敏审计包：

```powershell
python scripts\export_redacted_audit_package.py --db data\campus_agent.db --label handoff
```

导出 care queue 快照：

```powershell
python scripts\export_care_queue_snapshot.py --db data\campus_agent.db --role counselor --include-low-priority
```

生成验收报告：

```powershell
python scripts\generate_acceptance_report.py --test-summary "384 passed"
```

运行处理层 live smoke：

```powershell
python scripts\smoke_processing_acceptance.py --base-url http://127.0.0.1:8000
```

## 7. GitHub 提交边界

应提交：

- `src/`
- `tests/`
- `scripts/` 中可复现的正式脚本
- `docs/` 中正式说明文档
- `README.md`
- `.env.example`
- `requirements.txt`

不应提交：

- `.env`
- `.venv/`
- `logs/`
- `reports/`
- `data/campus_agent.db`
- `data/backups/`
- `data/audit_exports/`
- `data/care_queue_exports/`
- `training/ms_swift/outputs/`

## 8. 常见问题

端口被占用时，换端口启动：

```powershell
uvicorn campus_support_agent.main:app --app-dir src --host 127.0.0.1 --port 8001
```

PowerShell 中 `$env:KEY='value'` 报错时，说明当前在 `cmd.exe`，使用：

```cmd
set LLM_PROVIDER=mock
set STT_PROVIDER=mock
```

生产环境 readiness blocked，优先检查：

- `ADMIN_API_KEY` 是否配置；
- `DATABASE_PATH` 父目录是否存在；
- `CAMPUS_KB_PATH` 是否存在；
- local checkpoint 与 base model 路径是否存在；
- `SESSION_DATA_RETENTION_DAYS` 和 `AUDIT_LOG_RETENTION_DAYS` 是否为正数。
