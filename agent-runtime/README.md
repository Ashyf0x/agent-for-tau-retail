# Research Agent

面向 Agent 算法学习与实习申请的轻量 Python 单 Agent 项目。重点是可靠执行、公开评测与失败轨迹分析；通用核心不包含 Retail 业务逻辑，不提供完整 Multi-Agent 调度。

当前状态（2026-11-10 收尾）：官方 `retail_50` 子集的 **50 任务 × 3 trial = 150 trial** 全量评测已完成，150/150 达评分；pass^1 = 76.7%（115/150，Wilson 95% 69.3–82.7）、pass^3 = 50.0%（25/50）、23 个 flaky 任务，总调用 2937 次、成本 41.77 元。正式报告与失败归因见 [run-50x3.md](experiments/retail-qwen-tier-order-20261108/run-50x3.md)，所有真实调用已停止，不因失败自动补跑；无横向 baseline。

开发规则见 [根目录 AGENTS.md](../AGENTS.md)，当前设计见 [ARCHITECTURE.md](ARCHITECTURE.md)，目录导航见 [工作区 README](../README.md)。

## 阅读与目录

建议从 `app/cli.py` 的 `demo`、`core/agent.py` 的 `Agent.run` 开始，再看执行器、Context、Memory 和轨迹重建。

```text
agentcore/
  core/           Agent 主循环、类型、模型协议适配和错误策略
  context/        本轮输入、历史视图、摘要和观察引用
  memory/         本地候选提交/检索、可选有界批次提炼
  tools/          工具定义、统一执行器、官方 SDK stdio MCP 接入
  storage/        SQLite 会话、运行、事件、观察和记忆
  observability/  运行报告、只读轨迹重建与导出
  app/            离线 CLI 与受限真实冒烟入口
examples/         离线 Agent、Memory、MCP 示例
tests/            行为测试
evals/            固定官方 Retail 环境的薄适配与接线 fixture
```

`.vendor/` 是固定官方评测依赖；`.research-data/` 与 `.offline-validation/` 保留早期验证依据，不作为待清理缓存；`.python/` 是 `.venv313` 的解释器来源，不可删除。历史探索实验、旧日志与外层 `.archive/` 已清理，不参与任何结论。当前实验入口统一见 [experiments/README.md](experiments/README.md)。

## 安装与离线运行

核心要求 Python 3.12+；固定 Retail 环境要求 Python 3.12–3.13。项目现场解释器是 `.venv313/Scripts/python.exe`，命令中的 `python` 可替换为它。

```powershell
python -m pip install -e ".[mcp,smoke,dev]"
python -m agentcore
python -m agentcore --database demo.sqlite3
python examples/offline_demo.py
python examples/mcp_demo.py
python examples/memory_demo.py
```

默认 CLI 是 `app/cli.py` 中的脚本模型示例：跨会话提交中文偏好，再调用本地 add 计算 2 + 3；成功时输出计算答案及运行报告。它不是通用交互终端，不读取 `.env`、不调用真实模型。Memory 示例展示显式提炼、阈值跳过和触发去重；MCP 示例启动真实本地 stdio 子进程。需要保留 Memory 示例数据库时可传 `--database <新路径>`。

离线测试入口：

```powershell
python -m pytest -q tests/test_runtime.py tests/test_context_memory.py tests/test_storage_boundaries.py
python -m pytest -q tests/test_memory_extraction.py tests/test_trajectory.py tests/test_model_mcp.py tests/test_protocol_review.py
python -m pytest -q tests/test_smoke.py
python -m pytest -q
```

测试中的 `offline_guard` 禁止外部网络连接和读取真实 `.env`，本地 HTTP fixture 仅走 loopback。Retail 安装与单独验证见 [evals/README.md](evals/README.md)。回归结论以当前全量 `pytest -q` 输出为准，不引用历史批次或旧聊天中的通过数量。`tests/fixtures/legacy/` 只保存离线测试输入，不是实验结果。

smoke 地址配置现已实现 `.env` 的 `AGENT_BASE_URL` 解析，无硬编码默认地址；相关测试区分确定性 fixture 与用户本地真实配置检查。真实 `.env` 的生效值应通过下面的 `--check-config` 命令核对，自动化 fixture 通过不证明真实文件配置正确或线上服务可用。

## 配置与真实冒烟入口

普通 `HTTPModelClient` 由应用显式传入模型、地址、认证和 provider，不自动读取环境配置。当前 `app/smoke.py` 的有效配置如下：

| 配置 | 当前来源与覆盖规则 |
| --- | --- |
| 模型 | `MODEL = "qwen3.7-flash"` 应用常量；没有 `--model` 参数，不读取 `AGENT_MODEL` |
| 服务地址 | 显式 `--base-url` 优先，否则读取 `--env-file` 中的 `AGENT_BASE_URL`；缺失/空值直接报错，无源码或进程环境回退 |
| 认证文件 | `--env-file` 指定，默认相对于工作目录的 `.env` |
| 认证值 | 仅 `--execute` 校验计划后，由 `_load_key` 调用 `dotenv_values(..., interpolate=False)` 取 `DASHSCOPE_API_KEY`；无进程环境回退 |
| 协议 | `make_client` 固定 Qwen、thinking/preserve_thinking 开启、非流式、总输出上限 2048 |
| 次数与预算 | 最多 2 次请求；`--budget-yuan` 默认 0.05 元且只能降低；仅只读 add，重试/纠错/Memory 提炼关闭 |

模型保持用户当前别名，不切换日期快照。地址检查仍限制本 smoke 既有北京 HTTPS 服务范围，因价卡按北京冻结；不支持的地址明确报错，不自动换地址。`--check-config` 仅将地址赋值交给 dotenv 解析，关闭变量插值，不加载认证、不创建客户端、不发请求；检查成功只证明实际配置被解析且通过本地格式校验。

```powershell
# 真实本地配置检查：显示 .env 中实际 AGENT_BASE_URL，不加载认证、不发请求
python -m agentcore.app.smoke --check-config --env-file .env
# 只冻结计划：新目录，读取 .env 地址、不加载认证、不发请求
python -m agentcore.app.smoke --output .offline-validation/qwen-smoke-new --budget-yuan 0.05 --env-file .env
# 仅获得新的真实调用授权后执行；若显式指定 --base-url，冻结和执行须一致
python -m agentcore.app.smoke --output .offline-validation/qwen-smoke-new --budget-yuan 0.05 --execute --env-file .env
# 已启动实验只读检查，不能续跑
python -m agentcore.app.smoke --output .offline-validation/qwen-smoke-new --inspect
```

该预算保护仅用于此冒烟实验，不是 Retail Pilot 的现成预算控制。派发前预留、未知用量保留预留；异常停止后续派发。费用采用冻结价卡估算，字节代理、输出容差和供应商结算使其无法保证绝对账单上限。

## Retail 直接运行

Retail入口已移除手动冻结计划步骤。参数直接随`--execute`传入，实际配置和源码快照自动保存在新运行目录；模型、地址及认证继续按现有`.env`与显式参数规则读取。预算和原始证据保护保留。当前默认Context为容量保留B；原始A使用`--context-history recent --context-compaction legacy`，C使用`--context-history capacity --context-compaction task`。这次仅增加Context选项，不改可靠执行循环、模型、工具及官方评分。

失败原因投影默认关闭。`--failure-reason-chars 0`（默认）保持既有契约：失败工具的`value`被置空，模型只看到`error`/`category`/`decision`，拒绝原文不入模（`tests/test_retail.py`对此有断言）。设为`1..240`时额外投影一个有界摘录`failure_reason`：截断到指定长度、折叠空白、去控制字符，并标注不可信、非指令、不确认任何状态变更。`status="unknown"`、`category="arguments"`及结构化payload一律不摘录。该选项用于tau-bench Retail：否则业务拒绝原因（如`Insufficient gift card balance to pay for the price difference`）完全不可见，105因此编造原因并转人工。关闭时`config["context"]`与冻结快照逐字节一致。

```powershell
.\.venv313\Scripts\python.exe -m evals.retail_pilot --execute --task-id 6 `
  --agent-max-completion-tokens 9060 --max-request-bytes 131072 `
  --resource-profile relaxed --budget-yuan 3.00 --output ".offline-validation/retail-$(Get-Date -Format yyyyMMdd-HHmmss)"
```

此命令会产生真实调用，须另获当批范围和总预算授权。新候选资源档为每任务全角色80次、每Agent.run内部24步、每任务3元；历史40/12/1元用`--resource-profile legacy --budget-yuan 1.00`。这是项目执行选择，未验证最优；官方60步、输出及Context边界不变，仅已有请求状态`run_state.limit`随实际步数变化。默认费率仅沿用Flash历史保守估价，换模请显式核实费率。检查配置、离线接线及结果查看见[Retail运行说明](evals/README.md)。上文smoke是独立入口，不使用此Retail流程。

## 轨迹与已有证据

```powershell
python -m agentcore.observability.trajectory --database <数据库路径> --run-id <运行ID> --format markdown --output <新文件.md>
python -m agentcore.observability.trajectory --database <数据库路径> --run-id <运行ID> --format json --output <新文件.json>
```

可加 `--user-id` 限定用户；导出只读、不覆盖文件，保留事件、实际请求 JSON、解析响应、调用 ID、工具证据及真实终态。未记录信息明确列出。导出含原始对话和已记录的思考内容，不是脱敏报告。

| 证据 | 可以得出的结论 |
| --- | --- |
| [.offline-validation/qwen-smoke-user-run-20261105-181537/summary.json](.offline-validation/qwen-smoke-user-run-20261105-181537/summary.json) 及同目录 SQLite、plan、trajectory.md/json | 当前别名和北京地址下，两次真实调用完成 add(19,23)→42；reasoning 接收及第二次回传已核对。输入 992、输出 163 tokens，冻结价卡估价 0.0003288 元；无未知用量、遗留调用或重试。不是日期快照验证或 Retail 成绩 |
| [evals/offline-wiring.json](evals/offline-wiring.json) | 固定 v2.0.0/text-v1/en/retail：加载 50 个任务、环境提供 16 个工具；自己的 Agent 经脚本模型执行两次官方工具。不是运行 50 个任务，官方分数为 null |
| `tests/` | 当前确定性契约与离线验证入口；本轮仅阅读，不代表当前全套执行通过 |

程序 `completed`、工具 `ok` 和用户目标完成各自独立；没有业务评分时 `goal_status=unverified`。

## 当前限制

- Context 保留 `recent/full` 基线，以实际请求 UTF-8 字节为界，不证明服务端 token 窗口；尚无算法收益结论。
- Memory 是用户隔离的词项检索与来源校验；自动提炼默认关闭，须应用主动接 hook，每批最多额外一次模型调用；语义正确性与收益未验证，Retail 基线不启用。
- QUERY 是可信状态查询回调，尚无真实业务服务；未知写结果不盲目重放。远程 MCP、真实 DeepSeek 服务和跨进程强一致执行未验证。
- Retail 官方 orchestrator、UserSimulator 与 simulation/ALL 评分已接线，并已在 50×3 全量评测中真实运行（用户模拟器、NL 裁判调用均有账本与证据）。`evals.retail_pilot --execute` 是真实执行入口，`--check-config` / `--offline-check` / `--inspect` 为只读或离线入口。离线脚本接线与确定性测试不等于模型能力成绩；成绩口径见 [run-50x3.md](experiments/retail-qwen-tier-order-20261108/run-50x3.md)，接线说明见 [evals/README.md](evals/README.md)。

保留 50×3 原始数据、预算账本、冻结配置与源码快照；不恢复早期探索计划，不复用历史授权。实验已收尾：不再新增重跑，不为单次模型失败追加措辞特判或 prompt 补丁。后续只做有可信外部判据的机制改动，且须先说明对模型请求字节与既有 150 trial 可比性的影响；结论与取舍见 [run-50x3.md §17](experiments/retail-qwen-tier-order-20261108/run-50x3.md)。
