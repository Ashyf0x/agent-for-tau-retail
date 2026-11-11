# 官方 Retail 评测接线

> **证据口径**：本文件记录接线与协议设计的历史约定。2026-11-10 已清理早期探索实验目录，
> 下文涉及的旧实验记录不再保留；当前唯一权威结果与配置以
> [experiments/README.md](../experiments/README.md) 和
> [run-50x3.md](../experiments/retail-qwen-tier-order-20261108/run-50x3.md) 为准。
> 测试所需的最小离线输入已迁至 `tests/fixtures/legacy/`。

## 协议与验收约定

- **最终回复验收**：自然回复的 `reasoning_content` 须为字符串、允许空串；首轮工具回复及其他保护不变。付费实验仍须另获当次授权。
- **`bailian_deepseek` 适配**：`--agent-model deepseek-v4.1-flash-1017` 选择独立适配，思考保持开启，省略不适用的 `preserve_thinking` 服务端参数，已有 `reasoning_content` 仍独立回传；User/NL/COMM 的 Qwen 协议不变。该协议要求显式 `--input-price/--output-price`，不套用 Qwen 旧默认估价。对应离线 fixture 在 `tests/fixtures/legacy/deepseek-protocol/`。
- **决策快照准备器**：`python -m evals.decision_snapshots --candidate <candidate.json> --output <新目录> --audit <审计记录.json> --runs <历史run目录>` 仅离线生成输入，不读取 `.env`、不执行模型/工具；`--audit` 与 `--runs` 为必需参数，准备器不再引用任何实验目录。测试用数据在 `tests/fixtures/legacy/decision-study/`、`decision-audit-requests.json` 与 `decision-source-runs/`。

固定官方环境、自己的 Agent.run、官方用户模拟器和 ALL 评分已接线。当前支持单进程顺序运行 1–2 个任务。**直接指定参数执行，无需先生成或传入冻结计划。** 程序自动保存实际配置、源码和全部证据。脚本传输验证只证明接线，不是模型能力或 Baseline 成绩。

## 配置与直接运行

从 `agent-runtime/` 运行。公共配置必须有 `AGENT_MODEL` 和 `AGENT_BASE_URL`，保留用户当前选择，没有源码或进程环境回退。默认四角色共用这两个值；如明确需要分别配置，可设置 `RETAIL_AGENT_MODEL/BASE_URL`、`RETAIL_USER_MODEL/BASE_URL`、`RETAIL_COMMUNICATE_MODEL/BASE_URL`、`RETAIL_NL_MODEL/BASE_URL`。优先级为显式 `--model/--base-url`（覆盖全部角色）> 对应角色字段 > 公共字段。显式空值报错，不回退。

模型对照使用 `--agent-model <模型ID>`，只覆盖 Agent 模型，优先于该角色其他模型来源；各角色地址以及用户模拟器、裁判的模型保持既有配置。`--model <模型ID>`覆盖全部四角色模型；两者均不修改 `.env`。资源配置由下表选择，`--budget-yuan`仍是本次运行共用的估价上限（最高10元），可低于单任务额度；每次真实运行须遵守当次授权。没有跨进程共享账本，不应并发运行来规避总预算。

|资源档|CLI选项|每任务全角色调用|每次Agent.run的Core步数|每任务估价上限|未显式给budget时的运行总上限|
|---|---|---:|---:|---:|---:|
|宽松候选（默认）|`--resource-profile relaxed`|80|24|3元|3元|
|历史复现|`--resource-profile legacy`|40|12|1元|1元|

数值唯一入口是`evals/pilot_config.py:RESOURCE_PROFILES`，由`build_config`进入自动保存的`config.json.resources`（版本`retail-direct-pilot-v3-resources`），传入共享Calls与Retail的AgentConfig；通用Core缺省仍为12。全角色范围仍为Agent、用户、COMM和NL，包含适用的评分调用；Core步数按模型循环迭代计，一步可执行多条工具调用。两个任务分别受各自调用/费用额度约束，同时共用运行总预算，该预算不乘任务数。

这是项目选择的宽松执行配置，不是官方要求，未验证最优。单任务提高额度不构成批次预算或真实调用授权；此前36会话的12元授权已结束，不继承。后续批次另行确认任务范围、次数和总预算，由当批控制器遵守总预算。候选清单为早期资源候选记录（已随探索实验清理）。Baseline和优化版须使用同一套新资源配置及同一批次预算约束；旧40/12/1元结果仅作历史结果，不能直接与新结果比较并将差异归因于Context。

支持既有Qwen与精确型号`deepseek-v4.1-flash-1017`的百炼非流式Chat Completions协议；Agent开启thinking/思考回传，用户与裁判关闭thinking，temperature=0。Qwen Agent仍发送preserve_thinking，百炼DeepSeek适配不发送该参数；不切换到DeepSeek原生API的thinking对象。未知新型号不继承已核查兼容结论。用户模拟器seed=42原样送到服务端。NL裁判通过有界作用域的generate传输接线采用冻结角色配置，官方提示、任务和评分逻辑不改动，作用域退出恢复绑定。本实现用于顺序Pilot，不支持并发任务。

`--check-config`可选，只筛选公共配置行，不解析认证值、不发请求。直接执行时先解析实际配置、校验资源限制及固定官方版本，再加载 DASHSCOPE_API_KEY；配置自动存入新运行目录的`config.json`。不再接受`--plan`，也不通过外部JSON启动运行。只读`--inspect`不读取.env，兼容历史`plan.json`。离线`--offline-check`必须显式给出fixture模型和地址，使用本地HTTP MockTransport，不读取.env或认证、不验证线上服务。

```powershell
Set-Location A:\Github\Agent-Project\agent-runtime
$py = ".\.venv313\Scripts\python.exe"
& $py -m evals.retail_pilot --check-config --env-file .env
$pilotRun = ".offline-validation/retail-pilot-$(Get-Date -Format yyyyMMdd-HHmmss)"
# 以下一条会产生真实调用；本例沿用Flash历史保守费率，换模型需核实并替换费率
& $py -m evals.retail_pilot --execute --env-file .env --task-id 6 `
  --agent-max-completion-tokens 9060 --max-request-bytes 131072 `
  --resource-profile relaxed --budget-yuan 3.00 --input-price 1.2 --output-price 4.8 --output $pilotRun
# 无推理、无配置读取；失败后只检查，不续跑
& $py -m evals.retail_pilot --inspect --output $pilotRun
```

示例只用已见任务6；需要两个任务时可再加一个已获授权的`--task-id`，不要用预留15项调试。每任务1 trial、60官方步骤，所有角色共享每任务80次请求的总计上限。复现历史资源限制使用`--resource-profile legacy --budget-yuan 1.00`，并保留原Context选项和其他显式参数。输出必须是新路径，不覆盖、不恢复、不自动补跑失败实验；改变配置后新开目录。入口默认输出仍2048、请求体28000字节；上例显式选择当前研究的Agent9060/其他角色2048和128KiB。

Core原本把实际步数上限放入系统消息的`run_state.limit`；用户本轮明确允许该字段随12→24变化。除此之外，未触及上限时的模型请求内容及Context构造保持一致；Prompt、Context A/B/C、Memory及工具反馈投影源码未改。

仅验证接线、不产生真实调用：

```powershell
& $py -m evals.retail_pilot --offline-check --task-id 6 `
  --model offline-fixture --base-url https://fixture.invalid/v1 `
  --output ".offline-validation/retail-fixture-$(Get-Date -Format yyyyMMdd-HHmmss)"
```

## Context配置：A/B/C

模型、地址、9060/2048、Policy、工具和评分与Context选项独立。新Retail运行默认B；通用`ContextManager`的缺省仍为原始recent，既有非Retail调用不被隐式切换。

|组别|命令行选项|行为|
|---|---|---|
|A Baseline|`--context-history recent --context-compaction legacy`|原始3个用户轮次、180字符/最后6项旧摘要；实际请求可逐字节对照历史|
|B Capacity-Aware（默认）|`--context-history capacity --context-compaction legacy`|先保留全部会话，超预算才删旧完整轮次，沿用原摘要处理|
|C Capacity + Task Handoff|`--context-history capacity --context-compaction task`|同B保留；实际预算压力导致移除旧轮次时，生成确定性交接|

只需把对应选项附到上文直接运行命令中，无需外部计划文件。两个开关可独立配置；`recent+task`消融也仅在额外预算压力下使用任务交接，不因每轮对话自动生成它。

B/C保留131072字节硬上限，并在其中留2048字节安全余量，不扩大输入或输出限制；最终序列化包含Policy、工具Schema、历史思考回传和摘要。Token窗口只有在显式提供可信计数器时才单独硬检查；没有Token上限或tokenizer时不按模型名猜测，不把字节代理或9060输出当作输入窗口。

交接保留完整旧用户原文、相邻Agent回复（标为未验证陈述）、工具名称/参数/结果状态及来源。成功工具值小于1KiB时保留，较大值提取有路径的标量并标记partial、保留原observation引用；失败value仍按原策略屏蔽。不同时间的观察按原顺序保留，不把旧状态覆盖新状态。目标、确认范围、撤回和未完成业务不作语义推断，需结合引用重新判断。未知结果不给写重试授权，工具成功不标记目标完成。

交接上限16KiB；完整用户约束或当前轮次仍放不下时，明确报ContextBudgetError，不静默截断或关闭保护。交接每次只从当前任务会话记录构造，不建立跨任务缓存/长期Memory，也不在单次Agent.run结束时清空原任务历史。任务结束后新任务使用独立Store；旧证据保留。

Core的`model_request`事件记录窗口策略、保留量、裁剪轮数、交接字节、计数器状态、输出预留及reasoning字节。`input_token_byte_proxy`仅为粗略观测量，不是provider计数，也不替代原费用保护。详见本轮[离线报告](../.offline-validation/context-abc-implementation-20261106/report.md)。

可信Token预算的Python接线示意（不自动查询服务，也没有默认tokenizer）：

```python
context = ContextManager(policy_text, policy="capacity", compaction="task",
    context_window_tokens=verified_window, token_counter=count_full_request_tokens,
    output_reserve_tokens=actual_agent_output_cap, token_margin=256)
```

计数器须覆盖最终请求及工具Schema；失败、无效计数或输出预留小于真实序列化输出上限时停止，不回退到近似计数。

## 全角色预算与调用证据

四角色共用 `Calls`：每次先检查 STOP 文件、遗留预留、请求总数、实际协议/模型/地址、请求字节与可用估价预算，持久化并 fsync 派发意图后才发请求，零自动重试。用量未知或调用异常停止全部后续角色和任务，未知预留保留；已知用量结算后超界也停止。交互、裁判或证据存储失败不能自动续跑。可创建 `<run>/STOP` 阻止下一次派发，但不取消已发送请求。

默认估价每百万输入/输出1.2/4.8元，沿用历史smoke统一平价假设，未重新核价；可通过`--input-price/--output-price`指定，必须覆盖所选全部角色的实际费率。默认请求体上限28000字节，可显式指定至131072；输入预留同步取max(32000, 请求字节上限+2048)。输出按对应角色的上限+10预留：上例Agent9070、其他角色2058；保留原预算、调用和未知用量停止保护。字节代理不是tokenizer，估价保护不保证账户绝对账单。

128KiB对应输入代理133120；按上例费率，Agent9060每次派发前预留0.20328元，其他角色2048预留0.1696224元。3元能容纳单次最坏情况预留，但不承诺能用满80次；预留、实际usage结算/释放、未知用量保护、实际用量超界停止均保留。这不是服务端窗口声明。

调用上限在账本记为`call_limit`，费用预留拦截区分`task_budget_limit`和`run_budget_limit`；Core耗尽步数仍为`failed/step_limit`。Calls异常在Core内仍按原契约记录为`internal_error`，以关联账本辨明具体限制；用户模拟器派发前被拦截时可能没有失败Core.run。官方交互异常的部分simulation仍为`infrastructure_error`、reward=null，不补成官方零分。官方正常终止、Core结束及费用停止分别判断。提高调用上限不修复转人工后的交互循环；该问题另行处理，本轮不增加转人工后强制结束规则。

2026-11-05 首次真实任务 6 在原 28000 字节配置下以 ContextBudgetError 中断，8 次调用用量均已知、未评分；只读重建下一请求最少需 31692 字节。已增加上述显式参数与同步预留校验，没有改核心上下文策略，也未自动重跑。原始计划、源码快照和失败轨迹保留在 `.offline-validation/retail-user-authorized-20261105-195358/`。

Windows 下若官方依赖由 Codex 沙箱用户创建，校验器先检查锁定文件哈希，再只给本次只读 `git rev-parse` 设置精确 checkout 的 `safe.directory`；不需要修改全局 Git 信任配置，也没有使用通配信任。

每次调用的 `calls.jsonl` 保存角色、任务、调用 ID、请求 JSON/哈希、最终地址、HTTP 状态及解码后响应正文（base64）/哈希、解析响应、用量已知性、费用结算和停止原因，不保存认证头。Agent 额外关联 core run_id/model_call_id；provider 工具 ID 映射和完整工具证据在 core.sqlite3。若进程异常退出，`--inspect` 从已落盘意图保守重建未知用量和预留；尾行写入不完整会明确标记。

每次真实或脚本运行自动保存`config.json`、source-snapshot、任务与call context、官方未评分/已评分SimulationRun、逐run轨迹JSON/Markdown、各角色统计和summary。历史计划、源码快照与结果原位保留；旧CLI冻结流程已移除，历史实验脚本及快照是复核依据，不代表当前入口。官方agent_cost保持null，CNY估价单独记录。交互/裁判失败保留未评分轨迹，不制造零分；NL响应必须覆盖全部待评断言。

评分保持官方 reward_basis、semantic 沟通判定与 ALL evaluator 的严格 DB 回放。某任务无需沟通/NL 调用时，对应调用数为 0 属于正常分支；全部四角色的共享传输与评分分支通过合成任务测试验证，不表示该任务有真实裁判成绩。当前产物是逐任务 SimulationRun，未包装为官方批量 Results 提交文件。

离线行为测试（新目录）：

```powershell
& $py -m pytest -q tests/test_pilot_calls.py tests/test_pilot_config.py tests/test_retail.py tests/test_retail_interaction.py --basetemp ".offline-validation/pilot-tests-$(Get-Date -Format yyyyMMdd-HHmmss)"
```

后续失败分析先区分调用/预算停止、Agent 终态、工具错误、DB/沟通/NL 分项失败与证据缺失，再核对原始调用、官方轨迹与 core 轨迹。不从脚本验证推导算法收益，不将一两个任务成绩当作 Baseline。

## 固定版本与历史环境 fixture

范围固定为 [sierra-research/tau2-bench v2.0.0](https://github.com/sierra-research/tau2-bench/tree/v2.0.0)，提交 `989d3285ae669e49503324bdb9981bf14fccedeb`，`text-v1`、English/Retail。`retail.lock.json` 固定提交、text manifest、English pack、修订任务、50-task subset、环境数据库/政策与源码哈希；运行前拒绝版本/文件漂移。

[官方版本说明](https://github.com/sierra-research/tau2-bench/blob/v2.0.0/README.md)明确区分 2.0 和 1.x：English Retail 使用修订任务，模拟器提示和沟通评分也有变化。因此本项目不把旧 τ² 或 1.x 成绩与 2.0 混比。

## 安装

必须使用 Python 3.12–3.13。以下是公开代码和依赖下载，不触发模型调用；不要运行上游 `tau2 run` 或 `tau2 judges`。

```powershell
git clone --depth 1 --branch v2.0.0 --filter=blob:none --sparse https://github.com/sierra-research/tau2-bench.git .vendor/tau2-bench
git -C .vendor/tau2-bench sparse-checkout set src/tau2 docs data/tau2/domains/retail data/tau2/benchmarks data/tau2/task_subsets data/tau2/multilingual/en
python -m pip install -e ".[mcp,dev]" -e .vendor/tau2-bench -r evals/requirements.txt
```

上游 tau2 的包初始化会导入一部分音频代码，因此 `requirements.txt` 补充 `audioop-lts`、`pydub`、`scipy` 的本轮已验证版本；它们不进入通用核心依赖。官方源码未修改。现场这些步骤已完成，解释器为 `.venv313/Scripts/python.exe`。

## 运行与证据

从项目根目录：

```powershell
python -m evals.retail_fixture
python -m pytest -q tests/test_retail.py
```

`--output` 可写入一个新报告路径，已存在路径会拒绝覆盖。已生成的 `offline-wiring.json` 是本轮实际运行记录：

- 官方语言加载器解析固定的 50 个 English/Retail 任务；没有运行这 50 个任务。
- 官方环境实际提供 16 个工具；fixture 调用 `find_user_id_by_email`、`get_user_details`，只读前后官方 DB hash 相同。
- 模型是 ScriptedModel；自己的 `Agent.run` 实际执行三次模型步骤及两次环境工具调用，没有使用上游 LLMAgent 循环。
- 第二次调用参数来自第一次工具观察；模型输入不含用户隐藏 scenario、evaluation_criteria、参考动作或评分里程碑。
- 真实模型、用户模拟器和裁判调用均为 0。`official_score=null`。

额外测试验证官方业务失败仍为 returned/ok=false、不同环境数据库副本互相隔离，以及锁定文件漂移拒绝。原始业务失败正文保留在本地记录，模型默认只见简短反馈。

## 尚未验证

尚未接通完整官方半双工 orchestrator 与用户模拟器，不生成官方 simulation 提交文件；本轮交付的是环境最小薄适配与固定交互证据。默认测试在未安装 checkout 时明确 skip Retail 测试，不能据此宣称该环境已验证。本现场已安装，最终回归无 skip。

[官方评分说明](https://github.com/sierra-research/tau2-bench/blob/v2.0.0/docs/evaluation.md)区分 DB、COMMUNICATE、NL_ASSERTION 等组件，最终 reward 由任务 reward_basis 决定。参考动作是生成目标数据库的参考轨迹，不应作为 Agent 的隐藏提示。新版沟通语义判定与自然语言断言可能调用模型。本轮未运行任何官方评分（包括确定性的 DB reward），未替换裁判、未删除组件，也未生成部分评分冒充完整成绩。

若以后进行真实评测，需要单独完成完整交互/评分适配、再次固定配置并取得新的推理授权。旧 ClarifyBench 预算与确认计划均不得复用。
