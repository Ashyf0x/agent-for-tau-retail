# Agent-Project 工作区

当前开发项目是 **agent-runtime**：轻量单 Agent 运行核心，以「程序管状态与执行、模型只管语言与工具选择」为边界，接入固定版本 τ²-Bench Retail 官方评测链路。50 任务 × 3 trial 的全量评测已完成并收尾；工作区根目录只保留长期规范与导航，历史交接和探索批次已清理。

## 目录导航

```text
Agent-Project/
├── AGENTS.md                  统一开发规范（长期规则）
├── README.md                  工作区导航与架构概览（本文件）
├── .gitignore                 忽略实验产物、环境与工作区状态
└── agent-runtime/            当前项目，所有运行命令的工作目录
    ├── README.md              安装、配置、运行与验证
    ├── ARCHITECTURE.md        模块设计、运行链路与状态语义
    ├── pyproject.toml         Python 包与开发工具配置
    ├── agentcore/             通用运行核心及应用入口
    │   ├── core/              Agent 循环、模型协议、类型与错误策略
    │   ├── context/           本轮可见输入与上下文边界
    │   ├── memory/            跨会话候选存储、检索与可选提炼
    │   ├── tools/             工具契约、执行器与 MCP
    │   ├── storage/           SQLite 状态与原始证据
    │   ├── observability/     报告、轨迹重建与导出
    │   └── app/               离线 CLI、受限真实冒烟入口
    ├── evals/                 固定官方 Retail 环境、交互桥与评分接线
    ├── examples/              离线运行与 MCP/Memory 示例
    ├── experiments/           唯一权威证据：50×3 trial 结果与报告
    │   └── retail-qwen-tier-order-20261108/
    │       ├── run-50x3.md            正式实验报告
    │       ├── run-50x3.py            批次驱动（执行需显式授权）
    │       └── runs/t50x3/            150 个 trial 原始轨迹
    ├── tests/                 确定性测试、离线集成测试与架构约束
    │   └── fixtures/legacy/   从已清理实验提取的最小离线测试输入
    ├── .vendor/               固定官方评测源码与数据
    ├── .venv313/、.python/    当前解释器和运行环境
    ├── .research-data/        早期研究数据与源码快照，保留
    └── .offline-validation/   离线验证依据，保留
```

`.python/` 是 `.venv313` 的解释器来源，**不可删除**；`.vendor/` 是评测依赖，同样保留。
`.env` 不参与任何整理，离线工作不读取其内容。
实验证据的唯一入口见 [experiments/README.md](agent-runtime/experiments/README.md)。

## 架构概览

应用入口创建模型、工具和 Store，再调用唯一的 `Agent.run` 循环。Context 准备模型输入，Memory 提供跨会话候选，模型通过原生 Tool Calling 决定行动；工具执行器负责校验、派发和结果已知性。SQLite 保留原文、调用关联、工具结果和真实终态，Observability 只读重建轨迹。

Retail 评测位于核心之外：官方 orchestrator 负责用户轮次、环境执行和终止，消息桥复用自己的 Agent 循环，官方 ALL evaluator 按原任务评分。程序结束（run 级 `completed`）、工具成功与用户目标达成分别记录；L3 会话终态由外部权威声明，运行时不推断。离线脚本接线不能作为真实模型成绩。

详细调用图见 [ARCHITECTURE.md](agent-runtime/ARCHITECTURE.md)，评测接线说明见 [evals/README.md](agent-runtime/evals/README.md)，正式结果见 [run-50x3.md](agent-runtime/experiments/retail-qwen-tier-order-20261108/run-50x3.md)。

## 开始工作

先读 [AGENTS.md](AGENTS.md)，再读 [项目 README](agent-runtime/README.md)。从本工作区根目录进入项目：

```powershell
Set-Location .\agent-runtime
.\.venv313\Scripts\python.exe -m agentcore
```

默认入口是脚本模型离线示例。真实模型、用户模拟器和裁判调用需要另获当次授权。
