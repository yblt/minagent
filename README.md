# minagent — 从零实现的最小可用 Agent

不依赖任何 Agent 框架（无 langgraph/openhands 等），核心 Runtime 全部手写；
运行期仅用 Python 标准库，测试用 pytest。支持任意 OpenAI 兼容 LLM API。

## 快速开始

```bash
pip install pytest          # 仅测试需要，Agent 本体零第三方依赖
python -m pytest tests -q   # 36 个用例，全 mock、可离线运行
python demo.py              # 离线演示（MockLLM 复现考题双窗口场景）
```

接真实 LLM（DeepSeek / 通义 / OpenAI 均可）：

```bash
export MINAGENT_API_KEY=sk-xxx
export MINAGENT_BASE_URL=https://api.deepseek.com/v1
export MINAGENT_MODEL=deepseek-chat
python demo.py --real       # 命令行交互模式
```

## 系统设计

```
用户输入 ──► Agent.chat(user, window, input)
              │
              ├─ SessionStore: 按 "<user>::<window>" 取/建独立 session
              ├─ ContextManager: system(工具协议+待办快照) + 历史(超限压缩)
              ▼
            ┌── Loop（≤ max_iterations）────────────────────┐
            │ LLM.chat(messages)                            │
            │   └ LLMError → 重试1次 → 优雅降级返回          │
            │ parse_llm_output(raw)                         │
            │   ├ answer → 结束返回                          │
            │   ├ action → ToolRegistry.call                │
            │   │    └ ToolError → "error: …" 作为           │
            │   │       observation 回喂，LLM 自行修正重试     │
            │   └ 协议违规 → 回喂纠错提示，计入轮次            │
            │ 每步写 JSONL trace                             │
            └───────────────────────────────────────────────┘
```

模块划分（`minagent/`）：

| 文件 | 职责 |
|---|---|
| `agent.py` | 主循环、异常兜底、轮次上限 |
| `tools.py` | 工具注册表（name/description/JSON Schema）+ calculator/search/weather/todo 四个工具 |
| `parser.py` | LLM 输出解析：提取 thought / action / answer，容忍代码围栏与杂文本 |
| `session.py` | session 隔离与持久化（JSON 落盘，重启可续聊） |
| `context.py` | 上下文组装、长度控制、基础压缩、工具结果截断 |
| `llm.py` | MockLLM（测试/演示）与 OpenAICompatClient（真实 API） |
| `trace.py` | 每事件一行 JSONL 的执行 trace |

### 关键设计点

**工具协议**：system prompt 注入各工具 JSON Schema，模型每轮输出
`{"thought":…,"action":{"tool":…,"args":…}}` 或 `{"thought":…,"answer":…}` 二选一。
注册表按 schema 校验必填参数；`calculator` 用 AST 白名单求值，拒绝任意代码执行。

**异常处理三层兜底**：LLM 层（网络/限流→重试一次→降级话术）、协议层（非法输出→回喂纠错提示）、
工具层（ToolError→作为 observation 回喂，让模型改参数重试或换思路）。任何一层都不炸主循环。

**max_iterations**：默认 8，触顶返回 `finished=False` 的收尾回复，防无限工具循环。

## Memory：召回时机与放置方式

三类信息走三条路，这是本实现最核心的取舍：

1. **对话历史（messages）**——召回时机：每轮组 context 时全量（或压缩后）带上。
   放置：`user / assistant / tool` 角色消息按时间序。工具调用与其结果**都保留**：
   结果支撑"带着工具的追问"（多少度来着→直接引用历史 tool 消息），
   调用参数保留支撑"为什么这么做"的回溯。
2. **结构化状态（session.state，如待办列表）**——召回时机：不进对话历史，
   而是每轮渲染成**快照注入 system prompt**（`当前会话待办快照：…`）。
   理由：状态是"当前值"而非"发生过什么"，放历史里会被旧值干扰且随压缩丢失；
   放 system 里永远最新、成本恒定。
3. **压缩后的远期记忆**——触发：历史超 `max_messages`（默认 24）时，
   保留最近 `keep_recent` 条，更早的压成一条 `[历史摘要]` system 消息
   （默认逐条 80 字截断的朴素压缩；`summarizer` 可注入 LLM 摘要实现复杂压缩）。
   工具结果超长单独截断（信息头部最密）。

**隔离**：session_id = `user::window`，同用户不同窗口历史与 state 完全独立、互不污染，
各自可随时续聊（见 `tests/test_session_isolation.py` 复现考题场景）。

## Trace

`trace_dir` 开启后每 session 一个 JSONL：`llm_request / llm_raw / tool_call(含参数与成败) /
parse_error / llm_error / final`，可完整复盘一次对话为什么走了这条路径。

## 测试覆盖

`tests/`：工具注册与校验、注入攻击拒绝、解析容错（围栏/杂文本/非法JSON）、
单工具与多工具链、错误回喂修正、协议纠错、轮次上限、LLM 不可用降级、
双窗口隔离、续聊、持久化往返、压缩与截断、trace 事件。

## 已知边界（最小实现的有意取舍）

- 一次只调一个工具，无并行 tool_calls；无流式输出；无多 Agent。
- 压缩是截断式摘要，非语义摘要（接口已留 `summarizer`）。
- 真实 API 未在本仓库跑通端到端（离线交付），代码路径与 mock 完全一致，仅换 LLM 客户端。
