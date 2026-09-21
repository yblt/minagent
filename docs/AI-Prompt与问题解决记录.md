# 开发过程 AI 使用记录（Prompt 与问题解决）

> 考题要求提交"AI Prompt 与问题解决记录"。本仓库开发过程中 AI 参与方式的如实记录。

## 1. 设计先行（人主导）

动手前先确定的关键决策（这些是答题核心，不由 AI 代劳）：

- 输出协议选"单 JSON 对象、thought/action/answer 三选一"，而非 OpenAI function-calling——
  从零实现题应不依赖特定厂商能力；
- session 粒度定为 `user::window`，正好对应考题"同一用户两个窗口"场景；
- 待办这类**状态**与**历史**分开：状态注入 system 快照，不占对话历史——
  这是 memory 放置方式的答案来源。

## 2. AI 辅助编码（Prompt 示例与用途）

| Prompt（摘要） | 用途 | 采纳情况 |
|---|---|---|
| "用标准库实现 OpenAI 兼容 chat/completions 客户端，urllib，超时与 HTTP 错误转成 LLMError" | llm.py | 采纳，错误分类保留 |
| "写一个只允许数字和四则运算的表达式求值器，禁止任意代码执行" | calculator | 采纳 AST 白名单方案，但补了指数上限与除零检查 |
| "解析模型输出：可能有 ```json 围栏或前后杂文本，手写大括号配平提取器" | parser.py | 采纳，配平需处理字符串内引号/转义 |
| "为 session 隔离写 pytest：同一 user 两个 window 各记一条待办，断言互不可见" | tests | 采纳并扩展为持久化往返测试 |

## 3. AI 产出被修正的记录（真实踩坑）

1. **解析器宽松边界**：AI 初版把"含 `{` 但 JSON 非法"也当纯文本答案兜底——
   协议纠错回路失效。修正为：无 `{` 才走宽松 answer；有 `{` 不成法一律 error 回喂。
   （对应测试 `test_bad_llm_output_gets_corrected`）
2. **demo 形参遮蔽**：辅助函数 `j(action, tool, **args)` 与 todo 工具的
   `action` 参数撞名，`TypeError: multiple values`。改名为 `thought`。
3. **压缩数量算错**：测试断言写"5条被压缩"，实际 10−keep_recent(2)=8。
   以代码语义为准修正测试并加注释。
4. **状态注入位置**：AI 曾把待办列表追加为一条 user 消息——多轮后旧待办堆进历史、
   被压缩后丢失。改为每轮从 `session.state` 现算，渲染进 system prompt。

## 4. 验证方式

- `python -m pytest tests -q` → 36 passed（全 mock，可离线复现）；
- `python demo.py` → 复现考题双窗口场景 + 隔离断言 + trace 落盘检查；
- 真实 API 路径未跑通端到端（离线交付，见 README"已知边界"）。
