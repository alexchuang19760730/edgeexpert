# OpenHands Agent Canvas + Prime Agent (ACP) 接入 Checklist

> 目标：把 prime-agent 注册为 OpenHands Agent Canvas 的 Custom ACP agent，
> 获得 UI 可见性 + 自动化 + 隔离，同时保留 prime-agent 的 RLM 持久状态。
> 用途：用 Canvas 驱动 prime-agent 执行 `MOE_STREAMING_TASK.md`（MoE expert streaming 优化）。

---

## 0. 前置条件确认

- [ ] macOS（本机已满足，Canvas 支持 macOS/Linux/Windows）
- [ ] `prime-agent` 已安装并可用：`which prime-agent`（本机已装 v0.7.1）
- [ ] ACP 模式已验证：`printf '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}\n' | prime-agent --mode acp`
      应返回 JSON-RPC 错误（如 `-32602 Invalid params`）= 握手正常
- [ ] 有至少一个 LLM API key（OpenAI / Anthropic / Gemini 等，prime-agent 用）
- [ ] 磁盘空间 ≥ 5GB（Canvas + prime-agent + 工作树编译产物）

---

## 1. 安装 OpenHands Agent Canvas

- [ ] 到 GitHub Releases 下载 macOS 版：`https://github.com/All-Hands-AI/OpenHands/releases`（搜 "Agent Canvas"）
  - 备选：`https://openhands.dev/product/canvas` 页面上的下载入口
- [ ] 打开 App，首次启动会自动拉起本地 `agent-server`（无需 Docker）
- [ ] 确认状态：右上角 backend 显示 "Local machine" 且为健康状态

> 若安装受阻（网络）：Canvas 本体是 Electron app，下载源为 GitHub releases，
> 国内可尝试镜像或稍后重试。本 checklist 其余步骤不受影响。

---

## 2. 注册 prime-agent 为 Custom ACP agent

- [ ] Canvas → **Settings → Agent**
- [ ] 确认当前 Agent 类型为 **ACP**（不是 OpenHands 内置）
- [ ] 点击 **Custom**（自定义 ACP server）或 Provider 下拉选 "Custom"
- [ ] **Command** 填：`prime-agent --mode acp`
  - 若 `prime-agent` 不在 Canvas 的 PATH 里，填绝对路径：
    `/Users/alexchuang/.workbuddy/binaries/node/versions/22.22.2/bin/prime-agent --mode acp`
- [ ] **Model**：留空或填 prime-agent 默认模型（它会自己从登录状态读；留空最稳）
- [ ] 保存（会写 `agent_settings_diff`：agent_kind=acp, acp_command=...）

---

## 3. 配置 API 凭据（Secrets）

- [ ] Canvas → **Settings → Secrets**
- [ ] 添加与你的 provider 对应的环境变量（名字必须精确匹配，Canvas 会 export 给 ACP 子进程）：
  - OpenAI: `OPENAI_API_KEY`
  - Anthropic: `ANTHROPIC_API_KEY`（+ 可选 `ANTHROPIC_BASE_URL` 指向代理）
  - Gemini: `GEMINI_API_KEY`
  - 或你自建的 LiteLLM 代理：`OPENAI_API_KEY` + `OPENAI_BASE_URL`
- [ ] 保存后，**重启 agent-server**（Canvas → 设置 → 重启 backend，确保 env 生效）

> 注意：prime-agent 自己有独立的 `/login` 配置（存在它自己的配置目录）。
> 两条路径二选一即可，建议统一走 Canvas Secrets（env 优先）。

---

## 4. 首次对话验证（最小冒烟测试）

- [ ] Canvas → New Conversation
- [ ] Agent 选择器确认是 **prime-agent (ACP)**
- [ ] 发送一条极简消息，例如：`pwd && ls`（让它在工作树里跑）
- [ ] 预期：看到 assistant 消息 + 工具调用流（IPython cell 会以 `tool_call` 出现）
- [ ] 若工具流里出现 `ipython` cell = 整合成功（prime-agent 的 RLM 内核已被 Canvas 驱动）
- [ ] 若报错，查：Secrets 是否生效 / command 路径 / Canvas 日志（见 §7 排错）

---

## 5. 接入 MoE 工作树 + 启动任务

- [ ] 在 Canvas 的 Conversation 设置里把工作目录指向隔离工作树：
  `/Users/alexchuang/Documents/flashkv0516/prime-agent-worktrees/turbo-fieldfare`
  （ACP 的 cwd 在进程启动时固定，所以**先在 Canvas 里切好目录再开新会话**）
- [ ] 新建 Conversation，粘贴任务入口（二选一）：
  - 直接把 `MOE_STREAMING_TASK.md` 全文粘贴进首条消息，要求先读它再执行
  - 或：`cat MOE_STREAMING_TASK.md 然后按其中 P0/P1 顺序执行`
- [ ] 先让它只做 **P0**（跑基线、汇报数字），你 review 它怎么用环境
- [ ] P0 验收通过后，再允许它进入 P1（改代码 + 交错 A/B）

---

## 6. 验证清单（整合成功的标志）

- [ ] 会话里能看到 prime-agent 的持久 IPython 状态（变量跨轮存活：让它 `x=1` 然后下轮 `print(x)`）
- [ ] 工具调用流可见（IPython cell / shell 输出在 UI 中实时显示）
- [ ] 能并行开多个 Conversation（Canvas 每会话一个独立进程/工作树）
- [ ] （可选）配一个 cron/webhook 自动化，验证 Canvas 的调度层能唤醒 prime-agent

---

## 7. 排错速查

| 症状 | 处理 |
|---|---|
| 握手失败 / 无响应 | 先手动跑 §0 的 ACP 验证命令；确认 `prime-agent --mode acp` 能启动 |
| command not found | 改用绝对路径（见 §2） |
| API key 没生效 | 检查 Secrets 变量名精确匹配；重启 agent-server 让 env 重新 export |
| cwd 不对 | ACP 模式 cwd 启动即固定——关掉旧会话，先切目录再 New Conversation |
| Canvas 看不到子 agent/refine | 正常：这些走 `_meta` 扩展，Canvas 会忽略；深度控制回 TUI |
| 想回退 | Settings → Agent 切回 "OpenHands" 内置，或 "Claude Code"；prime-agent 的 ACP 模式无副作用，随时可停 |

---

## 8. 已知限制（整合边界）

- **UI 深度**：Canvas 只能看到标准 ACP 事件流（消息/工具调用）。prime-agent 的子 agent 树、`/refine`、goal 走 `_meta` 信封，Canvas 目前忽略——这些仍要在 TUI 里操作。
- **双配置**：prime-agent 的 `/login` 与 Canvas Secrets 并存，统一走 Canvas 更省心。
- **单会话单连接**：ACP 模式一个进程一个 session；Canvas 开多个 Conversation 会各起一个进程（内存占用线性，注意 16GB 机器上别开太多）。
- **16GB 内存**：Canvas + agent-server + prime-agent + 可能的编译，同时跑会有压力；建议一次只开 1-2 个 agent 会话。

---

## 9. 建议执行顺序

1. §1 装 Canvas → 2. §2 注册 agent → 3. §3 配 Secrets → 4. §4 冒烟测试
5. §5 接工作树跑 P0 → 6. 验收后放行 P1 → 7. 顺手把本文件归档到工作树 docs/
