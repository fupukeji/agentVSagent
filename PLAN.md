# 立回斗士 → AI 竞技任务包：开发计划与任务书

> 文档版本：v1.0（2026-10-05）
> 执行者：pi agent（在 `/Users/natefier/projects/fighting-rl` 目录下工作）
> 委托人：项目发起人
> 阅读要求：**请先完整读完本文档再动手**，按任务顺序执行，每完成一个任务勾选对应复选框并自测通过后再进入下一个。

---

## 0. 给执行 Agent 的工作约定

1. 工作目录：`/Users/natefier/projects/fighting-rl`，所有新文件放这里，**不要改动** `/Users/natefier/projects/dahua-rl`（姊妹项目，仅供参考）。
2. 一律使用 `.venv/bin/python` 运行（依赖已装好：gymnasium、stable-baselines3、torch、numpy、pygame）。不要另建虚拟环境，不要安装新依赖，除非任务明确要求。
3. 现有文件中 `fighting_env.py` 允许**向后兼容地**扩展（见 Task 1），但不得破坏现有接口：`FightingEnv.step/step_both/reset`、`FootsiesBot`、`MOVES`、常量名。`train.py` 的训练结果必须保持可复现。
4. 每个任务完成后：运行该任务「验收」小节列出的命令，全部通过才视为完成；然后 `git commit`（Task 0 会初始化仓库）。
5. 遇到规格未覆盖的设计细节，按「平台文档原则」一节的精神自行决策并在 commit message / 代码注释中说明理由，**不要停下来等待确认**，除非遇到会破坏向后兼容的必要变更。
6. 所有用户可见的输出（打印、报告）用中文；代码注释中英不限。

---

## 1. 背景与使命（为什么做这件事）

### 1.1 平台愿景（一页纸摘要）

上级项目是一个「**AI Agent 任务挑战与评测平台**」（完整文档见
`/Users/natefier/projects/dabidou/project_documents/AI任务与共创竞技平台*.docx`，
可用 `textutil -convert txt -stdout 文件.docx` 阅读）：

- 用户**带着自己的 AI**（RL 策略 / LLM Agent / 脚本 Bot）参加别人创建的任务，平台负责**执行、评分、回放、排名**。
- 核心用户循环：发现任务 → 带 AI 进入 → 执行并观察 → 获得结果 → 查看回放 → 修改 AI 或规则 → 再次挑战 → 分享。
- 首期三类任务：**娱乐对抗**、能力挑战、真实微任务。本项目属于第一类，且是两周验证冲刺的展示核心。
- 平台原则（必须遵守）：**任务先于平台、评分先于排行、模板先于自由、结果先于热度**。

### 1.2 本项目的使命

把现有的单机训练器 `fighting-rl` 封装成平台可用的第一个「**任务包**」：

> 两个 AI 在 1D 格斗环境中对战 → 无头执行 → 生成回放 → 多种子/隐藏对手评分 → Elo 榜单。

完成后的演示语：任何用户上传一个符合接口的策略，30 秒内得到「胜率 + 速度 + 稳定性」三维评分、可回放的对局记录和在榜单上的位置。

### 1.3 游戏机制速查（现状，已实现并验证）

- 舞台一维 0~1，每步 = 1 帧（60fps 语义），双方**同时**决策（`step_both(a1, a2)` 接口已预留）。
- 招式（帧数据 = 启动/判定/收招）：

| 招式 | 启动 | 判定 | 收招 | 距离 | 伤害 | 硬直(命中/被防) |
|------|-----|-----|-----|------|-----|----------------|
| 轻击 | 3 | 2 | 5 | 0.13 | 7 | 9 / 4 |
| 重击 | 8 | 3 | 14 | 0.19 | 18 | 15 / 4 |
| 投技 | 5 | 2 | 10 | 0.07 | 14 | 18 / - |

- 猜拳环：打击克投技、投技克防御、防御克打击（防御只吃 15% 削减伤害，但被投技破坏）。
- 其他常量：前进/后撤速度 0.009/0.007，最小间距 0.05，命中/被防击退 0.030/0.018，默认 900 帧超时。
- 已有成果：PPO（`fighting_ppo.zip`）对默认 `FootsiesBot` 50 场 **48 胜 2 负**；训练仅需 17 秒。

---

## 2. 现有资产盘点

| 文件 | 内容 | 状态 |
|------|------|------|
| `fighting_env.py` | Gymnasium 环境 + 帧数据 + `FootsiesBot`，`__main__` 自带冒烟测试 | ✅ 可用，随机策略 1胜19负 |
| `train.py` | PPO 训练 + 50 场评估 | ✅ 可用 |
| `play.py` | pygame 实时对战画面（调速/暂停/距离仪表） | ✅ 可用 |
| `fighting_ppo.zip` | 已训练模型（96% 胜率） | ✅ 可用 |
| `README.md` | 项目说明 | ✅ 需小更新（见 Task 8） |

**尚缺**：无头对局执行、回放导出与复放、隐藏对手池、外部 Agent 接入、Elo 榜单、综合评分 —— 即本任务书的全部内容。

---

## 3. 目标架构

```
fighting-rl/
├── fighting_env.py      # 环境核心（已有，向后兼容扩展）
├── rules/
│   └── default.json     # 规则包：帧数据/速度/常量 外置（Task 1）
├── arena.py             # 无头对局执行器：A vs B → 结果 + 回放 JSON（Task 2）
├── replay.py            # 回放读写/校验/统计（Task 3）
├── hidden_bots.py       # 隐藏对手池：8+ 个风格化 Bot 变体（Task 4）
├── agents/              # Agent 接入适配器（Task 5）
│   ├── __init__.py      #   统一加载器 load_agent(spec)
│   ├── base.py          #   Agent 协议基类 + 决策窗口包装器
│   ├── ppo_agent.py     #   包装 fighting_ppo.zip（示例一）
│   ├── script_agent.py  #   包装 FootsiesBot 等脚本（示例二）
│   └── http_agent.py    #   HTTP/LLM Agent 适配器（决策窗口制）
├── score.py             # 综合评分：胜率65% + 速度20% + 稳定性15%（Task 6）
├── elo.py               # 循环赛 + Elo 榜单（Task 7）
├── play.py              # pygame 演示（已有，加 --replay 复放模式，Task 3）
├── train.py             # 训练（已有，不动）
├── replays/             # 回放输出目录（运行时生成）
└── PLAN.md              # 本文档
```

---

## 4. 任务清单（按顺序执行）

### Task 0：工程初始化 —— P0，10 分钟

- [ ] `git init`，创建 `.gitignore`（忽略 `.venv/`、`__pycache__/`、`*.zip` 不忽略——模型要提交、`replays/` 忽略）。
- [ ] 首次提交全部现有文件，commit message：`init: 立回斗士单机版（env/train/play）`。

**验收**：`git log` 有一条提交；`git status` 干净。

---

### Task 1：规则包外置（moves.json）—— P0，半天

**目标**：把 `fighting_env.py` 里硬编码的帧数据和常量抽到 `rules/default.json`，环境从 JSON 加载。这是平台「规则包」概念的最小实现：改一个数字 = 一个规则变体，用户可 fork。

**规格**：

1. JSON 结构（同时是平台的规则包 schema v1）：

```json
{
  "name": "default",
  "version": 1,
  "walk": {"forward": 0.009, "back": 0.007},
  "chip": 0.15,
  "min_gap": 0.05,
  "push": {"hit": 0.03, "block": 0.018},
  "max_hp": 100,
  "max_ticks": 900,
  "moves": [
    {"slot": 0, "name": "后撤", "kind": "move", "dir": "back"},
    {"slot": 1, "name": "前进", "kind": "move", "dir": "forward"},
    {"slot": 2, "name": "轻击", "kind": "strike", "startup": 3, "active": 2,
     "recovery": 5, "reach": 0.13, "damage": 7, "hitstun": 9, "blockstun": 4},
    {"slot": 3, "name": "重击", "kind": "strike", "startup": 8, "active": 3,
     "recovery": 14, "reach": 0.19, "damage": 18, "hitstun": 15, "blockstun": 4},
    {"slot": 4, "name": "防御", "kind": "guard"},
    {"slot": 5, "name": "投技", "kind": "throw", "startup": 5, "active": 2,
     "recovery": 10, "reach": 0.07, "damage": 14, "hitstun": 18, "blockstun": 0}
  ]
}
```

2. `FightingEnv.__init__(..., rules=None)`：`rules` 可为 dict / JSON 路径 / None（None 时加载 `rules/default.json`；文件缺失时回退到内置常量并打 warning，保证向后兼容）。
3. `FootsiesBot` **参数化**（为 Task 4 铺路）：`__init__(self, punish=0.8, want_dist=0.155, mixup=None, seed=None)`，`mixup` 为 `{"throw": p, "light": p, "heavy": p, "block": p, "back": p}` 概率表（剩余概率给 light，默认值保持现行为不变）。
4. 所有常量改为从 rules 读取（模块级常量保留为默认值，供旧代码 import）。

**验收**：
- [ ] `.venv/bin/python fighting_env.py` 冒烟测试结果与改造前一致（随机 20 场，约 1胜19负，允许种子内一致即可）。
- [ ] 临时改一份 JSON（如轻击 reach 0.13→0.20）加载后对局行为明显变化，恢复后一致。
- [ ] `train.py` 不改一行仍可运行。

---

### Task 2：无头对局执行器（arena.py）—— P0，半天

**目标**：脱离 pygame，纯模拟执行「Agent vs Agent」一局或一批，输出结果与回放。这是平台「执行层」的最小内核。

**规格**：

1. CLI：
   - `python arena.py run <agent1> <agent2> [--seed N] [--rules PATH] [--out replays/]` → 打印一行结果，写出一个回放 JSON。
   - `python arena.py batch <agent1> <agent2> [--games 20] [--seeds 0-19]` → 打印汇总（胜/负/判、平均用时帧、平均回报），批量写回放到 `replays/`。
   - `python arena.py demo` → 内置演示：`ppo` vs `footsies` 打 3 场。
2. Agent 用**名字引用**（由 Task 5 的 `load_agent` 解析），本任务先支持 `random`（内置随机）与 `footsies`（默认 FootsiesBot），接口预留 `ppo`。
3. 内部使用 `step_both`（双方公平，AI 侧不再有「环境替对手出招」的特殊地位）。
4. 单场结果结构（同时也是回放 JSON 的 `outcome` 字段）：

```json
{"winner": 0, "hp": [82, 0], "ticks": 341, "result": "win0"}
```

`winner` ∈ 0/1/null(判平)；`result` ∈ `win0|win1|timeout0|timeout1|draw`（timeout 按血量判）。

**验收**：
- [ ] `python arena.py demo` 正常运行并生成 3 个回放文件。
- [ ] 相同 seed 重复运行 `run`，两次结果完全一致（确定性）。
- [ ] `random` vs `footsies` 20 场胜率 < 30%（与已知随机强度一致）。

---

### Task 3：回放格式与复放（replay.py + play.py 扩展）—— P0，半天

**目标**：确定性格式：**种子 + 双方动作序列 = 完整复现**。校验器保证「平台产出的回放不可伪造」的最弱保证（重放一致性）。

**规格**：

1. 回放 JSON schema v1：

```json
{
  "format_version": 1,
  "game": "foosies-fighter",
  "rules": {"name": "default", "sha256": "<规则包内容哈希>"},
  "seed": 42,
  "max_ticks": 900,
  "players": [
    {"name": "ppo", "kind": "ppo", "meta": {}},
    {"name": "footsies", "kind": "script", "meta": {"punish": 0.8}}
  ],
  "ticks": [[3, 1], [3, 1], [0, 0]],
  "outcome": {"winner": 0, "hp": [82, 0], "ticks": 341, "result": "win0"}
}
```

   `ticks` 为每帧 `[a1, a2]` 动作对。文件名建议：`replays/{timestamp}_{p1}_vs_{p2}_s{seed}.json`。
2. `replay.py` 提供：`save_replay(path, data)`、`load_replay(path)`、`verify_replay(path)`（用种子+动作序列重放 `step_both`，校验 HP 轨迹与 outcome 完全一致，返回 bool + 差异说明）。
3. `play.py` 加 `--replay <file>` 参数：复放模式，不加载模型，按 `ticks` 逐帧驱动 `step_both` 渲染，结尾显示 outcome；UI 上标注「回放模式」与双方名字。原有实时对战模式行为不变。
4. `arena.py` 写出的每个回放都自动 `verify_replay` 自检（成本可忽略），失败则报错。

**验收**：
- [ ] `python play.py --replay replays/某个文件.json` 能完整复放并正确显示结果。
- [ ] `python replay.py verify replays/某个文件.json` 输出 `OK`。
- [ ] 手工篡改回放中一个动作后 verify 返回不一致（负例测试）。

---

### Task 4：隐藏对手池（hidden_bots.py）—— P1，半天

**目标**：落实平台反过拟合原则：**公开榜打明牌 Bot，终榜打隐藏池**。同时是「稳定性」评分的数据来源。

**规格**：

1. 提供 `HIDDEN_POOL`：至少 8 个参数化 `FootsiesBot` 变体，风格要覆盖极端策略：

| 代号 | punish | want_dist | mixup 特征 |
|------|--------|-----------|-----------|
| balanced | 0.8 | 0.155 | 默认均衡 |
| rushdown | 0.7 | 0.10 | 重攻击+投技占比高 |
| turtle | 0.9 | 0.185 | 防御占比高，远距离 |
| thrower | 0.6 | 0.08 | 投技占比极高 |
| punisher | 0.95 | 0.17 | 确反极强，少主动进攻 |
| spacer | 0.5 | 0.175 | 游走拉扯，很少出招 |
| chancer | 0.3 | 0.15 | 大量随机重击 |
| mirror | 0.8 | 0.155 | 与 balanced 同参但不同种子 |

   （mixup 具体概率自行设计，保证各自风格可从对局统计中辨认。）
2. 提供 `PUBLIC_POOL`（公开榜用，1~2 个，参数写入 README 榜单说明）。
3. 池内 Bot 的种子在评测时固定派生（如 `seed = base_seed*100 + bot_index`），保证可复现。
4. `python hidden_bots.py` 自测：每个 Bot 与 `random` 打 20 场，打印各自胜率（都应显著 > 50%，用于确认没有废 Bot）。

**验收**：
- [ ] 自测命令运行通过，8 个 Bot 对 random 胜率全部 > 60%。
- [ ] `arena.py` 能用 `hidden:thrower` 这类名字引用池内 Bot。

---

### Task 5：Agent 接入层（agents/）—— P1，1 天

**目标**：平台「Agent 接入：至少两种方式」的最小实现。统一协议 + 三个适配器。

**规格**：

1. **统一协议**（`agents/base.py`）：

```python
class Agent:
    name: str
    def reset(self, seed: int) -> None: ...
    def act(self, obs: np.ndarray) -> int: ...        # 0~5
    def info(self) -> dict: ...                        # 写入回放 players[].meta
```

   `load_agent(spec: str)` 解析引用：`random` / `footsies` / `hidden:<代号>` / `ppo:<模型路径>` / `file:<py路径>` / `http:<url>`。
2. `ppo_agent.py`：包装 SB3 PPO 模型（示例：`ppo` → `ppo:fighting_ppo.zip`）。
3. `script_agent.py`：包装 `FootsiesBot` 及其变体。注意 `step_both` 下没有「环境侧」Bot，脚本 Bot 需作为正式 Agent 参战（观察自己=哪一侧要处理好，建议协议加 `side` 参数：`reset(seed, side)`，0=左/1=右，Bot 的方向逻辑按 side 翻转）。
4. `http_agent.py`：**LLM/HTTP 决策窗口制**——解决「LLM 无法逐帧决策」：
   - `WindowAgent(agent, window=15)` 包装器：每 `window` 帧调用一次内部 agent，期间每帧重复上次动作（重复轻击=空闲时自动再出，重复防御=持续防御，天然合法）。
   - HTTP 协议：POST `{"obs": [21维], "tick": n, "side": 0, "legal_hint": "..."}` → 期望 `{"action": 0-5}`；超时 5 秒默认返回防御(4)并记录。
   - 附带一个内置假 LLM（`agents/mock_llm.py` 或 http_agent 内实现）：起本地端口、按简单启发式返回动作，用于演示与压测。
   - 记录每次调用的延迟与次数到 `info()`（成本可观测，对应平台的成本预算要求）。
5. `file:<py路径>`：用户上传策略文件的最简形态——python 文件内含 `class Agent(BaseAgent)`。文档写明：MVP 阶段无沙箱，信任运行，沙箱化属于平台工程（本文档范围外）。

**验收**：
- [ ] `python arena.py run ppo footsies` 与 `python arena.py run ppo hidden:mirror` 均可运行。
- [ ] `python arena.py run random file:agents/examples/always_block.py` 可运行（需附带这个示例文件：永远防御的 Bot）。
- [ ] 启动 mock LLM 后 `python arena.py run http:127.0.0.1:8080 random` 可完成一局，回放 meta 中记录了调用次数与平均延迟。
- [ ] 决策窗口演示：`WindowAgent(ppo, window=15)` vs 原生 `ppo` 各打 10 场，打印对比（预期：窗口版胜率下降但仍可观，说明机制可用）。

---

### Task 6：综合评分（score.py）—— P1，半天

**目标**：平台评分规范：**胜率 65% + 速度 20% + 稳定性 15%**（对应平台文档「准确率 60~70%、速度 15~25%、稳定性 10~20%」区间）。

**规格**：

1. `python score.py <agent> [--pool hidden] [--games 24] [--report out.json]`：
   - 对 `HIDDEN_POOL` 每个 Bot 打 N 局（默认每 Bot 3 局、种子派生固定），共 24 局。
   - 三个分项：
     - **胜率分** = 总胜场占比（判平按 0.5 计），权重 0.65；
     - **速度分** = `1 - mean(结束tick / max_ticks)`，权重 0.20；
     - **稳定性分** = `1 - std(逐Bot胜率)`（各 Bot 局数相同时 std∈[0,0.5]，做 `min(1, 2*std)` 归一后取 `1-x`），权重 0.15。
   - 总分 0~100，输出分项明细表 + 总分，可选写 JSON。
2. 附 `python score.py --compare ppo random`：并排对比两个 Agent 的报告（演示用）。

**验收**：
- [ ] `python score.py ppo` 输出报告，总分 ≥ 85（PPO 已知很强，若低于预期需检查实现）。
- [ ] `python score.py random` 总分显著低（预期 < 30）。
- [ ] 报告包含逐 Bot 胜率明细，可看出风格克制关系。

---

### Task 7：锦标赛与 Elo（elo.py）—— P2，半天

**目标**：多 Agent 循环赛 + Elo 榜单，输出排行榜 JSON（平台榜单模块的最小内核）。

**规格**：

1. `python elo.py <agent1> <agent2> ... [--games 10] [--out leaderboard.json]`：
   - 两两循环赛，每对 `--games` 局（种子配对派生，双方轮流左右侧以消除位置偏差——若 Task 5 的 side 逻辑正确，左右应无差异，此处同时是回归测试）。
   - Elo：初始 1500，K=32，按局内逐场更新；胜 1 / 负 0 / 平 0.5。
   - 输出排序榜单：`elo | 胜 | 负 | 平 | 场均tick`，并写 JSON。
2. 至少能跑：`python elo.py ppo random footsies hidden:punisher hidden:turtle always-block(用 file:)`。

**验收**：
- [ ] 上述命令完整运行并产出 `leaderboard.json`。
- [ ] `ppo` 排第一；`random` 排最后（合理性检查）。
- [ ] 同一命令重跑，Elo 结果一致（种子固定）。

---

### Task 8：收尾与文档 —— P2，1 小时

- [ ] `README.md` 更新：新增「竞技场用法」一节（arena/score/elo/play --replay 的最小示例命令）、「Agent 接入协议」一节、公开榜对手说明。
- [ ] 在 README 顶部加一行：`详细开发计划见 PLAN.md（已完成任务见其中勾选状态）`。
- [ ] `PLAN.md` 中勾掉所有已完成项。
- [ ] 最终 commit：`feat: 竞技任务包完成（arena/replay/score/elo/agents）`。

---

## 5. 端到端验收演示（全部完成后的标准剧本）

依次运行并向委托人展示输出：

```bash
python arena.py demo                                   # 1. 无头对局 + 回放
python play.py --replay replays/<最新文件>.json        # 2. 复放一场（含 K.O. 画面）
python score.py ppo                                    # 3. 三维评分报告（隐藏池）
python score.py --compare ppo random                   # 4. 强弱对比
python elo.py ppo random footsies hidden:punisher      # 5. Elo 榜单
# 6.（可选，若 mock LLM 演示成功）
python agents/mock_llm.py &  python arena.py run http:127.0.0.1:8080 random
```

演示叙事（给需求方/投资人讲）：**上传你的 AI → 30 秒得到胜率/速度/稳定性三维分 → 看回放找弱点 → 改完再战 → 上榜**。这正好是平台核心用户循环的最小闭环。

---

## 6. 范围红线（明确不做）

依据平台文档「首期不做」原则，本阶段**禁止**：

1. 不加新招式、新机制（跳/蹲/2D 舞台/飞行道具/连招）——平衡性调优留到有真实对局数据之后。
2. 不做 Web 前端、不接数据库、不做用户系统——那是平台 MVP 的事，本任务包只产出**可被平台调用的内核与格式**。
3. 不做沙箱/权限隔离——`file:` 接入的信任运行问题在 README 中声明即可。
4. 不自我对弈训练新模型——`fighting_ppo.zip` 已够强作为演示；self-play 是下一阶段任务。
5. 不引入新 Python 依赖。
6. 不动 `train.py` 的训练逻辑与超参。

## 7. 已知风险与决策记录

| 风险/决策 | 说明 |
|-----------|------|
| PPO 96% 胜率 vs 隐藏池 | 单一 Bot 上 96% 不代表池上高分；若 score.py 显示 PPO 对 turtle/thrower 显著更弱，**这是特性不是 bug**（风格克制正是卖点），在报告中如实呈现 |
| 左右侧对称性 | `step_both` 中 p1 恒在左；若发现左右胜率系统偏差 > 5%，优先修 env 而非在 elo 里绕过 |
| 重复动作=窗口制的语义 | 「重复上次动作」在硬直中会被忽略，天然安全；此设计已确认，勿改为清零/待机 |
| 回放含动作但不含观察 | 观察由确定性规则推导，无需存储；若未来规则引入随机事件，需升级 format_version |

## 8. 参考资料

- 平台可行性报告 / 实施方案：`/Users/natefier/projects/dabidou/project_documents/AI任务与共创竞技平台*.docx`（`textutil -convert txt -stdout` 阅读；重点节：4.3 反作弊、四 任务与评分、五 技术架构）
- 姊妹项目（回合制前作，热力图可视化思路的来源）：`/Users/natefier/projects/dahua-rl`
- 学术参照：FightingICE（格斗 AI 平台，其「AI 反应延迟」思想已列入本项目路线图第 3 条，本阶段不实现）
