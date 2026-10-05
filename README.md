# 立回斗士（Foosies Fighter）—— RL 格斗 AI

> 详细开发计划见 [PLAN.md](PLAN.md)（已完成任务见其中勾选状态）。
>
> 当前阶段：本项目已封装为 AI 竞技平台的「**任务包**」——
> 无头对局执行 / 确定性回放 / 隐藏对手池评分 / Elo 榜单 / 多方式 Agent 接入。

用 PPO 学会 1D 极简格斗的两个核心博弈：**立回（距离管控）** 与 **打投择（打击/投技/防御猜拳）**。

从姊妹项目 [dahua-rl]（回合制战斗）进化而来：从「回合制」走向「帧级实时」。

## 安装依赖

```bash
pip install gymnasium stable-baselines3 torch numpy pygame
```

## 快速开始（单机版）

```bash
python fighting_env.py   # 冒烟测试：随机策略 vs 立回机器人
python train.py          # 训练 PPO（CPU 几分钟）
python play.py           # 观看 AI vs 立回机器人
```

## 竞技场用法（任务包）

```bash
# 无头对局 + 回放（种子固定，可确定性重放）
python arena.py run ppo footsies --seed 0
python arena.py batch ppo hidden:turtle --games 20
python arena.py demo

# 复放一场（空格暂停，+/- 调速，结尾显示结果）
python play.py --replay replays/<回放文件>.json

# 回放校验（种子+动作序列重放，逐帧 HP 轨迹比对，防篡改）
python replay.py verify replays/<回放文件>.json

# 综合评分：胜率 65% + 速度 20% + 稳定性 15%（隐藏池 8 风格化对手）
python score.py ppo
python score.py --compare ppo random

# 循环赛 + Elo 榜单（含对阵明细矩阵）
python elo.py ppo random footsies hidden:punisher hidden:turtle \
    file:agents/examples/always_block.py

# 隐藏池自检（每个风格 Bot 对 random 的胜率）
python hidden_bots.py

# LLM/HTTP Agent 演示（决策窗口制：每 15 帧决策一次）
python agents/mock_llm.py &              # 终端 1：假 LLM 服务
python arena.py run http:127.0.0.1:8080 random   # 终端 2
python -m agents.base                    # 窗口宽度退化曲线演示
```

演示叙事：**上传你的 AI → 30 秒得到胜率/速度/稳定性三维分 → 看回放找弱点
→ 改完再战 → 上榜**。（实测案例：PPO 对默认 Bot 96% 胜率，但被隐藏池的
投技狂/压制流克制、对纯防御 Bot 十连平——看回放即可定位「不会破防」的盲区，
这正是「稳定性」维度的价值。）

## Agent 接入协议

所有 Agent 通过 `load_agent(spec)` 名字引用，统一协议（`agents/base.py`）：

```python
class Agent:
    name: str
    def reset(self, seed: int, side: int = 0) -> None: ...  # 每局开始；side 0=左/1=右
    def act(self, obs) -> int: ...      # 每帧（或每个决策窗口）→ 0~5
    def info(self) -> dict: ...         # 写入回放 players[].meta
```

Agent 只能看到**自己视角**的 21 维观测（右侧为镜像坐标），与人类玩家等价。

| 引用格式 | 说明 |
|---|---|
| `random` | 内置随机策略（基线） |
| `footsies` | 默认立回机器人 |
| `hidden:<代号>` | 隐藏池风格化 Bot（见 `hidden_bots.py`） |
| `ppo[:<模型路径>]` | SB3 PPO 模型（默认 `fighting_ppo.zip`） |
| `file:<py路径>` | 用户策略文件，含 `class Agent`（见 `agents/examples/`） |
| `http:<url>` | HTTP/LLM Agent，自动套 15 帧决策窗口 |

**决策窗口制**（LLM 无法逐帧决策的解法）：每 15 帧调用一次远端服务，
期间逐帧重复上次动作（重复防御=持续防御、重复轻击=空闲自动再出、
硬直中重复指令被环境忽略——天然合法）。

**file: 接入声明**：MVP 阶段无沙箱，策略文件被信任执行；沙箱化属于平台工程范围。

## 规则包（可 fork 的规则变体）

帧数据与常量外置于 [rules/default.json](rules/default.json)（schema v1）。
改一个数字 = 一个规则变体：

```bash
python arena.py run ppo footsies --rules rules/default.json
```

回放记录规则包内容哈希（sha256）；规则变更后旧回放校验将失败（预期行为）。

## 公开榜对手（参数公示）

公开榜（PUBLIC_POOL）使用两个明牌 Bot，参数如下：

| 代号 | punish | want_dist | mixup（throw/light/heavy/block/back） |
|---|---|---|---|
| public-balanced | 0.8 | 0.155 | 原版均衡（throw 15%仅贴身 / light 35% / heavy 12% / block 20% / back 18%） |
| public-chancer | 0.3 | 0.15 | 10%/25%/45%/10%/10% |

终榜使用 8 个隐藏池 Bot（`hidden_bots.py`：龟防流/投技狂/确反大师/游走流等），
风格覆盖极端策略，专门检验泛化性与稳定性（反过拟合）。

## 机制

| 要素 | 实现 |
|------|------|
| 舞台 | 一维直线 0~1，距离决定一切 |
| 时间 | 每步 = 1 帧（60fps 语义），双方**同时**决策 |
| 帧数据 | 每招分 启动/判定/收招；收招硬直中被惩罚是大亏 |
| 猜拳环 | 打击克投技 · 投技克防御 · 防御克打击 |
| 互拼 | 双方同时进入判定帧 → Trade，各吃伤害 |

### 招式表

| 招式 | 启动 | 判定 | 收招 | 距离 | 伤害 | 备注 |
|------|-----|-----|-----|------|-----|------|
| 轻击 | 3 | 2 | 5 | 0.13 | 7 | 快而短，立回主力 |
| 重击 | 8 | 3 | 14 | 0.19 | 18 | 慢而长，确反用 |
| 投技 | 5 | 2 | 10 | 0.07 | 14 | 防御不能，但怕打击 |
| 防御 | - | - | - | - | 15%削减 | 需持续按住；被投技破坏 |

## 强化学习设计

| 要素 | 实现 |
|------|------|
| 状态 | 21 维：双方血量/距离/位置/状态one-hot/是否防御/硬直进度 |
| 动作 | 离散 6 个 |
| 奖励 | 伤害 ±0.08/点，K.O. ±10，超时按血量差 ±3 |
| 对手 | FootsiesBot：维持交战距离 + 概率读取状态确反/破防 |
| 算法 | PPO（stable-baselines3） |

## 路线图

1. **立回热力图**：策略在 (距离 × 敌方状态) 平面的切片 → 看 AI 的距离管控哲学
2. **自我对弈**：用 `step_both` 接口做 PPO vs PPO，加对手池
3. **公平性**：给 AI 加反应延迟（如 8~13 帧），模拟人类
4. **扩展**：2D 舞台（跳/蹲）、更多招式（升龙、飞行道具、确反康）
5. **迁移**：gym-fightingice（学术平台）或 gym-retro 街霸2
