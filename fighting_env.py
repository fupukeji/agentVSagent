"""
立回斗士（Foosies Fighter）—— 1D 极简格斗环境（Gymnasium 接口）
================================================================
致敬街霸类格斗的两个核心博弈：
1. 立回（Footsies）：距离管控，用前进/后撤诱骗对方出招
2. 打投择（Mixup）：打击 克 投技，投技 克 防御，防御 克 打击

机制要点
--------
- 舞台是一维直线 0~1，每步 = 1 帧（60fps 语义），双方同时决策
- 每招有帧数据：启动帧 → 判定帧 → 收招帧；收招硬直中被惩罚是大亏
- 防御可挡打击（仅吃 15% 削减伤害）但会被投技破坏
- 投技对「防御中 / 待机 / 收招 / 防御硬直」的目标生效；
  对方正在出招（启动/判定）时投技落空 → 打击克投技
- 双方同时进入判定帧时互拼（Trade），各吃伤害
- 时间到按剩余血量判定胜负

规则包
------
帧数据与常量外置于 rules/default.json（平台「规则包」schema v1）；
`FightingEnv` / `FootsiesBot` 均可传 `rules=dict|路径` 自定义。
文件缺失时回退到内置常量并打 warning（保证向后兼容）。

动作空间（6）
------------
0 后撤 | 1 前进 | 2 轻击 | 3 重击 | 4 防御 | 5 投技
"""

import json
import os
import sys

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ---------------------------------------------------------------
# 帧数据（Frame Data）
# ---------------------------------------------------------------
class Move:
    """一招的帧数据：启动 / 判定 / 收招"""

    def __init__(self, name, startup, active, recovery, reach, damage,
                 kind, hitstun=8, blockstun=4):
        self.name = name
        self.startup = startup    # 启动帧：出招到产生判定
        self.active = active      # 判定帧：能命中对方的窗口
        self.recovery = recovery  # 收招帧：硬直，可被惩罚
        self.reach = reach        # 判定距离
        self.damage = damage
        self.kind = kind          # "strike" 打击 / "throw" 投技
        self.hitstun = hitstun    # 命中后对方硬直
        self.blockstun = blockstun  # 被防御后对方硬直

    @property
    def total(self):
        return self.startup + self.active + self.recovery

    def __repr__(self):
        return (f"{self.name}(启动{self.startup}/判定{self.active}/收招{self.recovery} "
                f"距离{self.reach:.2f} 伤害{self.damage})")


MOVES = [
    Move("后撤", 0, 0, 0, 0.0, 0, "strike"),                     # 0 移动
    Move("前进", 0, 0, 0, 0.0, 0, "strike"),                     # 1 移动
    Move("轻击", 3, 2, 5, 0.13, 7, "strike", hitstun=9),         # 2 快而短
    Move("重击", 8, 3, 14, 0.19, 18, "strike", hitstun=15),      # 3 慢而长，大硬直
    Move("防御", 0, 0, 0, 0.0, 0, "strike"),                     # 4 持续按住才防御
    Move("投技", 5, 2, 10, 0.07, 14, "throw", hitstun=18),       # 5 防御不能，但距离极短
]
ACT_NAMES = [m.name for m in MOVES]

WALK_F, WALK_B = 0.009, 0.007   # 前进/后撤速度（每帧位移）
CHIP = 0.15                     # 防御时承受的削减伤害比例
MIN_GAP = 0.05                  # 双方最小间距（不可穿身）
STAGE = (0.0, 1.0)
HIT_PUSH, BLOCK_PUSH = 0.030, 0.018  # 命中/被防后的击退

# 角色状态机
S_IDLE, S_STARTUP, S_ACTIVE, S_RECOVERY, S_BLOCKSTUN, S_HITSTUN = range(6)
STATE_NAMES = ["待机", "启动", "判定", "收招", "防御硬直", "受击硬直"]

# 动作索引
A_BACK, A_FWD, A_LIGHT, A_HEAVY, A_BLOCK, A_THROW = range(6)


# ---------------------------------------------------------------
# 规则包（Rules Pack）—— 平台「规则包」schema v1
# ---------------------------------------------------------------
RULES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rules")
DEFAULT_RULES_PATH = os.path.join(RULES_DIR, "default.json")

# 内置常量：与 rules/default.json 完全一致，文件缺失时回退用
BUILTIN_RULES = {
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
         "recovery": 10, "reach": 0.07, "damage": 14, "hitstun": 18, "blockstun": 0},
    ],
}


class Rules:
    """解析后的规则包：环境 / Bot / 角色共享的常量与帧数据。"""

    def __init__(self, data: dict):
        if "moves" not in data:
            raise ValueError("规则包缺少 moves 字段")
        self.raw = data
        self.name = data.get("name", "custom")
        self.version = int(data.get("version", 1))
        walk = data.get("walk", {})
        self.walk_f = float(walk.get("forward", WALK_F))
        self.walk_b = float(walk.get("back", WALK_B))
        self.chip = float(data.get("chip", CHIP))
        self.min_gap = float(data.get("min_gap", MIN_GAP))
        push = data.get("push", {})
        self.push_hit = float(push.get("hit", HIT_PUSH))
        self.push_block = float(push.get("block", BLOCK_PUSH))
        self.max_hp = int(data.get("max_hp", 100))
        self.max_ticks = int(data.get("max_ticks", 900))
        n = len(data["moves"])
        moves: list = [None] * n
        for m in data["moves"]:
            mv = Move(m["name"], m.get("startup", 0), m.get("active", 0),
                      m.get("recovery", 0), m.get("reach", 0.0),
                      m.get("damage", 0), m.get("kind", "strike"),
                      hitstun=m.get("hitstun", 8), blockstun=m.get("blockstun", 4))
            mv.slot = int(m.get("slot", -1))
            if not (0 <= mv.slot < n) or moves[mv.slot] is not None:
                raise ValueError(f"非法或重复的 slot: {mv.slot} ({mv.name})")
            moves[mv.slot] = mv
        if any(m is None for m in moves):
            raise ValueError("moves 的 slot 不连续")
        self.moves = moves

    def __repr__(self):
        return f"Rules(name={self.name!r}, version={self.version}, moves={len(self.moves)})"


def load_rules(rules=None) -> Rules:
    """rules: dict → 直接解析；str/Path → JSON 路径；None → rules/default.json
    （文件缺失时回退内置常量并打 warning，保证向后兼容）。"""
    if rules is None:
        path = DEFAULT_RULES_PATH
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return Rules(json.load(f))
        print("[warn] 未找到 rules/default.json，回退到内置常量", file=sys.stderr)
        return Rules(BUILTIN_RULES)
    if isinstance(rules, (str, os.PathLike)):
        with open(rules, encoding="utf-8") as f:
            return Rules(json.load(f))
    return Rules(rules)


# 模块级默认规则（default.json 存在时与其一致，否则为内置常量）
_DEFAULT_RULES = load_rules()


# ---------------------------------------------------------------
# 角色
# ---------------------------------------------------------------
class Fighter:
    def __init__(self, name, rules=None):
        r = rules if isinstance(rules, Rules) else _DEFAULT_RULES
        self.name = name
        self.max_hp = r.max_hp
        self.moves = r.moves          # 出招表（与规则包绑定）
        self.hp = self.max_hp
        self.x = 0.5
        self.state = S_IDLE
        self.frames_left = 0   # 当前状态剩余帧
        self.move = None       # 当前招式
        self.has_hit = False   # 本招是否已命中（防多段）
        self.blocking = False
        # 每帧结算用（奖励计算）
        self.hit_taken = 0
        self.dealt = 0

    def busy(self):
        return self.state in (S_STARTUP, S_ACTIVE, S_RECOVERY,
                              S_BLOCKSTUN, S_HITSTUN)

    def start(self, action):
        self.move = self.moves[action]
        self.has_hit = False
        self.state = S_STARTUP
        self.frames_left = self.move.startup


# ---------------------------------------------------------------
# 脚本对手：立回机器人
# ---------------------------------------------------------------
class FootsiesBot:
    """
    会打立回的脚本对手：
    - 维持理想交战距离（刚好贴着轻击判定外沿）
    - 概率性读取玩家状态：收招→重击确反、出招→拉防、防御→投技破防
    - 交战距离内随机打投择
    """

    def __init__(self, punish=0.8, want_dist=0.155, mixup=None, seed=None,
                 rules=None, band=0.035):
        r = rules if isinstance(rules, Rules) else _DEFAULT_RULES
        self.rules = r
        self.moves = r.moves
        self.punish = punish          # 读取成功率（模拟人类反应，<1 更真实）
        self.want_dist = want_dist    # 理想交战距离（立回核心参数）
        self.band = band              # 立回滞回带半径（默认 0.035 保持原行为）
        self.mixup = mixup            # 打投择概率表；None = 保持原版行为
        if mixup is not None:
            self._mix = self._parse_mixup(mixup)
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _parse_mixup(mixup: dict):
        """校验并补全打投择概率表：{"throw"/"light"/"heavy"/"block"/"back": p}。
        未给出的键按 0 处理，剩余概率全部补给 light（throw 仍受投技距离门控）。"""
        mix = {"throw": 0.0, "light": 0.0, "heavy": 0.0, "block": 0.0, "back": 0.0}
        unknown = set(mixup) - set(mix)
        if unknown:
            raise ValueError(f"mixup 含未知键: {unknown}")
        mix.update({k: float(v) for k, v in mixup.items()})
        others = mix["throw"] + mix["heavy"] + mix["block"] + mix["back"]
        if others + mix["light"] > 1.0 + 1e-9:
            raise ValueError(f"mixup 概率之和超过 1: {mixup}")
        mix["light"] += max(0.0, 1.0 - others)   # 剩余概率给 light
        return mix

    def set_rules(self, rules):
        """更换规则包（供 FightingEnv 在加载自定义规则时同步自身）。"""
        r = rules if isinstance(rules, Rules) else load_rules(rules)
        self.rules = r
        self.moves = r.moves

    def act(self, opp, me):
        d = abs(me.x - opp.x)        # 交战距离（与站位无关，side 0/1 通用）
        mv_light, mv_heavy, mv_throw = (self.moves[A_LIGHT],
                                        self.moves[A_HEAVY], self.moves[A_THROW])
        rng = self.rng.random

        if me.busy():
            return A_BLOCK            # 硬直中指令无效

        # ---- 概率性读取（模拟人类反应延迟/失误）----
        if rng() < self.punish:
            if opp.state == S_RECOVERY and d <= mv_heavy.reach:
                return A_HEAVY        # 确反：惩罚收招
            if opp.state == S_STARTUP and d <= mv_heavy.reach:
                return A_BLOCK        # 对方出招 → 拉防
            if (opp.blocking or opp.state == S_BLOCKSTUN) and d <= mv_throw.reach:
                return A_THROW        # 破防投

        # ---- 立回：维持理想交战距离 ----
        if d > self.want_dist + self.band:
            return A_FWD
        if d < self.want_dist - self.band:
            return A_BACK

        # ---- 交战距离内的打投择 ----
        r = rng()
        if self.mixup is None:        # 原版默认行为（向后兼容）
            if d < 0.09 and r < 0.15:
                return A_THROW
            if r < 0.50:
                return A_LIGHT
            if r < 0.62:
                return A_HEAVY
            if r < 0.82:
                return A_BLOCK
            return A_BACK
        m = self._mix                 # 参数化打投择（投技仍受距离门控）
        if d <= mv_throw.reach + 0.02 and r < m["throw"]:
            return A_THROW
        if r < m["throw"] + m["light"]:
            return A_LIGHT
        if r < m["throw"] + m["light"] + m["heavy"]:
            return A_HEAVY
        if r < m["throw"] + m["light"] + m["heavy"] + m["block"]:
            return A_BLOCK
        return A_BACK


# ---------------------------------------------------------------
# 环境
# ---------------------------------------------------------------
class FightingEnv(gym.Env):
    """
    我方（p1，AI 控制，在左） vs 机器（p2，FootsiesBot，在右）。
    step(action)：AI 出招，机器人同时出招，结算一帧。
    step_both(a1, a2)：自我对弈接口（留给 PPO vs PPO）。
    """

    metadata = {"render_modes": ["human"]}
    NUM_ACTIONS = 6
    OBS_DIM = 21

    def __init__(self, max_ticks=None, bot=None, rules=None):
        super().__init__()
        self.rules = load_rules(rules)
        # 显式传入的 max_ticks 覆盖规则包（None 时用规则包的 max_ticks）
        self.max_ticks = self.rules.max_ticks if max_ticks is None else int(max_ticks)
        self.bot = bot if bot is not None else FootsiesBot(rules=self.rules)
        if hasattr(self.bot, "set_rules"):   # 自带 Bot 同步本环境规则包
            self.bot.set_rules(self.rules)
        self.action_space = spaces.Discrete(self.NUM_ACTIONS)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, (self.OBS_DIM,), np.float32)
        self.p1 = Fighter("AI", rules=self.rules)
        self.p2 = Fighter("机器", rules=self.rules)
        self.t = 0

    # ---------- Gym API ----------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        for f in (self.p1, self.p2):
            f.hp = f.max_hp
            f.state = S_IDLE
            f.frames_left = 0
            f.move = None
            f.has_hit = False
            f.blocking = False
        self.p1.x, self.p2.x = 0.38, 0.62
        self.t = 0
        if seed is not None:
            self.bot.rng = np.random.default_rng(seed + 7)
        return self._obs(), {}

    def step(self, action):
        a_p = int(action)
        a_e = self.bot.act(self.p1, self.p2)
        return self.step_both(a_p, a_e)

    def step_both(self, a_p, a_e):
        p, e = self.p1, self.p2
        r = self.rules
        p.hit_taken = e.hit_taken = 0
        p.dealt = e.dealt = 0

        # ---- 1. 状态推进（各状态倒计时）----
        nxt = {S_STARTUP: S_ACTIVE, S_ACTIVE: S_RECOVERY, S_RECOVERY: S_IDLE,
               S_BLOCKSTUN: S_IDLE, S_HITSTUN: S_IDLE}
        for f in (p, e):
            if f.busy():
                f.frames_left -= 1
                if f.frames_left <= 0:
                    f.state = nxt[f.state]
                    if f.state == S_ACTIVE:
                        f.frames_left = f.move.active
                    elif f.state == S_RECOVERY:
                        f.frames_left = f.move.recovery
                    else:  # 回到待机
                        f.frames_left = 0
                        f.move = None

        # ---- 2. 接受新指令（硬直中指令作废）----
        for f, a in ((p, a_p), (e, a_e)):
            if f.busy():
                continue
            if a == A_BLOCK:
                f.blocking = True
                continue
            f.blocking = False
            if a in (A_LIGHT, A_HEAVY, A_THROW):
                f.start(a)
            elif a == A_FWD:
                f.x += r.walk_f if f is p else -r.walk_f
            elif a == A_BACK:
                f.x += -r.walk_b if f is p else r.walk_b

        # ---- 3. 碰撞与舞台边界 ----
        lo, hi = STAGE
        p.x = float(np.clip(p.x, lo, hi))
        e.x = float(np.clip(e.x, lo, hi))
        if e.x - p.x < r.min_gap:
            mid = (p.x + e.x) / 2
            p.x = max(lo, mid - r.min_gap / 2)
            e.x = min(hi, mid + r.min_gap / 2)
        d = e.x - p.x

        # ---- 4. 打击判定（先收集再结算 → 支持互拼 Trade）----
        strikes = []
        for atk, dfn in ((p, e), (e, p)):
            if (atk.state == S_ACTIVE and not atk.has_hit
                    and atk.move.kind == "strike" and d <= atk.move.reach):
                strikes.append((atk, dfn))
        for atk, dfn in strikes:
            atk.has_hit = True
            if dfn.blocking:
                dmg = max(1, int(atk.move.damage * r.chip))
                dfn.state = S_BLOCKSTUN
                dfn.frames_left = atk.move.blockstun
                self._push(dfn, r.push_block)
            else:
                dmg = atk.move.damage
                dfn.state = S_HITSTUN
                dfn.frames_left = atk.move.hitstun
                self._push(dfn, r.push_hit)
            dfn.hp = max(0, dfn.hp - dmg)
            dfn.hit_taken += dmg
            atk.dealt += dmg

        # ---- 5. 投技判定（在打击之后结算）----
        for atk, dfn in ((p, e), (e, p)):
            if (atk.state == S_ACTIVE and not atk.has_hit
                    and atk.move.kind == "throw" and d <= atk.move.reach):
                # 只对 防御中/待机/收招/防御硬直 的目标生效
                if dfn.blocking or dfn.state in (S_IDLE, S_RECOVERY, S_BLOCKSTUN):
                    atk.has_hit = True
                    dmg = atk.move.damage
                    dfn.hp = max(0, dfn.hp - dmg)
                    dfn.hit_taken += dmg
                    atk.dealt += dmg
                    dfn.state = S_HITSTUN
                    dfn.frames_left = atk.move.hitstun
                    dfn.blocking = False
                    self._push(dfn, r.push_hit)

        # ---- 6. 回合结束与奖励 ----
        self.t += 1
        term = p.hp <= 0 or e.hp <= 0
        trunc = (not term) and self.t >= self.max_ticks
        reward = (p.dealt - p.hit_taken) * 0.08
        if term or trunc:
            if e.hp <= 0 < p.hp:
                reward += 10.0
            elif p.hp <= 0 < e.hp:
                reward += -10.0
            else:
                reward += 3.0 * np.sign(p.hp - e.hp)  # 超时按血量判

        info = {"tick": self.t, "dist": d,
                "p_state": STATE_NAMES[p.state], "e_state": STATE_NAMES[e.state],
                "p_dealt": p.dealt, "e_dealt": e.dealt}
        return self._obs(), reward, term, trunc, info

    # ---------- 内部 ----------
    def obs_for(self, side: int):
        """从某一侧视角返回观测（side=0 左 / 1 右）。
        右侧为镜像视角：交换双方字段并翻转坐标，使两个 Agent
        看到语义对称的 obs（「前进」始终指向对手）。"""
        if side == 0:
            return self._obs()
        p, e = self.p1, self.p2
        mx, ox = 1.0 - e.x, 1.0 - p.x   # 镜像坐标
        return np.concatenate([
            np.array([e.hp / e.max_hp, p.hp / p.max_hp,
                      ox - mx, mx, ox], dtype=np.float32),
            self._onehot(e), self._onehot(p),
            np.array([1.0 * e.blocking, 1.0 * p.blocking,
                      self._busy_frac(e), self._busy_frac(p)], dtype=np.float32),
        ]).astype(np.float32)

    def outcome(self) -> dict:
        """终局判定：winner ∈ 0/1/None；result ∈ win0|win1|timeout0|timeout1|draw。"""
        p, e = self.p1, self.p2
        if p.hp <= 0 or e.hp <= 0:
            if p.hp <= 0 and e.hp <= 0:
                winner, result = None, "draw"
            elif e.hp <= 0:
                winner, result = 0, "win0"
            else:
                winner, result = 1, "win1"
        elif p.hp > e.hp:
            winner, result = 0, "timeout0"
        elif p.hp < e.hp:
            winner, result = 1, "timeout1"
        else:
            winner, result = None, "draw"
        return {"winner": winner, "hp": [int(p.hp), int(e.hp)],
                "ticks": int(self.t), "result": result}

    def _push(self, dfn, amount):
        sign = 1.0 if dfn is self.p2 else -1.0
        dfn.x = float(np.clip(dfn.x + sign * amount, *STAGE))

    @staticmethod
    def _onehot(f):
        v = np.zeros(6, dtype=np.float32)
        v[f.state] = 1.0
        return v

    @staticmethod
    def _busy_frac(f):
        if f.move and f.busy() and f.move.total > 0:
            return min(1.0, f.frames_left / f.move.total)
        return 0.0

    def _obs(self):
        p, e = self.p1, self.p2
        return np.concatenate([
            np.array([p.hp / p.max_hp, e.hp / e.max_hp,
                      e.x - p.x, p.x, e.x], dtype=np.float32),
            self._onehot(p), self._onehot(e),
            np.array([1.0 * p.blocking, 1.0 * e.blocking,
                      self._busy_frac(p), self._busy_frac(e)], dtype=np.float32),
        ]).astype(np.float32)


if __name__ == "__main__":
    # 冒烟测试：随机策略 vs 立回机器人
    env = FightingEnv()
    wins = losses = draws = 0
    for ep in range(20):
        obs, _ = env.reset(seed=ep)
        assert obs.shape == (FightingEnv.OBS_DIM,)
        done, ret = False, 0.0
        while not done:
            obs, r, term, trunc, info = env.step(env.action_space.sample())
            ret += r
            done = term or trunc
        if env.p2.hp <= 0 < env.p1.hp:
            wins += 1
        elif env.p1.hp <= 0 < env.p2.hp:
            losses += 1
        else:
            draws += 1
        print(f"第{ep + 1:2d}场 AI HP{env.p1.hp:3d} vs 机器 HP{env.p2.hp:3d} "
              f"回合回报 {ret:+7.2f}")
    print(f"\n随机策略 vs 立回机器人：{wins}胜 {losses}负 {draws}判")
