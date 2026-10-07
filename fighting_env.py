"""
立回斗士 2D（Foosies Fighter 2D）—— 跳跃/蹲伏扩展版（Gymnasium 接口）
================================================================
在 1D 立回基础上新增纵轴博弈（跳跃/蹲伏），猜拳环从 3 元素扩展到 6 元素。

动作空间（10）
------------
0 后撤 | 1 前进 | 2 轻击(高) | 3 重击(高) | 4 站防 | 5 投技
6 跳跃(自动跳攻) | 7 蹲伏 | 8 下段轻击(低) | 9 蹲防

高低段判定矩阵
--------------
| 攻击\\防守 | 站立 | 站防 | 蹲伏 | 蹲防 | 空中 |
| 轻击(高)   | HIT  | BLOCK| MISS | BLOCK| HIT  |
| 重击(高)   | HIT  | BLOCK| MISS | BLOCK| HIT  |
| 投技       | HIT  | HIT  | HIT  | HIT  | MISS |
| 下段(低)   | HIT  | LEAK | HIT  | BLOCK| MISS |
| 跳攻(顶)   | HIT  | BLOCK| HIT  | LEAK | TRADE|
"""

import json
import math
import os
import sys

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ---------------------------------------------------------------
# 帧数据
# ---------------------------------------------------------------
class Move:
    def __init__(self, name, startup, active, recovery, reach, damage,
                 kind, hitstun=8, blockstun=4, height="mid", slot=0):
        self.name = name
        self.startup = startup
        self.active = active
        self.recovery = recovery
        self.reach = reach
        self.damage = damage
        self.kind = kind          # strike / throw / jump / move / guard / crouch_guard
        self.hitstun = hitstun
        self.blockstun = blockstun
        self.height = height      # "high" / "low" / "mid" / "air"(overhead)
        self.slot = slot

    @property
    def total(self):
        return self.startup + self.active + self.recovery

    def __repr__(self):
        return (f"{self.name}(启{self.startup}/判{self.active}/收{self.recovery} "
                f"距{self.reach:.2f} 伤{self.damage} 段{'高' if self.height=='high' else '低' if self.height=='low' else self.height})")


# 动作索引
A_BACK, A_FWD, A_LIGHT, A_HEAVY, A_BLOCK, A_THROW = range(6)
A_JUMP, A_CROUCH, A_LOW, A_CROUCH_BLOCK = 6, 7, 8, 9
NUM_ACTIONS = 10
ACT_NAMES = ["后撤", "前进", "轻击", "重击", "站防", "投技",
             "跳跃", "蹲伏", "下段", "蹲防"]

# 角色状态机
(S_IDLE, S_STARTUP, S_ACTIVE, S_RECOVERY,
 S_BLOCKSTUN, S_HITSTUN, S_JUMP_RISE, S_JUMP_FALL,
 S_CROUCH, S_LANDING) = range(10)
STATE_NAMES = ["待机", "启动", "判定", "收招", "防御硬直", "受击硬直",
               "跳升", "跳落", "蹲伏", "落地"]
NUM_STATES = 10

# 常量
WALK_F, WALK_B = 0.009, 0.007
CHIP = 0.15
MIN_GAP = 0.05
STAGE = (0.0, 1.0)
HIT_PUSH, BLOCK_PUSH = 0.030, 0.018
MAX_HP = 100
MAX_TICKS = 900
JUMP_FRAMES = 20
JUMP_HEIGHT = 0.06
JUMP_FWD = 0.005
JUMP_BACK = 0.003
OBS_DIM = 29

# 2D 默认规则（与 rules/default2d.json 同步）
BUILTIN_RULES_2D = {
    "name": "default2d", "version": 2,
    "walk": {"forward": WALK_F, "back": WALK_B},
    "chip": CHIP, "min_gap": MIN_GAP,
    "push": {"hit": HIT_PUSH, "block": BLOCK_PUSH},
    "max_hp": MAX_HP, "max_ticks": MAX_TICKS,
    "jump": {"frames": JUMP_FRAMES, "height": JUMP_HEIGHT,
             "fwd": JUMP_FWD, "back": JUMP_BACK},
    "moves": [
        {"slot": 0, "name": "后撤", "kind": "move", "dir": "back"},
        {"slot": 1, "name": "前进", "kind": "move", "dir": "forward"},
        {"slot": 2, "name": "轻击", "kind": "strike", "height": "high",
         "startup": 3, "active": 2, "recovery": 5,
         "reach": 0.13, "damage": 7, "hitstun": 9, "blockstun": 4},
        {"slot": 3, "name": "重击", "kind": "strike", "height": "high",
         "startup": 8, "active": 3, "recovery": 14,
         "reach": 0.19, "damage": 18, "hitstun": 15, "blockstun": 4},
        {"slot": 4, "name": "站防", "kind": "guard", "height": "high"},
        {"slot": 5, "name": "投技", "kind": "throw", "height": "mid",
         "startup": 5, "active": 2, "recovery": 10,
         "reach": 0.07, "damage": 14, "hitstun": 18, "blockstun": 0},
        {"slot": 6, "name": "跳跃", "kind": "jump", "height": "air",
         "startup": JUMP_FRAMES // 2, "active": 8, "recovery": 4,
         "reach": 0.10, "damage": 10, "hitstun": 12, "blockstun": 6},
        {"slot": 7, "name": "蹲伏", "kind": "crouch"},
        {"slot": 8, "name": "下段", "kind": "strike", "height": "low",
         "startup": 4, "active": 2, "recovery": 6,
         "reach": 0.15, "damage": 5, "hitstun": 6, "blockstun": 3},
        {"slot": 9, "name": "蹲防", "kind": "crouch_guard", "height": "low"},
    ],
}

# 1D 兼容规则（老版本）
BUILTIN_RULES = {
    "name": "default", "version": 1,
    "walk": {"forward": WALK_F, "back": WALK_B},
    "chip": CHIP, "min_gap": MIN_GAP,
    "push": {"hit": HIT_PUSH, "block": BLOCK_PUSH},
    "max_hp": MAX_HP, "max_ticks": MAX_TICKS,
    "moves": BUILTIN_RULES_2D["moves"][:6],  # 只取前 6 个动作
}

RULES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rules")
DEFAULT_RULES_PATH = os.path.join(RULES_DIR, "default.json")


class Rules:
    def __init__(self, data):
        d = dict(data)
        self.raw = d
        self.name = d.get("name", "custom")
        self.version = int(d.get("version", 1))
        walk = d.get("walk", {})
        self.walk_f = float(walk.get("forward", WALK_F))
        self.walk_b = float(walk.get("back", WALK_B))
        self.chip = float(d.get("chip", CHIP))
        self.min_gap = float(d.get("min_gap", MIN_GAP))
        push = d.get("push", {})
        self.push_hit = float(push.get("hit", HIT_PUSH))
        self.push_block = float(push.get("block", BLOCK_PUSH))
        self.max_hp = int(d.get("max_hp", MAX_HP))
        self.max_ticks = int(d.get("max_ticks", MAX_TICKS))
        jmp = d.get("jump", {})
        self.jump_frames = int(jmp.get("frames", JUMP_FRAMES))
        self.jump_height = float(jmp.get("height", JUMP_HEIGHT))
        self.jump_fwd = float(jmp.get("fwd", JUMP_FWD))
        self.jump_back = float(jmp.get("back", JUMP_BACK))
        n = len(d.get("moves", []))
        moves = [None] * max(n, 6)
        for m in d.get("moves", []):
            mv = Move(m["name"], m.get("startup", 0), m.get("active", 0),
                      m.get("recovery", 0), m.get("reach", 0.0),
                      m.get("damage", 0), m.get("kind", "strike"),
                      hitstun=m.get("hitstun", 8), blockstun=m.get("blockstun", 4),
                      height=m.get("height", "mid"), slot=m.get("slot", -1))
            if 0 <= mv.slot < len(moves):
                moves[mv.slot] = mv
        # 填充未定义的槽位为安全默认
        for i in range(6):
            if moves[i] is None:
                moves[i] = Move(ACT_NAMES[i], 0, 0, 0, 0, 0, "move", slot=i)
        self.moves = moves
        self.num_actions = len([m for m in moves if m is not None])

    def __repr__(self):
        return f"Rules({self.name!r}, v{self.version}, {self.num_actions} moves)"


def load_rules(rules=None):
    if rules is None:
        if os.path.exists(DEFAULT_RULES_PATH):
            with open(DEFAULT_RULES_PATH, encoding="utf-8") as f:
                return Rules(json.load(f))
        print("[warn] 未找到 rules/default.json，回退到内置常量", file=sys.stderr)
        return Rules(BUILTIN_RULES)
    if isinstance(rules, (str, os.PathLike)):
        with open(rules, encoding="utf-8") as f:
            return Rules(json.load(f))
    return Rules(rules)


_DEFAULT_RULES = load_rules()


# ---------------------------------------------------------------
# 角色
# ---------------------------------------------------------------
class Fighter:
    def __init__(self, name, rules=None):
        r = rules if isinstance(rules, Rules) else _DEFAULT_RULES
        self.name = name
        self.max_hp = r.max_hp
        self.moves = r.moves
        self.hp = self.max_hp
        self.x = 0.5
        self.y = 0.0            # 2D: 0=地面, >0=空中
        self.state = S_IDLE
        self.frames_left = 0
        self.move = None
        self.has_hit = False
        self.blocking = False
        self.block_low = False   # 蹲防
        self.crouching = False
        self.jump_frame = 0      # 跳跃帧计数
        self.jump_dir = 1        # 跳跃水平方向（1=前 / -1=后）
        self.hit_taken = 0
        self.dealt = 0

    def busy(self):
        return self.state in (S_STARTUP, S_ACTIVE, S_RECOVERY,
                              S_BLOCKSTUN, S_HITSTUN, S_JUMP_RISE,
                              S_JUMP_FALL, S_LANDING)

    def airborne(self):
        return self.state in (S_JUMP_RISE, S_JUMP_FALL)

    def start(self, action):
        self.move = self.moves[action]
        self.has_hit = False
        self.state = S_STARTUP
        self.frames_left = self.move.startup


# ---------------------------------------------------------------
# 脚本对手
# ---------------------------------------------------------------
class FootsiesBot:
    """1D 立回机器人（2D 兼容：不懂跳/蹲但能打地面战）"""

    def __init__(self, punish=0.8, want_dist=0.155, mixup=None, seed=None,
                 rules=None, band=0.035):
        r = rules if isinstance(rules, Rules) else _DEFAULT_RULES
        self.rules = r
        self.moves = r.moves
        self.punish = punish
        self.want_dist = want_dist
        self.band = band
        self.mixup = mixup
        if mixup is not None:
            self._mix = self._parse_mixup(mixup)
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _parse_mixup(mixup):
        mix = {"throw": 0.0, "light": 0.0, "heavy": 0.0, "block": 0.0, "back": 0.0}
        mix.update({k: float(v) for k, v in mixup.items()})
        others = mix["throw"] + mix["heavy"] + mix["block"] + mix["back"]
        if others + mix["light"] > 1.0 + 1e-9:
            raise ValueError(f"mixup 概率之和超过 1: {mixup}")
        mix["light"] += max(0.0, 1.0 - others)
        return mix

    def set_rules(self, rules):
        r = rules if isinstance(rules, Rules) else load_rules(rules)
        self.rules = r
        self.moves = r.moves

    def act(self, opp, me):
        d = abs(me.x - opp.x)
        mv_light = self.moves[A_LIGHT] if A_LIGHT < len(self.moves) else None
        mv_heavy = self.moves[A_HEAVY] if A_HEAVY < len(self.moves) else None
        mv_throw = self.moves[A_THROW] if A_THROW < len(self.moves) else None
        if not (mv_light and mv_heavy and mv_throw):
            return A_BLOCK
        rng = self.rng.random

        if me.busy():
            return A_BLOCK

        # 对空：对方跳起来 → 重击
        if hasattr(opp, 'airborne') and opp.airborne():
            if rng() < self.punish and d <= mv_heavy.reach:
                return A_HEAVY
            return A_BLOCK

        if rng() < self.punish:
            if opp.state == S_RECOVERY and d <= mv_heavy.reach:
                return A_HEAVY
            if opp.state == S_STARTUP and d <= mv_heavy.reach:
                return A_BLOCK
            if (opp.blocking or opp.state == S_BLOCKSTUN) and d <= mv_throw.reach:
                return A_THROW

        if d > self.want_dist + self.band:
            return A_FWD
        if d < self.want_dist - self.band:
            return A_BACK

        r = rng()
        if self.mixup is None:
            if d < 0.09 and r < 0.15:
                return A_THROW
            if r < 0.50:
                return A_LIGHT
            if r < 0.62:
                return A_HEAVY
            if r < 0.82:
                return A_BLOCK
            return A_BACK
        m = self._mix
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
    metadata = {"render_modes": ["human"]}
    NUM_ACTIONS = NUM_ACTIONS
    OBS_DIM = OBS_DIM

    def __init__(self, max_ticks=None, bot=None, rules=None):
        super().__init__()
        self.rules = load_rules(rules)
        self.max_ticks = self.rules.max_ticks if max_ticks is None else int(max_ticks)
        self.bot = bot if bot is not None else FootsiesBot(rules=self.rules)
        if hasattr(self.bot, "set_rules"):
            self.bot.set_rules(self.rules)
        self.action_space = spaces.Discrete(self.NUM_ACTIONS)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, (self.OBS_DIM,), np.float32)
        self.p1 = Fighter("AI", rules=self.rules)
        self.p2 = Fighter("机器", rules=self.rules)
        self.t = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        for f in (self.p1, self.p2):
            f.hp = f.max_hp
            f.state = S_IDLE
            f.frames_left = 0
            f.move = None
            f.has_hit = False
            f.blocking = False
            f.block_low = False
            f.crouching = False
            f.y = 0.0
            f.jump_frame = 0
            f.jump_dir = 1
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

        # ---- 1. 状态推进 ----
        nxt = {S_STARTUP: S_ACTIVE, S_ACTIVE: S_RECOVERY, S_RECOVERY: S_IDLE,
               S_BLOCKSTUN: S_IDLE, S_HITSTUN: S_IDLE,
               S_JUMP_RISE: S_JUMP_FALL, S_LANDING: S_IDLE}
        for f in (p, e):
            if f.state == S_JUMP_RISE or f.state == S_JUMP_FALL:
                f.jump_frame += 1
                jf = f.jump_frame
                total = r.jump_frames
                if jf <= total // 2:
                    f.state = S_JUMP_RISE
                else:
                    f.state = S_JUMP_FALL
                # Y 位置（抛物线弧）
                f.y = r.jump_height * math.sin(math.pi * min(jf, total) / total)
                # 水平移动
                speed = r.jump_fwd if f.jump_dir > 0 else r.jump_back
                f.x += f.jump_dir * speed * (1 if f is p else -1)
                # 落地
                if jf >= total:
                    f.y = 0.0
                    f.state = S_LANDING
                    f.frames_left = 4  # 落地硬直
                    f.move = None
                continue

            if f.state == S_LANDING:
                f.frames_left -= 1
                if f.frames_left <= 0:
                    f.state = S_IDLE
                    f.frames_left = 0
                    f.crouching = False
                continue

            if f.busy():
                f.frames_left -= 1
                if f.frames_left <= 0:
                    f.state = nxt.get(f.state, S_IDLE)
                    if f.state == S_ACTIVE:
                        f.frames_left = f.move.active
                    elif f.state == S_RECOVERY:
                        f.frames_left = f.move.recovery
                    else:
                        f.frames_left = 0
                        if f.state == S_IDLE:
                            f.move = None
                            f.crouching = False

        # ---- 2. 接受新指令 ----
        for f, a in ((p, a_p), (e, a_e)):
            if f.busy():
                continue
            # 清除上一帧的防御/蹲伏标志
            f.blocking = False
            f.block_low = False

            if a == A_BLOCK:
                f.blocking = True
                f.crouching = False
                continue
            elif a == A_CROUCH_BLOCK:
                f.block_low = True
                f.crouching = True
                f.blocking = True  # 也算 blocking（广义）
                continue
            elif a == A_CROUCH:
                f.crouching = True
                continue

            f.crouching = False

            if a == A_JUMP:
                f.jump_frame = 0
                f.jump_dir = 1  # 默认前跳；简化：不支持后跳
                f.y = 0.001  # 标记离地
                f.state = S_JUMP_RISE
                f.move = f.moves[A_JUMP] if A_JUMP < len(f.moves) else None
                f.has_hit = False
                continue

            if a in (A_LIGHT, A_HEAVY, A_THROW, A_LOW):
                slot = a
                if slot < len(f.moves) and f.moves[slot]:
                    f.start(slot)
            elif a == A_FWD:
                f.x += r.walk_f if f is p else -r.walk_f
            elif a == A_BACK:
                f.x += -r.walk_b if f is p else r.walk_b

        # ---- 3. 碰撞与边界 ----
        lo, hi = STAGE
        p.x = float(np.clip(p.x, lo, hi))
        e.x = float(np.clip(e.x, lo, hi))
        if e.x - p.x < r.min_gap:
            mid = (p.x + e.x) / 2
            p.x = max(lo, mid - r.min_gap / 2)
            e.x = min(hi, mid + r.min_gap / 2)
        d = e.x - p.x

        # ---- 4. 打击判定（含高低段/对空/蹲伏躲避）----
        strikes = []
        for atk, dfn in ((p, e), (e, p)):
            if atk.state != S_ACTIVE or not atk.move or atk.has_hit:
                continue
            mv = atk.move
            if mv.kind not in ("strike", "jump"):
                continue

            # 距离检查
            if d > mv.reach:
                continue

            # 高度判定
            dfn_air = dfn.airborne()
            dfn_crouch = dfn.crouching

            if mv.height == "high":
                # 高段：蹲伏（纯蹲不蹲防）完全躲开
                if dfn_crouch and not dfn_air:
                    continue  # whiff over head
                # 高段可以对空（打空中对手）
                # 空中对手在射程内即命中

            elif mv.height == "low":
                # 低段：打不到空中
                if dfn_air:
                    continue
                # 低段：站防挡不住（leak），蹲防能挡

            elif mv.height == "air":
                # 跳攻（overhead）：蹲防挡不住
                pass  # 由下方 blocking 逻辑处理

            strikes.append((atk, dfn))

        # ---- 5. 结算打击 ----
        for atk, dfn in strikes:
            atk.has_hit = True
            mv = atk.move

            # 判断防御结果
            if dfn.block_low and mv.height == "low":
                # 蹲防挡低段
                dmg = max(1, int(mv.damage * r.chip))
                dfn.state = S_BLOCKSTUN
                dfn.frames_left = mv.blockstun
                self._push(dfn, r.push_block)
            elif dfn.blocking and not dfn.block_low and mv.height in ("high", "air"):
                # 站防挡高段+跳攻
                dmg = max(1, int(mv.damage * r.chip))
                dfn.state = S_BLOCKSTUN
                dfn.frames_left = mv.blockstun
                self._push(dfn, r.push_block)
            elif dfn.blocking and not dfn.block_low and mv.height == "low":
                # 站防漏低段 → 全额伤害
                dmg = mv.damage
                dfn.state = S_HITSTUN
                dfn.frames_left = mv.hitstun
                self._push(dfn, r.push_hit)
            elif dfn.block_low and mv.height == "air":
                # 蹲防漏跳攻 → 全额伤害
                dmg = mv.damage
                dfn.state = S_HITSTUN
                dfn.frames_left = mv.hitstun
                self._push(dfn, r.push_hit)
            elif dfn.blocking and mv.height == "mid":
                # 中段（投技等）：所有防御都挡不住
                dmg = mv.damage
                dfn.state = S_HITSTUN
                dfn.frames_left = mv.hitstun
                self._push(dfn, r.push_hit)
            else:
                # 无防御 → 全额
                dmg = mv.damage
                dfn.state = S_HITSTUN
                dfn.frames_left = mv.hitstun
                self._push(dfn, r.push_hit)

            dfn.hp = max(0, dfn.hp - dmg)
            dfn.hit_taken += dmg
            atk.dealt += dmg
            dfn.crouching = False  # 被打站直
            dfn.blocking = False
            dfn.block_low = False

        # ---- 6. 投技判定 ----
        for atk, dfn in ((p, e), (e, p)):
            if (atk.state in (S_ACTIVE, S_JUMP_FALL) and not atk.has_hit
                    and atk.move and atk.move.kind == "throw"
                    and d <= atk.move.reach):
                # 投技打不到空中
                if dfn.airborne():
                    continue
                if dfn.blocking or dfn.state in (S_IDLE, S_RECOVERY, S_BLOCKSTUN):
                    atk.has_hit = True
                    dmg = atk.move.damage
                    dfn.hp = max(0, dfn.hp - dmg)
                    dfn.hit_taken += dmg
                    atk.dealt += dmg
                    dfn.state = S_HITSTUN
                    dfn.frames_left = atk.move.hitstun
                    dfn.blocking = False
                    dfn.block_low = False
                    dfn.crouching = False
                    self._push(dfn, r.push_hit)

        # ---- 7. 跳攻判定（overhead，自动触发）----
        for atk, dfn in ((p, e), (e, p)):
            if (atk.state == S_JUMP_FALL and not atk.has_hit
                    and atk.move and atk.move.kind == "jump"):
                jf = atk.jump_frame
                total = r.jump_frames
                # 跳攻在下降段帧 8-15 激活
                if total // 2 < jf <= total - 4 and d <= atk.move.reach:
                    atk.has_hit = True
                    mv = atk.move
                    if dfn.block_low:
                        # 蹲防漏跳攻
                        dmg = mv.damage
                        dfn.state = S_HITSTUN
                        dfn.frames_left = mv.hitstun
                        self._push(dfn, r.push_hit)
                    elif dfn.blocking:
                        dmg = max(1, int(mv.damage * r.chip))
                        dfn.state = S_BLOCKSTUN
                        dfn.frames_left = mv.blockstun
                        self._push(dfn, r.push_block)
                    else:
                        dmg = mv.damage
                        dfn.state = S_HITSTUN
                        dfn.frames_left = mv.hitstun
                        self._push(dfn, r.push_hit)
                    dfn.hp = max(0, dfn.hp - dmg)
                    dfn.hit_taken += dmg
                    atk.dealt += dmg
                    dfn.blocking = False
                    dfn.block_low = False
                    dfn.crouching = False

        # ---- 8. 回合结束 ----
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
                reward += 3.0 * np.sign(p.hp - e.hp)

        info = {"tick": self.t, "dist": d,
                "p_state": STATE_NAMES[p.state], "e_state": STATE_NAMES[e.state],
                "p_dealt": p.dealt, "e_dealt": e.dealt}
        return self._obs(), reward, term, trunc, info

    def obs_for(self, side):
        if side == 0:
            return self._obs()
        p, e = self.p1, self.p2
        mx, ox = 1.0 - e.x, 1.0 - p.x
        return np.concatenate([
            np.array([e.hp / e.max_hp, p.hp / p.max_hp,
                      ox - mx, mx, ox], dtype=np.float32),
            self._onehot(e), self._onehot(p),
            np.array([1.0 * e.blocking, 1.0 * p.blocking,
                      self._busy_frac(e), self._busy_frac(p),
                      e.y, p.y,
                      1.0 * e.crouching, 1.0 * p.crouching,
                      1.0 * e.airborne(), 1.0 * p.airborne(),
                      float(e.jump_frame), float(p.jump_frame)], dtype=np.float32),
        ]).astype(np.float32)

    def outcome(self):
        p, e = self.p1, self.p2
        if p.hp <= 0 or e.hp <= 0:
            if p.hp <= 0 and e.hp <= 0:
                return {"winner": None, "hp": [int(p.hp), int(e.hp)],
                        "ticks": int(self.t), "result": "draw"}
            elif e.hp <= 0:
                return {"winner": 0, "hp": [int(p.hp), int(e.hp)],
                        "ticks": int(self.t), "result": "win0"}
            else:
                return {"winner": 1, "hp": [int(p.hp), int(e.hp)],
                        "ticks": int(self.t), "result": "win1"}
        elif p.hp > e.hp:
            return {"winner": 0, "hp": [int(p.hp), int(e.hp)],
                    "ticks": int(self.t), "result": "timeout0"}
        elif p.hp < e.hp:
            return {"winner": 1, "hp": [int(p.hp), int(e.hp)],
                    "ticks": int(self.t), "result": "timeout1"}
        return {"winner": None, "hp": [int(p.hp), int(e.hp)],
                "ticks": int(self.t), "result": "draw"}

    def _push(self, dfn, amount):
        sign = 1.0 if dfn is self.p2 else -1.0
        dfn.x = float(np.clip(dfn.x + sign * amount, *STAGE))

    @staticmethod
    def _onehot(f):
        """6 维旧版 one-hot（向后兼容）：新状态映射到最近的旧状态。"""
        v = np.zeros(6, dtype=np.float32)
        old_state = f.state
        if old_state >= 6:  # 新状态映射
            mapping = {S_JUMP_RISE: S_IDLE, S_JUMP_FALL: S_ACTIVE,
                       S_CROUCH: S_IDLE, S_LANDING: S_RECOVERY}
            old_state = mapping.get(old_state, S_IDLE)
        v[old_state] = 1.0
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
                      self._busy_frac(p), self._busy_frac(e),
                      p.y, e.y,
                      1.0 * p.crouching, 1.0 * e.crouching,
                      1.0 * p.airborne(), 1.0 * e.airborne(),
                      float(p.jump_frame), float(e.jump_frame)], dtype=np.float32),
        ]).astype(np.float32)


if __name__ == "__main__":
    # 冒烟测试：随机策略 vs 立回机器人（2D）
    env = FightingEnv()
    wins = losses = draws = 0
    for ep in range(20):
        obs, _ = env.reset(seed=ep)
        assert obs.shape == (FightingEnv.OBS_DIM,), f"obs 维度错误: {obs.shape}"
        done, ret = False, 0.0
        while not done:
            # 随机策略偶尔用新动作
            a = env.action_space.sample()
            obs, r, term, trunc, info = env.step(a)
            ret += r
            done = term or trunc
        if env.p2.hp <= 0 < env.p1.hp:
            wins += 1
        elif env.p1.hp <= 0 < env.p2.hp:
            losses += 1
        else:
            draws += 1
        if ep < 3:
            print(f"第{ep + 1}场 AI HP{env.p1.hp:3d} vs 机器 HP{env.p2.hp:3d} "
                  f"回合回报 {ret:+7.2f}")
    print(f"\n2D 冒烟测试：随机策略 vs 立回机器人：{wins}胜 {losses}负 {draws}判")
    print(f"观测维度: {FightingEnv.OBS_DIM} | 动作数: {FightingEnv.NUM_ACTIONS}")
