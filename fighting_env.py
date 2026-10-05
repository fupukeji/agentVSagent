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

动作空间（6）
------------
0 后撤 | 1 前进 | 2 轻击 | 3 重击 | 4 防御 | 5 投技
"""

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
# 角色
# ---------------------------------------------------------------
class Fighter:
    def __init__(self, name):
        self.name = name
        self.max_hp = 100
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
        self.move = MOVES[action]
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

    def __init__(self, punish=0.8, seed=None):
        self.punish = punish          # 读取成功率（模拟人类反应，<1 更真实）
        self.rng = np.random.default_rng(seed)

    def act(self, opp, me):
        d = me.x - opp.x              # 玩家在左，机器在右，d > 0
        rng = self.rng.random

        if me.busy():
            return A_BLOCK            # 硬直中指令无效

        # ---- 概率性读取（模拟人类反应延迟/失误）----
        if rng() < self.punish:
            if opp.state == S_RECOVERY and d <= MOVES[A_HEAVY].reach:
                return A_HEAVY        # 确反：惩罚收招
            if opp.state == S_STARTUP and d <= MOVES[A_HEAVY].reach:
                return A_BLOCK        # 对方出招 → 拉防
            if (opp.blocking or opp.state == S_BLOCKSTUN) and d <= MOVES[A_THROW].reach:
                return A_THROW        # 破防投

        # ---- 立回：维持理想交战距离 ----
        want = 0.155                  # 刚好在轻击(0.13)与重击(0.19)之间
        if d > want + 0.035:
            return A_FWD
        if d < want - 0.035:
            return A_BACK

        # ---- 交战距离内的打投择 ----
        r = rng()
        if d < 0.09 and r < 0.15:
            return A_THROW
        if r < 0.50:
            return A_LIGHT
        if r < 0.62:
            return A_HEAVY
        if r < 0.82:
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

    def __init__(self, max_ticks=900, bot=None):
        super().__init__()
        self.max_ticks = max_ticks
        self.bot = bot or FootsiesBot()
        self.action_space = spaces.Discrete(self.NUM_ACTIONS)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, (self.OBS_DIM,), np.float32)
        self.p1 = Fighter("AI")
        self.p2 = Fighter("机器")
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
                f.x += WALK_F if f is p else -WALK_F
            elif a == A_BACK:
                f.x += -WALK_B if f is p else WALK_B

        # ---- 3. 碰撞与舞台边界 ----
        lo, hi = STAGE
        p.x = float(np.clip(p.x, lo, hi))
        e.x = float(np.clip(e.x, lo, hi))
        if e.x - p.x < MIN_GAP:
            mid = (p.x + e.x) / 2
            p.x = max(lo, mid - MIN_GAP / 2)
            e.x = min(hi, mid + MIN_GAP / 2)
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
                dmg = max(1, int(atk.move.damage * CHIP))
                dfn.state = S_BLOCKSTUN
                dfn.frames_left = atk.move.blockstun
                self._push(dfn, BLOCK_PUSH)
            else:
                dmg = atk.move.damage
                dfn.state = S_HITSTUN
                dfn.frames_left = atk.move.hitstun
                self._push(dfn, HIT_PUSH)
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
                    self._push(dfn, HIT_PUSH)

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
