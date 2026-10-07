"""Agent 统一协议（平台「Agent 接入层」的协议层）。

所有 Agent 只能看到「自己视角」的 21 维观测（右侧为镜像坐标），
与人类玩家等价——平台不泄漏对手内部状态。
"""

import numpy as np

from fighting_env import (
    S_STARTUP, S_ACTIVE, S_RECOVERY, S_BLOCKSTUN, S_HITSTUN,
)

# 观测布局常量（与 FightingEnv._obs 一致）
OBS_DIM = 29
_BUSY_STATES = (S_STARTUP, S_ACTIVE, S_RECOVERY, S_BLOCKSTUN, S_HITSTUN)

# 2D 扩展观测下标
OBS_MY_Y = 21
OBS_OP_Y = 22
OBS_MY_CROUCH = 23
OBS_OP_CROUCH = 24
OBS_MY_AIR = 25
OBS_OP_AIR = 26
OBS_MY_JUMP_FRAME = 27
OBS_OP_JUMP_FRAME = 28


class BaseAgent:
    """平台 Agent 协议。

    - name:       展示名（写入回放 players[].name）
    - reset:      每局开始时调用，seed 派生固定 → 可复现；side 0=左 / 1=右
    - act(obs):   每帧调用，返回 0~5
    - info():     写入回放 players[].meta 的元信息
    """

    name = "agent"

    def reset(self, seed: int, side: int = 0) -> None:  # pragma: no cover
        pass

    def act(self, obs: np.ndarray) -> int:  # pragma: no cover
        raise NotImplementedError

    def info(self) -> dict:  # pragma: no cover
        return {"kind": type(self).__name__}


class RandomAgent(BaseAgent):
    """内置随机策略（基线强度参照）。"""

    def __init__(self, name="random"):
        self.name = name
        self.rng = np.random.default_rng(0)

    def reset(self, seed: int, side: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray) -> int:
        return int(self.rng.integers(6))

    def info(self) -> dict:
        return {"kind": "random"}


class _ObsFighter:
    """从观测重建的轻量角色视图（含 2D 蹲伏/空中字段）。"""

    def __init__(self, x, state, blocking, crouching=False, airborne=False, y=0.0):
        self.x = x
        self.state = state
        self.blocking = blocking
        self.crouching = crouching
        self.y = y
        self.move = None

    def busy(self):
        return self.state in _BUSY_STATES

    def airborne(self):
        return self._air_flag or self.y > 0.005

    _air_flag = False


def fighters_from_obs(obs: np.ndarray):
    """29 维观测 → (me, opp) 轻量角色视图。自动兼容 21 维旧观测。"""
    dim = len(obs)
    me_cr = bool(obs[23] > 0.5) if dim > 23 else False
    opp_cr = bool(obs[24] > 0.5) if dim > 24 else False
    me_air = bool(obs[25] > 0.5) if dim > 25 else False
    opp_air = bool(obs[26] > 0.5) if dim > 26 else False
    me_y = float(obs[21]) if dim > 21 else 0.0
    opp_y = float(obs[22]) if dim > 22 else 0.0

    me = _ObsFighter(float(obs[3]), int(np.argmax(obs[5:11])),
                     bool(obs[17] > 0.5), crouching=me_cr,
                     airborne=me_air, y=me_y)
    me._air_flag = me_air
    opp = _ObsFighter(float(obs[4]), int(np.argmax(obs[11:17])),
                      bool(obs[18] > 0.5), crouching=opp_cr,
                      airborne=opp_air, y=opp_y)
    opp._air_flag = opp_air
    return me, opp


class WindowAgent(BaseAgent):
    """决策窗口制包装器：解决 LLM 无法逐帧决策的问题。

    每 window 帧调用一次内部 agent，期间逐帧重复上次动作。
    语义天然合法：重复轻击 = 空闲时自动再出，重复防御 = 持续防御，
    硬直中重复指令会被环境忽略（见 PLAN §7 决策记录，勿改为清零）。
    """

    def __init__(self, agent: BaseAgent, window: int = 15):
        self.agent = agent
        self.window = max(1, int(window))
        self.name = f"{agent.name}·w{self.window}"
        self._last_action = 4  # 默认防御（安全动作）
        self._frames_left = 0
        self.calls = 0

    def reset(self, seed: int, side: int = 0) -> None:
        self.agent.reset(seed, side)
        self._last_action = 4
        self._frames_left = 0
        self.calls = 0

    def act(self, obs: np.ndarray) -> int:
        if self._frames_left <= 0:
            self._last_action = int(self.agent.act(obs))
            self._frames_left = self.window
            self.calls += 1
        self._frames_left -= 1
        return self._last_action

    def info(self) -> dict:
        d = self.agent.info()
        d["window"] = self.window
        return d


if __name__ == "__main__":
    # 决策窗口演示（运行：python -m agents.base）
    # 1) 窗口版 ppo vs 原生 ppo 互打（左右侧各 10 场）
    # 2) 窗口宽度退化曲线（vs footsies 各 10 场）
    from agents import load_agent
    from arena import play_one

    ppo = load_agent("ppo")
    windowed = WindowAgent(ppo, window=15)
    footsies = load_agent("footsies")

    print("== 窗口版 ppo(w15) vs 原生 ppo 互打（各侧10场） ==")
    for left, right in ((windowed, ppo), (ppo, windowed)):
        w = sum(1 for s in range(10)
                if play_one(left, right, s)[0]["winner"] == 0)
        print(f"  {left.name}(左) vs {right.name}(右): 左侧 {w}/10")

    print("== 窗口宽度退化曲线（vs footsies 各10场，w=1 应等价原生） ==")
    for win_w in (1, 3, 5, 8, 15):
        wa = WindowAgent(ppo, window=win_w)
        w = sum(1 for s in range(10)
                if play_one(wa, footsies, s)[0]["winner"] == 0)
        print(f"  w={win_w:2d}: {w}/10")
