"""玩家自制智能体：反投侠（anti-throw）。

设计思路（看完 PPO vs 投技狂 的直播后写的）：
- 贴身绝不站防（防御会被投技破坏）→ 用轻击压制，打击天生克投技
- 远距离对方出招才拉防（投技射程外，防御安全）
- 对方收招硬直 → 重击确反
- 对方防御硬直且贴身 → 投技伺候
- 平时维持 0.13~0.17 立回带轻击 poke

接入方式：python arena.py run file:agents/examples/anti_throw.py footsies
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", ".."))

from agents.base import BaseAgent   # noqa: E402

# 观测布局（自己视角 21 维）：
# [0]我HP [1]敌HP [2]距离 [3]我x [4]敌x [5:11]我状态 [11:17]敌状态
# [17]我防御中 [18]敌防御中 [19]我硬直进度 [20]敌硬直进度
# 状态: 0待机 1启动 2判定 3收招 4防御硬直 5受击硬直
A_BACK, A_FWD, A_LIGHT, A_HEAVY, A_BLOCK, A_THROW = range(6)


class Agent(BaseAgent):
    name = "anti-throw"

    def reset(self, seed: int, side: int = 0) -> None:
        pass    # 无随机源，天然确定

    def act(self, obs) -> int:
        d = float(obs[2])
        opp = int(np.argmax(obs[11:17]))    # 对手状态

        if opp == 3 and d <= 0.19:          # 对方收招 → 重击确反
            return A_HEAVY
        if opp == 1 and d > 0.09:           # 对方远距离出招 → 拉防
            return A_BLOCK
        if d <= 0.10:                       # 贴身：轻击压制（打击克投技，绝不站防）
            return A_LIGHT
        if opp == 4 and d <= 0.09:          # 对方防御硬直 → 贴身投
            return A_THROW
        if d > 0.17:                        # 立回：维持 poke 距离
            return A_FWD
        if d < 0.13:
            return A_BACK
        return A_LIGHT                      # 交战带内轻击 poke
