"""我的参赛智能体（官方模板）——从这里开始你的冠军之路。

参赛：官网「参赛教学」页上传本文件，或本地
    python arena.py run file:my_agent_template.py footsies

==========================================================
观测速查（21 维，全是你被允许看到的）
----------------------------------------------------------
obs[0]  我方HP/100        obs[1]  敌方HP/100
obs[2]  交战距离(0~1)     obs[3]  我方x      obs[4] 敌方x
        （坐标为自己视角：无论你在左侧还是右侧，「前进」都指向对手）
obs[5:11] 我方状态 one-hot    obs[11:17] 敌方状态 one-hot
        状态编码: 0待机 1启动 2判定 3收招 4防御硬直 5受击硬直
obs[17] 我方防御中(0/1)  obs[18] 敌方防御中(0/1)
obs[19] 我方硬直进度(0~1) obs[20] 敌方硬直进度(0~1)

动作空间（返回其中一个整数）
----------------------------------------------------------
0 后撤 | 1 前进 | 2 轻击 | 3 重击 | 4 防御 | 5 投技
==========================================================
"""

import numpy as np

from agents.base import BaseAgent

A_BACK, A_FWD, A_LIGHT, A_HEAVY, A_BLOCK, A_THROW = range(6)
IDLE, STARTUP, ACTIVE, RECOVERY, BLOCKSTUN, HITSTUN = range(6)


class Agent(BaseAgent):
    name = "我的智能体"

    def reset(self, seed: int, side: int = 0) -> None:
        pass    # 需要随机数时：self.rng = np.random.default_rng(seed)

    def act(self, obs) -> int:
        d = float(obs[2])                       # 交战距离
        opp = int(np.argmax(obs[11:17]))        # 对手状态
        opp_blocking = obs[18] > 0.5

        # TODO 在这里写下你的立回哲学。几个进阶起手式（把注释变成代码即可变强）：
        # 反投：贴身时绝不能傻站（会被投），用轻击压制
        #   if d <= 0.10: return A_LIGHT
        # 破防：对方在防御且贴身 → 投技伺候
        #   if opp_blocking and d <= 0.09: return A_THROW
        # 拉防：对方出招（远距离）→ 防住然后惩罚
        #   if opp == STARTUP and d > 0.09: return A_BLOCK

        # —— 默认策略（内置第一课「确反」；想登顶就继续改造）——
        if opp == RECOVERY and d <= 0.19:   # 对方收招硬直 → 重击惩罚
            return A_HEAVY
        if d > 0.14:                        # 走进轻击射程（0.13）内
            return A_FWD
        return A_LIGHT                      # 交战带内轻击试探
