"""我的参赛智能体 2D 版（官方模板）——从这里开始你的冠军之路。

参赛：官网「参赛教学」页上传本文件，或本地
    python arena.py run file:my_agent.py footsies

==========================================================
观测速查（29 维，前 21 维与 1D 完全兼容）
----------------------------------------------------------
obs[0]  我方HP/100        obs[1]  敌方HP/100
obs[2]  交战距离(0~1)     obs[3]  我方x      obs[4] 敌方x
obs[5:11] 我方状态 one-hot（0待机 1启动 2判定 3收招 4防硬 5受硬）
obs[11:17] 敌方状态 one-hot
obs[17] 我方防御中(0/1)  obs[18] 敌方防御中(0/1)
obs[19] 我方硬直进度     obs[20] 敌方硬直进度
--- 2D 新增 ---
obs[21] 我方Y位置(0=地面) obs[22] 敌方Y位置
obs[23] 我方蹲伏中(0/1)  obs[24] 敌方蹲伏中
obs[25] 我方空中(0/1)    obs[26] 敌方空中 ← 对空时机！
obs[27] 我方跳跃帧       obs[28] 敌方跳跃帧

动作空间（10）
----------------------------------------------------------
0 后撤 | 1 前进 | 2 轻击(高) | 3 重击(高·对空) | 4 站防
5 投技 | 6 跳跃(自动跳攻) | 7 蹲伏 | 8 下段(低) | 9 蹲防

高低段判定速查
----------------------------------------------------------
蹲伏(7)   躲高段(2/3)     站防(4) 漏低段(8)和投(5)
蹲防(9)   挡低段(8)+高段  蹲防(9) 漏跳攻(6)和投(5)
跳跃(6)   躲投+低段       重击(3) 可打空中对手(对空)
==========================================================
"""

import numpy as np

from agents.base import BaseAgent

A_BACK, A_FWD, A_LIGHT, A_HEAVY, A_BLOCK, A_THROW = range(6)
A_JUMP, A_CROUCH, A_LOW, A_CROUCH_BLOCK = 6, 7, 8, 9
IDLE, STARTUP, ACTIVE, RECOVERY, BLOCKSTUN, HITSTUN = range(6)


class Agent(BaseAgent):
    name = "我的智能体"

    def reset(self, seed: int, side: int = 0) -> None:
        pass

    def act(self, obs) -> int:
        d = float(obs[2])
        opp = int(np.argmax(obs[11:17]))
        opp_crouch = obs[24] > 0.5   # 对方蹲着
        opp_air = obs[26] > 0.5      # 对方跳着（对空时机！）

        # TODO 在这里写下你的立回哲学。2D 起手式：

        # 对空：对方跳起来 → 重击打下来（18伤害）
        #   if opp_air and d <= 0.19: return A_HEAVY
        # 确反：对方收招 → 重击
        #   if opp == RECOVERY and d <= 0.19: return A_HEAVY
        # 对方蹲防 → 跳攻穿透（蹲防挡不住跳攻）
        #   if opp_crouch and d >= 0.10: return A_JUMP
        # 对方站防 → 下段穿透（站防挡不住低段）
        #   if obs[18] > 0.5 and d <= 0.15: return A_LOW
        # 贴身反投：绝不站防挨投
        #   if d <= 0.10: return A_LIGHT

        # 默认策略：走到轻击射程内 poke
        if d > 0.12:
            return A_FWD
        return A_LIGHT
