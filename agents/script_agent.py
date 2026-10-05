"""脚本 Bot 适配器：把 FootsiesBot 及其变体包装为正式 Agent。

脚本 Bot 作为参战方时，从「自己视角」的观测中重建轻量角色视图，
再调用 bot.act(opp, me)——与在环境侧运行时看到的博弈信息严格等价。
"""

import numpy as np

from .base import BaseAgent, fighters_from_obs


class ScriptAgent(BaseAgent):
    def __init__(self, bot, name=None):
        self.bot = bot
        self.name = name or type(bot).__name__

    def reset(self, seed: int, side: int = 0) -> None:
        self.bot.rng = np.random.default_rng(seed)

    def act(self, obs) -> int:
        me, opp = fighters_from_obs(obs)
        return int(self.bot.act(opp, me))

    def info(self) -> dict:
        return {
            "kind": "script",
            "bot": type(self.bot).__name__,
            "punish": getattr(self.bot, "punish", None),
            "want_dist": getattr(self.bot, "want_dist", None),
            "mixup": getattr(self.bot, "mixup", None),
        }
