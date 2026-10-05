"""PPO 模型适配器：包装 stable-baselines3 模型（示例一：RL 策略接入）。

模型在 p1（左）视角训练；作为右侧 Agent 参战时，arena 传入镜像观测，
「前进」语义自动对齐，无需任何模型侧修改。
"""

import os

from .base import BaseAgent


class PPOAgent(BaseAgent):
    def __init__(self, path="fighting_ppo.zip", name=None):
        from stable_baselines3 import PPO
        self.path = path
        self.model = PPO.load(path)
        self.name = name or os.path.splitext(os.path.basename(path))[0]

    def reset(self, seed: int, side: int = 0) -> None:
        pass  # 确定性预测，无需重置

    def act(self, obs) -> int:
        a, _ = self.model.predict(obs, deterministic=True)
        return int(a)

    def info(self) -> dict:
        return {"kind": "ppo", "model": os.path.basename(self.path)}
