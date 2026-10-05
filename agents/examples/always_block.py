"""示例用户策略：永远防御的 Bot（file: 接入的最简演示）。

接入方式：python arena.py run random file:agents/examples/always_block.py

用户策略文件只需包含 `class Agent(BaseAgent)`，实现 act(obs) → 0~5。
注意：MVP 阶段无沙箱，策略文件被信任执行（沙箱化属于平台工程范围）。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", ".."))

from agents.base import BaseAgent   # noqa: E402


class Agent(BaseAgent):
    name = "always-block"

    def reset(self, seed: int, side: int = 0) -> None:
        pass

    def act(self, obs) -> int:
        return 4                     # 防御！防御！防御！

    def info(self) -> dict:
        return {"kind": "file", "note": "永远防御（示例策略）"}
