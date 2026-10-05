"""Agent 接入层：统一加载器 load_agent(spec)。

支持的引用格式（平台「Agent 用名字引用」的最小实现）：
- random               内置随机策略
- footsies             默认立回机器人
- hidden:<代号>        隐藏池风格化 Bot（hidden_bots.HIDDEN_POOL）
- ppo[:<模型路径>]     SB3 PPO 模型（默认 fighting_ppo.zip）
- file:<py路径>        用户上传的策略文件（含 class Agent(BaseAgent)）
- http:<url>           HTTP/LLM Agent（决策窗口制适配）
"""

import os

from .base import BaseAgent, RandomAgent, WindowAgent
from .script_agent import ScriptAgent
from .ppo_agent import PPOAgent


def load_agent(spec: str) -> BaseAgent:
    if spec == "random":
        return RandomAgent()
    if spec == "footsies":
        from fighting_env import FootsiesBot
        return ScriptAgent(FootsiesBot(), name="footsies")
    if spec.startswith("hidden:"):
        from hidden_bots import get_hidden_bot
        code = spec.split(":", 1)[1]
        return ScriptAgent(get_hidden_bot(code), name=spec)
    if spec == "ppo" or spec.startswith("ppo:"):
        path = spec.split(":", 1)[1] if ":" in spec else "fighting_ppo.zip"
        return PPOAgent(path)
    if spec.startswith("file:"):
        return _load_file_agent(spec.split(":", 1)[1])
    if spec.startswith("http:"):
        from .http_agent import HttpAgent
        return HttpAgent(spec)
    raise ValueError(
        f"无法解析 Agent 引用: {spec!r}（支持 random/footsies/hidden:*/ppo[:路径]/file:*/http:*）")


def _load_file_agent(path: str) -> BaseAgent:
    """用户上传策略文件的最简形态：py 文件内含 class Agent(BaseAgent)。

    注意：MVP 阶段无沙箱，信任运行（沙箱化属于平台工程，见 PLAN §6 红线 3）。
    """
    import importlib.util
    path = path.rstrip("/")
    if not os.path.exists(path):
        raise FileNotFoundError(f"策略文件不存在: {path}")
    mod_name = "user_agent_" + os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    agent_cls = getattr(mod, "Agent", None)
    if agent_cls is None:
        raise ValueError(f"{path} 中未找到 class Agent")
    agent = agent_cls()
    if not hasattr(agent, "act"):
        raise ValueError(f"{path} 的 Agent 缺少 act 方法")
    # 未自定义名字（继承了基类默认值）时用文件名
    if getattr(agent, "name", None) in (None, "agent"):
        agent.name = os.path.splitext(os.path.basename(path))[0]
    return agent


__all__ = ["load_agent", "BaseAgent", "RandomAgent", "WindowAgent",
           "ScriptAgent", "PPOAgent"]
