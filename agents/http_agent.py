"""HTTP/LLM Agent 适配器（决策窗口制）。

解决「LLM 无法逐帧决策」：WindowAgent 每 15 帧调用一次远端服务，
期间逐帧重复上次动作（重复轻击=空闲时自动再出，重复防御=持续防御，
硬直中重复指令被环境忽略——天然合法，见 PLAN §7）。

HTTP 协议：
    POST <url>  {"obs": [21维], "tick": n, "side": 0, "legal_hint": "..."}
    期望响应     {"action": 0~5}
    超时（默认 5 秒）→ 返回防御(4)并记录；其它网络/解析错误同样兜底。

url 规范化：`http:127.0.0.1:8080` → `http://127.0.0.1:8080/act`；
带 scheme/path 的完整 URL 原样使用。
"""

import json
import socket
import time
import urllib.error
import urllib.request

from .base import BaseAgent, WindowAgent

LEGAL_HINT = ("动作 0后撤/1前进/2轻击/3重击/4防御/5投技；"
              "窗口内重复动作=持续该行为（重复防御=持续防御）")


def normalize_url(url: str) -> str:
    if "://" not in url:
        url = "http://" + url
    host_part = url.split("://", 1)[1]
    if "/" not in host_part:          # 只有主机无路径 → 默认 /act
        url = url.rstrip("/") + "/act"
    return url


class HttpPolicy(BaseAgent):
    """逐决策点调用远端 HTTP 服务的策略（必须配合 WindowAgent 使用）。"""

    def __init__(self, url, timeout=5.0):
        self.url = normalize_url(url)
        self.timeout = float(timeout)
        self.calls = 0
        self.timeouts = 0
        self.errors = 0
        self._lat_sum = 0.0
        self._tick = 0        # 决策点计数（非帧计数）
        self._side = 0

    def reset(self, seed: int, side: int = 0) -> None:
        self._tick = 0
        self._side = side

    def act(self, obs) -> int:
        payload = {
            "obs": [round(float(x), 4) for x in obs],
            "tick": self._tick,
            "side": self._side,
            "legal_hint": LEGAL_HINT,
        }
        action = 4  # 兜底：防御
        t0 = time.perf_counter()
        try:
            req = urllib.request.Request(
                self.url, data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                a = int(body["action"])
                if 0 <= a <= 5:
                    action = a
        except Exception as e:  # noqa: BLE001 — 网络/超时/解析错误统一兜底
            msg = str(e).lower()
            if ("timed out" in msg or "timeout" in msg
                    or isinstance(e, (TimeoutError, socket.timeout))):
                self.timeouts += 1
            else:
                self.errors += 1
        self._lat_sum += (time.perf_counter() - t0) * 1000.0
        self.calls += 1
        self._tick += 1
        return action

    def info(self) -> dict:
        return {
            "kind": "http",
            "url": self.url,
            "calls": self.calls,
            "timeouts": self.timeouts,
            "errors": self.errors,
            "avg_latency_ms": round(self._lat_sum / self.calls, 1) if self.calls else None,
        }


class HttpAgent(WindowAgent):
    """http:<url> 引用的完整 Agent = HTTP 策略 + 决策窗口。"""

    def __init__(self, spec, window=15, timeout=5.0, name=None):
        url = spec[5:] if spec.startswith("http:") else spec
        self.policy = HttpPolicy(url, timeout=timeout)
        super().__init__(self.policy, window=window)
        self.name = name or f"llm@{self.policy.url.split('//', 1)[1]}"
