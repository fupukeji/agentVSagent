"""内置假 LLM 服务：起本地端口，按简单启发式返回动作。

用于演示与压测 HTTP Agent 接入（模拟 30~80ms 的 LLM 推理延迟）。
配合：python arena.py run http:127.0.0.1:8080 random

运行：python agents/mock_llm.py [端口，默认 8080]
"""

import json
import random
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

# 观测布局（与 fighting_env.FightingEnv._obs 一致）
# [0]我HP [1]敌HP [2]距离 [3]我x [4]敌x [5:11]我状态onehot [11:17]敌状态onehot
# [17]我防御中 [18]敌防御中 [19]我硬直进度 [20]敌硬直进度
S_IDLE, S_STARTUP, S_ACTIVE, S_RECOVERY, S_BLOCKSTUN, S_HITSTUN = range(6)


def decide(obs: np.ndarray) -> int:
    """假 LLM 的「思考」：简单但合法的立回启发式。"""
    d = float(obs[2])
    opp_state = int(np.argmax(obs[11:17]))
    opp_blocking = obs[18] > 0.5

    if opp_state == S_RECOVERY and d <= 0.19:
        return 3                      # 确反：对方收招 → 重击
    if opp_state == S_STARTUP and d <= 0.19:
        return 4                      # 对方出招 → 拉防
    if opp_state in (S_HITSTUN, S_BLOCKSTUN) and d <= 0.09:
        return 5                      # 对方硬直且贴身 → 投技
    if opp_blocking and d <= 0.09:
        return 5                      # 破防投
    if d > 0.16:
        return 1                      # 太远 → 前进
    if d < 0.10:
        return 2                      # 贴身 → 轻击压制
    return 2 if random.random() < 0.7 else 4   # 交战带内： poke 为主


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path.rstrip("/") != "/act":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        obs = np.asarray(payload.get("obs", [0.5] * 21), dtype=np.float32)
        action = decide(obs)
        time.sleep(random.uniform(0.03, 0.08))   # 模拟 LLM 延迟
        body = json.dumps({"action": int(action)}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):     # 静默访问日志
        pass


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"假 LLM 已启动: POST http://127.0.0.1:{port}/act （Ctrl+C 停止）")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n假 LLM 已停止")


if __name__ == "__main__":
    main()
