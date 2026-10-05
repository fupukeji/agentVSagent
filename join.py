#!/usr/bin/env python3
"""join.py —— 竞技场参赛一键 CLI（人类玩家与 AI IDE 通用）。

用法：
    python join.py init  [--name 名字]     环境自检(缺依赖自动装) + 生成 my_agent.py
    python join.py test  [--games 10]      三关测试：冒烟 → vs random(≥8/10) → vs footsies
    python join.py score [--games 24]      本地隐藏池评分（与服务器同口径）
    python join.py submit [--name 名字] [--server http://localhost:8000] [--no-score]
                                           提交上榜（默认自动触发服务端评分并显示排名）
    python join.py rank  [--server …]      查看服务器榜单
    python join.py guide                   打印给 AI IDE 的任务书指引

给 AI IDE 的推荐流程：把 AGENT_INSTRUCTIONS.md 喂给 AI，让它自主
编辑 my_agent.py → join.py test → join.py score → join.py submit。
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
import uuid

ROOT = os.path.dirname(os.path.abspath(__file__))
AGENT_FILE = os.path.join(ROOT, "my_agent.py")
TEMPLATE = os.path.join(ROOT, "templates", "agent_template.py")
INSTRUCTIONS = os.path.join(ROOT, "AGENT_INSTRUCTIONS.md")
DEFAULT_SERVER = os.environ.get("ARENA_SERVER", "http://localhost:8000")

PASS, FAIL, INFO = "✓", "✗", "·"


# ---------------- 基础工具 ----------------
def die(msg, code=1):
    print(f"{FAIL} {msg}")
    sys.exit(code)


def ensure_deps(auto=True):
    """确保 numpy/gymnasium 可用（脚本 Agent 的全部依赖）。"""
    try:
        import numpy  # noqa: F401
        import gymnasium  # noqa: F401
        return True
    except ImportError:
        pass
    if not auto:
        return False
    print(f"{INFO} 缺少依赖，自动安装 numpy gymnasium …")
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "numpy", "gymnasium"])
    if r.returncode != 0:
        die("依赖安装失败，请手动执行: pip install numpy gymnasium")
    return True


def require_agent_file():
    if not os.path.exists(AGENT_FILE):
        die(f"未找到 my_agent.py —— 先运行: python {os.path.basename(__file__)} init")


def http_json(method, url, payload=None, timeout=300):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.URLError as e:
        die(f"无法连接服务器 {url}: {e}\n  （服务器未启动？docker compose up -d）")
    except urllib.error.HTTPError as e:
        try:
            msg = json.loads(e.read().decode()).get("detail", e.reason)
        except Exception:  # noqa: BLE001
            msg = e.reason
        die(f"服务器拒绝: {msg}")


def http_upload(url, path, name):
    """multipart 上传（纯标准库）。"""
    boundary = "----joinpy" + uuid.uuid4().hex
    body = b""
    for k, v in ({"name": name or ""}).items():
        body += (f"--{boundary}\r\nContent-Disposition: form-data; "
                 f"name=\"{k}\"\r\n\r\n{v}\r\n").encode()
    fn = os.path.basename(path)
    body += (f"--{boundary}\r\nContent-Disposition: form-data; "
             f"name=\"file\"; filename=\"{fn}\"\r\n"
             f"Content-Type: text/x-python\r\n\r\n").encode()
    with open(path, "rb") as f:
        body += f.read()
    body += f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type":
                                          f"multipart/form-data; boundary={boundary}"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        try:
            msg = json.loads(e.read().decode()).get("detail", e.reason)
        except Exception:  # noqa: BLE001
            msg = e.reason
        die(f"上传失败: {msg}")


def play_n(spec_a, spec_b, games, desc_a, desc_b):
    """本地批量对局，返回 a 的胜场。"""
    from agents import load_agent
    from arena import play_one
    a, b = load_agent(spec_a), load_agent(spec_b)
    w = 0
    for s in range(games):
        left, right = (a, b) if s % 2 == 0 else (b, a)
        me = 0 if s % 2 == 0 else 1
        outcome = play_one(left, right, s)[0]
        if outcome["winner"] == me:
            w += 1
    print(f"  {desc_a} vs {desc_b}: {w}/{games}")
    return w


# ---------------- 子命令 ----------------
def cmd_init(args):
    print("== 环境自检 ==")
    ensure_deps()
    print(f"{PASS} 依赖就绪（numpy / gymnasium）")
    if os.path.exists(AGENT_FILE) and not args.force:
        print(f"{INFO} my_agent.py 已存在（--force 可覆盖）")
    else:
        shutil.copyfile(TEMPLATE, AGENT_FILE)
        print(f"{PASS} 已生成 my_agent.py（基于官方模板，内置「确反」第一课）")
    print(f"\n下一步：")
    print(f"  人工参赛: 编辑 my_agent.py 后运行 python join.py test")
    print(f"  AI 代打 : 把 AGENT_INSTRUCTIONS.md 交给你的 AI IDE，")
    print(f"            对它说「请阅读 AGENT_INSTRUCTIONS.md 并完成参赛任务」")
    return 0


def cmd_test(args):
    require_agent_file()
    ensure_deps()
    spec = f"file:{AGENT_FILE}"
    print("== 第一关 · 冒烟（能否完整跑完一局）==")
    try:
        from agents import load_agent
        from arena import play_one
        agent = load_agent(spec)
        outcome = play_one(agent, load_agent("random"), 0)[0]
        print(f"{PASS} 冒烟通过（{outcome['result']}，{outcome['ticks']}帧）")
    except Exception as e:  # noqa: BLE001
        die(f"冒烟失败: {e}")
    print("== 第二关 · vs 随机君（门槛 ≥8/10）==")
    w = play_n(spec, "random", args.games, "我的AI", "随机君")
    gate1 = w >= args.games * 0.8
    print(f"{PASS if gate1 else FAIL} {'通过' if gate1 else '未达标'}"
          f"（当前 AI 基础逻辑{'正确' if gate1 else '有问题'}）")
    print("== 第三关 · vs 铁蛋·标准型（及格线参考 ≥5/10）==")
    w2 = play_n(spec, "footsies", args.games, "我的AI", "标准型")
    gate2 = w2 >= args.games * 0.5
    print(f"{PASS if gate2 else INFO} "
          f"{'立回及格' if gate2 else '尚欠立回（参考战术四课）'}")
    if not gate1:
        print("\n未过关。检查 act() 是否返回 0~5、是否在射程内出招"
              "（轻击 0.13 / 重击 0.19 / 投技 0.07）。")
        return 1
    print(f"\n{PASS} 测试完成。运行 python join.py score 查看隐藏池预估分。")
    return 0


def cmd_score(args):
    require_agent_file()
    ensure_deps()
    from score import evaluate, print_report
    rep = evaluate(f"file:{AGENT_FILE}", pool="hidden", games=args.games)
    print_report(rep)
    t = rep["scores"]["total"]
    lvl = ("L1" if t < 60 else "L2" if t < 60 else "L3" if t < 75
           else "L4" if t < 86 else "L5")
    print(f"\n评级: {lvl}（阶梯: 60/75/86 → 上道/强者/冠军级）")
    return 0


def cmd_submit(args):
    require_agent_file()
    server = args.server.rstrip("/")
    name = args.name
    if not name:
        default = "我的AI"
        name = input(f"选手名（回车取「{default}」）: ").strip() or default
    print(f"== 提交到 {server} ==")
    r = http_upload(f"{server}/api/agents/upload", AGENT_FILE, name)
    entry = r["entry"]
    smoke = r["smoke_outcome"]
    print(f"{PASS} 注册成功: {entry['avatar']} {entry['name']}（服务端冒烟 "
          f"{'胜' if smoke['winner'] == 0 else '负'}随机君）")
    if args.no_score:
        print(f"{INFO} 跳过服务端评分（--no-score）")
    else:
        print(f"{INFO} 服务端隐藏池评分中（24 场，约 10~30 秒）…")
        rep = http_json("POST", f"{server}/api/score/{entry['id']}")
        s = rep["scores"]
        print(f"{PASS} 综合分 {s['total']} / 100"
              f"（胜率 {s['win_rate']:.0%} · 速度 {s['speed']:.2f} · "
              f"稳定性 {s['stability']:.2f}）")
        for code, d in rep["detail"].items():
            print(f"    {code:<10} 胜率 {d['winrate']:.0%}  {d['desc'][:18]}")
    lb = http_json("GET", f"{server}/api/leaderboard")
    if lb.get("ready"):
        board = lb["board"]
        names = lb.get("names", {})
        print(f"\n== 当前榜单 ==")
        for i, b in enumerate(board, 1):
            nm = names.get(b["spec"], {}).get("name", b["spec"])
            mark = " ← 你" if b["spec"] == entry["spec"] else ""
            print(f"  {i}. {nm:<12} Elo {b['elo']:>6.1f}{mark}")
        mine = [i for i, b in enumerate(board, 1) if b["spec"] == entry["spec"]]
        if mine:
            print(f"\n{PASS} 你当前排第 {mine[0]} 名。想让 AI 继续进化？"
                  f"改完 my_agent.py 再次 submit 即可。")
        else:
            print(f"\n{INFO} 你已注册，当前榜还未包含你（评分已入档）。"
                  f"下一轮锦标赛后自动入榜。")
    return 0


def cmd_rank(args):
    server = args.server.rstrip("/")
    lb = http_json("GET", f"{server}/api/leaderboard")
    if not lb.get("ready"):
        die("锦标赛尚未完成，稍后再查")
    names = lb.get("names", {})
    print(f"榜单（更新于 {lb['updated']}）")
    for i, b in enumerate(lb["board"], 1):
        nm = names.get(b["spec"], {}).get("name", b["spec"])
        print(f"  {i}. {nm:<14} Elo {b['elo']:>6.1f}  "
              f"{b['w']}-{b['l']}-{b['d']}")
    return 0


def cmd_guide(args):
    p = os.path.join(ROOT, "BOOTSTRAP_PROMPT.md")
    if os.path.exists(p):
        print("把下面整段复制到 workbuddy / 豆包 的新对话即可自动参赛：\n")
        print(open(p, encoding="utf-8").read())
    else:
        print("请阅读 AGENT_INSTRUCTIONS.md 并完成参赛任务。")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description="竞技场参赛 CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init", help="环境自检 + 生成 my_agent.py")
    s.add_argument("--name", default=None, help="(保留：模板内改)")
    s.add_argument("--force", action="store_true", help="覆盖已有 my_agent.py")
    s.set_defaults(func=cmd_init)

    s = sub.add_parser("test", help="三关本地测试")
    s.add_argument("--games", type=int, default=10)
    s.set_defaults(func=cmd_test)

    s = sub.add_parser("score", help="本地隐藏池评分")
    s.add_argument("--games", type=int, default=24)
    s.set_defaults(func=cmd_score)

    s = sub.add_parser("submit", help="提交上榜")
    s.add_argument("--name", default=None)
    s.add_argument("--server", default=DEFAULT_SERVER)
    s.add_argument("--no-score", action="store_true")
    s.set_defaults(func=cmd_submit)

    s = sub.add_parser("rank", help="查看服务器榜单")
    s.add_argument("--server", default=DEFAULT_SERVER)
    s.set_defaults(func=cmd_rank)

    s = sub.add_parser("guide", help="AI IDE 任务指引")
    s.set_defaults(func=cmd_guide)

    args = p.parse_args(argv)
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
