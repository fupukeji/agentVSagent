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
import time
import urllib.error
import urllib.request
import uuid

ROOT = os.path.dirname(os.path.abspath(__file__))
AGENT_FILE = os.path.join(ROOT, "my_agent.py")
TEMPLATE = os.path.join(ROOT, "templates", "agent_template.py")
INSTRUCTIONS = os.path.join(ROOT, "AGENT_INSTRUCTIONS.md")
CRED_FILE = os.path.join(ROOT, ".arena-credentials")
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


def http_json(method, url, payload=None, timeout=300, headers=None):
    data = json.dumps(payload).encode() if payload is not None else None
    hdrs = {"Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, data=data, method=method, headers=hdrs)
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


def http_upload(url, path, fields):
    """multipart 上传（纯标准库）。fields: {字段名: 值}"""
    boundary = "----joinpy" + uuid.uuid4().hex
    body = b""
    for k, v in fields.items():
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
    # 不做交互式询问（Agent 全自主）：更新选手时不传名则沿用现名；新建时服务端自动取名
    fields = {"token": require_cred()["token"]}
    if args.name:
        fields["name"] = args.name
    print(f"== 提交到 {server} ==")
    if args.skin:
        try:
            json.loads(args.skin)      # 早失败：JSON 非法立即提示
        except Exception:
            die(f"--skin 不是合法 JSON: {args.skin}")
        fields["skin"] = args.skin
    r = http_upload(f"{server}/api/agents/upload", AGENT_FILE, fields)
    entry = r["entry"]
    smoke = r["smoke_outcome"]
    skin = entry.get("skin") or {}
    base = "少年格斗家" if skin.get("base") == "hero" else "机器人"
    acc_cn = {"none": "无饰品", "headband": "发带", "crown": "皇冠", "ahoge": "呆毛",
              "shades": "墨镜", "bow": "蝴蝶结", "scarf": "围巾",
              "antenna": "天线"}.get(skin.get("acc", "none"), "?")
    verb = "策略已更新（选手身份不变）" if r.get("created") is False else "注册成功"
    print(f"{PASS} {verb}: {entry['avatar']} {entry['name']}（首秀测试"
          f"{'胜' if smoke['winner'] == 0 else '负'}随机君）")
    print(f"{PASS} 战斗形象: {base} · {acc_cn} · 主色 {skin.get('main', '?')}"
          f"（官网动画回放中生效；--skin 可定制，见任务书「形象定制」）")
    if args.no_score:
        print(f"{INFO} 跳过评分（--no-score）")
    else:
        print(f"{INFO} 服务端自动评分 + 锦标赛刷新已排队，等待评分结果…")
        rep = None
        for _ in range(60):                      # 轮询至多 ~150 秒
            agents = http_json("GET", f"{server}/api/agents")
            me = next((x for x in agents if x["id"] == entry["id"]), None)
            if me and me.get("score"):
                rep = me["score"]
                break
            time.sleep(2.5)
        if rep:
            s = rep["scores"]
            print(f"{PASS} 综合分 {s['total']} / 100"
 f"（胜率 {s['win_rate']:.0%} · 速度 {s['speed']:.2f} · "
                  f"稳定性 {s['stability']:.2f}）")
            for code, d in rep["detail"].items():
                print(f"    {code:<10} 胜率 {d['winrate']:.0%}  {d['desc'][:18]}")
        else:
            print(f"{INFO} 评分仍在排队（服务器忙），稍后自动完成，可运行 "
                  f"join.py rank 查看榜单")
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


def save_cred(server, player_id, name, token):
    with open(CRED_FILE, "w", encoding="utf-8") as f:
        json.dump({"server": server, "player_id": player_id,
                   "name": name, "token": token}, f, ensure_ascii=False)
    os.chmod(CRED_FILE, 0o600)


def load_cred():
    if not os.path.exists(CRED_FILE):
        return None
    return json.load(open(CRED_FILE, encoding="utf-8"))


def require_cred():
    c = load_cred()
    if not c:
        die("未找到玩家凭证 .arena-credentials —— 先运行: "
            f"python {os.path.basename(__file__)} register --name 玩家名")
    return c


def cmd_register(args):
    server = args.server.rstrip("/")
    name = args.name or input("玩家名: ").strip()
    r = http_json("POST", f"{server}/api/players/register", {"name": name})
    save_cred(server, r["player_id"], r["name"], r["token"])
    print(f"{PASS} 玩家「{r['name']}」注册成功，令牌已保存到 .arena-credentials")
    print(f"{INFO} 令牌即账号：之后 submit/skin/report/comment 自动登录；"
          f"把它交给你的 AI（workbuddy 等）即可全权代理")
    return 0


def cmd_whoami(args):
    c = require_cred()
    r = http_json("GET", f"{c['server']}/api/players/me",
                  headers={"Authorization": f"Bearer {c['token']}"})
    f = r.get("fighter")
    print(f"{PASS} 玩家: {r['name']}（注册于 {r['created']}）")
    print(f"{PASS} 选手: {f['name'] if f else '尚无——去 submit 创建'}")
    return 0


def cmd_skin(args):
    require_agent_file()
    c = require_cred()
    try:
        json.loads(args.skin)
    except Exception:
        die("--skin 不是合法 JSON")
    r = http_json("POST", f"{c['server']}/api/players/me/skin",
                  json.loads(args.skin),
                  headers={"Authorization": f"Bearer {c['token']}"})
    s = r["entry"]["skin"]
    print(f"{PASS} 形象已更新: {s.get('base')} · {s.get('acc')} · 主色 {s.get('main')}")
    return 0


def cmd_report(args):
    c = require_cred()
    r = http_json("GET", f"{c['server']}/api/me/report",
                  headers={"Authorization": f"Bearer {c['token']}"})
    print(f"== 战报：{r['player']} 的选手「{r['fighter']}」 ==")
    sc = r.get("score")
    if sc:
        s = sc["scores"]
        print(f"综合分 {s['total']}（胜率 {s['win_rate']:.0%} · 速度 {s['speed']:.2f} · "
              f"稳定性 {s['stability']:.2f}）")
        print("逐对手:", " · ".join(
            f"{k} {v[0]}胜{v[1]}负{v[2]}平" for k, v in r["per_opponent"].items()))
    print(f"近期战绩: {r['recent_form'] or '暂无'}")
    if r["losses"]:
        print("败局回放（交给 AI 分析弱点）:")
        for l in r["losses"]:
            print(f"  vs {l['opp']} ({l['result']}) → "
                  f"python play.py --replay data/replays/{l['replay']}")
    print(f"{INFO} {r['hint']}")
    return 0


def cmd_comment(args):
    c = load_cred()
    headers = {"Authorization": f"Bearer {c['token']}"} if c else {}
    r = http_json("POST", f"{(c or {}).get('server', args.server).rstrip('/')}/api/comments",
                  {"target": args.target, "body": args.text,
                   "author": args.author}, headers=headers)
    tag = "✓认证" if r["certified"] else "观众"
    print(f"{PASS} 评论已发布（{tag}）: {r['body'][:40]}")
    return 0


def cmd_challenge(args):
    c = require_cred()
    print("== 王座挑战（Bo7 · 左右侧轮换）==")
    r = http_json("POST", f"{c['server']}/api/challenge", {},
                  headers={"Authorization": f"Bearer {c['token']}"})
    if r["result"] == "you_are_champ":
        print(f"{INFO} {r['msg']}")
    else:
        print(f"{PASS if r['result'] == 'win' else FAIL} "
              f"{r['challenger']} {r['score']} {r['champion']}")
        print(f"{INFO} {r['msg']}")
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
    s.add_argument("--skin", default=None, metavar="JSON",
                   help="战斗形象 JSON（base/main/trim/head/eye/acc），见任务书「形象定制」；缺省由策略哈希自动生成")
    s.add_argument("--no-score", action="store_true")
    s.set_defaults(func=cmd_submit)

    s = sub.add_parser("rank", help="查看服务器榜单")
    s.add_argument("--server", default=DEFAULT_SERVER)
    s.set_defaults(func=cmd_rank)

    s = sub.add_parser("guide", help="AI IDE 任务指引")
    s.set_defaults(func=cmd_guide)

    s = sub.add_parser("register", help="注册玩家（获得长期令牌，令牌即账号）")
    s.add_argument("--name", default=None)
    s.add_argument("--server", default=DEFAULT_SERVER)
    s.set_defaults(func=cmd_register)

    s = sub.add_parser("whoami", help="验证凭证与查看自己的选手")
    s.add_argument("--server", default=DEFAULT_SERVER)
    s.set_defaults(func=cmd_whoami)

    s = sub.add_parser("skin", help="更新自己选手的战斗形象（需令牌）")
    s.add_argument("--skin", required=True, metavar="JSON")
    s.set_defaults(func=cmd_skin)

    s = sub.add_parser("report", help="战报反馈：战绩/逐对手/败局回放（供 AI 迭代）")
    s.set_defaults(func=cmd_report)

    s = sub.add_parser("comment", help="以玩家（有凭证时认证）或观众身份评论")
    s.add_argument("--target", required=True, help="match:<对局id> 或 player:<选手id>")
    s.add_argument("--text", required=True)
    s.add_argument("--author", default=None, help="无凭证时的昵称")
    s.add_argument("--server", default=DEFAULT_SERVER)
    s.set_defaults(func=cmd_comment)

    s = sub.add_parser("challenge", help="王座挑战：对现任第一 Bo7，胜则触发全量重排")
    s.set_defaults(func=cmd_challenge)

    args = p.parse_args(argv)
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
