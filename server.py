"""立回斗士·AI 竞技场 —— 平台层最小服务器（Docker 本地演示）。

职责（对应 PLAN「平台愿景」的可信执行层）：
- 对局执行 POST /api/match        （无头、确定性、写回放）
- 隐藏池评分 POST /api/score/{id} （评分在服务端发生——演示镜像内置隐藏池，
                                   生产环境参数不下发）
- 循环赛榜   POST /api/tournament
- 选手管理   POST /api/agents/upload（file: 信任运行，无沙箱——平台工程范围）
- 回放/榜单/对局历史查询

首次启动自动：注册内置选手 → 开幕锦标赛 → 逐个隐藏池评分（后台线程）。
数据落 ./data（docker 挂载到宿主机，可直接用 play.py 复放回放）。
"""

import colorsys
import hashlib
import json
import os
import re
import sys
import threading
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from agents import load_agent          # noqa: E402
from arena import play_one             # noqa: E402
from elo import tournament             # noqa: E402
from fighting_env import FightingEnv, load_rules  # noqa: E402
from replay import load_replay as load_replay_json, rules_digest  # noqa: E402
from score import evaluate             # noqa: E402

DATA = ROOT / "data"
UPLOADS = DATA / "uploads"
REPLAYS = DATA / "replays"
SCORES = DATA / "scores"
for d in (DATA, UPLOADS, REPLAYS, SCORES):
    d.mkdir(parents=True, exist_ok=True)

LOCK = threading.Lock()          # 串行化所有对局计算（内核快，无需并行）
_CACHE = {}                      # spec → 已加载 Agent（predict 无状态可复用）
TOUR = {"running": False, "log": ""}   # 锦标赛状态（防并发重入）

BUILTIN = [
    dict(spec="ppo", name="阿焰·PPO", avatar="🧑‍🦰", note="强化学习 · PPO 训练",
         skin={"base": "hero", "main": "#f8f8fa", "trim": "#ff7a36",
               "head": "#603c26", "eye": "#282837", "acc": "headband"}),
    dict(spec="file:agents/examples/anti_throw.py", name="反投侠",
         avatar="🥋", note="玩家手写 · 克制投技流",
         skin={"base": "hero", "main": "#3f6b4f", "trim": "#ffd166",
               "head": "#2c4a3a", "eye": "#282837", "acc": "shades"}),
    dict(spec="footsies", name="铁蛋·标准型", avatar="🤖", note="脚本 · 均衡立回",
         skin={"base": "robot", "main": "#acc4ce", "trim": "#d6e8ec",
               "head": "#acc4ce", "eye": "#5ae6f0", "acc": "antenna"}),
    dict(spec="random", name="随机君", avatar="🎲", note="基线 · 纯随机",
         skin={"base": "robot", "main": "#8f7acd", "trim": "#cbbef0",
               "head": "#8f7acd", "eye": "#ff8cdf", "acc": "crown"}),
]

KIND_AVATAR = {"ppo": "🧑‍🦰", "script": "🤖", "file": "🥋",
               "random": "🎲", "http": "🧠"}

SKIN_ACCS = ("none", "headband", "crown", "ahoge", "shades",
             "bow", "scarf", "antenna", "horns")


def validate_skin(s):
    """形象配置校验：{base, main, trim, head, eye, acc}，非法值直接报错。"""
    if not isinstance(s, dict):
        raise ValueError("skin 必须是 JSON 对象")
    base = s.get("base", "hero")
    if base not in ("hero", "robot"):
        raise ValueError(f"base 只能是 hero/robot: {base}")
    acc = s.get("acc", "none")
    if acc not in SKIN_ACCS:
        raise ValueError(f"未知饰品: {acc}（可选 {SKIN_ACCS}）")
    out = {"base": base, "acc": acc}
    for k in ("main", "trim", "head", "eye"):
        v = s.get(k)
        if v is not None:
            if not re.fullmatch(r"#[0-9a-fA-F]{6}", str(v)):
                raise ValueError(f"颜色 {k} 需为 #RRGGBB: {v}")
            out[k] = str(v).lower()
    sc = s.get("scale")            # 体型倍率（可选）：>1 更高大壮硕
    if sc is not None:
        try:
            sc = float(sc)
        except Exception:          # noqa: BLE001
            raise ValueError(f"scale 需为数字: {sc}")
        if not 0.7 <= sc <= 1.35:
            raise ValueError(f"scale 需在 0.7~1.35: {sc}")
        out["scale"] = round(sc, 2)
    return out


ACC_CN = {"none": "无饰品", "headband": "发带", "crown": "皇冠", "ahoge": "呆毛",
          "shades": "墨镜", "bow": "蝴蝶结", "scarf": "围巾", "antenna": "天线",
          "horns": "犄角"}


def auto_skin(seed_bytes: bytes) -> dict:
    """由内容哈希确定性生成专属形象（同一份策略永远同一个形象）。"""
    h = hashlib.sha256(seed_bytes).digest()
    base = "hero" if h[0] % 2 == 0 else "robot"
    accs = [a for a in SKIN_ACCS if a != "none"]
    acc = accs[h[1] % len(accs)]

    def col(i, s, v):
        r, g, b = colorsys.hsv_to_rgb(h[i] / 255.0, s, v)
        return "#{:02x}{:02x}{:02x}".format(round(r * 255), round(g * 255),
                                            round(b * 255))

    return {"base": base, "acc": acc,
            "main": col(2, 0.45, 0.85), "trim": col(3, 0.65, 0.90),
            "head": col(4, 0.50, 0.50), "eye": col(5, 0.75, 0.95)}


def skin_desc(s: dict) -> str:
    base = "少年" if s.get("base") == "hero" else "机器人"
    return f"{base}·{ACC_CN.get(s.get('acc', 'none'), '?')}·主色{s.get('main', '?')}"

app = FastAPI(title="立回斗士·AI 竞技场")


# ---------------- 状态存取 ----------------
def load_roster():
    p = DATA / "roster.json"
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8"))


def save_roster(r):
    (DATA / "roster.json").write_text(
        json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")


def append_match(rec):
    with open(DATA / "matches.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_matches(limit=20):
    p = DATA / "matches.jsonl"
    if not p.exists():
        return []
    lines = p.read_text(encoding="utf-8").splitlines()
    return [json.loads(x) for x in lines[-limit:]][::-1]


def get_agent(spec):
    if spec not in _CACHE:
        _CACHE[spec] = load_agent(spec)
    return _CACHE[spec]


def by_id(aid):
    for e in load_roster():
        if e["id"] == aid:
            return e
    raise HTTPException(404, f"选手不存在: {aid}")


def entry_public(e):
    """选手条目 + 最新评分合并。"""
    out = dict(e)
    sp = SCORES / f"{e['id']}.json"
    out["score"] = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else None
    owner = player_by_id(e.get("owner"))
    out["owner_name"] = owner["name"] if owner else None
    return out


# ---------------- 玩家账号（令牌即账号） ----------------
def load_players():
    p = DATA / "players.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []


def save_players(pl):
    (DATA / "players.json").write_text(
        json.dumps(pl, ensure_ascii=False, indent=2), encoding="utf-8")


def player_by_id(pid):
    return next((p for p in load_players() if p["id"] == pid), None)


def _hash_token(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def _auth(token: str):
    """令牌 → 玩家；无效返回 None。"""
    if not token:
        return None
    h = _hash_token(token.strip())
    return next((p for p in load_players() if p["token_hash"] == h), None)


def _auth_bearer(authorization: str):
    p = _auth((authorization or "").removeprefix("Bearer ").strip())
    if not p:
        raise HTTPException(401, "无效令牌：请先 join.py register（或检查 .arena-credentials）")
    return p


def _fighter_of(pid):
    return next((e for e in load_roster() if e.get("owner") == pid), None)


@app.post("/api/players/register")
def api_player_register(body: dict):
    """注册玩家：返回一次性明文令牌（服务端只存哈希，遗失需重新注册）。"""
    name = (body.get("name") or "").strip()[:24]
    if not name:
        raise HTTPException(400, "玩家名不能为空")
    if any(p["name"] == name for p in load_players()):
        raise HTTPException(409, f"玩家名已被占用: {name}")
    token = "pa_" + uuid.uuid4().hex + uuid.uuid4().hex[:8]
    player = {"id": uuid.uuid4().hex[:8], "name": name,
              "token_hash": _hash_token(token),
              "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    save_players(load_players() + [player])
    print(f"[players] 新玩家注册: {name}", flush=True)
    return {"player_id": player["id"], "name": name, "token": token,
            "note": "令牌仅此一次返回，请妥善保存（join.py 会自动写入 .arena-credentials）"}


@app.get("/api/players/me")
def api_player_me(authorization: str = Header(None)):
    p = _auth_bearer(authorization)
    f = _fighter_of(p["id"])
    return {"player_id": p["id"], "name": p["name"],
            "created": p["created"],
            "fighter": {"id": f["id"], "name": f["name"]} if f else None}


# ---------------- API ----------------
@app.get("/api/agents")
def api_agents():
    return [entry_public(e) for e in load_roster()]


@app.get("/api/leaderboard")
def api_leaderboard():
    p = DATA / "leaderboard.json"
    if not p.exists():
        return {"ready": False, "msg": "锦标赛尚未开赛（首次启动约 1 分钟）"}
    data = json.loads(p.read_text(encoding="utf-8"))
    data["ready"] = True
    return data


@app.get("/api/rules")
def api_rules():
    """公开规则包：官网规则页与执行内核同源（规则透明 + 内容哈希）。"""
    raw = dict(load_rules(None).raw)
    raw["sha256"] = rules_digest(raw)
    return raw


@app.get("/my_agent_template.py")
def api_template():
    """官方参赛模板下载（join.py init 与官网下载同源）。"""
    return FileResponse(ROOT / "templates" / "agent_template.py",
                        media_type="text/x-python",
                        filename="my_agent_template.py")


@app.get("/agent_instructions.md")
def api_instructions():
    """给 AI IDE 的任务书（豆包/workbuddy/Cursor 粘贴用）。"""
    return FileResponse(ROOT / "AGENT_INSTRUCTIONS.md",
                        media_type="text/markdown",
                        filename="AGENT_INSTRUCTIONS.md")


@app.get("/bootstrap", response_class=PlainTextResponse)
def api_bootstrap():
    """一键参赛自举指令：复制全文 → 粘到 workbuddy/豆包 新对话 → 自动参赛。"""
    return (ROOT / "BOOTSTRAP_PROMPT.md").read_text(encoding="utf-8")


@app.get("/api/matches")
def api_matches(limit: int = 20):
    return read_matches(limit)


@app.get("/api/stats")
def api_stats():
    p = DATA / "matches.jsonl"
    n = len(p.read_text(encoding="utf-8").splitlines()) if p.exists() else 0
    return {"agents": len(load_roster()), "matches": n,
            "replays": len(list(REPLAYS.glob("*.json")))}


@app.get("/api/me/report")
def api_report(authorization: str = Header(None)):
    """玩家战报（供 Agent 迭代策略的反馈环）：近期战绩、逐对手胜率、败局回放清单。"""
    p = _auth_bearer(authorization)
    e = _fighter_of(p["id"])
    if not e:
        raise HTTPException(404, "你还没有选手：先 join.py submit")
    ms = [m for m in read_matches(300)
          if m["a"]["id"] == e["id"] or m["b"]["id"] == e["id"]]
    per_opp = {}
    losses = []
    form = ""
    for m in ms[::-1]:
        me_a = m["a"]["id"] == e["id"]
        opp = m["b"] if me_a else m["a"]
        w = m["outcome"]["winner"]
        res = "d" if w is None else ("w" if w == (0 if me_a else 1) else "l")
        d = per_opp.setdefault(opp["name"], [0, 0, 0])
        d["wld".index(res)] += 1
        form += {"w": "胜", "l": "负", "d": "平"}[res]
        if res in ("l", "d") and m.get("replay"):
            losses.append({"opp": opp["name"], "result": m["outcome"]["result"],
                           "replay": m["replay"], "seed": m["seed"]})
    sp = SCORES / f"{e['id']}.json"
    return {"player": p["name"], "fighter": e["name"],
            "score": json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else None,
            "recent_form": form[-20:], "per_opponent": per_opp,
            "losses": losses[:5],
            "hint": "让 Agent 分析败局回放（join.py report 打印复放命令），针对性改进后重新 submit"}


# ---------------- 评论（观众人类 / Agent 双通道） ----------------
_CMT_TS = {}


def _comments():
    p = DATA / "comments.jsonl"
    if not p.exists():
        return []
    return [json.loads(x) for x in
            p.read_text(encoding="utf-8").splitlines() if x.strip()]


@app.get("/api/comments")
def api_comments(target: str):
    cs = [c for c in _comments() if c["target"] == target]
    return cs[-50:]


@app.post("/api/comments")
def api_comment(body: dict, authorization: str = Header(None)):
    """评论：匿名昵称（观众）或玩家令牌（认证发言，强制显示玩家名+✓）。"""
    target = (body.get("target") or "").strip()
    text = (body.get("body") or "").strip()
    if not target.startswith(("match:", "player:")):
        raise HTTPException(400, "target 需为 match:<id> 或 player:<id>")
    if not (1 <= len(text) <= 300):
        raise HTTPException(400, "评论长度 1~300 字")
    player = _auth((authorization or "").removeprefix("Bearer ").strip())
    author = player["name"] if player else (body.get("author") or "观众")[:24]
    key = author
    if time.time() - _CMT_TS.get(key, 0) < 8:
        raise HTTPException(429, "发言太快，歇 8 秒")
    _CMT_TS[key] = time.time()
    c = {"ts": time.strftime("%m-%d %H:%M"), "target": target,
         "author": author, "certified": bool(player), "body": text}
    with open(DATA / "comments.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")
    return c


@app.get("/api/replays/{name}")
def api_replay(name: str):
    if not re.fullmatch(r"[\w.-]+\.json", name):
        raise HTTPException(400, "非法文件名")
    p = REPLAYS / name
    if not p.exists():
        raise HTTPException(404, "回放不存在")
    return json.loads(p.read_text(encoding="utf-8"))


@app.get("/api/replay_frames/{name}")
def api_replay_frames(name: str):
    """把回放展开成逐帧状态轨（服务端确定性重放导出，供官网动画播放器）。
    frames: [x1, x2, hp1, hp2, state1, state2, blocking1, blocking2] / 帧"""
    if not re.fullmatch(r"[\w.-]+\.json", name):
        raise HTTPException(400, "非法文件名")
    p = REPLAYS / name
    if not p.exists():
        raise HTTPException(404, "回放不存在")
    data = load_replay_json(p)
    rules = load_rules(None)
    if rules_digest(rules.raw) != data["rules"].get("sha256"):
        raise HTTPException(409, "规则包已变更，该回放对当前规则无效")
    env = FightingEnv(rules=rules.raw, max_ticks=int(data["max_ticks"]))
    env.reset(seed=int(data["seed"]))
    frames = []
    for a1, a2 in data["ticks"]:
        env.step_both(int(a1), int(a2))
        frames.append([round(env.p1.x, 4), round(env.p2.x, 4),
                       int(env.p1.hp), int(env.p2.hp),
                       env.p1.state, env.p2.state,
                       int(env.p1.blocking), int(env.p2.blocking),
                       round(env.p1.y, 4), round(env.p2.y, 4),
                       int(env.p1.crouching), int(env.p2.crouching)])
    return {"players": data["players"], "seed": data["seed"],
            "max_ticks": data["max_ticks"], "outcome": data["outcome"],
            "actions": data["ticks"], "frames": frames,
            "skins": [p.get("skin") for p in data["players"]],
            "frame_fields": ["x1", "x2", "hp1", "hp2", "s1", "s2",
                             "b1", "b2", "y1", "y2", "c1", "c2"]}


@app.post("/api/players/me/skin")
def api_skin(body: dict, authorization: str = Header(None)):
    """修改自己选手的形象（需玩家令牌，只能动自己的）。"""
    p = _auth_bearer(authorization)
    with LOCK:
        roster = load_roster()
        e = next((x for x in roster if x.get("owner") == p["id"]), None)
        if not e:
            raise HTTPException(404, "你还没有选手：先 join.py submit")
        try:
            e["skin"] = validate_skin(body)
        except Exception as ex:  # noqa: BLE001
            raise HTTPException(400, f"形象配置无效: {ex}")
        save_roster(roster)
    return {"entry": entry_public(e)}


def run_match(a, b, seed, tag=None):
    """执行并记录一场对局（擂台赛/挑战/战书共用）。需持有 LOCK。"""
    ag_a, ag_b = get_agent(a["spec"]), get_agent(b["spec"])
    outcome, _, _, path = play_one(ag_a, ag_b, seed, out_dir=str(REPLAYS))
    if path:   # 把注册名与形象写入回放（复放/动画渲染用）
        rp = json.loads(Path(path).read_text(encoding="utf-8"))
        for pl, e in zip(rp["players"], (a, b)):
            pl["name"] = e["name"]
            if e.get("skin"):
                pl["skin"] = e["skin"]
        Path(path).write_text(json.dumps(rp, ensure_ascii=False), encoding="utf-8")
    rec = {"id": uuid.uuid4().hex[:8], "ts": time.strftime("%m-%d %H:%M:%S"),
           "a": {"id": a["id"], "name": a["name"], "avatar": a["avatar"],
                 "skin": a.get("skin")},
           "b": {"id": b["id"], "name": b["name"], "avatar": b["avatar"],
                 "skin": b.get("skin")},
           "seed": seed, "outcome": outcome,
           "replay": Path(path).name if path else None}
    if tag:
        rec["duel"] = tag
    append_match(rec)
    return rec


@app.post("/api/match")
def api_match(body: dict):
    a, b = by_id(body.get("a")), by_id(body.get("b"))
    seed = int(body.get("seed", int(time.time())) % 100000)
    with LOCK:
        rec = run_match(a, b, seed)
    return rec


@app.post("/api/score/{agent_id}")
def api_score(agent_id: str):
    e = by_id(agent_id)
    with LOCK:
        report = evaluate(e["spec"], pool="hidden", games=24)
    (SCORES / f"{e['id']}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


@app.post("/api/tournament")
def api_tournament(body: dict | None = None):
    games = int((body or {}).get("games", 6))
    roster = load_roster()
    if len(roster) < 2:
        raise HTTPException(400, "至少需要 2 名选手")
    if TOUR["running"]:
        raise HTTPException(409, f"锦标赛已在进行中（{TOUR['log']}），请稍候")
    TOUR["running"] = True
    TOUR["log"] = f"{len(roster)} 人 · 每对 {games} 局 · " + time.strftime("%H:%M:%S")
    try:
        with LOCK:
            board, pairwise = tournament([e["spec"] for e in roster], games=games)
    finally:
        TOUR["running"] = False
    names = {e["spec"]: {"name": e["name"], "avatar": e["avatar"]}
             for e in roster}
    data = {"games_per_pair": games, "board": board, "pairwise": pairwise,
            "names": names, "updated": time.strftime("%m-%d %H:%M:%S")}
    prev = json.loads((DATA / "leaderboard.json").read_text(encoding="utf-8")) \
        if (DATA / "leaderboard.json").exists() else {}
    hist = prev.get("history", [])[-19:]
    if board:
        hist.append({"ts": data["updated"], "spec": board[0]["spec"],
                     "name": names[board[0]["spec"]]["name"]})
    data["history"] = hist
    (DATA / "leaderboard.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def _load_smtp():
    """SMTP 配置（管理员放 data/smtp.json：host/port/user/pass/sender/ssl）。"""
    p = DATA / "smtp.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _send_mail(to, title, body):
    cfg = _load_smtp()
    if not cfg:
        print("[notify] ⚠️ 未配置 SMTP（data/smtp.json），邮件通道不可用", flush=True)
        return
    import smtplib
    from email.mime.text import MIMEText
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = title
    msg["From"] = cfg.get("sender") or cfg.get("user") or "arena@localhost"
    msg["To"] = to
    host, port = cfg["host"], int(cfg.get("port", 25))
    if cfg.get("ssl"):
        with smtplib.SMTP_SSL(host, port, timeout=6) as s:
            if cfg.get("user"):
                s.login(cfg["user"], cfg.get("pass", ""))
            s.sendmail(msg["From"], [to], msg.as_string())
    else:
        with smtplib.SMTP(host, port, timeout=6) as s:
            if cfg.get("user"):
                s.login(cfg["user"], cfg.get("pass", ""))
            s.sendmail(msg["From"], [to], msg.as_string())


def _notify(player, title, body):
    """向玩家配置的通知通道推送（email/bark/飞书/通用 JSON webhook）。尽力而为，不阻塞。"""
    cfg = player.get("notify")
    if not cfg:
        return
    def _send():
        typ = cfg.get("type", "json")
        try:
            if typ == "email":
                _send_mail(cfg["to"], title, body)
            else:
                url = cfg.get("url")
                if not url:
                    return
                if typ == "bark":
                    q = urllib.parse.urlencode({"title": title, "body": body,
                                                "group": "arena"})
                    sep = "&" if "?" in url else "?"
                    urllib.request.urlopen(
                        urllib.request.Request(f"{url}{sep}{q}"), timeout=4).read()
                elif typ == "feishu":
                    data = json.dumps({"msg_type": "text",
                                       "content": {"text": f"{title}\n{body}"}}).encode()
                    req = urllib.request.Request(url, data=data,
                                                 headers={"Content-Type": "application/json"})
                    urllib.request.urlopen(req, timeout=4).read()
                else:   # 通用 JSON webhook（Discord/Slack 中继、自建服务）
                    data = json.dumps({"event": "duel", "title": title,
                                       "body": body}, ensure_ascii=False).encode()
                    req = urllib.request.Request(url, data=data,
                                                 headers={"Content-Type": "application/json"})
                    urllib.request.urlopen(req, timeout=4).read()
            print(f"[notify] ✅ 已推送 → {player['name']} ({typ})", flush=True)
        except Exception as ex:  # noqa: BLE001
            print(f"[notify] ⚠️ 推送失败 {player['name']}: {ex}", flush=True)
    threading.Thread(target=_send, daemon=True).start()


@app.post("/api/players/me/notify")
def api_set_notify(body: dict, authorization: str = Header(None)):
    """注册/清除我的通知通道：{type: email, to} 或 {url, type: json|bark|feishu}；空则清除。"""
    p = _auth_bearer(authorization)
    typ = body.get("type", "json")
    players = load_players()
    me = next(x for x in players if x["id"] == p["id"])
    if typ == "email":
        to = (body.get("to") or "").strip()
        if to and not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", to):
            raise HTTPException(400, f"邮箱格式无效: {to}")
        if to:
            me["notify"] = {"type": "email", "to": to}
        else:
            me.pop("notify", None)
    else:
        url = (body.get("url") or "").strip()
        if url and typ not in ("json", "bark", "feishu"):
            raise HTTPException(400, "type 只能是 email/json/bark/feishu")
        if url:
            me["notify"] = {"url": url, "type": typ}
        else:
            me.pop("notify", None)
    save_players(players)
    return {"player": p["name"], "notify": me.get("notify")}


# ---------------- 战书（玩家约战） ----------------
def _load_duels():
    p = DATA / "duels.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []


def _save_duels(ds):
    (DATA / "duels.json").write_text(
        json.dumps(ds, ensure_ascii=False, indent=2), encoding="utf-8")


def _duel_public(d):
    return {"id": d["id"], "ts": d["ts"],
            "from": d["from"]["fighter"], "to": d["to"]["fighter"],
            "text": d.get("text", ""), "games": d["games"],
            "status": d["status"], "result": d.get("result"),
            "score_live": d.get("score_live"),
            "final": d.get("final_match")}


@app.get("/api/duels")
def api_duels():
    ds = _load_duels()
    now = time.time()
    changed = False
    for d in ds:
        if d["status"] == "pending" and now > d["expires"]:
            d["status"] = "expired"
            changed = True
    if changed:
        _save_duels(ds)
    return [_duel_public(d) for d in reversed(ds[-20:])]


@app.post("/api/duels")
def api_duel_send(body: dict, authorization: str = Header(None)):
    """下战书（需令牌）：to = 对方选手名或玩家名；附狠话；BoN（3/5/7）。"""
    p = _auth_bearer(authorization)
    me = _fighter_of(p["id"])
    if not me:
        raise HTTPException(404, "你还没有选手：先 join.py submit")
    target_name = (body.get("to") or "").strip()
    players = {pp["id"]: pp["name"] for pp in load_players()}
    target = next((x for x in load_roster()
                   if x["name"] == target_name
                   or players.get(x.get("owner")) == target_name), None)
    if not target:
        raise HTTPException(404, f"找不到选手或玩家: {target_name}")
    if target["id"] == me["id"]:
        raise HTTPException(400, "不能对自己下战书")
    games = int(body.get("games", 7))
    if games not in (3, 5, 7):
        raise HTTPException(400, "games 只能是 3/5/7")
    text = (body.get("text") or "堂堂正正一战！")[:120]
    d = {"id": uuid.uuid4().hex[:6], "ts": time.strftime("%m-%d %H:%M"),
         "created": time.time(), "expires": time.time() + 7 * 86400,
         "from": {"pid": p["id"], "fighter": me["name"], "fid": me["id"]},
         "to": {"pid": target.get("owner"), "fighter": target["name"],
                "fid": target["id"]},
         "text": text, "games": games, "status": "pending"}
    with LOCK:
        _save_duels(_load_duels() + [d])
    print(f"[duel] 📜 战书: {me['name']} → {target['name']} 「{text}」", flush=True)
    # 通知对方：已注册通知通道则推送；无论是否注册，下次其 Agent 跑任意命令都会看到信使横幅
    tp = player_by_id(target.get("owner"))
    if tp:
        _notify(tp, f"⚔️ 战书：{me['name']} 向你下书！",
                f"「{text}」\n应战: join.py accept {d['id']} · 拒战: join.py decline {d['id']}"
                f"（7 天内有效，Bo{games}）")
    return _duel_public(d)


@app.get("/api/duels/inbox")
def api_duel_inbox(authorization: str = Header(None)):
    """战书通知：我收到的 / 我发出的。"""
    p = _auth_bearer(authorization)
    out = []
    for d in reversed(_load_duels()[-30:]):
        role = None
        if d["to"]["pid"] == p["id"]:
            role = "收到"
        elif d["from"]["pid"] == p["id"]:
            role = "发出"
        if role:
            x = _duel_public(d)
            x["role"] = role
            out.append(x)
    return out


@app.post("/api/duels/{duel_id}/accept")
def api_duel_accept(duel_id: str, authorization: str = Header(None)):
    """应战（仅受战方可）：立即执行 BoN 荣誉决斗，逐场入档。不影响循环赛排名。"""
    p = _auth_bearer(authorization)
    ds = _load_duels()
    d = next((x for x in ds if x["id"] == duel_id), None)
    if not d:
        raise HTTPException(404, "战书不存在")
    if d["to"]["pid"] != p["id"]:
        raise HTTPException(403, "只有受战方本人能应战")
    if d["status"] != "pending":
        raise HTTPException(409, f"战书状态已是 {d['status']}")
    me = _fighter_of(p["id"])
    roster = load_roster()
    opp = next((x for x in roster if x["id"] == d["from"]["fid"]), None)
    if not (me and opp):
        raise HTTPException(404, "选手已离场")
    with LOCK:
        aw = bw = 0                       # a = 下书方
        d["status"] = "fighting"
        d["score_live"] = [0, 0]
        _save_duels(ds)
        final = None
        for j in range(d["games"]):      # 逐场入档并直播比分（官网实时可见）
            a, b = (opp, me) if j % 2 == 0 else (me, opp)
            rec = run_match(a, b, (int(time.time()) + j) % 100000, tag="战书决斗")
            if rec["outcome"]["winner"] is not None:
                if (rec["outcome"]["winner"] == 0) == (j % 2 == 0):
                    aw += 1
                else:
                    bw += 1
            d["score_live"] = [aw, bw]
            final = rec
            _save_duels(ds)
            print(f"  [战书决斗] 第{j + 1}场 {rec['a']['name']} vs {rec['b']['name']} "
                  f"→ {rec['outcome']['result']}（{d['from']['fighter']} {aw}:{bw} "
                  f"{d['to']['fighter']}）", flush=True)
        d["status"] = "finished"
        d["result"] = f"{d['from']['fighter']} {aw}:{bw} {d['to']['fighter']}"
        d["final_match"] = {"id": final["id"], "replay": final["replay"]}
        _save_duels(ds)
    winner = d["from"]["fighter"] if aw > bw else (
        d["to"]["fighter"] if bw > aw else None)
    print(f"[duel] ⚔️ 决斗完成: {d['result']}（胜者 {winner}）", flush=True)
    fp = player_by_id(d["from"].get("pid"))
    if fp:
        _notify(fp, f"⚔️ 决斗结果：{d['result']}",
                f"你下的战书已决出胜负，胜者 {winner or '平局'}。"
                f"官网恩怨台可看决胜局回放。")
    return {**_duel_public(d), "winner": winner or "平"}


@app.post("/api/duels/{duel_id}/decline")
def api_duel_decline(duel_id: str, authorization: str = Header(None)):
    p = _auth_bearer(authorization)
    ds = _load_duels()
    d = next((x for x in ds if x["id"] == duel_id), None)
    if not d:
        raise HTTPException(404, "战书不存在")
    if d["to"]["pid"] != p["id"]:
        raise HTTPException(403, "只有受战方能拒战")
    if d["status"] != "pending":
        raise HTTPException(409, f"战书状态已是 {d['status']}")
    d["status"] = "declined"
    _save_duels(ds)
    return _duel_public(d)


@app.post("/api/challenge")
def api_challenge(authorization: str = Header(None)):
    """王座挑战（需玩家令牌）：对现任第一 Bo7。获胜≠直接加冕，但会立即触发
    全量循环赛重排——王冠永远只能通过「击败当下所有人」的完整循环赛易主。"""
    p = _auth_bearer(authorization)
    me = _fighter_of(p["id"])
    if not me:
        raise HTTPException(404, "你还没有选手：先 join.py submit")
    lb_path = DATA / "leaderboard.json"
    if not lb_path.exists():
        raise HTTPException(409, "榜单尚未生成，无法挑战")
    lb = json.loads(lb_path.read_text(encoding="utf-8"))
    champ = next((x for x in load_roster() if x["spec"] == lb["board"][0]["spec"]), None)
    if not champ:
        raise HTTPException(409, "现任第一已离场")
    if champ["id"] == me["id"]:
        return {"result": "you_are_champ",
                "msg": f"你已是现任第一（{me['name']}）。卫冕靠实力守住每届循环赛。"}
    with LOCK:
        cw, cc = _best_of(me, champ, 7, tag="王座挑战")
    if cw > cc:   # 挑战成功：立即重排全量循环赛，王冠由新一届冠军获得
        threading.Thread(target=_auto_cycle, daemon=True).start()
    return {"result": "win" if cw > cc else "lose",
            "challenger": me["name"], "champion": champ["name"],
            "score": f"{cw}:{cc}",
            "msg": (f"挑战成功 {cw}:{cc}！全量循环赛已触发重排——"
                    f"只有击败在场所有人才能加冕。") if cw > cc else
                   (f"挑战失败 {cw}:{cc}。王者仍在王座——"
                    f"他能当第一，正是因为当下没人能全面赢他。")}


@app.post("/api/agents/upload")
async def api_upload(file: UploadFile = File(...), name: str = Form(None),
                     skin: str = Form(None), token: str = Form(None)):
    raw = await file.read()
    if len(raw) > 100_000:
        raise HTTPException(400, "策略文件过大（>100KB）")
    if not file.filename.endswith(".py"):
        raise HTTPException(400, "只接受 .py 策略文件")
    fname = re.sub(r"[^\w.-]", "_", file.filename)
    fname = f"{uuid.uuid4().hex[:6]}_{fname}"      # 唯一前缀：同一文件可多名重复参赛
    path = UPLOADS / fname
    path.write_text(raw.decode("utf-8"), encoding="utf-8")
    spec = f"file:data/uploads/{fname}"   # 相对 cwd（容器 /app 与本地项目根均成立）
    try:
        agent = load_agent(spec)
        with LOCK:     # 冒烟测试：与随机君打一局，确认能跑完全场
            outcome, _, _, _ = play_one(agent, get_agent("random"), 0)
    except Exception as ex:  # noqa: BLE001
        path.unlink(missing_ok=True)
        raise HTTPException(400, f"策略无法运行: {ex}")
    entry_skin = None
    if skin:
        try:
            entry_skin = validate_skin(json.loads(skin))
        except Exception as ex:  # noqa: BLE001
            raise HTTPException(400, f"形象配置无效: {ex}")
    kind = agent.info().get("kind", "file")
    player = _auth(token)
    if not player:
        raise HTTPException(401, "需要玩家令牌：先运行 join.py register --name 你的名字")
    with LOCK:
        roster = load_roster()
        mine = next((x for x in roster if x.get("owner") == player["id"]), None)
        if mine:                      # 同玩家再次提交 = 策略迭代（选手身份不变）
            _CACHE.pop(mine["spec"], None)   # 丢弃旧策略缓存
            mine["spec"] = spec
            if name:
                mine["name"] = name[:24]
            if entry_skin:              # 仅显式传入时才改形象；不传则保留
                mine["skin"] = entry_skin
            mine["note"] = f"@{player['name']} · 策略已更新 " \
                           f"{time.strftime('%m-%d %H:%M')}"
            entry = mine
            created = False
        else:
            entry = {"id": uuid.uuid4().hex[:8], "spec": spec,
                     "owner": player["id"],
                     "name": (name or agent.name or fname[:-3])[:24],
                     "avatar": KIND_AVATAR.get(kind, "🥋"),
                     "skin": entry_skin or auto_skin(raw),   # 新建未定制→哈希生成专属形象
                     "note": f"@{player['name']} · 首秀测试"
                             f"{'胜' if outcome['winner'] == 0 else '负'}随机君"}
            roster.append(entry)
            created = True
        save_roster(roster)
    # 全自动竞技场：提交即触发后台（补评分 → 刷新锦标赛），无需人工操作
    threading.Thread(target=_auto_cycle, args=(entry["id"],), daemon=True).start()
    return {"entry": entry_public(entry), "created": created,
            "player": player["name"],
            "smoke_outcome": outcome, "auto": "评分与锦标赛已自动排队"}


# ---------------- 启动 ----------------
def _kickoff():
    """首次启动：开幕锦标赛 + 逐个隐藏池评分（后台执行，日志可见进度）。"""
    print("[kickoff] 开幕锦标赛（每对 6 局）……", flush=True)
    try:
        api_tournament({"games": 6})
        print("[kickoff] 锦标赛完成", flush=True)
    except Exception as ex:  # noqa: BLE001
        print(f"[kickoff] 锦标赛失败: {ex}", flush=True)
    for e in load_roster():
        print(f"[kickoff] 隐藏池评分: {e['name']} ……", flush=True)
        try:
            api_score(e["id"])
        except Exception as ex:  # noqa: BLE001
            print(f"[kickoff] 评分失败 {e['name']}: {ex}", flush=True)
    print("[kickoff] 全部就绪", flush=True)


def _auto_cycle(new_agent_id=None):
    """上传后自动：入位战（vs 现任第一）→ 补评分 → 空闲则全量循环赛。后台线程执行。"""
    try:
        e = next((x for x in load_roster() if x["id"] == new_agent_id), None) \
            if new_agent_id else None
        if e:
            if not (SCORES / f"{e['id']}.json").exists():
                print(f"[arena] 自动评分: {e['name']} …", flush=True)
                api_score(e["id"])
            lb = json.loads((DATA / "leaderboard.json").read_text(encoding="utf-8")) \
                if (DATA / "leaderboard.json").exists() else {"ready": False}
            if lb.get("ready") and lb["board"]:
                top = lb["board"][0]
                champ = next((x for x in load_roster()
                              if x["spec"] == top["spec"]), None)
                if champ and champ["id"] != e["id"]:
                    print(f"[arena] 入位战: {e['name']} 挑战现任第一 "
                          f"{champ['name']}（Bo3）…", flush=True)
                    _best_of(e, champ, 3, tag="入位战")
        if not TOUR["running"]:
            print("[arena] 自动锦标赛刷新…", flush=True)
            api_tournament({"games": 6})
    except Exception as ex:  # noqa: BLE001
        print(f"[arena] 自动周期异常: {ex}", flush=True)


def _best_of(fa, fb, n, tag="对抗"):
    """BoN 对抗（左右侧轮换），逐场入档。需持有 LOCK。返回 (fa胜场, fb胜场)。"""
    aw = bw = 0
    for j in range(n):
        a, b = (fa, fb) if j % 2 == 0 else (fb, fa)
        rec = run_match(a, b, (int(time.time()) + j) % 100000, tag=tag)
        if rec["outcome"]["winner"] is not None:
            if rec["outcome"]["winner"] == 0:
                if j % 2 == 0: aw += 1
                else: bw += 1
            else:
                if j % 2 == 0: bw += 1
                else: aw += 1
        print(f"  [{tag}] 第{j + 1}场 {rec['a']['name']} vs {rec['b']['name']} "
              f"→ {rec['outcome']['result']}（{fa['name']} {aw}:{bw} {fb['name']}）",
              flush=True)
    return aw, bw


def _exhibition():
    """自动擂台赛：每 45 秒轮转捉对厮杀一场，保持战报流鲜活（无人操作）。"""
    idx = 0
    time.sleep(20)
    while True:
        time.sleep(45)
        if not LOCK.acquire(blocking=False):     # 评分/锦标赛优先，忙则跳过本轮
            continue
        try:
            roster = load_roster()
            if len(roster) >= 2:
                a = roster[idx % len(roster)]
                b = roster[(idx + 1) % len(roster)]
                idx += 1
                if a["id"] == b["id"]:
                    continue
                rec = run_match(a, b, int(time.time()) % 100000)
                o = rec["outcome"]
                w = "平" if o["winner"] is None else rec["a" if o["winner"] == 0 else "b"]["name"]
                print(f"[arena] 擂台赛 {rec['a']['name']} vs {rec['b']['name']} "
                      f"→ {o['result']}（{w}）", flush=True)
        except Exception as ex:  # noqa: BLE001
            print(f"[arena] 擂台赛异常: {ex}", flush=True)
        finally:
            LOCK.release()


@app.on_event("startup")
def on_startup():
    first = not load_roster()
    if first:
        seed = [dict(e, id=uuid.uuid4().hex[:8]) for e in BUILTIN]
        if os.environ.get("ARENA_SKIP_PPO"):      # 磁盘紧张的服务器可裁掉 torch
            seed = [e for e in seed if e["spec"] != "ppo"]
            print("[kickoff] ARENA_SKIP_PPO=1：不注册 PPO 选手（免装 torch）", flush=True)
        save_roster(seed)
        print(f"[kickoff] 已注册 {len(seed)} 名内置选手", flush=True)
        threading.Thread(target=_kickoff, daemon=True).start()
    else:                      # 迁移：为历史选手补上形象（内置选手用签名形象）
        builtin_skin = {e["spec"]: e.get("skin") for e in BUILTIN}
        roster = load_roster()
        changed = False
        for e in roster:
            if not e.get("skin"):
                e["skin"] = (builtin_skin.get(e["spec"])
                              or auto_skin(e["spec"].encode("utf-8")))
                changed = True
        if changed:
            save_roster(roster)
            print("[kickoff] 已为历史选手生成专属形象", flush=True)
    threading.Thread(target=_exhibition, daemon=True).start()
    print("[arena] 自动擂台赛线程已启动（每 45 秒一场）", flush=True)


app.mount("/", StaticFiles(directory=str(ROOT / "static"), html=True),
          name="static")
