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
import re
import sys
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
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
             "bow", "scarf", "antenna")


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
    return out


ACC_CN = {"none": "无饰品", "headband": "发带", "crown": "皇冠", "ahoge": "呆毛",
          "shades": "墨镜", "bow": "蝴蝶结", "scarf": "围巾", "antenna": "天线"}


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
    return out


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
                       int(env.p1.blocking), int(env.p2.blocking)])
    return {"players": data["players"], "seed": data["seed"],
            "max_ticks": data["max_ticks"], "outcome": data["outcome"],
            "actions": data["ticks"], "frames": frames,
            "skins": [p.get("skin") for p in data["players"]]}


@app.post("/api/agents/{agent_id}/skin")
@app.post("/api/agents/{agent_id}/skin")
def api_skin(agent_id: str, body: dict):
    """修改选手形象（无需重新上传策略）。"""
    with LOCK:
        roster = load_roster()
        e = next((x for x in roster if x["id"] == agent_id), None)
        if not e:
            raise HTTPException(404, f"选手不存在: {agent_id}")
        try:
            e["skin"] = validate_skin(body)
        except Exception as ex:  # noqa: BLE001
            raise HTTPException(400, f"形象配置无效: {ex}")
        save_roster(roster)
    return {"entry": e}


def run_match(a, b, seed):
    """执行并记录一场对局（演控台 API 与自动擂台赛共用）。需持有 LOCK。"""
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
    (DATA / "leaderboard.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


@app.post("/api/agents/upload")
async def api_upload(file: UploadFile = File(...), name: str = Form(None),
                     skin: str = Form(None)):
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
    else:                      # 未定制 → 由策略文件哈希生成专属形象
        entry_skin = auto_skin(raw)
    kind = agent.info().get("kind", "file")
    entry = {"id": uuid.uuid4().hex[:8], "spec": spec,
             "name": (name or agent.name or fname[:-3])[:24],
             "avatar": KIND_AVATAR.get(kind, "🥋"),
             "skin": entry_skin,
             "note": f"玩家上传 · 冒烟{'胜' if outcome['winner'] == 0 else '负'}"
                     f"随机君"}
    roster = load_roster() + [entry]
    save_roster(roster)
    # 全自动竞技场：注册即触发后台（补评分 → 刷新锦标赛），无需人工操作
    threading.Thread(target=_auto_cycle, args=(entry["id"],), daemon=True).start()
    return {"entry": entry, "smoke_outcome": outcome, "auto": "评分与锦标赛已自动排队"}


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
    """上传后自动：补评分（若无）→ 空闲则重跑锦标赛。后台线程执行。"""
    try:
        if new_agent_id:
            e = next((x for x in load_roster() if x["id"] == new_agent_id), None)
            if e and not (SCORES / f"{e['id']}.json").exists():
                print(f"[arena] 自动评分: {e['name']} …", flush=True)
                api_score(e["id"])
        if not TOUR["running"]:
            print("[arena] 自动锦标赛刷新…", flush=True)
            api_tournament({"games": 6})
    except Exception as ex:  # noqa: BLE001
        print(f"[arena] 自动周期异常: {ex}", flush=True)


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
        save_roster([dict(e, id=uuid.uuid4().hex[:8]) for e in BUILTIN])
        print(f"[kickoff] 已注册 {len(BUILTIN)} 名内置选手", flush=True)
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
