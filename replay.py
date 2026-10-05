"""回放格式与复放（确定性重放 + 一致性校验）。

格式 v1 核心思想：种子 + 双方动作序列 = 完整复现。
- 观察不存储：由确定性规则从 (seed, actions) 推导（见 PLAN §7 决策记录）；
  若未来规则引入随机事件，需升级 format_version。
- verify_replay：用种子+动作序列重放 step_both，校验 outcome 完全一致，
  是「平台产出的回放不可伪造」的最弱保证（重放一致性）。

CLI：
    python replay.py verify <file>   # 校验回放
    python replay.py info <file>     # 打印回放摘要
"""

import hashlib
import json
import sys

from fighting_env import FightingEnv, load_rules

FORMAT_VERSION = 1


def rules_digest(rules_raw: dict) -> str:
    """规则包内容哈希（规范化 JSON 序列化后取 sha256，与文件排版无关）。"""
    canonical = json.dumps(rules_raw, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def save_replay(path: str, data: dict) -> str:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path


def load_replay(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if data.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"不支持的回放版本: {data.get('format_version')}")
    for key in ("rules", "seed", "max_ticks", "players", "ticks", "outcome"):
        if key not in data:
            raise ValueError(f"回放缺少字段: {key}")
    return data


def replay_game(data: dict, rules_path=None):
    """按回放数据在无头环境重放整局，返回 (outcome, 逐帧HP轨迹)。"""
    rules = load_rules(rules_path)  # None → rules/default.json
    digest = rules_digest(rules.raw)
    if digest != data["rules"].get("sha256"):
        raise ValueError(
            f"规则包哈希不一致: 回放 {data['rules'].get('sha256')[:12]}… "
            f"vs 当前 {digest[:12]}…（规则已被修改，回放对当前规则无效）")
    env = FightingEnv(rules=rules.raw, max_ticks=int(data["max_ticks"]))
    env.reset(seed=int(data["seed"]))
    hp_track = []
    for a1, a2 in data["ticks"]:
        env.step_both(int(a1), int(a2))
        hp_track.append([int(env.p1.hp), int(env.p2.hp)])
    return env.outcome(), hp_track


def verify_replay(path: str, rules_path=None):
    """校验回放：重放一致 → (True, 'OK')；否则 (False, 差异说明)。

    比对两级：逐帧 HP 轨迹（hp_track，若存在）+ 终局 outcome。
    注：硬直中被忽略的指令不影响游戏状态，重放仍算一致（语义等价）。"""
    try:
        data = load_replay(path)
        got, hp_track = replay_game(data, rules_path=rules_path)
    except Exception as e:  # noqa: BLE001
        return False, f"无法重放: {e}"
    want_track = data.get("hp_track")
    if want_track is not None and want_track != hp_track:
        for t, (a, b) in enumerate(zip(want_track, hp_track)):
            if a != b:
                return False, f"不一致 → 第{t + 1}帧 HP: 回放记录 {a} vs 重放结果 {b}"
        return False, (f"不一致 → HP 轨迹长度: 回放记录 {len(want_track)} vs "
                       f"重放结果 {len(hp_track)}")
    want = data["outcome"]
    if got != want:
        diffs = []
        for k in ("winner", "hp", "ticks", "result"):
            if got.get(k) != want.get(k):
                diffs.append(f"{k}: 回放记录 {want.get(k)} vs 重放结果 {got.get(k)}")
        return False, "不一致 → " + "; ".join(diffs)
    return True, "OK"


def replay_summary(data: dict) -> str:
    p1, p2 = data["players"]
    n = len(data["ticks"])
    o = data["outcome"]
    return (f"对局: {p1['name']} vs {p2['name']} | 规则 {data['rules']['name']} "
            f"| seed {data['seed']} | {n} 帧 | 结果 {o['result']} "
            f"hp={o['hp']}")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2 or argv[0] not in ("verify", "info"):
        print("用法: python replay.py verify|info <回放文件>")
        return 2
    cmd, path = argv
    if cmd == "info":
        print(replay_summary(load_replay(path)))
        return 0
    ok, msg = verify_replay(path)
    print(("OK" if ok else "FAIL"), msg if msg != "OK" else "")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
