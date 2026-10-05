"""综合评分：胜率 65% + 速度 20% + 稳定性 15%（对齐平台文档评分区间）。

- 胜率分   = 总胜场占比（判平按 0.5 计），权重 0.65
- 速度分   = 1 - mean(结束tick / max_ticks)，权重 0.20
- 稳定性分 = 1 - min(1, 2*std(逐Bot胜率))，权重 0.15
  （各 Bot 局数相同时 std ∈ [0, 0.5]，线性归一到 [0,1]）

对手池默认 HIDDEN_POOL（反过拟合：公开榜打明牌，终榜打隐藏池）。
每 Bot 局数 = games // 池大小；左右侧逐场交替（同时回归检验对称性）。
种子派生固定：seed = base_seed*100 + bot_index*10 + j → 可复现。

CLI：
    python score.py ppo                       # 单 Agent 评分报告
    python score.py ppo --games 24 --report r.json
    python score.py --compare ppo random      # 并排对比
"""

import argparse
import json
import sys

import numpy as np

from agents import load_agent
from agents.script_agent import ScriptAgent
from arena import play_one
from hidden_bots import HIDDEN_POOL, PUBLIC_POOL, make_bot

W_WIN, W_SPEED, W_STAB = 0.65, 0.20, 0.15
BASE_SEED = 1000


def evaluate(spec: str, pool: str = "hidden", games: int = 24):
    """对整个对手池评测一个 Agent，返回报告 dict。"""
    pools = {"hidden": HIDDEN_POOL, "public": PUBLIC_POOL}
    if pool not in pools:
        raise ValueError(f"未知对手池: {pool}（可选: {sorted(pools)}）")
    bots = pools[pool]
    agent = load_agent(spec)

    n_bots = len(bots)
    per = max(1, games // n_bots)
    extra = games - per * n_bots
    counts = [per + (1 if i < extra else 0) for i in range(n_bots)]

    detail = {}
    all_ticks = []          # 每场 结束tick/max_ticks
    wins = draws = 0
    total = 0
    for i, (code, cfg) in enumerate(bots.items()):
        opp = ScriptAgent(make_bot(code), name=f"pool:{code}")
        w = l = d = 0
        ticks = []
        for j in range(counts[i]):
            seed = BASE_SEED * 100 + i * 10 + j
            # 左右侧逐场交替（消除位置偏差）
            if j % 2 == 0:
                left, right, my_side = agent, opp, 0
            else:
                left, right, my_side = opp, agent, 1
            outcome, data, _, _ = play_one(left, right, seed)
            if outcome["winner"] is None:
                d += 1
            elif outcome["winner"] == my_side:
                w += 1
            else:
                l += 1
            ticks.append(outcome["ticks"])
            all_ticks.append(outcome["ticks"] / data["max_ticks"])
        wins += w
        draws += d
        total += counts[i]
        detail[code] = {
            "desc": cfg["desc"], "games": counts[i], "w": w, "l": l, "d": d,
            "winrate": round((w + 0.5 * d) / counts[i], 4),
            "mean_ticks": round(float(np.mean(ticks)), 1),
        }
    win_rate = (wins + 0.5 * draws) / total
    speed = 1.0 - float(np.mean(all_ticks))
    wrs = [d["winrate"] for d in detail.values()]
    std = float(np.std(wrs))
    stab = 1.0 - min(1.0, 2.0 * std)

    scores = {
        "win_rate": round(win_rate, 4),
        "speed": round(speed, 4),
        "stability": round(stab, 4),
        "per_bot_std": round(std, 4),
        "win_pts": round(100 * W_WIN * win_rate, 1),
        "speed_pts": round(100 * W_SPEED * speed, 1),
        "stab_pts": round(100 * W_STAB * stab, 1),
        "total": round(100 * (W_WIN * win_rate + W_SPEED * speed
                              + W_STAB * stab), 1),
    }
    return {"agent": spec, "agent_name": agent.name, "pool": pool,
            "games": total, "detail": detail, "scores": scores}


def print_report(rep: dict):
    s = rep["scores"]
    print(f"评分对象: {rep['agent']}（{rep['agent_name']}）| "
          f"对手池: {rep['pool']} | 共 {rep['games']} 场（左右侧交替）")
    print(f"  {'Bot':<12}{'胜-负-判':>10}{'胜率':>8}{'平均tick':>10}  风格")
    for code, d in rep["detail"].items():
        wld = f"{d['w']}-{d['l']}-{d['d']}"
        print(f"  {code:<12}{wld:>10}{d['winrate']:>8.0%}"
              f"{d['mean_ticks']:>10.0f}  {d['desc']}")
    print(f"  胜率 {s['win_rate']:.0%} → {s['win_pts']}/65 | "
          f"速度 {s['speed']:.2f} → {s['speed_pts']}/20 | "
          f"稳定性 {s['stability']:.2f} → {s['stab_pts']}/15 "
          f"(逐Bot胜率std={s['per_bot_std']:.3f})")
    print(f"  总分: {s['total']} / 100")


def print_compare(rep1: dict, rep2: dict):
    print(f"对比: {rep1['agent']} vs {rep2['agent']}（对手池 "
          f"{rep1['pool']}/{rep2['pool']}）\n")
    rows = [("胜率", "win_rate", "{:.0%}"), ("速度", "speed", "{:.2f}"),
            ("稳定性", "stability", "{:.2f}"), ("总分", "total", "{:.1f}")]
    print(f"  {'指标':<8}{rep1['agent']:>16}{rep2['agent']:>16}")
    for label, key, fmt in rows:
        print(f"  {label:<8}{fmt.format(rep1['scores'][key]):>16}"
              f"{fmt.format(rep2['scores'][key]):>16}")
    print(f"\n  逐Bot胜率:")
    codes = list(rep1["detail"])
    print(f"  {'Bot':<12}{rep1['agent']:>14}{rep2['agent']:>14}")
    for code in codes:
        a = rep1["detail"][code]["winrate"]
        b = rep2["detail"].get(code, {}).get("winrate")
        b_str = f"{b:.0%}" if b is not None else "-"
        a_str = f"{a:.0%}"
        print(f"  {code:<12}{a_str:>14}{b_str:>14}")


def main(argv=None):
    p = argparse.ArgumentParser(description="综合评分（胜率/速度/稳定性）")
    p.add_argument("agent", nargs="?", help="被评 Agent 引用")
    p.add_argument("--pool", default="hidden", choices=["hidden", "public"])
    p.add_argument("--games", type=int, default=24)
    p.add_argument("--report", default=None, help="评分报告 JSON 输出路径")
    p.add_argument("--compare", nargs=2, metavar=("A", "B"),
                   help="并排对比两个 Agent")
    args = p.parse_args(argv)

    if args.compare:
        rep1 = evaluate(args.compare[0], pool=args.pool, games=args.games)
        rep2 = evaluate(args.compare[1], pool=args.pool, games=args.games)
        print_compare(rep1, rep2)
        if args.report:
            json.dump({"a": rep1, "b": rep2},
                      open(args.report, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=2)
            print(f"\n报告已写入 {args.report}")
        return 0

    if not args.agent:
        p.error("请提供 Agent 引用，或使用 --compare A B")
    rep = evaluate(args.agent, pool=args.pool, games=args.games)
    print_report(rep)
    if args.report:
        json.dump(rep, open(args.report, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print(f"报告已写入 {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
