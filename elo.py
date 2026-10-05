"""多 Agent 循环赛 + Elo 榜单（平台榜单模块的最小内核）。

- 两两循环赛，每对 --games 局；双方轮流左右侧（消除位置偏差，
  同时是 Task 5 side 逻辑的回归测试——若左右有系统偏差应修 env，见 PLAN §7）。
- Elo：初始 1500，K=32，按局内逐场更新；胜 1 / 负 0 / 平 0.5。
- 种子配对派生：crc32("a|b") + j（规范序）→ 同一命令重跑结果完全一致。

CLI：
    python elo.py ppo random footsies hidden:punisher hidden:turtle \
        file:agents/examples/always_block.py [--games 10] [--out leaderboard.json]
"""

import argparse
import itertools
import json
import sys
import zlib

from agents import load_agent
from arena import play_one

INITIAL_ELO = 1500.0
K = 32.0


def pair_seed(spec_a: str, spec_b: str, j: int) -> int:
    """配对种子：与参数顺序无关（规范序），跨运行确定。"""
    lo, hi = sorted((spec_a, spec_b))
    return zlib.crc32(f"{lo}|{hi}".encode("utf-8")) % 1_000_000 + j


def expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))


def tournament(specs, games=10):
    """循环赛；返回 (按 Elo 排序的榜单, 对阵明细)。"""
    uniq = list(dict.fromkeys(specs))       # 去重保序
    stats = {s: {"spec": s, "elo": INITIAL_ELO, "w": 0, "l": 0, "d": 0,
                 "ticks": []} for s in uniq}
    loaded = {s: load_agent(s) for s in uniq}
    pairwise = {}   # "a|b"(规范序) -> {a: 胜-负-判, b: 胜-负-判}

    for a_spec, b_spec in itertools.combinations(uniq, 2):
        key = f"{a_spec}|{b_spec}"
        pw = {a_spec: [0, 0, 0], b_spec: [0, 0, 0]}
        for j in range(games):
            seed = pair_seed(a_spec, b_spec, j)
            if j % 2 == 0:                   # 轮流左右侧
                left, right = loaded[a_spec], loaded[b_spec]
                a_side = 0
            else:
                left, right = loaded[b_spec], loaded[a_spec]
                a_side = 1
            outcome, data, _, _ = play_one(left, right, seed)

            sa = 0.5 if outcome["winner"] is None \
                else (1.0 if outcome["winner"] == a_side else 0.0)
            ra, rb = stats[a_spec]["elo"], stats[b_spec]["elo"]
            ea = expected(ra, rb)
            stats[a_spec]["elo"] = ra + K * (sa - ea)
            stats[b_spec]["elo"] = rb + K * ((1 - sa) - (1 - ea))

            for spec, s in ((a_spec, sa), (b_spec, 1.0 - sa)):
                st = stats[spec]
                st["ticks"].append(outcome["ticks"])
                if s == 1.0:
                    st["w"] += 1
                elif s == 0.0:
                    st["l"] += 1
                else:
                    st["d"] += 1
                pw[spec][0 if s == 1.0 else (2 if s == 0.5 else 1)] += 1
        pairwise[key] = pw

    board = sorted(stats.values(), key=lambda st: -st["elo"])
    for st in board:
        st["elo"] = round(st["elo"], 1)
        st["games"] = len(st["ticks"])
        st["mean_ticks"] = round(sum(st["ticks"]) / len(st["ticks"]), 1)
        del st["ticks"]
    return board, pairwise


def print_board(board, pairwise, games):
    print(f"循环赛榜单（每对 {games} 局，双方轮流左右侧，Elo 初值 "
          f"{INITIAL_ELO:.0f} K={K:.0f}）\n")
    print(f"  {'排名':<4}{'Agent':<40}{'Elo':>7}{'胜':>5}{'负':>5}{'平':>5}"
          f"{'场均tick':>10}")
    for rank, st in enumerate(board, 1):
        print(f"  {rank:<4}{st['spec']:<40}{st['elo']:>7.1f}{st['w']:>5}"
              f"{st['l']:>5}{st['d']:>5}{st['mean_ticks']:>10.1f}")
    print("\n  对阵明细（行 vs 列：胜-负-判）：")
    specs = [st["spec"] for st in board]
    short = {s: (s[:18] + "…" if len(s) > 19 else s) for s in specs}
    print(f"  {'':<20}" + "".join(f"{short[s]:>22}" for s in specs))
    for a in specs:
        cells = []
        for b in specs:
            if a == b:
                cells.append("-")
                continue
            key = f"{a}|{b}" if f"{a}|{b}" in pairwise else f"{b}|{a}"
            w, l, d = pairwise[key][a]
            cells.append(f"{w}-{l}-{d}")
        print(f"  {short[a]:<20}" + "".join(f"{c:>22}" for c in cells))


def main(argv=None):
    p = argparse.ArgumentParser(description="循环赛 + Elo 榜单")
    p.add_argument("agents", nargs="+", help="Agent 引用列表")
    p.add_argument("--games", type=int, default=10, help="每对局数")
    p.add_argument("--out", default="leaderboard.json", help="榜单 JSON 路径")
    args = p.parse_args(argv)

    board, pairwise = tournament(args.agents, games=args.games)
    print_board(board, pairwise, args.games)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"games_per_pair": args.games, "initial_elo": INITIAL_ELO,
                       "k": K, "board": board, "pairwise": pairwise},
                      f, ensure_ascii=False, indent=2)
        print(f"\n榜单已写入 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
