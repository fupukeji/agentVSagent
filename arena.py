"""无头对局执行器（平台「执行层」最小内核）。

脱离 pygame，纯模拟执行「Agent vs Agent」一局或一批，输出结果与回放。
内部使用 step_both：双方公平，AI 侧不再有「环境替对手出招」的特殊地位。

CLI：
    python arena.py run <agent1> <agent2> [--seed N] [--rules PATH] [--out replays/]
    python arena.py batch <agent1> <agent2> [--games 20] [--seeds 0-19]
    python arena.py demo        # 内置演示：ppo vs footsies 打 3 场
"""

import argparse
import os
import re
import sys
import time

import numpy as np

from fighting_env import FightingEnv, load_rules
from agents import load_agent
from replay import (
    FORMAT_VERSION, rules_digest, save_replay, verify_replay,
)

DEFAULT_OUT = "replays"


def _safe_name(s: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.+-]", "-", s)


def play_one(agent1, agent2, seed, rules=None, out_dir=None):
    """执行一局，返回 (outcome, 回放dict, p1回报)。out_dir 给定时写回放并自检。"""
    env = FightingEnv(rules=rules)
    env.reset(seed=seed)
    # 双方种子派生固定（相差大素数，避免 batch 相邻种子串场）→ 可复现
    agent1.reset(seed, side=0)
    agent2.reset(seed + 10007, side=1)

    ticks = []
    ret1 = 0.0
    done = False
    while not done:
        a1 = int(agent1.act(env.obs_for(0)))
        a2 = int(agent2.act(env.obs_for(1)))
        _, r, term, trunc, _ = env.step_both(a1, a2)
        ticks.append([a1, a2])
        ret1 += float(r)
        done = term or trunc

    outcome = env.outcome()
    rules_obj = env.rules
    data = {
        "format_version": FORMAT_VERSION,
        "game": "foosies-fighter",
        "rules": {"name": rules_obj.name, "sha256": rules_digest(rules_obj.raw)},
        "seed": int(seed),
        "max_ticks": int(env.max_ticks),
        "players": [
            {"name": agent1.name, "kind": agent1.info().get("kind", "?"),
             "meta": agent1.info()},
            {"name": agent2.name, "kind": agent2.info().get("kind", "?"),
             "meta": agent2.info()},
        ],
        "ticks": ticks,
        "outcome": outcome,
    }

    path = None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        fname = (f"{ts}_{_safe_name(agent1.name)}_vs_{_safe_name(agent2.name)}"
                 f"_s{seed}.json")
        path = save_replay(os.path.join(out_dir, fname), data)
        ok, msg = verify_replay(path)   # 写出即自检（成本可忽略）
        if not ok:
            raise RuntimeError(f"回放自检失败: {path}: {msg}")
    return outcome, data, ret1, path


def cmd_run(args):
    a1, a2 = load_agent(args.agent1), load_agent(args.agent2)
    outcome, _, ret1, path = play_one(a1, a2, args.seed, rules=args.rules,
                                      out_dir=args.out)
    p1n, p2n = a1.name, a2.name
    w = outcome["winner"]
    verdict = {0: p1n, 1: p2n, None: "平局"}[w]
    print(f"结果: {outcome['result']} | 胜者 {verdict} | "
          f"hp={outcome['hp']} | {outcome['ticks']}帧 | "
          f"回报 {ret1:+.2f}" + (f" | 回放 {path}" if path else ""))
    return 0


def _parse_seeds(args):
    if args.seeds:
        lo, _, hi = args.seeds.partition("-")
        return list(range(int(lo), int(hi or lo) + 1))
    return list(range(args.games))


def cmd_batch(args):
    a1, a2 = load_agent(args.agent1), load_agent(args.agent2)
    seeds = _parse_seeds(args)
    w = l = d = 0
    ticks_sum = 0.0
    ret_sum = 0.0
    for seed in seeds:
        outcome, _, ret1, _ = play_one(a1, a2, seed, rules=args.rules,
                                       out_dir=args.out)
        if outcome["winner"] == 0:
            w += 1
        elif outcome["winner"] == 1:
            l += 1
        else:
            d += 1
        ticks_sum += outcome["ticks"]
        ret_sum += ret1
    n = len(seeds)
    print(f"{n}场: {a1.name} {w}胜 {l}负 {d}判（胜率含判定 "
          f"{(w + 0.5 * d) / n:.0%}）| 平均 {ticks_sum / n:.1f} 帧 | "
          f"平均回报 {ret_sum / n:+.2f}")
    return 0


def cmd_demo(args):
    a1, a2 = load_agent("ppo"), load_agent("footsies")
    print("演示: ppo vs footsies，3 场（种子 0/1/2）")
    w = l = d = 0
    for seed in (0, 1, 2):
        outcome, _, ret1, path = play_one(a1, a2, seed, out_dir=DEFAULT_OUT)
        if outcome["winner"] == 0:
            w += 1
        elif outcome["winner"] == 1:
            l += 1
        else:
            d += 1
        print(f"  第{seed + 1}场 {outcome['result']} hp={outcome['hp']} "
              f"{outcome['ticks']}帧 回报{ret1:+.2f} → {path}")
    print(f"汇总: ppo {w}胜 {l}负 {d}判")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description="无头对局执行器")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("run", help="单局")
    sp.add_argument("agent1")
    sp.add_argument("agent2")
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--rules", default=None, help="规则包 JSON 路径")
    sp.add_argument("--out", default=DEFAULT_OUT, help="回放输出目录")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("batch", help="批量")
    sp.add_argument("agent1")
    sp.add_argument("agent2")
    sp.add_argument("--games", type=int, default=20)
    sp.add_argument("--seeds", default=None, help="如 0-19（优先于 --games）")
    sp.add_argument("--rules", default=None)
    sp.add_argument("--out", default=DEFAULT_OUT)
    sp.set_defaults(func=cmd_batch)

    sp = sub.add_parser("demo", help="内置演示")
    sp.set_defaults(func=cmd_demo)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
