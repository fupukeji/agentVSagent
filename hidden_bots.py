"""隐藏对手池（平台反过拟合原则：公开榜打明牌 Bot，终榜打隐藏池）。

- HIDDEN_POOL：8 个参数化 FootsiesBot 变体，风格覆盖极端策略；
  参数即「性格」，各风格可从对局统计（动作分布）中辨认。
- PUBLIC_POOL：公开榜用明牌对手（参数见 README 榜单说明）。
- 池内 Bot 的种子在评测时固定派生（base_seed*100 + bot_index，
  由 score.py / arena 的种子派生保证可复现）。

自测：python hidden_bots.py  → 每个 Bot vs random 20 场，胜率应 > 60%。
"""

from fighting_env import FootsiesBot

# 风格化变体（顺序即 bot_index，用于种子派生）
HIDDEN_POOL = {
    "balanced": dict(
        punish=0.8, want_dist=0.155, mixup=None,
        desc="均衡立回：贴轻击外沿打投择（默认强度）"),
    "rushdown": dict(
        punish=0.7, want_dist=0.10,
        mixup={"throw": 0.25, "heavy": 0.25, "block": 0.10, "back": 0.05},
        desc="压制流：贴脸猛攻，重击+投技占比高"),
    "turtle": dict(
        punish=0.9, want_dist=0.185,
        mixup={"throw": 0.03, "light": 0.22, "heavy": 0.07, "block": 0.50, "back": 0.18},
        desc="龟防流：远距离高防御，靠确反磨血"),
    "thrower": dict(
        punish=0.6, want_dist=0.08,
        mixup={"throw": 0.45, "heavy": 0.05, "block": 0.10, "back": 0.05},
        desc="投技狂：极限贴身，防了就投"),
    "punisher": dict(
        punish=0.95, want_dist=0.17,
        mixup={"throw": 0.10, "heavy": 0.08, "block": 0.30, "back": 0.25},
        desc="确反大师：读招极准，很少主动进攻"),
    "spacer": dict(
        punish=0.5, want_dist=0.175,
        mixup={"throw": 0.02, "light": 0.06, "heavy": 0.14, "block": 0.42, "back": 0.36},
        desc="游走流：拉扯距离，很少出招（带内仅重击够得着）"),
    "chancer": dict(
        punish=0.3, want_dist=0.15,
        mixup={"throw": 0.10, "heavy": 0.45, "block": 0.10, "back": 0.10},
        desc="赌徒流：读招很差，大量随机重击"),
    "mirror": dict(
        punish=0.8, want_dist=0.155, mixup=None,
        desc="镜像流：与 balanced 同参，种子不同"),
}

# 公开榜对手（参数在 README「公开榜对手」一节公示）
PUBLIC_POOL = {
    "public-balanced": dict(
        punish=0.8, want_dist=0.155, mixup=None,
        desc="公开榜·均衡立回（与默认 footsies 同参）"),
    "public-chancer": dict(
        punish=0.3, want_dist=0.15,
        mixup={"throw": 0.10, "heavy": 0.45, "block": 0.10, "back": 0.10},
        desc="公开榜·赌徒流（激进重击，读招差）"),
}


def make_bot(code: str, seed=None) -> FootsiesBot:
    pools = {**HIDDEN_POOL, **PUBLIC_POOL}
    if code not in pools:
        raise KeyError(f"未知 Bot 代号: {code}（可用: {sorted(pools)}）")
    cfg = pools[code]
    return FootsiesBot(punish=cfg["punish"], want_dist=cfg["want_dist"],
                       mixup=cfg["mixup"], seed=seed)


def get_hidden_bot(code: str) -> FootsiesBot:
    """供 load_agent('hidden:<代号>') 使用；种子由评测时的 reset 派生。"""
    return make_bot(code)


def bot_index(code: str) -> int:
    """隐藏池内序号（种子派生用）。"""
    return list(HIDDEN_POOL).index(code)


if __name__ == "__main__":
    # 自测：每个 Bot（含公开池）与 random 打 20 场，胜率应显著 > 50%
    from agents import load_agent
    from agents.script_agent import ScriptAgent
    from arena import play_one

    random = load_agent("random")
    print(f"{'代号':<18}{'胜场':>4}  风格")
    for code, cfg in {**HIDDEN_POOL, **PUBLIC_POOL}.items():
        bot = ScriptAgent(make_bot(code), name=code)
        w = sum(1 for s in range(20)
                if play_one(bot, random, s)[0]["winner"] == 0)
        mark = "✓" if w / 20 > 0.6 else "✗ 低于60%!"
        print(f"{code:<18}{w:>3}/20 {mark} {cfg['desc']}")
