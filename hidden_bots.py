"""隐藏对手池（平台反过拟合原则：公开榜打明牌 Bot，终榜打隐藏池）。

- HIDDEN_POOL：8 个参数化 FootsiesBot 变体，风格覆盖极端策略；
  参数即「性格」，各风格可从对局统计（动作分布）中辨认。
- PUBLIC_POOL：公开榜用明牌对手（参数见 README 榜单说明）。
- 池内 Bot 的种子在评测时固定派生（base_seed*100 + bot_index，
  由 score.py / arena 的种子派生保证可复现）。

自测：python hidden_bots.py  → 每个 Bot vs random 20 场，胜率应 > 60%。
"""

from fighting_env import FootsiesBot, A_BACK, A_FWD, A_LIGHT, A_HEAVY, A_BLOCK, A_THROW, A_JUMP, A_CROUCH, A_CROUCH_BLOCK, A_LOW, S_RECOVERY, S_STARTUP, S_BLOCKSTUN

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
        punish=0.6, want_dist=0.065, band=0.004,
        mixup={"throw": 0.45, "heavy": 0.05, "block": 0.10, "back": 0.05},
        desc="投技狂：极限贴身，防了就投（0.065+0.004 游走带完全进入投技射程 0.07）"),
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


# 2D 隐藏池（跳跃/蹲伏/高低段时代的 8 个新考官）
HIDDEN_POOL_2D = {
    "balanced2d": dict(
        punish=0.8, want_dist=0.155, mixup=None,
        desc="均衡立回2D版：地面扎实，偶尔防空"),
    "jumper": dict(
        punish=0.5, want_dist=0.12,
        mixup={"throw": 0.10, "light": 0.20, "heavy": 0.10, "block": 0.15, "back": 0.05},
        jump_rate=0.35, desc="跳入流：35%概率跳入，空中压制"),
    "antiair": dict(
        punish=0.9, want_dist=0.17,
        mixup={"throw": 0.03, "light": 0.10, "heavy": 0.30, "block": 0.40, "back": 0.15},
        jump_rate=0.0, crouch_rate=0.3, desc="对空流：等对方跳，重击对空"),
    "lowrider": dict(
        punish=0.6, want_dist=0.13,
        mixup={"throw": 0.10, "light": 0.10, "heavy": 0.05, "block": 0.25, "back": 0.10},
        jump_rate=0.05, crouch_rate=0.5, low_rate=0.35, desc="下段流：蹲着磨脚，专打站防"),
    "highlow": dict(
        punish=0.7, want_dist=0.14,
        mixup=None, read_stance=True,
        desc="高低择：读对方防守姿态选段位"),
    "airdodge": dict(
        punish=0.7, want_dist=0.14,
        mixup={"throw": 0.10, "light": 0.35, "heavy": 0.10, "block": 0.25, "back": 0.05},
        jump_rate=0.15, desc="闪空流：跳跃躲投/低，地面扎实"),
    "groundtech": dict(
        punish=0.85, want_dist=0.15,
        mixup={"throw": 0.08, "light": 0.35, "heavy": 0.15, "block": 0.20, "back": 0.10},
        jump_rate=0.08, desc="地面技术流：扎实1D+精准2D插入"),
    "chaos2d": dict(
        punish=0.5, want_dist=0.14,
        mixup={"throw": 0.12, "light": 0.25, "heavy": 0.20, "block": 0.20, "back": 0.08},
        jump_rate=0.15, crouch_rate=0.10, low_rate=0.15,
        desc="混沌2D：乱跳乱蹲打下段，不可预测"),
}


class FootsiesBot2D(FootsiesBot):
    """2D 扩展立回 Bot：在 1D 基础上增加跳跃/蹲伏/下段/防空。"""

    def __init__(self, punish=0.8, want_dist=0.155, mixup=None, seed=None,
                 rules=None, band=0.035, jump_rate=0.0, crouch_rate=0.0,
                 low_rate=0.0, read_stance=False):
        super().__init__(punish, want_dist, mixup, seed, rules, band)
        self.jump_rate = jump_rate
        self.crouch_rate = crouch_rate
        self.low_rate = low_rate
        self.read_stance = read_stance

    def act(self, opp, me):
        import numpy as np
        d = abs(me.x - opp.x)
        rng = self.rng.random

        if me.busy():
            return A_BLOCK

        # 对空（所有 2D Bot 都会）
        if hasattr(opp, 'airborne') and opp.airborne():
            if rng() < self.punish and d <= 0.19:
                return A_HEAVY  # 对空重击
            return A_BLOCK

        # 读招（确反）
        if rng() < self.punish:
            if opp.state == S_RECOVERY and d <= 0.19:
                return A_HEAVY
            if opp.state == S_STARTUP and d <= 0.19:
                return A_BLOCK
            if (opp.blocking or opp.state == S_BLOCKSTUN) and d <= 0.07:
                if hasattr(opp, 'crouching') and opp.crouching:
                    return A_JUMP  # 蹲防漏跳攻
                return A_THROW

        # 高低择模式
        if self.read_stance:
            if d <= 0.13:
                opp_crouch = hasattr(opp, 'crouching') and opp.crouching
                opp_block = opp.blocking
                r = rng()
                if opp_crouch:
                    return A_JUMP if r < 0.4 else A_THROW
                elif opp_block:
                    return A_LOW if r < 0.5 else A_THROW
                else:
                    if r < 0.3: return A_LIGHT
                    elif r < 0.5: return A_LOW
                    elif r < 0.65: return A_JUMP
                    else: return A_BLOCK

        # 跳跃决策
        if self.jump_rate > 0 and rng() < self.jump_rate and d >= 0.10:
            return A_JUMP

        # 蹲伏/下段决策
        if self.crouch_rate > 0 and rng() < self.crouch_rate:
            if self.low_rate > 0 and rng() < self.low_rate and d <= 0.15:
                return A_LOW  # 蹲着出下段
            return A_CROUCH_BLOCK

        # 下段直出
        if self.low_rate > 0 and rng() < self.low_rate and d <= 0.13:
            return A_LOW

        # 立回（1D 基础）
        if d > self.want_dist + self.band:
            return A_FWD
        if d < self.want_dist - self.band:
            return A_BACK

        # 打投择（1D 基础）
        r = rng()
        if self.mixup is None:
            if d < 0.09 and r < 0.15:
                return A_THROW
            if r < 0.50: return A_LIGHT
            if r < 0.62: return A_HEAVY
            if r < 0.82: return A_BLOCK
            return A_BACK
        m = self._mix
        if d <= 0.09 and r < m["throw"]:
            return A_THROW
        if r < m["throw"] + m["light"]: return A_LIGHT
        if r < m["throw"] + m["light"] + m["heavy"]: return A_HEAVY
        if r < m["throw"] + m["light"] + m["heavy"] + m["block"]: return A_BLOCK
        return A_BACK


def make_bot_2d(code, seed=None):
    """创建 2D 池 Bot。"""
    pools = {**HIDDEN_POOL_2D, **PUBLIC_POOL}
    if code not in pools:
        raise KeyError(f"未知 Bot 代号: {code}")
    cfg = pools[code]
    return FootsiesBot2D(
        punish=cfg.get("punish", 0.8),
        want_dist=cfg.get("want_dist", 0.155),
        mixup=cfg.get("mixup"),
        seed=seed,
        band=cfg.get("band", 0.035),
        jump_rate=cfg.get("jump_rate", 0),
        crouch_rate=cfg.get("crouch_rate", 0),
        low_rate=cfg.get("low_rate", 0),
        read_stance=cfg.get("read_stance", False))
    pools = {**HIDDEN_POOL, **PUBLIC_POOL}
    if code not in pools:
        raise KeyError(f"未知 Bot 代号: {code}（可用: {sorted(pools)}）")
    cfg = pools[code]
    return FootsiesBot(punish=cfg["punish"], want_dist=cfg["want_dist"],
                       mixup=cfg["mixup"], seed=seed,
                       band=cfg.get("band", 0.035))


def get_hidden_bot(code: str) -> FootsiesBot:
    """供 load_agent('hidden:<代号>') 使用；种子由评测时的 reset 派生。"""
def make_bot(code: str, seed=None) -> FootsiesBot:
    """创建池内 Bot。"""
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
