"""
观看 AI 打架：PPO AI（左） vs 立回机器人（右）
运行：.venv/bin/python play.py [模型名] [场数]
按键：空格 暂停 | +/- 调速（2~60 tick/秒，60=真实速度）| ESC 退出
"""

import os
import sys
import pygame

from fighting_env import (
    FightingEnv, MOVES, STATE_NAMES,
    S_IDLE, S_STARTUP, S_ACTIVE, S_RECOVERY, S_BLOCKSTUN, S_HITSTUN,
)

W, H = 1000, 520
STAGE_Y = 350
PX0, PX1 = 100, W - 100
FPS = 60
BG = (16, 16, 24)
WHITE = (235, 235, 235)
GOLD = (255, 200, 60)

STATE_COLORS = {
    S_IDLE: (150, 155, 170),
    S_STARTUP: (255, 210, 80),
    S_ACTIVE: (235, 80, 80),
    S_RECOVERY: (240, 140, 70),
    S_BLOCKSTUN: (90, 160, 230),
    S_HITSTUN: (170, 110, 220),
}

_FONT_CANDIDATES = [
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
]
_FONT_PATH = next((p for p in _FONT_CANDIDATES if os.path.exists(p)), None)


def get_font(size, bold=False):
    if _FONT_PATH:
        f = pygame.font.Font(_FONT_PATH, size)
        f.set_bold(bold)
        return f
    return pygame.font.SysFont("pingfangsc", size, bold=bold)


def to_px(x):
    return PX0 + x * (PX1 - PX0)


def draw_fighter(screen, font, f, facing):
    px = to_px(f.x)
    color = STATE_COLORS[f.state]
    body = pygame.Rect(0, 0, 46, 84)
    body.center = (px, STAGE_Y - 42)
    pygame.draw.rect(screen, color, body, border_radius=10)
    head = pygame.draw.circle(screen, color, (px, STAGE_Y - 100), 16)
    if f.blocking:
        pygame.draw.rect(screen, (80, 220, 230), body, 3, border_radius=10)
    if f.state == S_ACTIVE:  # 判定帧：白色闪光描边
        pygame.draw.rect(screen, WHITE, body.inflate(14, 14), 3, border_radius=12)
    # 眼睛（表示朝向）
    pygame.draw.circle(screen, BG, (px + facing * 6, STAGE_Y - 102), 3)
    # 状态文字
    if f.move:
        txt = f"{f.move.name} · {STATE_NAMES[f.state]} {f.frames_left}帧"
    elif f.blocking:
        txt = "防御中"
    else:
        txt = STATE_NAMES[S_IDLE]
    surf = font.render(f"{f.name}  {txt}", True, color)
    screen.blit(surf, (px - surf.get_width() // 2, STAGE_Y - 150))


def draw(screen, env, font, big, result, tps, paused):
    p, e = env.p1, env.p2
    screen.fill(BG)

    # 舞台
    pygame.draw.line(screen, (70, 70, 85), (PX0 - 30, STAGE_Y), (PX1 + 30, STAGE_Y), 4)

    # 血条
    for f, x0, w in ((p, 60, 340), (e, W - 60 - 340, 340)):
        pygame.draw.rect(screen, (60, 60, 70), (x0, 40, w, 20), border_radius=6)
        frac = f.hp / f.max_hp
        if f is e:  # 右侧从右往左减
            pygame.draw.rect(screen, (200, 80, 80),
                             (x0 + w * (1 - frac), 40, w * frac, 20), border_radius=6)
        else:
            pygame.draw.rect(screen, (120, 200, 110),
                             (x0, 40, w * frac, 20), border_radius=6)
        t = font.render(f"{f.name} HP {f.hp}/{f.max_hp}", True, WHITE)
        screen.blit(t, (x0 if f is p else x0 + w - t.get_width(), 68))

    # 计时
    sec = max(0, env.max_ticks - env.t) / 60
    t = big.render(f"{sec:.1f}s", True, GOLD)
    screen.blit(t, (W // 2 - t.get_width() // 2, 36))

    # 角色
    draw_fighter(screen, font, p, +1)
    draw_fighter(screen, font, e, -1)

    # 距离仪表（立回核心！）
    x0, x1, y = 220, W - 220, H - 90
    pygame.draw.rect(screen, (60, 60, 70), (x0, y, x1 - x0, 10), border_radius=5)
    d = e.x - p.x
    px = x0 + min(d, 1.0) * (x1 - x0)
    pygame.draw.circle(screen, GOLD, (int(px), y + 5), 8)
    for mv, col in ((MOVES[5], (170, 110, 220)), (MOVES[2], (255, 210, 80)),
                    (MOVES[3], (235, 80, 80))):
        mx = x0 + mv.reach * (x1 - x0)
        pygame.draw.line(screen, col, (mx, y - 8), (mx, y + 18), 2)
        lbl = font.render(f"{mv.name}{mv.reach:.2f}", True, col)
        screen.blit(lbl, (mx - lbl.get_width() // 2, y + 20))
    dl = font.render("← 距离", True, WHITE)
    screen.blit(dl, (x0 - 90, y - 4))

    # 操作提示
    hint = font.render(f"空格 暂停    +/- 调速（当前 {tps} t/s）"
                       + ("    ‖ 已暂停" if paused else ""), True, (150, 150, 160))
    screen.blit(hint, (W // 2 - hint.get_width() // 2, H - 34))

    # 结果
    if result:
        surf = big.render(result, True, GOLD if "胜" in result else (230, 90, 90))
        s = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
        s.fill((0, 0, 0, 170))
        screen.blit(s, (W // 2 - s.get_width() // 2, 110))
        screen.blit(surf, (W // 2 - surf.get_width() // 2, 110))


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "fighting_ppo"
    episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    model = None
    if os.path.exists(model_path):
        from stable_baselines3 import PPO
        model = PPO.load(model_path)
        print(f"已加载模型 {model_path}")
    else:
        print(f"未找到 {model_path}，使用随机策略演示环境机制")

    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("立回斗士 — PPO AI vs 立回机器人")
    clock = pygame.time.Clock()
    font, big = get_font(15), get_font(22, bold=True)

    env = FightingEnv()
    tps, paused = 10, False

    for ep in range(episodes):
        obs, _ = env.reset(seed=ep)
        done, result, frame = False, None, 0
        while not done or frame < 150:  # 结束后停 2.5 秒
            step_every = max(1, round(FPS / tps))
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        pygame.quit()
                        return
                    if event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key in (pygame.K_EQUALS, pygame.K_KP_PLUS):
                        tps = min(60, tps + 4)
                    elif event.key == pygame.K_MINUS:
                        tps = max(2, tps - 4)

            if not done and not paused and frame % step_every == 0:
                if model is not None:
                    a, _ = model.predict(obs, deterministic=True)
                else:
                    a = env.action_space.sample()
                obs, r, term, trunc, info = env.step(int(a))
                done = term or trunc
                if done:
                    if env.p2.hp <= 0 < env.p1.hp:
                        result = "K.O. 胜利!"
                    elif env.p1.hp <= 0 < env.p2.hp:
                        result = "K.O. 战败..."
                    else:
                        verdict = "胜" if env.p1.hp > env.p2.hp else ("负" if env.p1.hp < env.p2.hp else "平")
                        result = f"时间到 · 判{verdict}"
                frame = 0
            frame += 1

            draw(screen, env, font, big, result, tps, paused)
            pygame.display.flip()
            clock.tick(FPS)

        print(f"第 {ep + 1} 场: {result}  AI HP {env.p1.hp} vs 机器 HP {env.p2.hp}")

    pygame.quit()


if __name__ == "__main__":
    main()
