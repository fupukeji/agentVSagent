"""
观看 AI 打架：PPO AI（左） vs 立回机器人（右）
运行：.venv/bin/python play.py [模型名] [场数]
复放：.venv/bin/python play.py --replay replays/<回放文件>.json
按键：空格 暂停 | +/- 调速（2~60 tick/秒，60=真实速度）| ESC 退出
"""

import math
import os
import sys
import pygame

from fighting_env import (
    FightingEnv, MOVES, STATE_NAMES,
    S_IDLE, S_STARTUP, S_ACTIVE, S_RECOVERY, S_BLOCKSTUN, S_HITSTUN,
)
from replay import load_replay, verify_replay

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


# ---------------------------------------------------------------
# 卡通角色：左·阿焰（元气少年） / 右·铁蛋（圆脸机器人）
# 手脚/表情全部矢量绘制，无外部素材；状态驱动姿态，帧计数驱动小动画
# ---------------------------------------------------------------
HERO = dict(  # 阿焰：橙发带少年格斗家
    skin=(255, 219, 179), hair=(96, 60, 38), band=(255, 122, 54),
    cloth=(248, 248, 250), belt=(255, 122, 54), fist=(255, 200, 160),
    pants=(70, 90, 140), shoe=(240, 240, 245), ribbon=(255, 122, 54),
)
ROBOT = dict(  # 铁蛋：圆脸机器人
    metal=(172, 198, 206), dark=(96, 116, 130), visor=(24, 34, 44),
    eye=(90, 230, 240), belly=(214, 232, 236), fist=(150, 176, 186),
    pants=(120, 140, 152), shoe=(80, 96, 108), ribbon=None,
)


def _capsule(surf, color, a, b, r):
    """胶囊肢体：圆头粗线段 + 端点圆。"""
    pygame.draw.line(surf, color, a, b, r * 2)
    pygame.draw.circle(surf, color, a, r)
    pygame.draw.circle(surf, color, b, r)


def _star(surf, color, c, r, n=8):
    """星光迸射（命中特效）。"""
    import math
    pts = []
    for i in range(n * 2):
        rad = r if i % 2 == 0 else r * 0.4
        ang = math.pi * i / n
        pts.append((c[0] + rad * math.cos(ang), c[1] + rad * math.sin(ang)))
    pygame.draw.polygon(surf, color, pts)


def _swirl(surf, color, c, r):
    """眩晕涡旋（受击表情）。"""
    for k in range(3):
        pygame.draw.circle(surf, color, (c[0] - 4 + k * 4, c[1]),
                           r - k * 2, 1)


def _sweat(surf, c):
    """收招冷汗。"""
    x, y = c
    pygame.draw.circle(surf, (150, 210, 255), (int(x), int(y)), 4)
    pygame.draw.polygon(surf, (150, 210, 255),
                        [(x - 4, y - 4), (x + 4, y - 4), (x, y - 11)])


def _face_hero(surf, f, hx, hy, facing, t, mouth):
    """阿焰的脸：刺猬头 + 发带 + 大眼睛。"""
    # 后脑发带飘带（随帧飘动）
    fl = 3 * math.sin(t * 0.25)
    bx = hx - facing * 20
    pygame.draw.polygon(surf, HERO["ribbon"],
                        [(bx, hy - 8), (bx - facing * 16, hy - 12 + fl),
                         (bx - facing * 12, hy - 2 + fl)])
    # 头发：刺猬尖
    pygame.draw.circle(surf, HERO["hair"], (hx, hy - 6), 19)
    for i, dx in enumerate((-13, -4, 6, 14)):
        tip = 10 + (i % 2) * 4
        pygame.draw.polygon(surf, HERO["hair"], [
            (hx + dx - 4, hy - 14), (hx + dx + 4, hy - 14),
            (hx + dx + facing * 2, hy - 14 - tip)])
    pygame.draw.circle(surf, HERO["skin"], (hx, hy + 2), 17)
    # 发带
    pygame.draw.rect(surf, HERO["band"],
                     (hx - 18, hy - 12, 36, 7), border_radius=3)
    pygame.draw.circle(surf, HERO["band"], (hx + facing * 10, hy - 9), 4)
    _eyes(surf, f, hx, hy, facing, t, skin=HERO["skin"])
    my = hy + 10
    if mouth == "shout":
        pygame.draw.ellipse(surf, (120, 60, 50),
                            (hx + facing * 3 - 5, my - 2, 10, 8))
    elif mouth == "ouch":
        pygame.draw.lines(surf, (120, 60, 50), False,
                          [(hx - 5, my), (hx - 1, my + 2), (hx + 3, my),
                           (hx + 7, my + 2)], 2)
    else:
        pygame.draw.arc(surf, (120, 60, 50),
                        (hx + facing * 3 - 6, my - 4, 12, 8), 3.3, 6.1, 2)


def _face_robot(surf, f, hx, hy, facing, t, mouth):
    """铁蛋的脸：天线 + 面罩扫描眼。"""
    # 天线（顶端小球闪烁）
    pygame.draw.line(surf, ROBOT["dark"], (hx, hy - 18), (hx, hy - 30), 3)
    on = (t // 12) % 2 == 0
    pygame.draw.circle(surf, (255, 120, 110) if on else ROBOT["dark"],
                       (hx, hy - 32), 4)
    # 头
    pygame.draw.circle(surf, ROBOT["metal"], (hx, hy), 19)
    pygame.draw.circle(surf, ROBOT["belly"], (hx, hy + 4), 14)
    # 面罩
    vw, vh = 26, 12
    vr = pygame.Rect(hx - vw // 2, hy - 4, vw, vh)
    pygame.draw.rect(surf, ROBOT["visor"], vr, border_radius=6)
    # 眼：情绪变色 + 扫描滑动
    if f.hp <= 0:
        for ex in (hx - 6, hx + 6):
            pygame.draw.line(surf, (255, 90, 80), (ex - 3, hy - 4),
                             (ex + 3, hy + 8), 2)
            pygame.draw.line(surf, (255, 90, 80), (ex + 3, hy - 4),
                             (ex - 3, hy + 8), 2)
    elif f.state == S_HITSTUN:
        _swirl(surf, (255, 140, 120), (hx, hy + 2), 5)
    else:
        col = ROBOT["eye"]
        if f.state == S_ACTIVE:
            col = (255, 170, 70)
        elif f.state in (S_BLOCKSTUN,) or f.blocking:
            col = (120, 180, 255)
        scan = int(6 * math.sin(t * 0.2))
        pygame.draw.circle(surf, col, (hx - 5 + scan, hy + 2), 3)
        pygame.draw.circle(surf, col, (hx + 5 + scan, hy + 2), 3)
    # 嘴：格栅
    for gx in (-6, -2, 2, 6):
        pygame.draw.line(surf, ROBOT["dark"],
                         (hx + gx, hy + 11), (hx + gx, hy + 14), 2)
    if mouth == "shout":
        pygame.draw.circle(surf, (255, 170, 70), (hx + facing * 3, hy + 12), 3)


def _eyes(surf, f, hx, hy, facing, t, skin):
    """通用卡通眼（阿焰用）：睁/眨/怒/晕/X。"""
    ey = hy + 1
    blink = (t % 160) < 6 and f.state == S_IDLE
    for ex in (hx + facing * 3 - 7, hx + facing * 3 + 7):
        if f.hp <= 0:
            pygame.draw.line(surf, (60, 40, 30), (ex - 3, ey - 3), (ex + 3, ey + 3), 2)
            pygame.draw.line(surf, (60, 40, 30), (ex + 3, ey - 3), (ex - 3, ey + 3), 2)
        elif f.state == S_HITSTUN:
            _swirl(surf, (60, 40, 30), (ex, ey), 4)
        elif blink:
            pygame.draw.line(surf, (60, 40, 30), (ex - 4, ey), (ex + 4, ey), 2)
        else:
            pygame.draw.ellipse(surf, WHITE, (ex - 4, ey - 4, 8, 9))
            pygame.draw.circle(surf, (40, 40, 55),
                               (ex + facing * 1, ey + 1), 3)
            pygame.draw.circle(surf, WHITE, (ex + facing * 2, ey - 1), 1)
    # 怒眉（出招时）
    if f.state in (S_STARTUP, S_ACTIVE):
        bx = hx + facing * 3
        pygame.draw.line(surf, HERO["hair"],
                         (bx - 9, ey - 8), (bx - 2, ey - 6), 2)
        pygame.draw.line(surf, HERO["hair"],
                         (bx + 2, ey - 6), (bx + 9, ey - 8), 2)


def draw_fighter(screen, font, f, facing, t=0, y_off=0, crouch=False):
    """卡通角色绘制（2D：y_off=跳跃高度，crouch=蹲伏）。"""
    import math
    pal = HERO if facing > 0 else ROBOT
    is_hero = facing > 0
    px = to_px(f.x)

    dx = 0.0
    if getattr(f, "_last_t", None) == t - 1:
        dx = f.x - getattr(f, "_last_x", f.x)
    f._last_t, f._last_x = t, f.x
    if not hasattr(f, "_walk"):
        f._walk = 0.0
    f._walk += dx * 900
    lean = max(-1, min(1, dx * 120)) * facing

    ko = f.hp <= 0
    bob = 2 * math.sin(t * 0.15) if f.state == S_IDLE and not ko else 0

    # 2D：跳跃偏移 + 蹲伏压缩
    jy = int(y_off * 600)  # 游戏Y→像素
    cy = 22 if crouch else 0  # 蹲伏下压

    hx0 = px + (10 * -facing if ko else 0)
    hy0 = STAGE_Y - 30

    # 跳跃轨迹尾迹
    if jy > 3:
        s = pygame.Surface((80, 60), pygame.SRCALPHA)
        pygame.draw.arc(s, (255, 210, 100, 60), (10, 10, 60, 40), math.pi * 0.2, math.pi * 0.8, 3)
        screen.blit(s, (px - 40, STAGE_Y - jy - 60))

    # 地面阴影
    shadow_w = 20 if crouch else 26
    pygame.draw.ellipse(screen, (10, 10, 16),
                        (px - shadow_w, STAGE_Y - 6, shadow_w * 2, 10))

    if ko:                                       # K.O. 倒地姿态
        body_y = STAGE_Y - 12
        _capsule(screen, pal["cloth"] if is_hero else pal["metal"],
                 (px - facing * 6, body_y), (px + facing * 30, body_y), 14)
        _capsule(screen, pal["pants"],
                 (px + facing * 30, body_y), (px + facing * 46, body_y - 8), 8)
        head_c = (px - facing * 30, body_y - 4)
        if is_hero:
            _face_hero(screen, f, head_c[0], head_c[1], facing, t, "ouch")
        else:
            _face_robot(screen, f, head_c[0], head_c[1], facing, t, "ouch")
        for k in range(3):                       # 头顶绕圈星星
            ang = t * 0.2 + k * 2.09
            sx = head_c[0] + 22 * math.cos(ang)
            sy = head_c[1] - 26 + 6 * math.sin(ang)
            _star(screen, (255, 220, 90), (sx, sy), 5, 5)
        _name_tag(screen, font, f, px, STAGE_Y - 66)
        return

    # --- 站立姿态骨架 ---
    by = STAGE_Y - 62 + bob                     # 躯干中心
    bx = px + lean * 5
    head_c = (bx + lean * 4, STAGE_Y - 108 + bob)

    # 后臂（先画，被身体遮挡）
    sh_back = (bx - facing * 14, by - 16)
    # 后腿/前腿：迈步剪刀脚
    step = math.sin(f._walk * 0.25) * 7 if abs(dx) > 1e-6 else 0
    hip = (bx, by + 26)
    for foot_dx, leg_c in ((10 + step, pal["pants"]), (-8 - step, pal["pants"])):
        foot = (bx + facing * foot_dx, STAGE_Y - 3)
        _capsule(screen, leg_c, (bx + facing * (foot_dx * 0.3), by + 18), foot, 7)
        pygame.draw.ellipse(screen, pal["shoe"],
                            (foot[0] - 9, foot[1] - 5, 18, 9))

    # 躯干
    body_col = pal["cloth"] if is_hero else pal["metal"]
    pygame.draw.rect(screen, body_col,
                     (bx - 21, by - 28, 42, 52), border_radius=16)
    if is_hero:
        pygame.draw.rect(screen, pal["belt"],
                         (bx - 21, by + 8, 42, 9), border_radius=4)
        pygame.draw.circle(screen, (255, 220, 140), (bx, by + 12), 3)
    else:                                        # 机器人：肚皮 + 铆钉
        pygame.draw.circle(screen, pal["belly"], (bx, by + 4), 13)
        for gx in (-14, 14):
            pygame.draw.circle(screen, pal["dark"],
                               (bx + gx, by - 18), 2)

    # 头
    if is_hero:
        _face_hero(screen, f, head_c[0], head_c[1], facing, t,
                   "shout" if f.state in (S_STARTUP, S_ACTIVE) else
                   ("ouch" if f.state == S_HITSTUN else "idle"))
    else:
        _face_robot(screen, f, head_c[0], head_c[1], facing, t,
                    "shout" if f.state in (S_STARTUP, S_ACTIVE) else "idle")

    # 前臂姿态（按状态）
    sh = (bx + facing * 14, by - 16)
    heavy = f.move is not None and getattr(f.move, "kind", "") == "strike" \
        and f.move.name == "重击"
    fist_r = 9 if heavy else 7
    if f.state == S_ACTIVE and f.move is not None:
        if f.move.kind == "throw":              # 投技：双手前探
            for fy in (-78, -66):
                _capsule(screen, pal["skin"] if is_hero else pal["metal"],
                         sh, (bx + facing * 44, STAGE_Y + fy), 6)
                pygame.draw.circle(screen, pal["fist"],
                                   (bx + facing * 44, STAGE_Y + fy), 7)
        else:                                   # 打击：全伸展 + 速度线
            reach = 52 if heavy else 44
            ft = (bx + facing * reach, STAGE_Y - 80)
            _capsule(screen, pal["skin"] if is_hero else pal["metal"], sh, ft, 7)
            pygame.draw.circle(screen, pal["fist"], ft, fist_r)
            for ly in (-8, 0, 8):
                pygame.draw.line(screen, (200, 200, 215),
                                 (ft[0] + facing * 14, STAGE_Y - 80 + ly),
                                 (ft[0] + facing * 30, STAGE_Y - 80 + ly), 2)
    elif f.state == S_STARTUP:                  # 蓄力：拳收到耳边
        ft = (bx + facing * 4, STAGE_Y - 96)
        _capsule(screen, pal["skin"] if is_hero else pal["metal"], sh, ft, 7)
        pygame.draw.circle(screen, pal["fist"], ft, fist_r)
        _capsule(screen, pal["skin"] if is_hero else pal["metal"],
                 sh_back, (bx - facing * 10, by - 24), 6)  # 后臂张开平衡
    elif f.state == S_RECOVERY:                 # 收招：拳下垂 + 冷汗
        ft = (bx + facing * 34, STAGE_Y - 66)
        _capsule(screen, pal["skin"] if is_hero else pal["metal"], sh, ft, 7)
        pygame.draw.circle(screen, pal["fist"], ft, fist_r)
        _sweat(screen, (head_c[0] + facing * 20, head_c[1] - 14))
    elif f.blocking or f.state == S_BLOCKSTUN:  # 防御：双臂交叠 + 护盾
        for fy in (-84, -62):
            _capsule(screen, pal["skin"] if is_hero else pal["metal"],
                     (bx + facing * 10, by - 20),
                     (bx + facing * 18, STAGE_Y + fy), 6)
        shield = pygame.Surface((46, 100), pygame.SRCALPHA)
        pygame.draw.arc(shield, (90, 220, 230, 160),
                        (4, 4, 38, 92), -1.1, 1.1, 4)
        screen.blit(shield, (bx + facing * 26 - (0 if facing > 0 else 46),
                             STAGE_Y - 100))
        if f.state == S_BLOCKSTUN and f.frames_left > 0:  # 防住瞬间火花
            _star(screen, (140, 230, 255),
                  (bx + facing * 34, STAGE_Y - 76), 9, 6)
    elif f.state == S_HITSTUN:                  # 受击：后仰 + 迸射星
        for fy, dx2 in ((-92, -18), (-64, -12)):
            _capsule(screen, pal["skin"] if is_hero else pal["metal"],
                     sh, (bx + facing * dx2, STAGE_Y + fy), 6)
        for k in range(3):
            ang = 2.2 + k * 0.5
            sx = head_c[0] + facing * (14 + k * 7) * math.cos(ang)
            sy = head_c[1] - 10 - k * 6
            pygame.draw.line(screen, (255, 200, 90),
                             (sx, sy), (sx + facing * 8, sy - 6), 2)
    else:                                       # 待机：护架 + 后手收腰
        ft = (bx + facing * 12, STAGE_Y - 88)
        _capsule(screen, pal["skin"] if is_hero else pal["metal"], sh, ft, 7)
        pygame.draw.circle(screen, pal["fist"], ft, 7)
        _capsule(screen, pal["skin"] if is_hero else pal["metal"],
                 sh_back, (bx - facing * 6, by - 8), 6)

    # 命中判定的白色闪光描边（保留原有视觉语言）
    if f.state == S_ACTIVE:
        pygame.draw.rect(screen, WHITE,
                         (bx - 27, STAGE_Y - 132, 54, 130), 3,
                         border_radius=14)

    _name_tag(screen, font, f, px, STAGE_Y - 156)


def _name_tag(screen, font, f, cx, y):
    """头顶名牌胶囊。"""
    if f.move:
        txt = f"{f.name} · {f.move.name} {STATE_NAMES[f.state]} {f.frames_left}帧"
    elif f.blocking:
        txt = f"{f.name} · 防御中"
    else:
        txt = f"{f.name} · 待机" if f.hp > 0 else f"{f.name} · K.O.!"
    surf = font.render(txt, True, WHITE)
    pad = 8
    tag = pygame.Rect(0, 0, surf.get_width() + pad * 2, surf.get_height() + 6)
    tag.center = (int(cx), int(y - surf.get_height() // 2))
    pygame.draw.rect(screen, (28, 28, 40), tag, border_radius=tag.height // 2)
    screen.blit(surf, (tag.x + pad, tag.y + 3))


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

    # 角色（含 2D Y 轴/蹲伏）
    draw_fighter(screen, font, p, +1, t=env.t, y_off=getattr(p, 'y', 0), crouch=getattr(p, 'crouching', False))
    draw_fighter(screen, font, e, -1, t=env.t, y_off=getattr(e, 'y', 0), crouch=getattr(e, 'crouching', False))

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


def _event_loop(state):
    """处理窗口事件；返回 False 表示退出程序。state: {tps, paused}"""
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_ESCAPE, pygame.K_q):
                return False
            if event.key == pygame.K_SPACE:
                state["paused"] = not state["paused"]
            elif event.key in (pygame.K_EQUALS, pygame.K_KP_PLUS):
                state["tps"] = min(60, state["tps"] + 4)
            elif event.key == pygame.K_MINUS:
                state["tps"] = max(2, state["tps"] - 4)
    return True


def outcome_text(o, p1n, p2n):
    """outcome 字典 → 结果展示文案。"""
    r = o["result"]
    if r == "win0":
        return f"K.O. {p1n} 胜利!"
    if r == "win1":
        return f"K.O. {p2n} 胜利!"
    if r == "timeout0":
        return f"时间到 · 判 {p1n} 胜"
    if r == "timeout1":
        return f"时间到 · 判 {p2n} 胜"
    return "平局"


def run_replay(path):
    """复放模式：按回放 ticks 逐帧驱动 step_both 渲染，不加载模型。"""
    if not path:
        print("用法: python play.py --replay <回放文件>")
        return 2
    ok, msg = verify_replay(path)
    if not ok:
        print(f"回放校验未通过，拒绝复放：{msg}")
        return 1
    data = load_replay(path)
    p1n, p2n = data["players"][0]["name"], data["players"][1]["name"]
    o = data["outcome"]
    print(f"复放: {p1n} vs {p2n} | seed {data['seed']} | "
          f"{len(data['ticks'])}帧 | 记录结果 {o['result']}")

    env = FightingEnv(max_ticks=data["max_ticks"])
    env.p1.name, env.p2.name = p1n, p2n
    env.reset(seed=data["seed"])

    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption(f"立回斗士 — 回放模式：{p1n} vs {p2n}")
    clock = pygame.time.Clock()
    font, big = get_font(15), get_font(22, bold=True)
    state = {"tps": 10, "paused": False}

    ticks = iter(data["ticks"])
    done, result, frame = False, None, 0
    while not done or frame < 150:  # 结束后停 2.5 秒
        step_every = max(1, round(FPS / state["tps"]))
        if not _event_loop(state):
            break
        if not done and not state["paused"] and frame % step_every == 0:
            pair = next(ticks, None)
            if pair is not None:
                env.step_both(int(pair[0]), int(pair[1]))
            if pair is None or env.p1.hp <= 0 or env.p2.hp <= 0 \
                    or env.t >= env.max_ticks:
                done = True
                result = outcome_text(env.outcome(), p1n, p2n)
            frame = 0
        frame += 1

        draw(screen, env, font, big, result, state["tps"], state["paused"])
        badge = font.render(
            f"回放模式 | {p1n} vs {p2n} | seed {data['seed']}", True, GOLD)
        screen.blit(badge, (W // 2 - badge.get_width() // 2, 8))
        pygame.display.flip()
        clock.tick(FPS)

    print(f"回放结束: {result}")
    pygame.quit()
    return 0


def main():
    argv = sys.argv[1:]
    # 复放模式：兼容 --replay / -replay / -r
    if argv and argv[0] in ("--replay", "-replay", "-r"):
        return run_replay(argv[1] if len(argv) > 1 else None)
    model_path = argv[0] if argv else "fighting_ppo"
    episodes = 3
    if len(argv) > 1:
        try:
            episodes = int(argv[1])
        except ValueError:
            print(f"[提示] 忽略无法解析的场数参数: {argv[1]!r}（用法: "
                  f"python play.py [模型名] [场数] | --replay <回放文件>）")

    model = None
    if os.path.exists(model_path) or os.path.exists(model_path + ".zip"):
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
    state = {"tps": 10, "paused": False}

    for ep in range(episodes):
        obs, _ = env.reset(seed=ep)
        done, result, frame = False, None, 0
        while not done or frame < 150:  # 结束后停 2.5 秒
            step_every = max(1, round(FPS / state["tps"]))
            if not _event_loop(state):
                pygame.quit()
                return
            paused = state["paused"]

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

            draw(screen, env, font, big, result, state["tps"], state["paused"])
            pygame.display.flip()
            clock.tick(FPS)

        print(f"第 {ep + 1} 场: {result}  AI HP {env.p1.hp} vs 机器 HP {env.p2.hp}")

    pygame.quit()


if __name__ == "__main__":
    main()
