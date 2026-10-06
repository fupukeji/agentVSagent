"""前端交叉校验：JS 引用的 id 与 HTML 定义一致、onclick 函数已定义。

曾抓出「renderCards 写已删除的 #cards 导致 loadAll 每 8 秒异常、榜单假死」的半成品事故，
进 CI 后此类问题在推送时即被拦截。
"""
import re
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
html = (root / "static" / "index.html").read_text(encoding="utf-8")
js_blocks = re.findall(r"<script>(.*?)</script>", html, re.S)
js = "\n".join(js_blocks)

fail = False

ids_used = set(re.findall(r"\$\('([\w-]+)'\)", js))
ids_def = set(re.findall(r'id="([\w-]+)"', html))
missing = ids_used - ids_def
if missing:
    print(f"✗ JS 引用但 HTML 缺失的 id: {missing}")
    fail = True

funcs = set(re.findall(r'onclick="(\w+)\(', html))
undefined = [f for f in funcs
             if not re.search(rf"function {f}\b|const {f}\s*=", js)]
if undefined:
    print(f"✗ HTML 引用但未定义的函数: {undefined}")
    fail = True

# 剧场核心函数完整性（曾因误删导致动画打不开）
for key in ["function drawFighter", "function rpDraw", "function rpAdvance",
            "async function openReplay", "function outcomeText", "SFX.bell"]:
    if key not in js:
        print(f"✗ 剧场核心缺失: {key}")
        fail = True

if fail:
    sys.exit(1)
print(f"✓ 前端校验通过（{len(ids_used)} 个 id 引用、{len(funcs)} 个事件函数、剧场核心完整）")
