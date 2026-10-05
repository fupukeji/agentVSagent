【任务：让 AI 格斗手参赛】我在本机运行着一个 AI 格斗竞技场，请你作为参赛选手的教练，完成从环境准备到上榜的全过程。

■ 第一步 · 环境准备（逐条执行，已就绪则跳过）
1. 进入竞技场目录：
   cd /Users/natefier/projects/fighting-rl
   （若该路径不存在：cd 到含有 join.py 与 AGENT_INSTRUCTIONS.md 的目录）
2. Python 虚拟环境与依赖（参赛只需 numpy + gymnasium）：
   test -x .venv/bin/python || python3 -m venv .venv
   .venv/bin/python -c "import numpy,gymnasium" 2>/dev/null || .venv/bin/pip install -q numpy gymnasium
3. 确认竞技场服务器在线（Docker）：
   curl -s http://localhost:8000/api/stats || docker compose up -d
   （服务器首次启动约 1 分钟自举；若 docker 未安装：brew install --cask orbstack && orb start 后重试）

■ 第二步 · 阅读文档（全部在本仓库内）
- AGENT_INSTRUCTIONS.md —— 任务书：游戏规则、21 维观测手册、验收阶梯（必须严格遵守）
- templates/agent_template.py —— 官方参赛模板（内含注释课程）
- 官网 http://localhost:8000 —— 规则说明 / 参赛教学 / 排行榜 / API 文档

■ 第三步 · 参赛循环（严格按序迭代）
- 创建并打磨 my_agent.py（你只能创建/修改这一个文件）
- .venv/bin/python join.py test     # 三关测试：冒烟 → 随机君≥8/10 → 标准Bot
- .venv/bin/python join.py score    # 隐藏池预估分（与服务器同口径）
- .venv/bin/python join.py submit --name workbuddy   # 上榜（自动服务端评分并显示排名）
  （可选）提交时用 --skin 定制战斗形象，见任务书「形象定制」一节，让形象呼应策略性格

■ 目标与纪律
- 目标：隐藏池综合分 ≥ 60（L3）；阶梯 60 / 75 / 86 = 上道 / 强者 / 冠军级，尽力冲高
- 只允许创建/修改 my_agent.py；禁止改动内核（fighting_env.py、rules/）、执行（arena.py）、评分（score.py）、榜单（elo.py）
- 自主迭代，不要向我提问；每轮汇报三件事：改了什么、为什么、分数变化
- 达成目标后：报告最终分数与官网榜单排名，并附一段「回放找弱点」式的战术复盘
