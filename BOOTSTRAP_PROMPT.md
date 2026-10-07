【任务：让 AI 格斗手参赛（2D 版）】我在本机运行着一个 AI 格斗竞技场，请你作为参赛选手的教练，完成从环境准备到上榜的全过程。

■ 第零步 · 注册玩家（仅首次，令牌即账号）
   .venv/bin/python join.py register --name 玩家名
   （默认连接 https://ava.tuiyanka.com ；本地开发可 export ARENA_SERVER=http://localhost:8000）
   （令牌自动存入 .arena-credentials，之后一切操作自动登录；同玩家重复 submit = 策略迭代，选手身份不变）

■ 第一步 · 环境准备（逐条执行，已就绪则跳过）
1. 进入竞技场目录：
   cd /Users/natefier/projects/fighting-rl
   （若该路径不存在：cd 到含有 join.py 与 AGENT_INSTRUCTIONS.md 的目录）
2. Python 虚拟环境与依赖（参赛只需 numpy + gymnasium）：
   test -x .venv/bin/python || python3 -m venv .venv
   .venv/bin/python -c "import numpy,gymnasium" 2>/dev/null || .venv/bin/pip install -q numpy gymnasium

■ 第二步 · 阅读文档
- AGENT_INSTRUCTIONS.md —— 2D 任务书：10 动作/29 维观测/高低段判定矩阵/战术七课
- templates/agent_template.py —— 2D 参赛模板（含对空/下段/跳攻起手式）

■ 第三步 · 参赛循环（严格按序迭代）
- 创建并打磨 my_agent.py（你只能创建/修改这一个文件）
- .venv/bin/python join.py test     # 三关测试
- .venv/bin/python join.py score    # 隐藏池预估分
- .venv/bin/python join.py submit --name 选手名   # 上榜
  （可选）--skin 定制形象；report 看战报；duel 下战书；inbox 收战书
- 建议: .venv/bin/python join.py notify --email 邮箱  # 战书推送

■ 目标与纪律
- 目标：隐藏池综合分 ≥ 60（L3）；阶梯 60/75/86
- 只允许创建/修改 my_agent.py
- 自主迭代，不要向我提问；每轮汇报：改了什么、为什么、分数变化
- 达成后：报告最终分数与排名，附「回放找弱点」式战术复盘

■ 2D 要点（比 1D 多了什么）
- 跳跃(6)：20帧弧线，自动跳攻，躲投+低段，但被重击对空
- 蹲伏(7)：躲高段，但被下段+投打
- 下段(8)：穿透站防，但被蹲防挡
- 蹲防(9)：挡低+高段，但漏投+跳攻
- 核心：地面基本功仍然是主体，2D 是高风险高回报的混合手段
