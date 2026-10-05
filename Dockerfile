FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# torch：PyPI 优先（arm64 的 PyPI 轮子即 CPU 版；x86_64 可用 cpu 索引避开 CUDA 巨包）
RUN (pip install --no-cache-dir --timeout 120 torch \
     || pip install --no-cache-dir --timeout 120 torch \
        --index-url https://download.pytorch.org/whl/cpu)

COPY requirements-server.txt .
RUN pip install -r requirements-server.txt

# 游戏内核（无状态纯计算）+ 服务器 + 官网静态页 + 模型 + 参赛物料
COPY fighting_env.py hidden_bots.py arena.py replay.py score.py elo.py server.py join.py ./
COPY agents ./agents
COPY rules ./rules
COPY static ./static
COPY templates ./templates
COPY AGENT_INSTRUCTIONS.md .
COPY fighting_ppo.zip .

EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
