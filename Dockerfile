FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# torch CPU 优先（避免拉 CUDA 巨型镜像；aarch64 无 cpu 轮子时回退 PyPI）
RUN (pip install torch --index-url https://download.pytorch.org/whl/cpu \
     || pip install torch)

COPY requirements-server.txt .
RUN pip install -r requirements-server.txt

# 游戏内核（无状态纯计算）+ 服务器 + 官网静态页 + 模型
COPY fighting_env.py hidden_bots.py arena.py replay.py score.py elo.py server.py ./
COPY agents ./agents
COPY rules ./rules
COPY static ./static
COPY fighting_ppo.zip .

EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
