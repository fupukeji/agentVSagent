FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# 镜像源参数（国内服务器构建时传入；默认原源，不影响本地/CI）
ARG PIP_INDEX=https://pypi.org/simple
ARG TORCH_INDEX=""
ENV PIP_INDEX_URL=${PIP_INDEX}

# torch：优先指定源（如 CPU 轮子索引）；否则 PyPI（arm64 即 CPU 版）→ cpu 索引兑底
RUN if [ -n "$TORCH_INDEX" ]; then \
        pip install --timeout 120 torch --index-url "$TORCH_INDEX"; \
    else \
        (pip install --timeout 120 torch \
         || pip install --timeout 120 torch \
            --index-url https://download.pytorch.org/whl/cpu); \
    fi

COPY requirements-server.txt .
RUN pip install --timeout 120 -r requirements-server.txt

# 游戏内核（无状态纯计算）+ 服务器 + 官网静态页 + 模型 + 参赛物料
COPY fighting_env.py hidden_bots.py arena.py replay.py score.py elo.py server.py join.py ./
COPY agents ./agents
COPY rules ./rules
COPY static ./static
COPY templates ./templates
COPY AGENT_INSTRUCTIONS.md BOOTSTRAP_PROMPT.md .
COPY fighting_ppo.zip .

EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
