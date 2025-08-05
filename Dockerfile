FROM continuumio/miniconda3:latest

# 设置工作目录
WORKDIR /app

# 使用默认conda源（不配置国内源）

# 配置apt使用国内源
COPY sources.list /etc/apt/sources.list

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    ffmpeg \
    libsndfile1 \
    libportaudio2 \
    portaudio19-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 创建conda环境
RUN conda create -n GPTSoVits python=3.10 -y

# 激活conda环境
SHELL ["conda", "run", "-n", "GPTSoVits", "/bin/bash", "-c"]

# 使用默认pip源（不配置国内源）

# 复制requirements文件
COPY requirements.txt .
COPY extra-req.txt .

# 安装Python依赖（使用默认源）
RUN pip install -r extra-req.txt --no-deps
RUN pip install -r requirements.txt
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch==2.3.1 torchaudio==2.3.1 torchvision==0.18.1
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch-tensorrt==2.3.0
# 安装ffmpeg（使用默认源）
RUN conda install ffmpeg -y

# 复制应用代码
COPY . .

# 创建模型目录
RUN mkdir -p GPT_SoVITS/pretrained_models

# 设置环境变量
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=""
ENV DEVICE="cpu"

# 暴露端口
EXPOSE 9880

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:9880/docs || exit 1

# 启动命令（使用conda环境）
CMD ["conda", "run", "-n", "GPTSoVits", "python", "api_v2.py"]