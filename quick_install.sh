#!/bin/bash

# GPT-SoVITS 快速安装脚本（使用国内源）

set -e

echo "🚀 GPT-SoVITS 快速安装脚本（使用国内源）"
echo "=========================================="

# 检查conda是否安装
if ! command -v conda &> /dev/null; then
    echo "❌ Conda未安装，请先安装Miniconda或Anaconda"
    echo "   下载地址: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# 检查GPTSoVits环境是否存在
if ! conda env list | grep -q "GPTSoVits"; then
    echo "🔧 创建GPTSoVits环境..."
    conda create -n GPTSoVits python=3.10 -y
    echo "✅ GPTSoVits环境创建成功"
fi

# 激活conda环境
echo "🔧 激活GPTSoVits环境..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate GPTSoVits

# 配置国内源
echo "🔧 配置国内源..."

# 配置apt源（如果在Ubuntu/Debian系统上）
if command -v apt-get &> /dev/null; then
    echo "📝 配置apt源..."
    sudo cp /etc/apt/sources.list /etc/apt/sources.list.backup
    sudo sed -i 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list
    sudo sed -i 's/security.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list
    echo "✅ apt源配置完成"
fi

# 使用默认conda源（不配置国内源）
echo "📝 使用默认conda源..."

# 使用默认pip源（不配置国内源）
echo "📝 使用默认pip源..."

# 检查依赖是否安装
echo "📦 检查依赖..."
if ! python -c "import torch" 2>/dev/null; then
    echo "⚠️  依赖未完全安装，正在安装..."
    
    echo "📦 安装extra-req.txt依赖..."
    pip install -r extra-req.txt --no-deps
    
    echo "📦 安装requirements.txt依赖..."
    pip install -r requirements.txt
    
    echo "📦 安装ffmpeg..."
    conda install ffmpeg -y
    
    echo "✅ 依赖安装完成"
else
    echo "✅ 依赖已安装"
fi

# 检查模型文件
echo "🔍 检查模型文件..."
if [ ! -d "GPT_SoVITS/pretrained_models" ] || [ -z "$(ls -A GPT_SoVITS/pretrained_models 2>/dev/null)" ]; then
    echo "⚠️  模型文件不存在，请下载GPT-SoVITS模型文件到GPT_SoVITS/pretrained_models目录"
    echo "   模型文件结构应该如下："
    echo "   GPT_SoVITS/pretrained_models/"
    echo "   ├── s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
    echo "   ├── s2G2333k.pth"
    echo "   ├── chinese-hubert-base/"
    echo "   ├── chinese-roberta-wwm-ext-large/"
    echo "   └── ..."
    echo ""
    echo "   下载地址: https://huggingface.co/lj1995/GPT-SoVITS"
else
    echo "✅ 模型文件已存在"
fi

# 安装客户端SDK
echo "📦 安装客户端SDK..."
if [ -d "gpt_sovits_client" ]; then
    cd gpt_sovits_client
    pip install -e .
    cd ..
    echo "✅ 客户端SDK安装完成"
else
    echo "⚠️  客户端SDK目录不存在"
fi

echo ""
echo "🎉 安装完成！"
echo "=========================================="
echo "📝 下一步操作："
echo ""
echo "1. 启动服务端："
echo "   ./start_local_server.sh"
echo ""
echo "2. 测试功能："
echo "   python test_client_sdk.py"
echo ""
echo "3. 运行示例："
echo "   python gpt_sovits_client/examples/basic_usage.py"
echo ""
echo "4. 查看文档："
echo "   cat USAGE_GUIDE.md"
echo ""
echo "🔧 环境信息："
echo "   Python版本: $(python --version)"
echo "   Conda环境: GPTSoVits"
echo "   PyTorch版本: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo '未安装')"
echo ""
echo "📚 更多信息请查看 USAGE_GUIDE.md" 