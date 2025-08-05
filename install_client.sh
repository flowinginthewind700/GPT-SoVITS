#!/bin/bash

# GPT-SoVITS 客户端SDK安装脚本

set -e

echo "📦 GPT-SoVITS 客户端SDK安装脚本"
echo "================================"

# 检查Python是否安装
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3未安装，请先安装Python3"
    exit 1
fi

# 检查pip是否安装
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3未安装，请先安装pip3"
    exit 1
fi

echo "🔧 安装客户端SDK依赖..."

# 配置pip使用国内源
echo "🔧 配置pip使用国内源..."
pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip3 config set global.trusted-host pypi.tuna.tsinghua.edu.cn

# 安装基础依赖（使用国内源）
echo "📦 安装基础依赖（使用国内源）..."
pip3 install requests>=2.28.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install urllib3>=1.26.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install typing-extensions>=4.0.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "✅ 基础依赖安装完成"

# 检查是否在开发模式下安装
if [ "$1" = "--dev" ]; then
    echo "🔧 开发模式安装..."
    cd gpt_sovits_client
    pip3 install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
    echo "✅ 开发模式安装完成"
else
    echo "📦 生产模式安装..."
    cd gpt_sovits_client
    pip3 install . -i https://pypi.tuna.tsinghua.edu.cn/simple
    echo "✅ 生产模式安装完成"
fi

echo ""
echo "🎉 客户端SDK安装完成！"
echo "================================"
echo "📝 使用示例："
echo ""
echo "1. 基本使用："
echo "   python3 gpt_sovits_client/examples/basic_usage.py"
echo ""
echo "2. 高级使用："
echo "   python3 gpt_sovits_client/examples/advanced_usage.py"
echo ""
echo "3. 测试SDK："
echo "   python3 test_client_sdk.py"
echo ""
echo "4. 在Python中使用："
echo "   from gpt_sovits_client import GPTSoVITSClient, LanguageType"
echo "   client = GPTSoVITSClient()"
echo ""
echo "📚 更多信息请查看："
echo "   gpt_sovits_client/README.md" 